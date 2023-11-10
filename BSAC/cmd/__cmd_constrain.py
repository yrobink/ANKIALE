
## Copyright(c) 2023 Yoann Robin
## 
## This file is part of BSAC.
## 
## BSAC is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## BSAC is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with BSAC.  If not, see <https://www.gnu.org/licenses/>.

#############
## Imports ##
#############

import os
import logging
import itertools as itt
from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams

import numpy  as np
import xarray as xr
import statsmodels.gam.api as smg

from ..__XZarr import XZarr
from ..__XZarr import random_zfile


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

#TODO Bias CX

def _constrain_X_parallel( hpar , cov , Xo , A ):##{{{
	
	## Variance of obs
	cov_o = np.identity(Xo.size) * float(np.std(Xo))**2
	
	## gaussian conditionning theorem
	K0 = A @ cov
	K1 = ( cov @ A.T ) @ np.linalg.inv( K0 @ A.T + cov_o )
	hpar = hpar + K1 @ ( Xo.squeeze() - A @ hpar )
	cov  = cov  - K1 @ K0
	
	return hpar,cov
##}}}

## run_bsac_cmd_constrain_X ##{{{
@log_start_end(logger)
def run_bsac_cmd_constrain_X():
	
	## Parameters
	clim = bsacParams.clim
	d_spatial = clim.d_spatial
	c_spatial = clim.c_spatial
	
	## Load observations
	zXo = {}
	for inp in bsacParams.input:
		
		## Name and file
		name,ifile = inp.split(",")
		if not name in clim.namesX:
			raise ValueError( f"Unknown variable {name}" )
		
		## Open data
		idata = xr.open_dataset(ifile)
		
		## Time axis
		time = idata.time.dt.year.values
		time = xr.DataArray( time , dims = ["time"] , coords = [time] )
		
		## Init zarr file
		dims   = ("time",) + d_spatial
		coords = [time] + [c_spatial[d] for d in d_spatial]
		shape  = [c.size for c in coords]
		Xo     = XZarr.from_value( np.nan , dims = dims , coords = coords , shape = shape , zfile = random_zfile( os.path.join( bsacParams.tmp , f"Xo_{name}" ) ) )
		
		## Now copy data
		ndimXo = len([s for s in idata[name].shape if s > 1])
		bias   = xr.DataArray( np.nan , dims = d_spatial , coords = c_spatial )
		if ndimXo == 1:
			
			xXo  = idata[name]
			anom = float(xXo.sel( time = slice(*[str(y) for y in clim.bper]) ).mean( dim = "time" ))
			xXo  = xXo.values.squeeze() - anom
			bias[:] = anom
			for idx in itt.product(*[range(s) for s in clim.s_spatial]):
				Xo.set_orthogonal_selection( (slice(None),) + idx , xXo )
		else:
			for idx in itt.product(*[range(s) for s in clim.s_spatial]):
				xXo  = idata[name][(slice(None),)+idx]
				anom = float(xXo.sel( time = slice(*[str(y) for y in clim.bper]) ).mean( dim = "time" ))
				xXo  = xXo.values.squeeze() - anom
				Xo.set_orthogonal_selection( (slice(None),) + idx , xXo )
				bias[idx] = anom
		clim._birs[name] = bias
		
		## Store zarr file
		zXo[name] = Xo
	
	##
	time = clim.time
	spl  = smg.BSplines( time , df = clim.GAM_dof + 1 - 1 , degree = clim.GAM_degree , include_intercept = False ).basis
	lin  = np.stack( [np.ones(time.size),clim.XN.loc[time].values] ).T.copy()
	
	I = np.identity(clim.size)
	for name in clim.names:
		I[clim.isel("lin",name),clim.isel("lin",name)] *= len(clim.dpers)
	
	d_spatial = clim.d_spatial
	c_spatial = clim.c_spatial
	chpar     = clim.hpar_names
	hpar_CX = xr.DataArray( clim.mean_.copy() , dims = ("hpar",)         + d_spatial , coords = { **{ "hpar" : chpar } , **c_spatial } )
	cov_CX  = xr.DataArray( clim.cov_.copy()  , dims = ("hpar0","hpar1") + d_spatial , coords = { **{ "hpar0" : chpar , "hpar1" : chpar } , **c_spatial } )
	
	for name in zXo:
		
		##
		Xo = zXo[name]
		
		## Build the design_matrix for projection
		design_ = []
		for nameX in clim.namesX:
			if nameX == name:
				design_ = design_ + [spl for _ in range(len(clim.dpers))]
			else:
				design_ = design_ + [np.zeros_like(spl) for _ in range(len(clim.dpers))]
			design_ = design_ + [lin]
		design_ = design_ + [np.zeros( (time.size,clim.sizeY) )]
		design_ = np.hstack(design_)
		
		T = xr.DataArray( np.identity(design_.shape[0]) , dims = ["time0","time1"] , coords = [time,time] ).loc[Xo.coords[0],time].values
		A = T @ design_ @ I / len(clim.dpers)
		
		## Loop on spatial
		jump = max( 0 , int( np.power( bsacParams.n_jobs , 1. / len(clim.s_spatial) ) ) ) + 1
		for idx in itt.product(*[range(0,s,jump) for s in clim.s_spatial]):
			
			##
			s_idx = tuple([slice(s,s+jump,1) for s in idx])
			idx1d = (slice(None),) + s_idx
			idx2d = (slice(None),slice(None)) + s_idx
			
			## Extract data
			hpar = hpar_CX[idx1d].chunk( { d : 1 for d in hpar_CX.dims[1:] } )
			cov  =  cov_CX[idx2d].chunk( { d : 1 for d in hpar_CX.dims[1:] } )
			xXo  = Xo.get_orthogonal_selection( (slice(None),) + s_idx ).chunk( { d : 1 for d in hpar_CX.dims[1:] } )
			
			h,c = xr.apply_ufunc( _constrain_X_parallel , hpar , cov , xXo ,
			                    input_core_dims  = [["hpar"],["hpar0","hpar1"],["time"]],
			                    output_core_dims = [["hpar"],["hpar0","hpar1"]],
			                    output_dtypes    = [hpar.dtype,cov.dtype],
			                    vectorize        = True,
			                    dask             = "parallelized",
			                    kwargs           = { "A" : A }
			                    )
			h = h.transpose( *hpar.dims ).compute()
			c = c.transpose(  *cov.dims ).compute()
			
			hpar_CX[idx1d] = h.values
			cov_CX[idx2d]  = c.values
	
	clim.mean_ = hpar_CX.values
	clim.cov_  = cov_CX.values
	bsacParams.clim = clim
##}}}

## run_bsac_cmd_constrain_Y ##{{{
@log_start_end(logger)
def run_bsac_cmd_constrain_Y():
	raise NotImplementedError
##}}}

## run_bsac_cmd_constrain ##{{{
@log_start_end(logger)
def run_bsac_cmd_constrain():
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the fit command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["X","Y"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the fit command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "X":
		run_bsac_cmd_constrain_X()
	if bsacParams.arg[0] == "Y":
		run_bsac_cmd_constrain_Y()
##}}}

