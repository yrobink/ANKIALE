
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
import gc
from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams

import numpy  as np
import xarray as xr
import statsmodels.gam.api as smg
import scipy.stats as sc

from ..__XZarr import XZarr
from ..__XZarr import random_zfile

from ..__sys     import coords_samples
from ..stats.__tools import nslawid_to_class
from ..stats.__rvs   import rvs_multivariate_normal


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

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
		clim._bias[name] = bias
		
		## Store zarr file
		zXo[name] = Xo
	
	##
	time = clim.time
	spl  = smg.BSplines( time , df = clim.GAM_dof + 1 - 1 , degree = clim.GAM_degree , include_intercept = False ).basis
	lin  = np.stack( [np.ones(time.size),clim.XN.loc[time].values] ).T.copy()
	
	I = np.identity(clim.size)
	for name in clim.namesX:
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


def _constrain_Y_parallel( hpar , hcov , Yo , timeYo , clim , size , n_mcmc_min , n_mcmc_max ):##{{{
	
	## Build output
	ohpar = hpar.copy() + np.nan
	ohcov = hcov.copy() + np.nan
	
	##
	if np.any(~np.isfinite(hpar)):
		return ohpar,ohcov
	
	## Draw hpars
	samples = coords_samples(size)
	hpars   = xr.DataArray( rvs_multivariate_normal( size , hpar , hcov ) , dims = ["sample","hpar"] , coords = [samples,clim.hpar_names] )
	
	## Draw XF
	spl,lin,designF_,_ = clim.build_design_XFC()
	hpar_coords = designF_.hpar.values.tolist()
	name        = clim.namesX[-1]
	XF = xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designF_
	         for per in clim.dpers
	        ],
	        dim = "period"
	    ).assign_coords( { "period" : clim.dpers } ).transpose("sample","period","time")
	
	## Build the prior
	prior_hpar = hpar[-clim.sizeY:]
	prior_hcov = hcov[-clim.sizeY:,-clim.sizeY:]
	prior      = sc.multivariate_normal( mean = prior_hpar , cov = prior_hcov , allow_singular = True )
	
	##
	Yf     = Yo
	nslaw  = clim._nslaw_class()
	for s in samples:
		
		## Extract
		Xf = XF.loc[s,:,timeYo].mean( dim = "period" ).values
		
		## MCMC
		n_mcmc_drawn = np.random.choice( range(n_mcmc_min,n_mcmc_max) , replace = False )
		nslaw.fit_bayesian( Yf , Xf , prior = prior , n_mcmc_drawn = n_mcmc_drawn )
		hpars[:,-clim.sizeY:].loc[s,:] = nslaw.coef_.copy()
	
	del XF
	gc.collect()
	
	##
	ohpar  = np.mean( hpars.values , axis = 0 )
	ohcov  = np.cov(  hpars.values.T )
	
	return ohpar,ohcov
##}}}

## run_bsac_cmd_constrain_Y ##{{{
@log_start_end(logger)
def run_bsac_cmd_constrain_Y():
	
	##
	clim = bsacParams.clim
	size = bsacParams.n_samples
	n_mcmc_min = int(bsacParams.config.get("n-mcmc-min", 5000))
	n_mcmc_max = int(bsacParams.config.get("n-mcmc-max",10000))
	
	## Load observations
	name,ifile = bsacParams.input[0].split(",")
	Yo = xr.open_dataset(ifile)[name]
	Yo = xr.DataArray( Yo.values , dims = Yo.dims , coords = [Yo.time.dt.year.values] + [Yo.coords[d] for d in Yo.dims[1:]] )
	
	## Bias
	bias = Yo.sel( time = slice(*[str(y) for y in clim.bper]) ).mean( dim = "time" )
	Yo   = Yo - bias
	clim._bias[name] = bias
	
	## Extract parameters
	d_spatial = clim.d_spatial
	c_spatial = clim.c_spatial
	chpar     = clim.hpar_names
	ihpar     = xr.DataArray( clim.mean_.copy() , dims = ("hpar",)         + d_spatial , coords = { **{ "hpar"  : chpar }                   , **c_spatial } )
	ihcov     = xr.DataArray( clim.cov_.copy()  , dims = ("hpar0","hpar1") + d_spatial , coords = { **{ "hpar0" : chpar , "hpar1" : chpar } , **c_spatial } )
	ohpar     = xr.zeros_like(ihpar) + np.nan
	ohcov     = xr.zeros_like(ihcov) + np.nan
	
	## Loop on spatial variables
	jump = max( 0 , int( np.power( bsacParams.n_jobs , 1. / len(clim.s_spatial) ) ) ) + 1
	for idx in itt.product(*[range(0,s,jump) for s in clim.s_spatial]):
		
		##
		s_idx = tuple([slice(s,s+jump,1) for s in idx])
		idx1d = (slice(None),) + s_idx
		idx2d = (slice(None),slice(None)) + s_idx
		
		## Extract data
		shpar = ihpar[idx1d].chunk( { d : 1 for d in ihpar.dims[1:] } )
		shcov = ihcov[idx2d].chunk( { d : 1 for d in ihpar.dims[1:] } )
		sYo   = Yo[(slice(None),) + s_idx].chunk({ d : 1 for d in ihpar.dims[1:] })
		
		##
		h,c = xr.apply_ufunc( _constrain_Y_parallel , shpar , shcov , sYo ,
		                    input_core_dims  = [["hpar"],["hpar0","hpar1"],["time"]],
		                    output_core_dims = [["hpar"],["hpar0","hpar1"]],
		                    output_dtypes    = [shpar.dtype,shcov.dtype],
		                    vectorize        = True,
		                    dask             = "parallelized",
		                    kwargs           = { "timeYo" : Yo.time.values , "clim" : clim , "size" : size , "n_mcmc_min" : n_mcmc_min , "n_mcmc_max" : n_mcmc_max }
		                    )
		h = h.transpose( *shpar.dims ).compute()
		c = c.transpose( *shcov.dims ).compute()
		
		ohpar[idx1d] = h.values
		ohcov[idx2d] = c.values
	
	## And save
	clim.mean_ = ohpar.values
	clim.cov_  = ohcov.values
	
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

