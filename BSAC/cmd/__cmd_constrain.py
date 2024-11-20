
## Copyright(c) 2023 / 2024 Yoann Robin
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
import tempfile
import itertools as itt
import gc


from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams

import numpy  as np
import xarray as xr
import zxarray as zr
import statsmodels.gam.api as smg
import scipy.stats as sc

from ..__sys     import coords_samples
from ..stats.__tools import nslawid_to_class
from ..stats.__rvs   import rvs_multivariate_normal
from ..stats.__rvs   import robust_covariance
from ..stats.__constraint import gaussian_conditionning

##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############


def zgaussian_conditionning( *args , A = None ):##{{{
	
	ihpar = args[0]
	ihcov = args[1]
	lXo   = args[2:]
	ssp   = ihpar.shape[:-1]
	
	ohpar = np.zeros_like(ihpar) + np.nan
	ohcov = np.zeros_like(ihcov) + np.nan
	
	for idx in itt.product(*[range(s) for s in ssp]):
		idx1d = idx + tuple([slice(None) for _ in range(1)])
		idx2d = idx + tuple([slice(None) for _ in range(2)])
		ih    = ihpar[idx1d]
		ic    = ihcov[idx2d]
		iargs = [ih,ic] + [ Xo[idx1d] for Xo in lXo ]
		
		oh,oc = gaussian_conditionning( *iargs , A = A )
		ohpar[idx1d] = oh
		ohcov[idx2d] = oc
	
	return ohpar,ohcov
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
	for i,inp in enumerate(bsacParams.input):
		
		## Name and file
		name,ifile = inp.split(",")
		if not name in clim.namesX:
			raise ValueError( f"Unknown variable {name}" )
		
		## Open data
		idata = xr.open_dataset(ifile)
		
		## Time axis
		time = idata.time.dt.year.values
		time = xr.DataArray( time , dims = [f"time{i}"] , coords = [time] )
		
		## Init zarr file
		dims   = (f"time{i}",) + d_spatial
		coords = [time] + [c_spatial[d] for d in d_spatial]
		shape  = [c.size for c in coords]
		Xo     = zr.ZXArray( data = np.nan , dims = dims , coords = coords )
		
		## Now copy data
		ndimXo = len([s for s in idata[name].shape if s > 1])
		bias   = xr.DataArray( np.nan , dims = d_spatial , coords = c_spatial )
		if bias.ndim == 0:
			bias = xr.DataArray( [np.nan] )
		if ndimXo == 1:
			xXo  = idata[name]
			anom = float(xXo.sel( time = slice(*[str(y) for y in clim.bper]) ).mean( dim = "time" ))
			xXo  = xXo.values.squeeze() - anom
			bias[:] = anom
			for idx in itt.product(*[range(s) for s in clim.s_spatial]):
				Xo[(slice(None),) + idx] = xXo
		else:
			raise NotImplementedError
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
	nper = len(clim.dpers)
	
	## Create projection matrix A
	proj = []
	for iname,name in enumerate(zXo):
		
		##
		Xo = zXo[name]
		
		## Build the design_matrix for projection
		design_ = []
		for nameX in clim.namesX:
			if nameX == name:
				design_ = design_ + [spl for _ in range(nper)] + [nper * lin]
			else:
				design_ = design_ + [np.zeros_like(spl) for _ in range(nper)] + [np.zeros_like(lin)]
		design_ = design_ + [np.zeros( (time.size,clim.sizeY) )]
		design_ = np.hstack(design_)
		
		
		T = xr.DataArray( np.identity(design_.shape[0]) , dims = ["timeA","timeB"] , coords = [time,time] ).loc[Xo[f"time{iname}"],time].values
		A = T @ design_ / nper
		
		proj.append(A)
	
	
	A = np.vstack(proj)
	
	## Build apply arguments
	hpar_names       = clim.hpar_names
	c_spatial        = clim.c_spatial
	d_spatial        = clim.d_spatial
	args             = [clim.hpar,clim.hcov] + [ zXo[name] for name in zXo ]
	output_dims      = [ ("hpar",) + d_spatial   , ("hpar0","hpar1") + d_spatial ]
	output_coords    = [ [hpar_names] + [ c_spatial[d] for d in d_spatial ] , [hpar_names,hpar_names] + [ c_spatial[d] for d in d_spatial ] ]
	output_dtypes    = [float,float]
	dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] ] + [ [f"time{i}"] for i in range(len(zXo)) ],
	                     "output_core_dims" : [ ["hpar"] , ["hpar0","hpar1"] ],
	                     "kwargs" : { "A" : A } ,
	                     "dask" : "parallelized",
	                     "output_dtypes"  : [clim.hpar.dtype,clim.hcov.dtype]
	                    }
	
	##
	hpar,hcov = zr.apply_ufunc( zgaussian_conditionning , *args ,
	                            bdims = d_spatial , max_mem = bsacParams.total_memory,
	                            output_coords = output_coords,
	                            output_dims   = output_dims,
	                            output_dtypes = output_dtypes,
	                            dask_kwargs   = dask_kwargs )
	
	## Save
	clim.hpar = hpar
	clim.hcov = hcov
	bsacParams.clim = clim
##}}}


def _constrain_Y_parallel( hpar , hcov , Yo , timeYo , clim , size , size_chain ):##{{{
	
	## Build output
	ohpar = hpar.copy() + np.nan
	ohcov = hcov.copy() + np.nan
	
	##
	if np.any(~np.isfinite(hpar)):
		return ohpar,ohcov
	
	
	## 
	_,_,designF_,_ = clim.build_design_XFC()
	hpar_coords = designF_.hpar.values.tolist()
	name        = clim.namesX[-1]
	
	## Build the prior
	prior_hpar = hpar[-clim.sizeY:]
	prior_hcov = hcov[-clim.sizeY:,-clim.sizeY:]
	prior      = sc.multivariate_normal( mean = prior_hpar , cov = prior_hcov , allow_singular = True )
	
	##
	total_draw = 0
	
	## While loop
	samples = coords_samples(size)
	Yf      = Yo
	nslaw   = clim._nslaw_class()
	draw    = []
	for _ in range(10):
		
		## Draw hpars
		hpars = xr.DataArray( rvs_multivariate_normal( size , hpar , hcov ) , dims = ["sample","hpar"] , coords = [samples,clim.hpar_names] )
		mcmc  = np.zeros( (hpars.hpar.size,size_chain) )
		
		## Build covariates
		XF = xr.concat(
		        [
		         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designF_.sel( time = timeYo )
		         for per in clim.dpers
		        ],
		        dim = "period"
		    ).mean( dim = "period" )#.values
		
		## and MCMC on all samples
		for s in samples:
			## MCMC
			chain = nslaw.fit_bayesian( Yf , XF.loc[s,:].values , prior = prior , n_mcmc_drawn = size_chain , tmp = bsacParams.tmp_stan )
			
			## Build output
			mcmc[:-clim.sizeY,:] = hpars.loc[s,:][:-clim.sizeY].values.reshape(-1,1)
			mcmc[-clim.sizeY:,:] = chain.T
			valid   = np.isfinite(mcmc).all( axis = 0 )
			n_valid = valid.sum()
			if n_valid > 0:
				draw.append( mcmc[:,valid].reshape(-1,n_valid).copy() )
			total_draw += n_valid
			if total_draw > size * size_chain:
				break
		if total_draw > size * size_chain:
			break
	
	## And merge all drawn
	draw  = np.hstack(draw)
	
	## Compute final parameters
	ohpar = draw.mean(1)
	ohcov = robust_covariance( draw.T , index = slice(-clim.sizeY,None,1) )
	
	## Clean memory
	del hpars
	gc.collect()
	
	return ohpar,ohcov
##}}}

## run_bsac_cmd_constrain_Y ##{{{
@log_start_end(logger)
def run_bsac_cmd_constrain_Y():
	
	##
	clim = bsacParams.clim
	size = bsacParams.n_samples
	size_chain = int(bsacParams.config.get("size-chain", 10))
	
	## Load observations
	name,ifile = bsacParams.input[0].split(",")
	Yo = xr.open_dataset(ifile)[name]
	Yo = xr.DataArray( Yo.values , dims = Yo.dims , coords = [Yo.time.dt.year.values] + [Yo.coords[d] for d in Yo.dims[1:]] )
	
	## Bias
	bias = Yo.sel( time = slice(*[str(y) for y in clim.bper]) ).mean( dim = "time" )
	Yo   = Yo - bias
	try:
		bias = float(bias)
	except:
		pass
	clim._bias[clim.names[-1]] = bias
	
	## Extract parameters
	d_spatial = clim.d_spatial
	c_spatial = clim.c_spatial
	chpar     = clim.hpar_names
	ihpar     = xr.DataArray( clim.mean_.copy() , dims = ("hpar",)         + d_spatial , coords = { **{ "hpar"  : chpar }                   , **c_spatial } )
	ihcov     = xr.DataArray( clim.cov_.copy()  , dims = ("hpar0","hpar1") + d_spatial , coords = { **{ "hpar0" : chpar , "hpar1" : chpar } , **c_spatial } )
	ohpar     = xr.zeros_like(ihpar) + np.nan
	ohcov     = xr.zeros_like(ihcov) + np.nan
	
	## Init stan
	logger.info("Stan compilation...")
	clim._nslaw_class.init_stan( tmp = bsacParams.tmp_stan , force_compile = True )
	logger.info("Stan compilation. Done.")
	
	## Loop on spatial variables
	jump = max( 0 , int( np.power( bsacParams.n_jobs , 1. / max( len(clim.s_spatial) , 1 ) ) ) ) + 1
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
		isfin = np.all( np.isfinite(sYo) , axis = 0 ).values
		logger.info( f" * {idx} + {jump} / {clim.s_spatial} ({round( 100 * isfin.sum() / isfin.size , 2 )}%)" )
		
		##
		h,c = xr.apply_ufunc( _constrain_Y_parallel , shpar , shcov , sYo ,
		                    input_core_dims  = [["hpar"],["hpar0","hpar1"],["time"]],
		                    output_core_dims = [["hpar"],["hpar0","hpar1"]],
		                    output_dtypes    = [shpar.dtype,shcov.dtype],
		                    vectorize        = True,
		                    dask             = "parallelized",
		                    kwargs           = { "timeYo" : Yo.time.values , "clim" : clim , "size" : size , "size_chain" : size_chain }
		                    )
		h = h.transpose( *shpar.dims ).compute()
		c = c.transpose( *shcov.dims ).compute()
		
		ohpar[idx1d] = h.values
		ohcov[idx2d] = c.values
		
		## Clean memory
		del h
		del c
		gc.collect()
	
	## And save
	clim.mean_ = ohpar.values
	clim.cov_  = ohcov.values
	bsacParams.clim = clim
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

