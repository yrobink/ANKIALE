
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
import sys
import itertools as itt
import datetime as dt
import logging
import warnings
from ..__logs import LINE
from ..__logs import log_start_end
from ..__release import version

from ..__BSACParams import bsacParams

from ..__climatology import Climatology

from ..__XZarr import XZarr
from ..__XZarr import random_zfile

from ..__sys import SizeOf
from ..__sys import coords_samples

from ..stats.__rvs import rvs_climatology
from ..stats.__rvs import rvs_multivariate_normal


import numpy  as np
import xarray as xr
import netCDF4
import cftime


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

def _attribute_event_parallel( hpar , hcov , Yo , bias , t_attr , clim , side , mode , n_samples , ci ):##{{{
	
	## Extract parameters from clim
	_,_,designF_,designC_ = clim.build_design_XFC()
	nslaw_class = clim._nslaw_class
	nslaw   = clim._nslaw_class()
	n_pers  = len(clim.dpers)
	n_times = len(clim.time)
	
	## Init output
	n_modes = n_samples if mode == "sample" else 3
	
	## If nan, return
	if not np.isfinite(hpar).all():
		return tuple([ np.zeros((n_modes,n_pers,n_times)) for _ in range(8) ])
	
	## Draw parameters
	hpars = xr.DataArray( rvs_multivariate_normal( n_samples , hpar , hcov ) , dims = ["sample","hpar"] , coords = [range(n_samples),clim.hpar_names] )
	
	## Design
	_,_,designF_,designC_ = clim.build_design_XFC()
	hpar_coords = designF_.hpar.values.tolist()
	name        = clim.namesX[-1]
	
	## Build XFC
	XF = xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designF_
	         for per in clim.dpers
	        ],
	        dim = "period"
	    ).assign_coords( period = clim.dpers )
	XC = xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designC_
	         for per in clim.dpers
	        ],
	        dim = "period"
	    ).assign_coords( period = clim.dpers )
	
	## Build params
	kwargsF = nslaw.draw_params( XF , hpars )
	kwargsC = nslaw.draw_params( XC , hpars )
	
	## Attribution
	xYo = np.zeros( (n_samples,n_pers,n_times) ) + Yo
	pF  = nslaw.cdf_sf( xYo , side = side , **kwargsF )
	pC  = nslaw.cdf_sf( xYo , side = side , **kwargsC )
	
	## Remove 0 and 1
	e  = 10 * sys.float_info.epsilon
	pF = np.where( pF >     e , pF ,     e )
	pC = np.where( pC >     e , pC ,     e )
	pF = np.where( pF < 1 - e , pF , 1 - e )
	pC = np.where( pC < 1 - e , pC , 1 - e )
	
	it_attr = int(np.argwhere( clim.time == t_attr ).ravel())
	pf      = np.zeros_like(pF) + pF[:,:,it_attr].reshape((n_samples,n_pers,1))
	
	IF = nslaw.icdf_sf( pf , side = side , **kwargsF )
	IC = nslaw.icdf_sf( pf , side = side , **kwargsC )
	
	## Others variables
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		IF  = IF + bias
		IC  = IC + bias
		RF  = 1. / pF
		RC  = 1. / pC
		dI  = IF - IC
		PR  = pF / pC
	
	## Compute CI
	if mode == "quantile":
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			pF = np.quantile( pF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			pC = np.quantile( pC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			RF = np.quantile( RF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			RC = np.quantile( RC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			IF = np.quantile( IF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			IC = np.quantile( IC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			dI = np.quantile( dI , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			PR = np.quantile( PR , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
	
	return tuple([pF,pC,RF,RC,IF,IC,dI,PR])
##}}}

## run_bsac_cmd_attribute_event ##{{{
@log_start_end(logger)
def run_bsac_cmd_attribute_event():
	
	## Parameters
	clim      = bsacParams.clim
	time      = clim.time
	n_samples = bsacParams.n_samples
	tmp       = bsacParams.tmp
	side      = bsacParams.config.get("side","right")
	mode      = bsacParams.config.get("mode","sample")
	ci        = bsacParams.config.get("ci",0.05)
	t_attr    = int(bsacParams.config.get("time"))
	it_attr   = int(np.argwhere( time == t_attr ).ravel())
	nslaw     = clim._nslaw_class()
	sp_dims   = clim.d_spatial
	
	## Logs
	logger.info(  " * Configuration" )
	logger.info( f"   => mode: {mode}" )
	logger.info( f"   => side: {side}" )
	logger.info( f"   => ci  : {ci}" )
	logger.info( f"   => year: {t_attr}" )
	
	## Mode dimension
	if mode == "sample":
		modes = coords_samples( n_samples )
	elif mode == "quantile":
		modes = np.array(["QL","BE","QU"])
	else:
		raise ValueError( f"Invalid mode ({mode})" )
	n_modes = modes.size
	
	## Load observations
	Yo  = xr.open_dataset( bsacParams.input[0].split(",")[1] )[clim.names[-1]]
	
	## Select year
	if "time" in Yo.dims:
		Yo = Yo.assign_coords( time = Yo.time.dt.year )
		if Yo.time.size > 1:
			Yo = Yo.sel( time = int(t_attr) )
		else:
			Yo = Yo.sel( time = Yo.time[0] )
		Yo = Yo.drop_vars("time")
	
	## And remove bias
	Yo = Yo - clim.bias[clim.names[-1]]
	
	## Output
	time   = clim.time
	period = np.array(clim.dpers)
	dims   = [mode ] + ["time","period"] + list(clim.d_spatial)
	coords = [modes] + [ time , period ] + [ clim.c_spatial[d] for d in clim.c_spatial ]
	shape  = [len(c) for c in coords]
	out = {
	    "pF" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "pF" ) ) ),
	    "pC" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "pC" ) ) ),
	    "RF" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "RF" ) ) ),
	    "RC" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "RC" ) ) ),
	    "IF" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "IF" ) ) ),
	    "IC" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "IC" ) ) ),
	    "dI" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "dI" ) ) ),
	    "PR" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "PR" ) ) )
	}
	
	## Build a 'small' climatology
	## climatology must be pass at each threads, but duplication of mean, cov bias implies a memory leak
	climB = Climatology.init_from_file( bsacParams.load_clim )
	del climB._mean
	del climB._cov
	del climB._bias
	
	## Dask parameters
	dask_gufunc_kwargs = {}
	dask_gufunc_kwargs["output_sizes"] = { mode : modes.size , "period" : period.size , "time" : time.size }
	
	## Loop on spatial variables
	block = max( 0 , int( np.power( bsacParams.n_jobs , 1. / len(clim.s_spatial) ) ) ) + 1
	logger.info( " * Loop on spatial variables" )
	for idx in itt.product(*[range(0,s,block) for s in clim.s_spatial]):
		
		##
		s_idx = tuple([slice(s,s+block,1) for s in idx])
		f_idx = tuple( [slice(None) for _ in range(3)] ) + s_idx
		
		##
		sYo  = Yo[s_idx].chunk( { d : 1 for d in sp_dims } )
		bias = clim.bias[clim.names[-1]][s_idx]
		
		hpar = clim.xmean_[(slice(None),)+s_idx]
		hcov = clim.xcov_[(slice(None),slice(None))+s_idx]
		
		#
		res = xr.apply_ufunc( _attribute_event_parallel , hpar , hcov , sYo , bias ,
		                    input_core_dims    = [["hpar"],["hpar0","hpar1"],[],[]],
		                    output_core_dims   = [[mode,"period","time"] for _ in range(8)],
		                    output_dtypes      = [hpar.dtype for _ in range(8)],
		                    vectorize          = True,
		                    dask               = "parallelized",
		                    kwargs             = { "t_attr" : t_attr , "clim" : climB , "side" : side , "mode" : mode , "n_samples" : n_samples , "ci" : ci },
		                    dask_gufunc_kwargs = dask_gufunc_kwargs
		                    )
		
		res = [ r.transpose( *out["pF"].dims ).compute() for r in res ]
		for key,r in zip(out,res):
			out[key].set_orthogonal_selection( f_idx , r.values )
	
	## And save
	logger.info( " * Save in netcdf" )
	ofile = bsacParams.output
	with netCDF4.Dataset( ofile , "w" ) as ncf:
		
		## Define dimensions
		ncdims = {
		           mode : ncf.createDimension( mode , n_modes ),
		       "period" : ncf.createDimension( "period" , len(clim.dpers) ),
		       "time"   : ncf.createDimension(   "time" ),
		}
		spatial = ()
		if clim._spatial is not None:
			for d in clim._spatial:
				ncdims[d] = ncf.createDimension( d , clim._spatial[d].size )
			spatial = tuple([d for d in clim._spatial])
		
		## Define variables
		ncvars = {
		           mode : ncf.createVariable(     mode , str       , (mode,)     ),
		       "period" : ncf.createVariable( "period" , str       , ("period",) ),
		       "time"   : ncf.createVariable( "time"   , "float32" , ("time"  ,) )
		}
		ncvars[mode][:] = modes
		ncvars["period"][:] = period
		if clim._spatial is not None:
			for d in clim._spatial:
				ncvars[d] = ncf.createVariable( d , "double" , (d,) )
				ncvars[d][:] = np.array(clim._spatial[d]).ravel()
		
		## Fill time axis
		calendar = "standard"
		units    = "days since 1750-01-01 00:00"
		ncvars["time"][:]  = cftime.date2num( [cftime.DatetimeGregorian( int(y) , 1 , 1 ) for y in time] , units = units , calendar = calendar )
		ncvars["time"].setncattr( "standard_name" , "time"      )
		ncvars["time"].setncattr( "long_name"     , "time_axis" )
		ncvars["time"].setncattr( "units"         , units       )
		ncvars["time"].setncattr( "calendar"      , calendar    )
		ncvars["time"].setncattr( "axis"          , "T"         )
		
		## Variables
		for key in out:
			ncvars[key] = ncf.createVariable( key , "float32" , (mode,"time","period") + spatial , compression = "zlib" , complevel = 5 , chunksizes = (1,1,1) + clim.s_spatial )
		
		## Attributes
		if mode == "quantile":
			ncvars[mode].setncattr( "confidence_level" , ci )
			ncvars["quantile_level"] = ncf.createVariable( "quantile_levels" , "float32" , ("quantile") )
			ncvars["quantile_level"][:] = np.array([ci/2,0.5,1-ci/2])
		ncvars["pF"].setncattr( "description" , "Probability in the Factual world" )
		ncvars["pC"].setncattr( "description" , "Probability in the Counter factual world" )
		ncvars["RF"].setncattr( "description" , "Return time in the Factual world" )
		ncvars["RC"].setncattr( "description" , "Return time in the Counter factual world" )
		ncvars["IF"].setncattr( "description" , "Intensity in the Factual world" )
		ncvars["IC"].setncattr( "description" , "Intensity in the Counter factual world" )
		ncvars["PR"].setncattr( "description" , "Change in probability between Factual and Counter factual world" )
		ncvars["dI"].setncattr( "description" , "Change in intensity between Factual and Counter factual world" )
		
		## Find the blocks to write netcdf
		blocks  = [1,1,1]
		sizes   = [n_modes,time.size,len(clim.dpers)]
		nsizes  = [n_modes,time.size,len(clim.dpers)]
		sp_mem  = SizeOf( n = int(np.prod(clim.s_spatial) * np.finfo('float32').bits // SizeOf(n = 0).bits_per_octet) , unit = "o" )
		tot_mem = SizeOf( n = int(min( 0.8 , 3 * bsacParams.frac_memory_per_array ) * bsacParams.total_memory.o) , unit = "o" )
		nfind   = [True,True,True]
		while any(nfind):
			i = np.argmin(nsizes)
			blocks[i] = sizes[i]
			while int(np.prod(blocks)) * sp_mem > tot_mem:
				if blocks[i] < 2:
					blocks[i] = 1
					break
				blocks[i] = blocks[i] // 2
			nfind[i]  = False
			nsizes[i] = np.inf
		logger.info( f"   => Blocks size {blocks}" )
		## Fill
		idx_sp = tuple([slice(None) for _ in range(len(clim._spatial))])
		for idx in itt.product(*[range(0,s,block) for s,block in zip(sizes,blocks)]):
			
			s_idx = tuple([slice(s,s+block,1) for s,block in zip(idx,blocks)])
			idxs = s_idx + idx_sp
			
			for key in out:
				xdata = out[key].get_orthogonal_selection(idxs)
				ncvars[key][idxs] = xdata.values
		
		## Global attributes
		ncf.setncattr( "creation_date" , str(dt.datetime.utcnow())[:19] + " (UTC)" )
		ncf.setncattr( "BSAC_version"  , version )
		ncf.setncattr( "description" , f"Attribute event" )
	
##}}}


def _attribute_freturnt_parallel( hpar , hcov , bias , RT , clim , side , mode , n_samples , ci ):##{{{
	
	## Extract parameters from clim
	_,_,designF_,designC_ = clim.build_design_XFC()
	nslaw_class = clim._nslaw_class
	nslaw   = clim._nslaw_class()
	n_pers  = len(clim.dpers)
	n_times = len(clim.time)
	n_RT    = len(RT)
	
	## Init output
	n_modes = n_samples if mode == "sample" else 3
	
	## If nan, return
	if not np.isfinite(hpar).all():
		return tuple([ np.zeros((n_RT,n_modes,n_pers,n_times)) for _ in range(8) ])
	
	## Draw parameters
	hpars = xr.DataArray( rvs_multivariate_normal( n_samples , hpar , hcov ) , dims = ["sample","hpar"] , coords = [range(n_samples),clim.hpar_names] )
	
	## Design
	_,_,designF_,designC_ = clim.build_design_XFC()
	hpar_coords = designF_.hpar.values.tolist()
	name        = clim.namesX[-1]
	
	## Build XFC
	XF = xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designF_
	         for per in clim.dpers
	        ],
	        dim = "period"
	    ).assign_coords( period = clim.dpers )
	XC = xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designC_
	         for per in clim.dpers
	        ],
	        dim = "period"
	    ).assign_coords( period = clim.dpers )
	
	## Build params
	kwargsF = nslaw.draw_params( XF , hpars )
	kwargsC = nslaw.draw_params( XC , hpars )
	
	## Output
	lpF = []
	lpC = []
	lRF = []
	lRC = []
	lIF = []
	lIC = []
	ldI = []
	lPR = []
	
	## Loop on return time
	for rt in RT:
		
		## Set output
		pF  = np.zeros( (n_samples,n_pers,n_times) ) + 1. / rt
		pC  = np.zeros( (n_samples,n_pers,n_times) ) + np.nan
		IF  = np.zeros( (n_samples,n_pers,n_times) ) + np.nan
		IC  = np.zeros( (n_samples,n_pers,n_times) ) + np.nan
		
		## Attribution
		IF = nslaw.icdf_sf( pF , side = side , **kwargsF )
		IC = nslaw.icdf_sf( pF , side = side , **kwargsC )
		pC = nslaw.cdf_sf(  IF , side = side , **kwargsC )
		
		## Remove 0 and 1
		e  = 10 * sys.float_info.epsilon
		pC = np.where( pC >     e , pC ,     e )
		pC = np.where( pC < 1 - e , pC , 1 - e )
		
		## Others variables
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			IF  = IF + bias
			IC  = IC + bias
			RF  = 1. / pF
			RC  = 1. / pC
			dI  = IF - IC
			PR  = pF / pC
		
		## Compute CI
		if mode == "quantile":
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				pF = np.quantile( pF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				pC = np.quantile( pC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				RF = np.quantile( RF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				RC = np.quantile( RC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				IF = np.quantile( IF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				IC = np.quantile( IC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				dI = np.quantile( dI , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				PR = np.quantile( PR , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
		
		lpF.append(pF)
		lpC.append(pC)
		lRF.append(RF)
		lRC.append(RC)
		lIF.append(IF)
		lIC.append(IC)
		ldI.append(dI)
		lPR.append(PR)
	
	pF = np.stack( lpF , axis = 0 )
	pC = np.stack( lpC , axis = 0 )
	RF = np.stack( lRF , axis = 0 )
	RC = np.stack( lRC , axis = 0 )
	IF = np.stack( lIF , axis = 0 )
	IC = np.stack( lIC , axis = 0 )
	dI = np.stack( ldI , axis = 0 )
	PR = np.stack( lPR , axis = 0 )
	
	return tuple([pF,pC,RF,RC,IF,IC,dI,PR])
##}}}

def _attribute_creturnt_parallel( hpar , hcov , bias , RT , clim , side , mode , n_samples , ci ):##{{{
	
	## Extract parameters from clim
	_,_,designF_,designC_ = clim.build_design_XFC()
	nslaw_class = clim._nslaw_class
	nslaw   = clim._nslaw_class()
	n_pers  = len(clim.dpers)
	n_times = len(clim.time)
	
	## Init output
	n_modes = n_samples if mode == "sample" else 3
	
	## If nan, return
	if not np.isfinite(hpar).all():
		return tuple([ np.zeros((n_RT,n_modes,n_pers,n_times)) for _ in range(8) ])
	
	## Draw parameters
	hpars = xr.DataArray( rvs_multivariate_normal( n_samples , hpar , hcov ) , dims = ["sample","hpar"] , coords = [range(n_samples),clim.hpar_names] )
	
	## Design
	_,_,designF_,designC_ = clim.build_design_XFC()
	hpar_coords = designF_.hpar.values.tolist()
	name        = clim.namesX[-1]
	
	## Build XFC
	XF = xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designF_
	         for per in clim.dpers
	        ],
	        dim = "period"
	    ).assign_coords( period = clim.dpers )
	XC = xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designC_
	         for per in clim.dpers
	        ],
	        dim = "period"
	    ).assign_coords( period = clim.dpers )
	
	## Build params
	kwargsF = nslaw.draw_params( XF , hpars )
	kwargsC = nslaw.draw_params( XC , hpars )
	
	## Output
	lpF = []
	lpC = []
	lRF = []
	lRC = []
	lIF = []
	lIC = []
	ldI = []
	lPR = []
	
	## Loop on return time
	for rt in RT:
		
		## Set output
		pF  = np.zeros( (n_samples,n_pers,n_times) ) + np.nan
		pC  = np.zeros( (n_samples,n_pers,n_times) ) + 1. / rt
		IF  = np.zeros( (n_samples,n_pers,n_times) ) + np.nan
		IC  = np.zeros( (n_samples,n_pers,n_times) ) + np.nan
		
		## Attribution
		IF = nslaw.icdf_sf( pC , side = side , **kwargsF )
		IC = nslaw.icdf_sf( pC , side = side , **kwargsC )
		pF = nslaw.cdf_sf(  IC , side = side , **kwargsF )
		
		## Remove 0 and 1
		e  = 10 * sys.float_info.epsilon
		pF = np.where( pF >     e , pF ,     e )
		pF = np.where( pF < 1 - e , pF , 1 - e )
		
		## Others variables
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			IF  = IF + bias
			IC  = IC + bias
			RF  = 1. / pF
			RC  = 1. / pC
			dI  = IF - IC
			PR  = pF / pC
		
		## Compute CI
		if mode == "quantile":
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				pF = np.quantile( pF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				pC = np.quantile( pC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				RF = np.quantile( RF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				RC = np.quantile( RC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				IF = np.quantile( IF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				IC = np.quantile( IC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				dI = np.quantile( dI , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
				PR = np.quantile( PR , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
		
		lpF.append(pF)
		lpC.append(pC)
		lRF.append(RF)
		lRC.append(RC)
		lIF.append(IF)
		lIC.append(IC)
		ldI.append(dI)
		lPR.append(PR)
	
	pF = np.stack( lpF , axis = 0 )
	pC = np.stack( lpC , axis = 0 )
	RF = np.stack( lRF , axis = 0 )
	RC = np.stack( lRC , axis = 0 )
	IF = np.stack( lIF , axis = 0 )
	IC = np.stack( lIC , axis = 0 )
	dI = np.stack( ldI , axis = 0 )
	PR = np.stack( lPR , axis = 0 )
	
	return tuple([pF,pC,RF,RC,IF,IC,dI,PR])
##}}}

## run_bsac_cmd_attribute_fcreturnt ##{{{
@log_start_end(logger)
def run_bsac_cmd_attribute_fcreturnt(arg):
	
	## Parameters
	clim      = bsacParams.clim
	time      = clim.time
	n_samples = bsacParams.n_samples
	tmp       = bsacParams.tmp
	side      = bsacParams.config.get("side","right")
	mode      = bsacParams.config.get("mode","sample")
	ci        = bsacParams.config.get("ci",0.05)
	nslaw     = clim._nslaw_class()
	sp_dims   = clim.d_spatial
	
	## Logs
	logger.info(  " * Configuration" )
	logger.info( f"   => mode: {mode}" )
	logger.info( f"   => side: {side}" )
	logger.info( f"   => ci  : {ci}" )
	
	## Mode dimension
	if mode == "sample":
		modes = coords_samples( n_samples )
	elif mode == "quantile":
		modes = np.array(["QL","BE","QU"])
	else:
		raise ValueError( f"Invalid mode ({mode})" )
	n_modes = modes.size
	
	## Read return time
	inp = bsacParams.input[0]
	if ":" in inp:
		rp = [float(x) for x in inp.split(":")]
		RT = np.arange( rp[0] , rp[1] + rp[2] / 2 , rp[2] , dtype = float ) 
	else:
		RT = np.array([float(x) for x in inp.split(",")]).ravel()
	n_RT = RT.size
	logger.info( f" * Return time: {RT}" )
	
	## Output
	time   = clim.time
	period = np.array(clim.dpers)
	dims   = ["RT",mode ] + ["time","period"] + list(clim.d_spatial)
	coords = [ RT ,modes] + [ time , period ] + [ clim.c_spatial[d] for d in clim.c_spatial ]
	shape  = [len(c) for c in coords]
	out = {
	    "pF" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "pF" ) ) ),
	    "pC" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "pC" ) ) ),
	    "RF" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "RF" ) ) ),
	    "RC" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "RC" ) ) ),
	    "IF" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "IF" ) ) ),
	    "IC" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "IC" ) ) ),
	    "dI" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "dI" ) ) ),
	    "PR" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "PR" ) ) )
	}
	
	## Select the good function
	if arg == "freturnt":
		_attribute_fcreturnt_parallel = _attribute_freturnt_parallel
	else:
		_attribute_fcreturnt_parallel = _attribute_creturnt_parallel
	
	## Build a 'small' climatology
	## climatology must be pass at each threads, but duplication of mean, cov bias implies a memory leak
	climB = Climatology.init_from_file( bsacParams.load_clim )
	del climB._mean
	del climB._cov
	del climB._bias
	
	## Dask parameters
	dask_gufunc_kwargs = { "output_sizes" : { "RT" : n_RT } }
	if mode == "quantile":
		dask_gufunc_kwargs["output_sizes"] = { "RT" : n_RT , mode : modes.size , "time" : time.size }
	
	## Loop on spatial variables
	block = max( 0 , int( np.power( bsacParams.n_jobs , 1. / len(clim.s_spatial) ) ) ) + 1
	logger.info( " * Loop on spatial variables" )
	for idx in itt.product(*[range(0,s,block) for s in clim.s_spatial]):
		
		##
		s_idx = tuple([slice(s,s+block,1) for s in idx])
		f_idx = tuple( [slice(None) for _ in range(3)] ) + s_idx
		
		##
		bias = clim.bias[clim.names[-1]][s_idx]
		
		##
		hpar = clim.xmean_[(slice(None),)+s_idx]
		hcov = clim.xcov_[(slice(None),slice(None))+s_idx]
		
		#
		res = xr.apply_ufunc( _attribute_fcreturnt_parallel , hpar , hcov , bias ,
		                    input_core_dims    = [["hpar"],["hpar0","hpar1"],[]],
		                    output_core_dims   = [["RT",mode,"period","time"] for _ in range(8)],
		                    output_dtypes      = [hpar.dtype for _ in range(8)],
		                    vectorize          = True,
		                    dask               = "parallelized",
		                    kwargs             = { "RT" : RT , "clim" : climB , "side" : side , "mode" : mode , "n_samples" : n_samples , "ci" : ci },
		                    dask_gufunc_kwargs = dask_gufunc_kwargs
		                    )
		
		res = [ r.transpose( *out["pF"].dims ).compute() for r in res ]
		
		for key,r in zip(out,res):
			out[key].set_orthogonal_selection( (slice(None),) + f_idx , r.values )
	
	## And save
	logger.info( " * Save in netcdf" )
	ofile = bsacParams.output
	with netCDF4.Dataset( ofile , "w" ) as ncf:
		
		## Define dimensions
		ncdims = {
		  "return_time" : ncf.createDimension( "return_time" , n_RT ),
		           mode : ncf.createDimension(     mode      , n_modes ),
		       "period" : ncf.createDimension( "period"      , len(clim.dpers) ),
		       "time"   : ncf.createDimension(   "time" ),
		}
		spatial = ()
		if clim._spatial is not None:
			for d in clim._spatial:
				ncdims[d] = ncf.createDimension( d , clim._spatial[d].size )
			spatial = tuple([d for d in clim._spatial])
		
		## Define variables
		ncvars = {
		  "return_time" : ncf.createVariable( "return_time" , "float32" , ("return_time",) ),
		           mode : ncf.createVariable(     mode      , str       , (mode,)     ),
		       "period" : ncf.createVariable( "period"      , str       , ("period",) ),
		       "time"   : ncf.createVariable( "time"        , "float32" , ("time"  ,) )
		}
		ncvars["return_time"][:] = RT
		ncvars[mode    ][:]      = modes
		ncvars["period"][:]      = period
		if clim._spatial is not None:
			for d in clim._spatial:
				ncvars[d] = ncf.createVariable( d , "double" , (d,) )
				ncvars[d][:] = np.array(clim._spatial[d]).ravel()
		
		if mode == "quantile":
			ncvars[mode].setncattr( "confidence_level" , ci )
			ncvars["quantile_level"] = ncf.createVariable( "quantile_levels" , "float32" , ("quantile") )
			ncvars["quantile_level"][:] = np.array([ci/2,0.5,1-ci/2])
		
		## Fill time axis
		calendar = "standard"
		units    = "days since 1750-01-01 00:00"
		ncvars["time"][:]  = cftime.date2num( [cftime.DatetimeGregorian( int(y) , 1 , 1 ) for y in time] , units = units , calendar = calendar )
		ncvars["time"].setncattr( "standard_name" , "time"      )
		ncvars["time"].setncattr( "long_name"     , "time_axis" )
		ncvars["time"].setncattr( "units"         , units       )
		ncvars["time"].setncattr( "calendar"      , calendar    )
		ncvars["time"].setncattr( "axis"          , "T"         )
		
		## Variables
		for key in out:
			ncvars[key] = ncf.createVariable( key , "float32" , ("return_time",mode,"time","period") + spatial , compression = "zlib" , complevel = 5 , chunksizes = (1,1,1,1) + clim.s_spatial )
		
		## Attributes
		ncvars["pF"].setncattr( "description" , "Probability in the Factual world" )
		ncvars["pC"].setncattr( "description" , "Probability in the Counter factual world" )
		ncvars["RF"].setncattr( "description" , "Return time in the Factual world" )
		ncvars["RC"].setncattr( "description" , "Return time in the Counter factual world" )
		ncvars["IF"].setncattr( "description" , "Intensity in the Factual world" )
		ncvars["IC"].setncattr( "description" , "Intensity in the Counter factual world" )
		ncvars["PR"].setncattr( "description" , "Change in probability between Factual and Counter factual world" )
		ncvars["dI"].setncattr( "description" , "Change in intensity between Factual and Counter factual world" )
		
		## Find the blocks to write netcdf
		blocks  = [1,1,1,1]
		sizes   = [n_RT,n_modes,time.size,len(clim.dpers)]
		nsizes  = [n_RT,n_modes,time.size,len(clim.dpers)]
		sp_mem  = SizeOf( n = int(np.prod(clim.s_spatial) * np.finfo('float32').bits // SizeOf(n = 0).bits_per_octet) , unit = "o" )
		tot_mem = SizeOf( n = int(min( 0.8 , 3 * bsacParams.frac_memory_per_array ) * bsacParams.total_memory.o) , unit = "o" )
		nfind   = [True,True,True,True]
		while any(nfind):
			i = np.argmin(nsizes)
			blocks[i] = sizes[i]
			while int(np.prod(blocks)) * sp_mem > tot_mem:
				if blocks[i] < 2:
					blocks[i] = 1
					break
				blocks[i] = blocks[i] // 2
			nfind[i]  = False
			nsizes[i] = np.inf
		logger.info( f"   => Blocks size {blocks}" )
		## Fill
		idx_sp = tuple([slice(None) for _ in range(len(clim._spatial))])
		for idx in itt.product(*[range(0,s,block) for s,block in zip(sizes,blocks)]):
			
			s_idx = tuple([slice(s,s+block,1) for s,block in zip(idx,blocks)])
			idxs = s_idx + idx_sp
			
			for key in out:
				xdata = out[key].get_orthogonal_selection(idxs)
				ncvars[key][idxs] = xdata.values
		
		## Global attributes
		ncf.setncattr( "creation_date" , str(dt.datetime.utcnow())[:19] + " (UTC)" )
		ncf.setncattr( "BSAC_version"  , version )
		ncf.setncattr( "description" , f"Attribute {arg}" )
	
##}}}


def _attribute_fintensity_parallel( hpar , hcov , bias , xIF , clim , side , mode , n_samples , ci ):##{{{
	
	## Extract parameters from clim
	_,_,designF_,designC_ = clim.build_design_XFC()
	nslaw_class = clim._nslaw_class
	nslaw   = clim._nslaw_class()
	n_pers  = len(clim.dpers)
	n_times = len(clim.time)
	
	## Init output
	n_modes = n_samples if mode == "sample" else 3
	
	## If nan, return
	if not np.isfinite(hpar).all():
		return tuple([ np.zeros((n_modes,n_pers,n_times)) for _ in range(8) ])
	
	## Draw parameters
	hpars = xr.DataArray( rvs_multivariate_normal( n_samples , hpar , hcov ) , dims = ["sample","hpar"] , coords = [range(n_samples),clim.hpar_names] )
	
	## Design
	_,_,designF_,designC_ = clim.build_design_XFC()
	hpar_coords = designF_.hpar.values.tolist()
	name        = clim.namesX[-1]
	
	## Build XFC
	XF = xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designF_
	         for per in clim.dpers
	        ],
	        dim = "period"
	    ).assign_coords( period = clim.dpers )
	XC = xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designC_
	         for per in clim.dpers
	        ],
	        dim = "period"
	    ).assign_coords( period = clim.dpers )
	
	## Build params
	kwargsF = nslaw.draw_params( XF , hpars )
	kwargsC = nslaw.draw_params( XC , hpars )
	
	## Factual and counter factual probabilities
	IF = np.zeros( (n_samples,n_pers,n_times) ) + xIF
	pF = nslaw.cdf_sf( IF , side = side , **kwargsF )
	pC = nslaw.cdf_sf( IF , side = side , **kwargsC )
	
	## Remove 0 and 1
	e  = 10 * sys.float_info.epsilon
	pF = np.where( pF >     e , pF ,     e )
	pC = np.where( pC >     e , pC ,     e )
	pF = np.where( pF < 1 - e , pF , 1 - e )
	pC = np.where( pC < 1 - e , pC , 1 - e )
	
	## Factual and counter factual intensities
	IC = nslaw.icdf_sf( pF , side = side , **kwargsC )
	
	## Others variables
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		IF  = IF + bias
		IC  = IC + bias
		RF  = 1. / pF
		RC  = 1. / pC
		dI  = IF - IC
		PR  = pF / pC
	
	## Compute CI
	if mode == "quantile":
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			pF = np.quantile( pF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			pC = np.quantile( pC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			RF = np.quantile( RF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			RC = np.quantile( RC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			IF = np.quantile( IF , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			IC = np.quantile( IC , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			dI = np.quantile( dI , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
			PR = np.quantile( PR , [ci/2,0.5,1-ci/2] , axis = 0 , method = "median_unbiased" )
	
	return tuple([pF,pC,RF,RC,IF,IC,dI,PR])
##}}}

## run_bsac_cmd_attribute_fintensity ##{{{
@log_start_end(logger)
def run_bsac_cmd_attribute_fintensity(arg):
	
	## Parameters
	clim      = bsacParams.clim
	time      = clim.time
	n_samples = bsacParams.n_samples
	tmp       = bsacParams.tmp
	side      = bsacParams.config.get("side","right")
	mode      = bsacParams.config.get("mode","sample")
	ci        = bsacParams.config.get("ci",0.05)
	nslaw     = clim._nslaw_class()
	sp_dims   = clim.d_spatial
	
	## Logs
	logger.info(  " * Configuration" )
	logger.info( f"   => mode: {mode}" )
	logger.info( f"   => side: {side}" )
	logger.info( f"   => ci  : {ci}" )
	
	## Mode dimension
	if mode == "sample":
		modes = coords_samples( n_samples )
	elif mode == "quantile":
		modes = np.array(["QL","BE","QU"])
	else:
		raise ValueError( f"Invalid mode ({mode})" )
	n_modes = modes.size
	
	## Read the intensity
	name,ifile = bsacParams.input[0].split(",")
	xIF = xr.open_dataset(ifile)[name]
	
	## Check the spatial dimensions
	for d in clim._spatial:
		if d not in xIF.dims:
			raise Exception( f"Spatial dimension missing: {d}" )
		if not xIF[d].size == clim._spatial[d].size:
			raise Exception( f"Bad size of the dimension {d}: {xIF[d].size} != {clim._spatial[d].size}" )
	
	## Reorganize dimension
	sdims   = list(clim._spatial)
	cdims   = [ d for d in xIF.dims if d not in sdims ]
	ccoords = [xIF[d] for d in cdims]
	cshape  = [xIF[d].size for d in cdims]
	dims    = tuple(cdims + sdims)
	xIF     = xIF.transpose(*dims)
	
	for i,d in enumerate(cdims):
		if d in [mode,"time"]: # + list(zdraw[ovars[0]].dims)[1:]:
			xIF = xIF.rename( { d : "BSAC_" + d } )
			cdims[i] = "BSAC_" + d
	
	## Remove bias
	oB  = clim.bias[list(clim.bias)[-1]]
	xIF = xIF - oB
	
	## Output
	time   = clim.time
	period = np.array(clim.dpers)
	dims   = cdims   + [mode ] + ["time","period"] + list(clim.d_spatial)
	coords = ccoords + [modes] + [ time , period ] + [ clim.c_spatial[d] for d in clim.c_spatial ]
	shape  = [len(c) for c in coords]
	out = {
	    "pF" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "pF" ) ) ),
	    "pC" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "pC" ) ) ),
	    "RF" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "RF" ) ) ),
	    "RC" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "RC" ) ) ),
	    "IF" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "IF" ) ) ),
	    "IC" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "IC" ) ) ),
	    "dI" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "dI" ) ) ),
	    "PR" : XZarr.from_value( np.nan , shape = shape , dims = dims , coords = coords , zfile = random_zfile( os.path.join( tmp , "PR" ) ) )
	}
	
	## Build a 'small' climatology
	## climatology must be pass at each threads, but duplication of mean, cov bias implies a memory leak
	climB = Climatology.init_from_file( bsacParams.load_clim )
	del climB._mean
	del climB._cov
	del climB._bias
	
	## Select the good function
	if arg == "fintensity":
		_attribute_fcintensity_parallel = _attribute_fintensity_parallel
	
	## Dask parameters
	dask_gufunc_kwargs = {}
	dask_gufunc_kwargs["output_sizes"] = { mode : modes.size , "time" : time.size }
	
	## Loop on spatial variables
	block = max( 0 , int( np.power( bsacParams.n_jobs , 1. / len(clim.s_spatial) ) ) ) + 1
	logger.info( " * Loop on spatial variables" )
	for idx in itt.product(*[range(0,s,block) for s in clim.s_spatial]):
		
		##
		s_idx = tuple([slice(s,s+block,1) for s in idx])
		f_idx = tuple( [slice(None) for _ in range(3)] ) + s_idx
		I_idx = tuple( [slice(None) for _ in range(len(cdims))] ) + s_idx
		
		##
		bias = clim.bias[clim.names[-1]][s_idx]
		IF   = xIF[I_idx]
		
		hpar = clim.xmean_[(slice(None),)+s_idx]
		hcov = clim.xcov_[(slice(None),slice(None))+s_idx]
		
		#
		res = xr.apply_ufunc( _attribute_fcintensity_parallel , hpar , hcov , bias , IF ,
		                    input_core_dims    = [["hpar"],["hpar0","hpar1"],[],[]],
		                    output_core_dims   = [[mode,"period","time"] for _ in range(8)],
		                    output_dtypes      = [hpar.dtype for _ in range(8)],
		                    vectorize          = True,
		                    dask               = "parallelized",
		                    kwargs             = { "clim" : climB , "side" : side , "mode" : mode , "ci" : ci , "n_samples" : n_samples },
		                    dask_gufunc_kwargs = dask_gufunc_kwargs
		                    )
		
		res = [ r.transpose( *out["pF"].dims ).compute() for r in res ]
		
		for key,r in zip(out,res):
			out[key].set_orthogonal_selection( tuple( [slice(None) for _ in range(len(cdims)+3)] ) + s_idx , r.values )
	
	## And save
	logger.info( " * Save in netcdf" )
	ofile = bsacParams.output
	with netCDF4.Dataset( ofile , "w" ) as ncf:
		
		## Define dimensions
		ncdims = {
		           mode : ncf.createDimension(     mode      , n_modes ),
		       "period" : ncf.createDimension( "period"      , len(clim.dpers) ),
		       "time"   : ncf.createDimension(   "time" ),
		}
		for d,s in zip(cdims,cshape):
			ncdims[d] = ncf.createDimension( d , s )
		
		spatial = ()
		if clim._spatial is not None:
			for d in clim._spatial:
				ncdims[d] = ncf.createDimension( d , clim._spatial[d].size )
			spatial = tuple([d for d in clim._spatial])
		
		## Define variables
		ncvars = {
		           mode : ncf.createVariable(     mode      , str       , (mode,)     ),
		       "period" : ncf.createVariable( "period"      , str       , ("period",) ),
		       "time"   : ncf.createVariable( "time"        , "float32" , ("time"  ,) )
		}
		ncvars[mode    ][:]      = modes
		ncvars["period"][:]      = period
		if clim._spatial is not None:
			for d in clim._spatial:
				ncvars[d] = ncf.createVariable( d , "double" , (d,) )
				ncvars[d][:] = np.array(clim._spatial[d]).ravel()
		for d in cdims:
			if isinstance(xIF[d][0].values,tuple([dt.datetime,np.datetime64,cftime.datetime])):
				ncvars[d] = ncf.createVariable( d , "float32" , (d,) )
			else:
				ncvars[d] = ncf.createVariable( d , xIF[d].values.dtype , (d,) )
				ncvars[d][:] = xIF[d].values
		
		if mode == "quantile":
			ncvars[mode].setncattr( "confidence_level" , ci )
			ncvars["quantile_level"] = ncf.createVariable( "quantile_levels" , "float32" , ("quantile") )
			ncvars["quantile_level"][:] = np.array([ci/2,0.5,1-ci/2])
		
		## Fill time axis
		calendar = "standard"
		units    = "days since 1750-01-01 00:00"
		ncvars["time"][:]  = cftime.date2num( [cftime.DatetimeGregorian( int(y) , 1 , 1 ) for y in time] , units = units , calendar = calendar )
		ncvars["time"].setncattr( "standard_name" , "time"      )
		ncvars["time"].setncattr( "long_name"     , "time_axis" )
		ncvars["time"].setncattr( "units"         , units       )
		ncvars["time"].setncattr( "calendar"      , calendar    )
		ncvars["time"].setncattr( "axis"          , "T"         )
		
		## Variables
		for key in out:
			ncvars[key] = ncf.createVariable( key , "float32" , tuple(cdims) + (mode,"time","period") + spatial , compression = "zlib" , complevel = 5 , chunksizes = tuple([1 for _ in range(len(cdims))]) + (1,1,1) + clim.s_spatial )
		
		## Attributes
		ncvars["pF"].setncattr( "description" , "Probability in the Factual world" )
		ncvars["pC"].setncattr( "description" , "Probability in the Counter factual world" )
		ncvars["RF"].setncattr( "description" , "Return time in the Factual world" )
		ncvars["RC"].setncattr( "description" , "Return time in the Counter factual world" )
		ncvars["IF"].setncattr( "description" , "Intensity in the Factual world" )
		ncvars["IC"].setncattr( "description" , "Intensity in the Counter factual world" )
		ncvars["PR"].setncattr( "description" , "Change in probability between Factual and Counter factual world" )
		ncvars["dI"].setncattr( "description" , "Change in intensity between Factual and Counter factual world" )
		
		## Find the blocks to write netcdf
		blocks  = [1 for _ in range(len(cdims))] + [1,1,1]
		sizes   = list(cshape) + [n_modes,time.size,len(clim.dpers)]
		nsizes  = list(cshape) + [n_modes,time.size,len(clim.dpers)]
		sp_mem  = SizeOf( n = int(np.prod(clim.s_spatial) * np.finfo('float32').bits // SizeOf(n = 0).bits_per_octet) , unit = "o" )
		tot_mem = SizeOf( n = int(min( 0.8 , 3 * bsacParams.frac_memory_per_array ) * bsacParams.total_memory.o) , unit = "o" )
		nfind   = [True for _ in range(len(cdims))] + [True,True,True]
		while any(nfind):
			i = np.argmin(nsizes)
			blocks[i] = sizes[i]
			while int(np.prod(blocks)) * sp_mem > tot_mem:
				if blocks[i] < 2:
					blocks[i] = 1
					break
				blocks[i] = blocks[i] // 2
			nfind[i]  = False
			nsizes[i] = np.inf
		logger.info( f"   => Blocks size {blocks}" )
		
		## Fill
		idx_sp = tuple([slice(None) for _ in range(len(clim._spatial))])
		for idx in itt.product(*[range(0,s,block) for s,block in zip(sizes,blocks)]):
			
			s_idx = tuple([slice(s,s+block,1) for s,block in zip(idx,blocks)])
			idxs = s_idx + idx_sp
			
			for key in out:
				xdata = out[key].get_orthogonal_selection(idxs)
				ncvars[key][idxs] = xdata.values
		
		## Global attributes
		ncf.setncattr( "creation_date" , str(dt.datetime.utcnow())[:19] + " (UTC)" )
		ncf.setncattr( "BSAC_version"  , version )
		ncf.setncattr( "description" , f"Attribute {arg}" )
##}}}


## run_bsac_cmd_attribute ##{{{
@log_start_end(logger)
def run_bsac_cmd_attribute():
	
	avail_arg = ["event","freturnt","creturnt","fintensity"]
	try:
		arg = bsacParams.arg[0]
	except:
		raise ValueError( f"A argument must be given for the attribute command ({', '.join(avail_arg)})" )
	
	if not arg in avail_arg:
		raise ValueError( f"Bad argument for the attribute command ({', '.join(avail_arg)})" )
	
	if arg == "event":
		run_bsac_cmd_attribute_event()
	if arg in ["freturnt","creturnt"]:
		run_bsac_cmd_attribute_fcreturnt(arg)
	if arg in ["fintensity"]:
		run_bsac_cmd_attribute_fintensity(arg)
	
##}}}


