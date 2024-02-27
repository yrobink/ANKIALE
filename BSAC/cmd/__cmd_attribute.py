
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
import itertools as itt
import datetime as dt
import logging
import warnings
from ..__logs import LINE
from ..__logs import log_start_end
from ..__release import version

from ..__BSACParams import bsacParams

from ..__XZarr import XZarr
from ..__XZarr import random_zfile

from ..__sys import SizeOf

from ..stats.__rvs import rvs_climatology


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

def _attribute_event_parallel( *args , ovars , nslaw_class , it_attr , side ):##{{{
	
	## Extract
	pars    = { ovar : arg for ovar,arg in zip(ovars,args) } ## Parameters
	Yo      = args[-1]
	nslaw   = nslaw_class()
	kwargsF = { p : pars[p+"F"] for p in nslaw.coef_kind }
	kwargsC = { p : pars[p+"C"] for p in nslaw.coef_kind }
	
	## Set output
	pF  = np.zeros_like( pars[ovars[0]] ) + np.nan
	pC  = np.zeros_like( pars[ovars[0]] ) + np.nan
	IF  = np.zeros_like( pars[ovars[0]] ) + np.nan
	IC  = np.zeros_like( pars[ovars[0]] ) + np.nan
	xYo = np.zeros_like( pars[ovars[0]] ) + Yo
	
	## Attribution
	if np.isfinite(Yo):
		
		pF = nslaw.cdf_sf( xYo , side = side , **kwargsF )
		pC = nslaw.cdf_sf( xYo , side = side , **kwargsC )
		
		pf = np.zeros_like(pF) + pF[:,it_attr,:].reshape((pF.shape[0],1,pF.shape[2]))
		
		IF = nslaw.icdf_sf( pf , side = side , **kwargsF )
		IC = nslaw.icdf_sf( pf , side = side , **kwargsC )
	
	
	## Others variables
	RF  = 1. / pF
	RC  = 1. / pC
	dI  = IF - IC
	PR  = pF / pC
	
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
	t_attr    = int(bsacParams.config.get("time"))
	it_attr   = int(np.argwhere( time == t_attr ).ravel())
	nslaw     = clim._nslaw_class()
	sp_dims   = clim.d_spatial
	
	## Load observations
	Yo  = xr.open_dataset( bsacParams.input[0].split(",")[1] )[clim.names[-1]]
	Yo  = Yo.assign_coords( time = Yo.time.dt.year )
	bYo = Yo.sel( time = slice(*clim.bper) ).mean( dim = "time" )
	Yo  = (Yo - bYo).sel( time = int(t_attr) )
	
	## Draw parameters for the attribution
	logger.info( " * Draw parameters" )
#	zdraw   = clim.rvsY(n_samples)
	zdraw   = rvs_climatology( clim , n_samples , tmp = bsacParams.tmp , n_jobs = bsacParams.n_jobs , mem_limit = bsacParams.total_memory )
	ovars   = [key for key in zdraw if key not in ["XF","XC","XA"]]
	samples = np.array(zdraw[ovars[0]].coords[zdraw[ovars[0]].dims.index("sample")])
	periods = np.array(zdraw[ovars[0]].coords[zdraw[ovars[0]].dims.index("period")])
	
	## Output
	out = {
	    "pF" : XZarr.like( zdraw[ovars[0]] , random_zfile( os.path.join( tmp , "pF" ) ) ),
	    "pC" : XZarr.like( zdraw[ovars[0]] , random_zfile( os.path.join( tmp , "pC" ) ) ),
	    "RF" : XZarr.like( zdraw[ovars[0]] , random_zfile( os.path.join( tmp , "RF" ) ) ),
	    "RC" : XZarr.like( zdraw[ovars[0]] , random_zfile( os.path.join( tmp , "RC" ) ) ),
	    "IF" : XZarr.like( zdraw[ovars[0]] , random_zfile( os.path.join( tmp , "IF" ) ) ),
	    "IC" : XZarr.like( zdraw[ovars[0]] , random_zfile( os.path.join( tmp , "IC" ) ) ),
	    "dI" : XZarr.like( zdraw[ovars[0]] , random_zfile( os.path.join( tmp , "dI" ) ) ),
	    "PR" : XZarr.like( zdraw[ovars[0]] , random_zfile( os.path.join( tmp , "PR" ) ) )
	}
	
	## Loop on spatial variables
	block = max( 0 , int( np.power( bsacParams.n_jobs , 1. / len(clim.s_spatial) ) ) ) + 1
	logger.info( " * Loop on spatial variables" )
	for idx in itt.product(*[range(0,s,block) for s in clim.s_spatial]):
		
		##
		s_idx = tuple([slice(s,s+block,1) for s in idx])
		f_idx = tuple( [slice(None) for _ in range(3)] ) + s_idx
		
		##
		draw = { ovar : zdraw[ovar].get_orthogonal_selection(f_idx).chunk( { d : 1 for d in sp_dims } )  for ovar in ovars }
		sYo  = Yo[s_idx].chunk( { d : 1 for d in sp_dims } )
		
		
		#
		res = xr.apply_ufunc( _attribute_event_parallel , *[draw[ovar] for ovar in ovars] , sYo ,
		                    input_core_dims  = [["sample","time","period"] for _ in range(len(draw))] + [[]],
		                    output_core_dims = [["sample","time","period"] for _ in range(8)],
		                    output_dtypes    = [draw[ovars[0]].dtype for _ in range(8)],
		                    vectorize        = True,
		                    dask             = "parallelized",
		                    kwargs           = { "ovars" : ovars , "nslaw_class" : clim._nslaw_class , "it_attr" : it_attr , "side" : side }
		                    )
		
		res = [ r.transpose( *out["pF"].dims ).compute() for r in res ]
		for key,r in zip(out,res):
			b = 0
			if key in ["IF","IC"]:
				b = bYo[s_idx]
			out[key].set_orthogonal_selection( f_idx , (r+b).values )
	
	## And save
	logger.info( " * Save in netcdf" )
	ofile = bsacParams.output
	with netCDF4.Dataset( ofile , "w" ) as ncf:
		
		## Define dimensions
		ncdims = {
		       "sample" : ncf.createDimension( "sample" , n_samples ),
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
		       "sample" : ncf.createVariable( "sample" , str       , ("sample",) ),
		       "period" : ncf.createVariable( "period" , str       , ("period",) ),
		       "time"   : ncf.createVariable( "time"   , "float32" , ("time"  ,) )
		}
		ncvars["sample"][:] = samples
		ncvars["period"][:] = periods
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
			ncvars[key] = ncf.createVariable( key , "float32" , ("sample","time","period") + spatial , compression = "zlib" , complevel = 5 , chunksizes = (1,1,1) + clim.s_spatial )
		
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
		blocks  = [1,1,1]
		sizes   = [n_samples,time.size,len(clim.dpers)]
		nsizes  = [n_samples,time.size,len(clim.dpers)]
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


def _attribute_freturnt_parallel( *args , RT , ovars , nslaw_class , side , mode , ci ):##{{{
	
	## Extract
	pars    = { ovar : arg for ovar,arg in zip(ovars,args) } ## Parameters
	bias    = args[-1]
	nslaw   = nslaw_class()
	kwargsF = { p : pars[p+"F"] for p in nslaw.coef_kind }
	kwargsC = { p : pars[p+"C"] for p in nslaw.coef_kind }
	
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
		pF  = np.zeros_like( pars[ovars[0]] ) + 1. / rt
		pC  = np.zeros_like( pars[ovars[0]] ) + np.nan
		IF  = np.zeros_like( pars[ovars[0]] ) + np.nan
		IC  = np.zeros_like( pars[ovars[0]] ) + np.nan
		
		## Attribution
		IF = nslaw.icdf_sf( pF , side = side , **kwargsF ) + bias
		IC = nslaw.icdf_sf( pF , side = side , **kwargsC ) + bias
		pC = nslaw.cdf_sf(  IF , side = side , **kwargsC )
		
		## Others variables
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
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

def _attribute_creturnt_parallel( *args , RT , ovars , nslaw_class , side , mode , ci ):##{{{
	
	## Extract
	pars    = { ovar : arg for ovar,arg in zip(ovars,args) } ## Parameters
	bias    = args[-1]
	nslaw   = nslaw_class()
	kwargsF = { p : pars[p+"F"] for p in nslaw.coef_kind }
	kwargsC = { p : pars[p+"C"] for p in nslaw.coef_kind }
	
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
		pF  = np.zeros_like( pars[ovars[0]] ) + np.nan
		pC  = np.zeros_like( pars[ovars[0]] ) + 1. / rt
		IF  = np.zeros_like( pars[ovars[0]] ) + np.nan
		IC  = np.zeros_like( pars[ovars[0]] ) + np.nan
		
		## Attribution
		IF = nslaw.icdf_sf( pC , side = side , **kwargsF ) + bias
		IC = nslaw.icdf_sf( pC , side = side , **kwargsC ) + bias
		pF = nslaw.cdf_sf(  IC , side = side , **kwargsF )
		
		## Others variables
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
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
	
	## Draw parameters for the attribution
	logger.info( " * Draw parameters" )
	zdraw   = rvs_climatology( clim , n_samples , tmp = bsacParams.tmp , n_jobs = bsacParams.n_jobs , mem_limit = bsacParams.total_memory )
	ovars   = [key for key in zdraw if key not in ["XF","XC","XA","hpar"]]
	periods = np.array(zdraw["XF"].coords[zdraw["XF"].dims.index("period")])
	if mode == "sample":
		modes = np.array(zdraw[ovars[0]].coords[zdraw[ovars[0]].dims.index("sample")])
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
	dims   = ["RT",mode   ] + list(zdraw[ovars[0]].dims)[1:]
	coords = [ RT ,modes  ] + list(zdraw[ovars[0]].coords)[1:]
	shape  = [n_RT,n_modes] + list(zdraw[ovars[0]].shape)[1:]
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
	
	## Dask parameters
	dask_gufunc_kwargs = { "output_sizes" : { "RT" : n_RT } }
	if mode == "quantile":
		dask_gufunc_kwargs["output_sizes"] = { "RT" : n_RT , mode : modes.size }
	
	## Loop on spatial variables
	block = max( 0 , int( np.power( bsacParams.n_jobs , 1. / len(clim.s_spatial) ) ) ) + 1
	logger.info( " * Loop on spatial variables" )
	for idx in itt.product(*[range(0,s,block) for s in clim.s_spatial]):
		
		##
		s_idx = tuple([slice(s,s+block,1) for s in idx])
		f_idx = tuple( [slice(None) for _ in range(3)] ) + s_idx
		
		##
		draw = { ovar : zdraw[ovar].get_orthogonal_selection(f_idx).chunk( { d : 1 for d in sp_dims } )  for ovar in ovars }
		bias = clim.bias[clim.names[-1]][s_idx]
		
		#
		res = xr.apply_ufunc( _attribute_fcreturnt_parallel , *[draw[ovar] for ovar in ovars] , bias ,
		                    input_core_dims    = [[ "sample","time","period"] for _ in range(len(draw))] + [[]],
		                    output_core_dims   = [["RT",mode,"time","period"] for _ in range(8)],
		                    output_dtypes      = [draw[ovars[0]].dtype for _ in range(8)],
		                    vectorize          = True,
		                    dask               = "parallelized",
		                    kwargs             = { "RT" : RT , "ovars" : ovars , "nslaw_class" : clim._nslaw_class , "side" : side , "mode" : mode , "ci" : ci },
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
		ncvars["period"][:]      = periods
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


## run_bsac_cmd_attribute ##{{{
@log_start_end(logger)
def run_bsac_cmd_attribute():
	
	avail_arg = ["event","freturnt","creturnt"]
	try:
		arg = bsacParams.arg[0]
	except:
		raise ValueError( "A argument must be given for the attribute command ({', '.join(avail_arg)})" )
	
	if not arg in avail_arg:
		raise ValueError( "Bad argument for the attribute command ({', '.join(avail_arg)})" )
	
	if arg == "event":
		run_bsac_cmd_attribute_event()
	if arg in ["freturnt","creturnt"]:
		run_bsac_cmd_attribute_fcreturnt(arg)
	
##}}}


