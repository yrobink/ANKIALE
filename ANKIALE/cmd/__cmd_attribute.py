
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
from ..__logs import disable_warnings
from ..__release import version

from ..__BSACParams import bsacParams

from ..__climatology import Climatology

from ..__sys import coords_samples


import numpy  as np
import xarray as xr
import zxarray as zr
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

## zattribute_fcreturnt ##{{{

@disable_warnings
def zattribute_fcreturnt( hpar , hcov , bias , projF , projC , RT , nslaw_class , side , world , mode , n_samples , ci ):
	
	## Coordinates
	nhpar = hpar.shape[-1]
	ssp   = hpar.shape[:-3]
	nper  = projF.shape[0]
	ntime = projF.shape[-2]
	
	##
	projF = projF.reshape( (nper,) + projF.shape[-2:] )
	projC = projC.reshape( (nper,) + projF.shape[-2:] )
	nslaw = nslaw_class()
	
	## Prepare output
	pF = np.zeros( ssp + (nper,RT.size,ntime,n_samples) ) + np.nan
	pC = np.zeros( ssp + (nper,RT.size,ntime,n_samples) ) + np.nan
	IF = np.zeros( ssp + (nper,RT.size,ntime,n_samples) ) + np.nan
	IC = np.zeros( ssp + (nper,RT.size,ntime,n_samples) ) + np.nan
	
	## Copy RT
	if world.upper() == "F":
		pF[:] = 1. / RT.reshape( *([1 for _ in ssp] + [1,-1,1,1]) )
	else:
		pC[:] = 1. / RT.reshape( *([1 for _ in ssp] + [1,-1,1,1]) )
	
	## Loop
	for idx in itt.product(*[range(s) for s in ssp]):
		
		ih = hpar[idx+(0,0,slice(None))]
		ic = hcov[idx+(0,0,slice(None),slice(None))]
		
		if not np.isfinite(ih).all() or not np.isfinite(ic).all():
			continue
		
		## Find hpars
		hpars = np.random.multivariate_normal( mean = ih , cov = ic , size = (n_samples,RT.size,nper) )
		
		## Transform in XF: nper,samples,RT,time
		dims   = ["period","sample","RT","time"]
		coords = [range(nper),range(n_samples),range(RT.size),range(ntime)]
		XF     = xr.DataArray( np.array( [hpars[:,:,i,:] @ projF[i,:,:].T for i in range(nper)] ).reshape(nper,n_samples,RT.size,ntime) , dims = dims , coords = coords )
		XC     = xr.DataArray( np.array( [hpars[:,:,i,:] @ projC[i,:,:].T for i in range(nper)] ).reshape(nper,n_samples,RT.size,ntime) , dims = dims , coords = coords )
		
		## Find law parameters
		dims    = ["sample","RT","period","hpar"]
		coords  = [range(n_samples),range(RT.size),range(nper),list(nslaw.h_name)]
		nspars  = xr.DataArray( hpars[:,:,:,-nslaw.nhpar:] , dims = dims , coords = coords )
		kwargsF = nslaw.draw_params( XF , nspars )
		kwargsC = nslaw.draw_params( XC , nspars )
		
		## Transpose
		kwargsF = { key : kwargsF[key].transpose("period","RT","time","sample") for key in kwargsF }
		kwargsC = { key : kwargsC[key].transpose("period","RT","time","sample") for key in kwargsF }
		
		## Attribution
		idxp = idx + tuple([slice(None) for _ in range(4)])
		if world.upper() == "F":
			IF[idxp] = nslaw.icdf_sf( pF[idxp] , side = side , **kwargsF )
			IC[idxp] = nslaw.icdf_sf( pF[idxp] , side = side , **kwargsC )
			pC[idxp] = nslaw.cdf_sf(  IF[idxp] , side = side , **kwargsC )
		else:
			IF[idxp] = nslaw.icdf_sf( pC[idxp] , side = side , **kwargsF )
			IC[idxp] = nslaw.icdf_sf( pC[idxp] , side = side , **kwargsC )
			pF[idxp] = nslaw.cdf_sf(  IC[idxp] , side = side , **kwargsF )
	
	## Remove 0 and 1
	e  = 10 * sys.float_info.epsilon
	if world.upper() == "F":
		pC = np.where( pC >     e , pC ,     e )
		pC = np.where( pC < 1 - e , pC , 1 - e )
	else:
		pF = np.where( pF >     e , pF ,     e )
		pF = np.where( pF < 1 - e , pF , 1 - e )
	
	## Others variables
	IF  = IF + bias.reshape( ssp + tuple([1 for _ in range(4)]) )
	IC  = IC + bias.reshape( ssp + tuple([1 for _ in range(4)]) )
	RF  = 1. / pF
	RC  = 1. / pC
	dI  = IF - IC
	PR  = pF / pC
	
	## Compute CI
	if mode == "quantile":
		trsp = tuple( [ i + 1 for i in range(pF.ndim-1) ] + [0] )
		pF = np.quantile( pF , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		pC = np.quantile( pC , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		RF = np.quantile( RF , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		RC = np.quantile( RC , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		IF = np.quantile( IF , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		IC = np.quantile( IC , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		dI = np.quantile( dI , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		PR = np.quantile( PR , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
	
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
	ci        = float(bsacParams.config.get("ci",0.05))
	nslaw     = clim._nslaw_class()
	
	## Logs
	logger.info(  " * Configuration" )
	logger.info( f"   => mode: {mode}" )
	logger.info( f"   => side: {side}" )
	logger.info( f"   => ci  : {ci}" )
	
	## Mode dimension
	if mode == "sample":
		modes = np.array(coords_samples( n_samples ))
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
	RT  = xr.DataArray( RT , dims = ["return_time"] , coords = [RT] )
	zRT = zr.ZXArray.from_xarray(RT)
	n_RT = RT.size
	logger.info( f" * Return time: {RT.values}" )
	
	## Build projection operator for the covariable
	projF,projC = clim.projection()
	zprojF = zr.ZXArray.from_xarray(projF.loc[clim.cname,:,:,:])
	zprojC = zr.ZXArray.from_xarray(projC.loc[clim.cname,:,:,:])
	
	## Samples
	n_samples = bsacParams.n_samples
	samples   = coords_samples(n_samples)
	
	##
	time        = np.array(clim.time)
	dpers       = np.array(clim.dpers)
	nslaw_class = clim._nslaw_class
	d_spatial   = clim.d_spatial
	c_spatial   = clim.c_spatial
	zbias       = zr.ZXArray.from_xarray(clim.bias[clim.vname])
	
	## Build arguments
	output_dims      = [ ("return_time",mode,"time","period") + d_spatial for _ in range(8) ]
	output_coords    = [ [RT,modes,time,dpers] + [ c_spatial[d] for d in d_spatial ] for _ in range(8) ]
	output_dtypes    = [clim.hpar.dtype for _ in range(8)]
	dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] , [] , ["time","hpar"] , ["time","hpar"] , [] ],
	                     "output_core_dims" : [ ["time",mode] for _ in range(8) ],
	                     "kwargs"           : { "nslaw_class" : nslaw_class , "side" : side , "world" : arg[0].upper() , "mode" : mode , "n_samples" : n_samples , "ci" : ci } ,
	                     "dask"             : "parallelized",
	                     "output_dtypes"    : [clim.hpar.dtype for _ in range(8)],
		                 "dask_gufunc_kwargs" : { "output_sizes"     : { mode : modes.size } }
	                    }
	
	## Block memory function
	nhpar = projF.hpar.size
	block_memory = lambda x : 2 * ( nhpar + nhpar**2 + 2 * time.size * nhpar + 8 * time.size * n_samples ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
	
	## Run with zxarray
	zargs = [clim.hpar,clim.hcov,zbias,zprojF,zprojC,zRT]
	with bsacParams.get_cluster() as cluster:
		out = zr.apply_ufunc( zattribute_fcreturnt , *zargs,
		                      block_dims         = ("period","return_time") + clim.d_spatial,
		                      total_memory       = bsacParams.total_memory,
		                      block_memory       = block_memory,
		                      output_dims        = output_dims,
		                      output_coords      = output_coords,
		                      output_dtypes      = output_dtypes,
		                      dask_kwargs        = dask_kwargs,
		                      n_workers          = bsacParams.n_workers,
		                      threads_per_worker = bsacParams.threads_per_worker,
		                      cluster            = cluster,
		                    )
	
	## Transform in dict
	keys = ["pF","pC","RF","RC","IF","IC","dI","PR"]
	out  = { key : out[ikey] for ikey,key in enumerate(keys) }
	
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
		if clim.has_spatial is not None and not clim.spatial_is_fake:
			for d in d_spatial:
				ncdims[d] = ncf.createDimension( d , c_spatial[d].size )
		
		## Define variables
		ncvars = {
		  "return_time" : ncf.createVariable( "return_time" , "float32" , ("return_time",) ),
		           mode : ncf.createVariable(     mode      , str       , (mode,)     ),
		       "period" : ncf.createVariable( "period"      , str       , ("period",) ),
		       "time"   : ncf.createVariable( "time"        , "float32" , ("time"  ,) )
		}
		ncvars["return_time"][:] = RT
		ncvars[mode    ][:]      = modes
		ncvars["period"][:]      = dpers
		if clim.has_spatial and not clim.spatial_is_fake:
			for d in d_spatial:
				ncvars[d] = ncf.createVariable( d , "double" , (d,) )
				ncvars[d][:] = np.array(c_spatial[d]).ravel()
		
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
			dims = ("return_time",mode,"time","period")
			chks = (1,1,1,1)
			if not clim.spatial_is_fake:
				dims = dims + d_spatial
				chks = chks + clim.s_spatial
			ncvars[key] = ncf.createVariable( key , "float32" , dims , compression = "zlib" , complevel = 5 , chunksizes = chks )
		
		## Attributes
		ncvars["pF"].setncattr( "description" , "Probability in the Factual world" )
		ncvars["pC"].setncattr( "description" , "Probability in the Counter factual world" )
		ncvars["RF"].setncattr( "description" , "Return time in the Factual world" )
		ncvars["RC"].setncattr( "description" , "Return time in the Counter factual world" )
		ncvars["IF"].setncattr( "description" , "Intensity in the Factual world" )
		ncvars["IC"].setncattr( "description" , "Intensity in the Counter factual world" )
		ncvars["PR"].setncattr( "description" , "Change in probability between Factual and Counter factual world" )
		ncvars["dI"].setncattr( "description" , "Change in intensity between Factual and Counter factual world" )
		
		## And fill variables
		idx = [slice(None) for _ in range(4)]
		if not clim.spatial_is_fake:
			idx = idx + [slice(None) for _ in range(len(clim.d_spatial))]
		idx = tuple(idx)
		for key in out:
			ncvars[key][:] = out[key]._internal.zdata.get_orthogonal_selection(idx)
		
		## Global attributes
		ncf.setncattr( "creation_date" , str(dt.datetime.utcnow())[:19] + " (UTC)" )
		ncf.setncattr( "BSAC_version"  , version )
		ncf.setncattr( "description" , f"Attribute {arg}" )
	
##}}}


## zattribute_fintensity ##{{{

@disable_warnings
def zattribute_fintensity( hpar , hcov , bias , xIF , projF , projC , nslaw_class , side , mode , n_samples , ci ):
	
	## Coordinates
	nhpar = hpar.shape[-1]
	ssp   = hpar.shape[:-2]
	nper  = projF.shape[0]
	ntime = projF.shape[-2]
	
	##
	projF = projF.reshape( (nper,) + projF.shape[-2:] )
	projC = projC.reshape( (nper,) + projF.shape[-2:] )
	nslaw = nslaw_class()
	
	## Prepare output
	pF = np.zeros( ssp + (nper,ntime,n_samples) ) + np.nan
	pC = np.zeros( ssp + (nper,ntime,n_samples) ) + np.nan
	IF = np.zeros( ssp + (nper,ntime,n_samples) ) + xIF.reshape( ssp + (1,1,1) )
	IC = np.zeros( ssp + (nper,ntime,n_samples) ) + np.nan
	
	
	## Loop
	for idx in itt.product(*[range(s) for s in ssp]):
		
		ih = hpar[idx+(0,slice(None))]
		ic = hcov[idx+(0,slice(None),slice(None))]
		
		if not np.isfinite(ih).all() or not np.isfinite(ic).all():
			continue
		
		## Find hpars
		hpars = np.random.multivariate_normal( mean = ih , cov = ic , size = (n_samples,nper) )
		
		## Transform in XF: nper,samples,time
		dims   = ["period","sample","time"]
		coords = [range(nper),range(n_samples),range(ntime)]
		XF     = xr.DataArray( np.array( [hpars[:,i,:] @ projF[i,:,:].T for i in range(nper)] ).reshape(nper,n_samples,ntime) , dims = dims , coords = coords )
		XC     = xr.DataArray( np.array( [hpars[:,i,:] @ projC[i,:,:].T for i in range(nper)] ).reshape(nper,n_samples,ntime) , dims = dims , coords = coords )
		
		## Find law parameters
		dims    = ["sample","period","hpar"]
		coords  = [range(n_samples),range(nper),list(nslaw.h_name)]
		nspars  = xr.DataArray( hpars[:,:,-nslaw.nhpar:] , dims = dims , coords = coords )
		kwargsF = nslaw.draw_params( XF , nspars )
		kwargsC = nslaw.draw_params( XC , nspars )
		
		## Transpose
		kwargsF = { key : kwargsF[key].transpose("period","time","sample") for key in kwargsF }
		kwargsC = { key : kwargsC[key].transpose("period","time","sample") for key in kwargsF }
		
		## Attribution
		idxp = idx + tuple([slice(None) for _ in range(3)])
		pF[idxp] = nslaw.cdf_sf( IF[idxp] , side = side , **kwargsF )
		pC[idxp] = nslaw.cdf_sf( IF[idxp] , side = side , **kwargsC )
		
		## Remove 0 and 1
		e  = 10 * sys.float_info.epsilon
		pF[idxp] = np.where( pF[idxp] >     e , pF[idxp] ,     e )
		pC[idxp] = np.where( pC[idxp] >     e , pC[idxp] ,     e )
		pF[idxp] = np.where( pF[idxp] < 1 - e , pF[idxp] , 1 - e )
		pC[idxp] = np.where( pC[idxp] < 1 - e , pC[idxp] , 1 - e )
		
		## Factual and counter factual intensities
		IC[idxp] = nslaw.icdf_sf( pF[idxp] , side = side , **kwargsC )
	
	## Others variables
	IF  = IF + bias.reshape( ssp + tuple([1 for _ in range(3)]) )
	IC  = IC + bias.reshape( ssp + tuple([1 for _ in range(3)]) )
	RF  = 1. / pF
	RC  = 1. / pC
	dI  = IF - IC
	PR  = pF / pC
	
	## Compute CI
	if mode == "quantile":
		trsp = tuple( [ i + 1 for i in range(pF.ndim-1) ] + [0] )
		pF = np.quantile( pF , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		pC = np.quantile( pC , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		RF = np.quantile( RF , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		RC = np.quantile( RC , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		IF = np.quantile( IF , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		IC = np.quantile( IC , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		dI = np.quantile( dI , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		PR = np.quantile( PR , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
	
	
	out = [pF,pC,RF,RC,IF,IC,dI,PR]
	
	return tuple(out)
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
	ci        = float(bsacParams.config.get("ci",0.05))
	nslaw     = clim._nslaw_class()
	
	## Logs
	logger.info(  " * Configuration" )
	logger.info( f"   => mode: {mode}" )
	logger.info( f"   => side: {side}" )
	logger.info( f"   => ci  : {ci}" )
	
	## Mode dimension
	if mode == "sample":
		modes = np.array(coords_samples( n_samples ))
	elif mode == "quantile":
		modes = np.array(["QL","BE","QU"])
	else:
		raise ValueError( f"Invalid mode ({mode})" )
	n_modes = modes.size
	
	## Read the intensity
	try:
		xIF = float(bsacParams.input[0]) + clim.bias[clim.vname] * 0
	except:
		name,ifile = bsacParams.input[0].split(",")
		xIF = xr.open_dataset(ifile)[name]
	
	## Remove bias
	xIF = xIF - clim.bias[clim.vname]
	
	## Check the spatial dimensions
	for d in clim.d_spatial:
		if d not in xIF.dims:
			raise Exception( f"Spatial dimension missing: {d}" )
		if not xIF[d].size == clim._spatial[d].size:
			raise Exception( f"Bad size of the dimension {d}: {xIF[d].size} != {clim._spatial[d].size}" )
	
	## Reorganize dimension and transform in zarr
	zIF = zr.ZXArray.from_xarray( xIF.transpose(*clim.d_spatial).copy() )
	
	## Build projection operator for the covariable
	projF,projC = clim.projection()
	zprojF = zr.ZXArray.from_xarray(projF.loc[clim.cname,:,:,:])
	zprojC = zr.ZXArray.from_xarray(projC.loc[clim.cname,:,:,:])
	
	## Samples
	n_samples = bsacParams.n_samples
	samples   = coords_samples(n_samples)
	
	##
	time        = np.array(clim.time)
	dpers       = np.array(clim.dpers)
	nslaw_class = clim._nslaw_class
	d_spatial   = clim.d_spatial
	c_spatial   = clim.c_spatial
	zbias       = zr.ZXArray.from_xarray(clim.bias[clim.vname])
	
	## Build arguments
	output_dims      = [ (mode,"time","period") + d_spatial for _ in range(8) ]
	output_coords    = [ [modes,time,dpers] + [ c_spatial[d] for d in d_spatial ] for _ in range(8) ]
	output_dtypes    = [clim.hpar.dtype for _ in range(8)]
	dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] , [] , [] , ["time","hpar"] , ["time","hpar"] ],
	                     "output_core_dims" : [ ["time",mode] for _ in range(8) ],
	                     "kwargs"           : { "nslaw_class" : nslaw_class , "side" : side , "mode" : mode , "n_samples" : n_samples , "ci" : ci } ,
	                     "dask"             : "parallelized",
	                     "output_dtypes"    : [clim.hpar.dtype for _ in range(8)],
		                 "dask_gufunc_kwargs" : { "output_sizes"     : { mode : modes.size } }
	                    }
	
	## Block memory function
	nhpar = projF.hpar.size
	block_memory = lambda x : 2 * ( nhpar + 2 * nhpar + 2 * time.size * nhpar + 8 * time.size * n_samples ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
	
	## Run with zxarray
	zargs = [clim.hpar,clim.hcov,zbias,zIF,zprojF,zprojC]
	with bsacParams.get_cluster() as cluster:
		out = zr.apply_ufunc( zattribute_fintensity , *zargs,
		                      block_dims         = ("period",) + clim.d_spatial,
		                      total_memory       = bsacParams.total_memory,
		                      block_memory       = block_memory,
		                      output_dims        = output_dims,
		                      output_coords      = output_coords,
		                      output_dtypes      = output_dtypes,
		                      dask_kwargs        = dask_kwargs,
		                      n_workers          = bsacParams.n_workers,
		                      threads_per_worker = bsacParams.threads_per_worker,
		                      cluster            = cluster,
		                    )
	
	## Transform in dict
	keys = ["pF","pC","RF","RC","IF","IC","dI","PR"]
	out  = { key : out[ikey] for ikey,key in enumerate(keys) }
	
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
		spatial = ()
		if clim.has_spatial and not clim.spatial_is_fake:
			for d in d_spatial:
				ncdims[d] = ncf.createDimension( d , c_spatial[d].size )
		
		## Define variables
		ncvars = {
		           mode : ncf.createVariable(     mode      , str       , (mode,)     ),
		       "period" : ncf.createVariable( "period"      , str       , ("period",) ),
		       "time"   : ncf.createVariable( "time"        , "float32" , ("time"  ,) )
		}
		ncvars[mode    ][:]      = modes
		ncvars["period"][:]      = dpers
		if clim.has_spatial and not clim.spatial_is_fake:
			for d in d_spatial:
				ncvars[d] = ncf.createVariable( d , "double" , (d,) )
				ncvars[d][:] = np.array(c_spatial[d]).ravel()
		
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
			dims = (mode,"time","period")
			chks = (1,1,1)
			if not clim.spatial_is_fake:
				dims = dims + d_spatial
				chks = chks + clim.s_spatial
			ncvars[key] = ncf.createVariable( key , "float32" , dims , compression = "zlib" , complevel = 5 , chunksizes = chks )
		
		## Attributes
		ncvars["pF"].setncattr( "description" , "Probability in the Factual world" )
		ncvars["pC"].setncattr( "description" , "Probability in the Counter factual world" )
		ncvars["RF"].setncattr( "description" , "Return time in the Factual world" )
		ncvars["RC"].setncattr( "description" , "Return time in the Counter factual world" )
		ncvars["IF"].setncattr( "description" , "Intensity in the Factual world" )
		ncvars["IC"].setncattr( "description" , "Intensity in the Counter factual world" )
		ncvars["PR"].setncattr( "description" , "Change in probability between Factual and Counter factual world" )
		ncvars["dI"].setncattr( "description" , "Change in intensity between Factual and Counter factual world" )
		
		## And fill variables
		idx = [slice(None) for _ in range(3)]
		if not clim.spatial_is_fake:
			idx = idx + [slice(None) for _ in range(len(clim.d_spatial))]
		idx = tuple(idx)
		for key in out:
			ncvars[key][:] = out[key]._internal.zdata.get_orthogonal_selection(idx)
		
		## Global attributes
		ncf.setncattr( "creation_date" , str(dt.datetime.utcnow())[:19] + " (UTC)" )
		ncf.setncattr( "BSAC_version"  , version )
		ncf.setncattr( "description" , f"Attribute {arg}" )
	
##}}}


## zattribute_event ##{{{

@disable_warnings
def zattribute_event( hpar , hcov , bias , event , projF , projC , idx_event , nslaw_class , side , mode , n_samples , ci ):
	
	## Coordinates
	nhpar = hpar.shape[-1]
	ssp   = hpar.shape[:-2]
	nper  = projF.shape[0]
	ntime = projF.shape[-2]
	
	##
	projF = projF.reshape( (nper,) + projF.shape[-2:] )
	projC = projC.reshape( (nper,) + projF.shape[-2:] )
	nslaw = nslaw_class()
	
	## Prepare output
	pF = np.zeros( ssp + (nper,ntime,n_samples) ) + np.nan
	pC = np.zeros( ssp + (nper,ntime,n_samples) ) + np.nan
	IF = np.zeros( ssp + (nper,ntime,n_samples) ) + np.nan
	IC = np.zeros( ssp + (nper,ntime,n_samples) ) + np.nan
	
	##
	event = event.reshape( ssp + (1,1,1) )
	
	## Loop
	for idx in itt.product(*[range(s) for s in ssp]):
		
		ih = hpar[idx+(0,slice(None))]
		ic = hcov[idx+(0,slice(None),slice(None))]
		
		if not np.isfinite(ih).all() or not np.isfinite(ic).all():
			continue
		
		## Find hpars
		hpars = np.random.multivariate_normal( mean = ih , cov = ic , size = (n_samples,nper) )
		
		## Transform in XF: nper,samples,time
		dims   = ["period","sample","time"]
		coords = [range(nper),range(n_samples),range(ntime)]
		XF     = xr.DataArray( np.array( [hpars[:,i,:] @ projF[i,:,:].T for i in range(nper)] ).reshape(nper,n_samples,ntime) , dims = dims , coords = coords )
		XC     = xr.DataArray( np.array( [hpars[:,i,:] @ projC[i,:,:].T for i in range(nper)] ).reshape(nper,n_samples,ntime) , dims = dims , coords = coords )
		
		## Find law parameters
		dims    = ["sample","period","hpar"]
		coords  = [range(n_samples),range(nper),list(nslaw.h_name)]
		nspars  = xr.DataArray( hpars[:,:,-nslaw.nhpar:] , dims = dims , coords = coords )
		kwargsF = nslaw.draw_params( XF , nspars )
		kwargsC = nslaw.draw_params( XC , nspars )
		
		## Transpose
		kwargsF = { key : kwargsF[key].transpose("period","time","sample") for key in kwargsF }
		kwargsC = { key : kwargsC[key].transpose("period","time","sample") for key in kwargsF }
		
		## Attribution
		idxp = idx + tuple([slice(None) for _ in range(3)])
		pF[idxp] = nslaw.cdf_sf( event[idxp] , side = side , **kwargsF )
		pC[idxp] = nslaw.cdf_sf( event[idxp] , side = side , **kwargsC )
		
		## Remove 0 and 1
		e  = 10 * sys.float_info.epsilon
		pF[idxp] = np.where( pF[idxp] >     e , pF[idxp] ,     e )
		pC[idxp] = np.where( pC[idxp] >     e , pC[idxp] ,     e )
		pF[idxp] = np.where( pF[idxp] < 1 - e , pF[idxp] , 1 - e )
		pC[idxp] = np.where( pC[idxp] < 1 - e , pC[idxp] , 1 - e )
		
		##
		pf = np.zeros_like(pF) + pF[ idx + (slice(None),idx_event,slice(None)) ].reshape( *([1 for _ in range(len(ssp))] + [nper,1,n_samples]) )
		
		## Factual and counter factual intensities
		IF[idxp] = nslaw.icdf_sf( pf[idxp] , side = side , **kwargsF )
		IC[idxp] = nslaw.icdf_sf( pf[idxp] , side = side , **kwargsC )
	
	## Others variables
	IF  = IF + bias.reshape( ssp + tuple([1 for _ in range(3)]) )
	IC  = IC + bias.reshape( ssp + tuple([1 for _ in range(3)]) )
	RF  = 1. / pF
	RC  = 1. / pC
	dI  = IF - IC
	PR  = pF / pC
	
	## Compute CI
	if mode == "quantile":
		trsp = tuple( [ i + 1 for i in range(pF.ndim-1) ] + [0] )
		pF = np.quantile( pF , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		pC = np.quantile( pC , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		RF = np.quantile( RF , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		RC = np.quantile( RC , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		IF = np.quantile( IF , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		IC = np.quantile( IC , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		dI = np.quantile( dI , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
		PR = np.quantile( PR , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose(trsp)
	
	
	out = [pF,pC,RF,RC,IF,IC,dI,PR]
	
	return tuple(out)
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
	ci        = float(bsacParams.config.get("ci",0.05))
	t_event   = int(bsacParams.config["time"])
	idx_event = int(np.argwhere( time == t_event ).ravel())
	nslaw     = clim._nslaw_class()
	
	## Logs
	logger.info(  " * Configuration" )
	logger.info( f"   => mode: {mode}" )
	logger.info( f"   => side: {side}" )
	logger.info( f"   => ci  : {ci}" )
	
	## Mode dimension
	if mode == "sample":
		modes = np.array(coords_samples( n_samples ))
	elif mode == "quantile":
		modes = np.array(["QL","BE","QU"])
	else:
		raise ValueError( f"Invalid mode ({mode})" )
	n_modes = modes.size
	
	## Load observations
	name,ifile = bsacParams.input[0].split(",")
	Yo  = xr.open_dataset(ifile)[name]
	
	## Select year
	if "time" in Yo.dims:
		Yo = Yo.assign_coords( time = Yo.time.dt.year )
		if Yo.time.size > 1:
			Yo = Yo.sel( time = int(t_event) )
		else:
			Yo = Yo.sel( time = Yo.time[0] )
		Yo = Yo.drop_vars("time")
	
	## Remove bias
	Yo = Yo - clim.bias[clim.vname]
	
	## Check the spatial dimensions
	for d in clim.d_spatial:
		if d not in Yo.dims:
			raise Exception( f"Spatial dimension missing: {d}" )
		if not Yo[d].size == clim._spatial[d].size:
			raise Exception( f"Bad size of the dimension {d}: {Yo[d].size} != {clim._spatial[d].size}" )
	
	## Reorganize dimension and transform in zarr
	zYo = zr.ZXArray.from_xarray( Yo.transpose(*clim.d_spatial).copy() )
	
	## Build projection operator for the covariable
	projF,projC = clim.projection()
	zprojF = zr.ZXArray.from_xarray(projF.loc[clim.cname,:,:,:])
	zprojC = zr.ZXArray.from_xarray(projC.loc[clim.cname,:,:,:])
	
	## Samples
	n_samples = bsacParams.n_samples
	samples   = coords_samples(n_samples)
	
	##
	time        = np.array(clim.time)
	dpers       = np.array(clim.dpers)
	nslaw_class = clim._nslaw_class
	d_spatial   = clim.d_spatial
	c_spatial   = clim.c_spatial
	zbias       = zr.ZXArray.from_xarray(clim.bias[clim.vname])
	
	## Build arguments
	output_dims      = [ (mode,"time","period") + d_spatial for _ in range(8) ]
	output_coords    = [ [modes,time,dpers] + [ c_spatial[d] for d in d_spatial ] for _ in range(8) ]
	output_dtypes    = [clim.hpar.dtype for _ in range(8)]
	dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] , [] , [] , ["time","hpar"] , ["time","hpar"] ],
	                     "output_core_dims" : [ ["time",mode] for _ in range(8) ],
	                     "kwargs"           : { "idx_event" : idx_event , "nslaw_class" : nslaw_class , "side" : side , "mode" : mode , "n_samples" : n_samples , "ci" : ci } ,
	                     "dask"             : "parallelized",
	                     "output_dtypes"    : [clim.hpar.dtype for _ in range(8)],
		                 "dask_gufunc_kwargs" : { "output_sizes"     : { mode : modes.size } }
	                    }
	
	## Block memory function
	nhpar = projF.hpar.size
	block_memory = lambda x : 2 * ( nhpar + 2 * nhpar + 2 * time.size * nhpar + 8 * time.size * n_samples ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
	
	## Run with zxarray
	zargs = [clim.hpar,clim.hcov,zbias,zYo,zprojF,zprojC]
	with bsacParams.get_cluster() as cluster:
		out = zr.apply_ufunc( zattribute_event , *zargs,
		                      block_dims         = ("period",) + clim.d_spatial,
		                      total_memory       = bsacParams.total_memory,
		                      block_memory       = block_memory,
		                      output_dims        = output_dims,
		                      output_coords      = output_coords,
		                      output_dtypes      = output_dtypes,
		                      dask_kwargs        = dask_kwargs,
		                      n_workers          = bsacParams.n_workers,
		                      threads_per_worker = bsacParams.threads_per_worker,
		                      cluster            = cluster,
		                    )
	
	## Transform in dict
	keys = ["pF","pC","RF","RC","IF","IC","dI","PR"]
	out  = { key : out[ikey] for ikey,key in enumerate(keys) }
	
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
		spatial = ()
		if clim.has_spatial and not clim.spatial_is_fake:
			for d in d_spatial:
				ncdims[d] = ncf.createDimension( d , c_spatial[d].size )
		
		## Define variables
		ncvars = {
		           mode : ncf.createVariable(     mode      , str       , (mode,)     ),
		       "period" : ncf.createVariable( "period"      , str       , ("period",) ),
		       "time"   : ncf.createVariable( "time"        , "float32" , ("time"  ,) )
		}
		ncvars[mode    ][:]      = modes
		ncvars["period"][:]      = dpers
		if clim.has_spatial and not clim.spatial_is_fake:
			for d in d_spatial:
				ncvars[d] = ncf.createVariable( d , "double" , (d,) )
				ncvars[d][:] = np.array(c_spatial[d]).ravel()
		
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
			dims = (mode,"time","period")
			chks = (1,1,1)
			if not clim.spatial_is_fake:
				dims = dims + d_spatial
				chks = chks + clim.s_spatial
			ncvars[key] = ncf.createVariable( key , "float32" , dims , compression = "zlib" , complevel = 5 , chunksizes = chks )
		
		## Attributes
		ncvars["pF"].setncattr( "description" , "Probability in the Factual world" )
		ncvars["pC"].setncattr( "description" , "Probability in the Counter factual world" )
		ncvars["RF"].setncattr( "description" , "Return time in the Factual world" )
		ncvars["RC"].setncattr( "description" , "Return time in the Counter factual world" )
		ncvars["IF"].setncattr( "description" , "Intensity in the Factual world" )
		ncvars["IC"].setncattr( "description" , "Intensity in the Counter factual world" )
		ncvars["PR"].setncattr( "description" , "Change in probability between Factual and Counter factual world" )
		ncvars["dI"].setncattr( "description" , "Change in intensity between Factual and Counter factual world" )
		
		## And fill variables
		idx = [slice(None) for _ in range(3)]
		if not clim.spatial_is_fake:
			idx = idx + [slice(None) for _ in range(len(clim.d_spatial))]
		idx = tuple(idx)
		for key in out:
			ncvars[key][:] = out[key]._internal.zdata.get_orthogonal_selection(idx)
		
		## Global attributes
		ncf.setncattr( "creation_date" , str(dt.datetime.utcnow())[:19] + " (UTC)" )
		ncf.setncattr( "BSAC_version"  , version )
		ncf.setncattr( "description" , f"Attribute event {t_event}" )
	
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


