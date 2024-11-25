
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
import gc
import logging
import datetime as dt
import itertools as itt
from ..__logs import LINE
from ..__logs import log_start_end
from ..__logs import disable_warnings

from ..__BSACParams import bsacParams
from ..__climatology import Climatology

import numpy  as np
import xarray as xr
import zxarray as zr

from ..__release import version
from ..__sys import coords_samples
import netCDF4


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############


## zwpe ##{{{
##TODO: add dI

@disable_warnings
def zwpe( hpar , hcov , bias , proj , pwpe , nslaw_class , side , n_samples , mode , ci ):
	
	##
	## hpar = ssp + (1,1,) + (nhpar,)
	## hcov = ssp + (1,1,) + (nhpar,nhpar)
	## bias = ssp + (1,1,)
	## proj = (nper,1,ntime,nhpar)
	## pwpe = (npwpe,)
	##
	
	## Find parameters
	ssp     = hpar.shape[:-3]
	nhpar   = hpar.shape[-1]
	ntime   = proj.shape[-2]
	nper    = proj.shape[0]
	npwpe   = pwpe.size
	n_modes = n_samples if mode == "sample" else 3
	nslaw   = nslaw_class()
	
	## Output
	IFC = np.zeros( ssp + (nper,npwpe,n_modes) ) + np.nan
	dI  = np.zeros( ssp + (nper,npwpe,n_modes) ) + np.nan
	
	## Loop on spatial coordinates
	for idx0 in itt.product( *[range(s) for s in ssp] ):
		
		## Draw parameters
		idx1d = idx0 + (0,0) + tuple([slice(None) for _ in range(1)])
		idx2d = idx0 + (0,0) + tuple([slice(None) for _ in range(2)])
		h     = hpar[idx1d]
		c     = hcov[idx1d]
		b     = bias[idx0 + (0,0)]
		
		if not np.isfinite(h).all() or not np.isfinite(c).all():
			continue
		
		hpars = np.random.multivariate_normal( mean = h , cov = c , size = npwpe * n_samples )
		
		## Transform in XFC
		dims   = ["period","time","pwpe","sample"]
		coords = [range(nper),range(ntime),range(npwpe),range(n_samples)]
		XFC    = xr.DataArray( np.array( [ proj[i,0,:,:] @ hpars.T for i in range(nper) ] ).reshape(nper,ntime,npwpe,n_samples) , dims = dims , coords = coords )
		
		## Find law parameters
		dims   = ["sample","pwpe","hpar"]
		coords = [range(n_samples),range(npwpe),nslaw.coef_name]
		nspars = xr.DataArray( hpars.reshape(n_samples,npwpe,nhpar)[:,:,-nslaw.n_coef:] , dims = dims , coords = coords )
		kwargs = nslaw.draw_params( XFC , nspars )
		kwargs = { key : kwargs[key].transpose("period","time","pwpe","sample") for key in kwargs }
		
		## Init eventL, eventH and eventM
		eventL = nslaw.icdf_sf( min(     1e-6 ,         pwpe.min()  / 100 ) , side , **kwargs )
		eventH = nslaw.icdf_sf( max( 1 - 1e-6 ,1 - (1 - pwpe.max()) / 100 ) , side , **kwargs )
		if side == "right":
			eventL,eventH = eventH,eventL
		eventL[:] = np.nanmin( eventL )
		eventH[:] = np.nanmax( eventH )
		eventM    = ( eventL + eventH ) / 2
		
		## Loop for optimization
		res = float(np.nanmax(eventH)) - float(np.nanmin(eventL))
		eps = min( 0.01 * res , 1 ) * 1e-4
		while res > eps:
			
			## Probability of the central event
			pM = 1 - np.prod( 1 - nslaw.cdf_sf( eventM , side , **kwargs ) , axis = 1 )
			
			## Update
			eventH = np.where( pM.reshape(nper,1,npwpe,n_samples) > pwpe.reshape(1,1,-1,1) , eventH , eventM )
			eventL = np.where( pM.reshape(nper,1,npwpe,n_samples) < pwpe.reshape(1,1,-1,1) , eventL , eventM )
			eventM = ( eventL + eventH ) / 2
			
			##
			res = float(np.nanmax(np.abs(eventH - eventL)))
		
		## Remove time axis (in fact all values are the same now along time axis)
		event  = np.nanmean( eventM , axis = 1 )
		devent = event - event[0,:,:].reshape(1,npwpe,n_samples)
		
		## Quantile ?
		if mode == "quantile":
			event  = np.quantile(  event , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose( 1 , 2 , 0 )
			devent = np.quantile( devent , [ci/2,0.5,1-ci/2] , axis = -1 , method = "median_unbiased" ).transpose( 1 , 2 , 0 )
		
		## And store
		idx3d = idx0 + tuple([slice(None) for _ in range(3)])
		IFC[idx3d] =  event + b
		dI[idx3d]  = devent
	
	return IFC,dI
##}}}

## run_bsac_cmd_misc_wpe ##{{{
@log_start_end(logger)
def run_bsac_cmd_misc_wpe():
	
	## Parameters
	clim      = bsacParams.clim
	n_samples = bsacParams.n_samples
	tmp       = bsacParams.tmp
	side      = bsacParams.config.get("side","right")
	mode      = bsacParams.config.get("mode","quantile")
	ci        = float(bsacParams.config.get("ci",0.05))
	nslaw     = clim._nslaw_class()
	time_wpe  = bsacParams.config.get("period")
	
	##
	if time_wpe is None:
		time_wpe = [dt.datetime.now(dt.UTC).year,int(time[-1])]
	else:
		time_wpe = [int(y) for y in time_wpe.split("/")]
	
	## Find the probabilities
	pwpe = bsacParams.input[0]
	if pwpe == "IPCC":
		pwpe = [0.01,0.1,0.33,0.5,0.66,0.9,0.99]
	else:
		try:
			pwpe = [float(s) for s in pwpe.split(":")]
		except:
			raise ValueError("Bad format for input probabilities of wpe")
	pwpe  = xr.DataArray( pwpe , dims = ["pwpe"] , coords = [pwpe] )
	zpwpe = zr.ZXArray.from_xarray(pwpe)
	
	logger.info( " * Probabilities found: {}".format(pwpe.values) )
	
	## Extract parameters
	d_spatial   = clim.d_spatial
	c_spatial   = clim.c_spatial
	hpar_names  = clim.hpar_names
	nslaw_class = clim._nslaw_class
	
	ihpar     = clim.hpar
	ihcov     = clim.hcov
	zbias     = zr.ZXArray.from_xarray(clim.bias[clim.names[-1]])
	samples   = np.array(coords_samples(n_samples))
	
	## Find mode
	if mode == "quantile":
		modes = np.array(["QL","BE","QU"])
	else:
		modes = samples
	
	## Projection matrix
	projF,projC = clim.projection()
	projC = projC.assign_coords( period = [ f"c_{p}" for p in projC.period.values.tolist() ] )
	proj  = xr.concat( (projC,projF) , dim = "period" ).sel( period = [projC.period.values.tolist()[0]] + projF.period.values.tolist() ).assign_coords( period = ["cfactual"] + projF.period.values.tolist() )
	proj  = proj[-1,:,:,:].drop("name")
	proj  = proj.sel( time = slice(*time_wpe) )
	zproj = zr.ZXArray.from_xarray(proj)
	
	## zxarray parameters
	logger.info( " * zxarray parameters..." )
	zargs = [ihpar,ihcov,zbias,zproj,zpwpe]
	output_dims      = [ (mode,"pwpe","period") + d_spatial for _ in range(2) ]
	output_coords    = [ [modes,pwpe,proj.period] + [ c_spatial[d] for d in d_spatial ] for _ in range(2) ]
	output_dtypes    = [clim.hpar.dtype for _ in range(2)]
	dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] , [] , ["time","hpar"] , [] ],
	                     "output_core_dims" : [[mode],[mode]],
	                     "kwargs"           : { "nslaw_class" : nslaw_class , "side" : side , "n_samples" : n_samples , "mode" : mode , "ci" : ci } ,
	                     "dask"             : "parallelized",
	                     "output_dtypes"    : [ihpar.dtype,ihpar.dtype],
		                 "dask_gufunc_kwargs" : { "output_sizes"     : { mode : modes.size} }
	                    }
	
	## Run
	logger.info( " * and run" )
	zIFC,zdI = zr.apply_ufunc( zwpe , *zargs,
	                           bdims         = d_spatial + ("pwpe",),
	                           max_mem       = bsacParams.total_memory,
	                           output_dims   = output_dims,
	                           output_coords = output_coords,
	                           output_dtypes = output_dtypes,
	                           dask_kwargs   = dask_kwargs
	                        )
	
	##
	periods = proj.period.values
	
	## And save in netcdf
	logger.info( " * Save in netcdf" )
	with netCDF4.Dataset( bsacParams.output , "w" ) as ncf:
		
		## Define dimensions
		ncdims = {
		       "probability" : ncf.createDimension( "probability"   , len(pwpe) ),
		           mode : ncf.createDimension(  mode    , len(modes)   ),
		       "period" : ncf.createDimension( "period" , len(periods) )
		}
		if clim.has_spatial:
			for d in clim.d_spatial:
				ncdims[d] = ncf.createDimension( d , clim._spatial[d].size )
			spatial = tuple([d for d in clim.d_spatial])
		
		## Define variables
		ncvars = {
		  "probability" : ncf.createVariable( "probability" , "float32" , ("probability",) ),
		           mode : ncf.createVariable(     mode , str       , (mode,) ),
		       "period" : ncf.createVariable( "period" , str       , ("period",) )
		}
		ncvars["probability"][:] = np.array([pwpe]).squeeze()
		ncvars[mode][:]          = np.array([modes]).squeeze()
		ncvars["period"][:]      = np.array([periods]).squeeze()
		if clim.has_spatial:
			for d in clim.d_spatial:
				ncvars[d] = ncf.createVariable( d , "double" , (d,) )
				ncvars[d][:] = np.array(clim._spatial[d]).ravel()
		if mode == "quantile":
			ncvars[mode].setncattr( "confidence_level" , ci )
			ncvars["quantile_level"] = ncf.createVariable( "quantile_levels" , "float32" , ("quantile") )
			ncvars["quantile_level"][:] = np.array([ci/2,0.5,1-ci/2])
		
		## Variables
		ncvars["IFC"] = ncf.createVariable( "IFC" , zIFC.dtype , (mode,"probability","period") + spatial , compression = "zlib" , complevel = 5 , chunksizes = (1,1,1) + clim.s_spatial )
		ncvars["dI"]  = ncf.createVariable( "dI"  ,  zdI.dtype , (mode,"probability","period") + spatial , compression = "zlib" , complevel = 5 , chunksizes = (1,1,1) + clim.s_spatial )
		
		## Attributes
		ncvars["IFC"].setncattr( "description" , "Intensity in Factual / Counter factual world of the Worst Possible Event with Probability probability" )
		ncvars["dI"].setncattr( "description" , "Change in intensity in between factual and counter factual world of the Worst Possible Event with Probability probability" )
		
		## Fill
		ncvars["IFC"][:] = zIFC._internal.zdata[:]
		ncvars["dI"][:]  =  zdI._internal.zdata[:]
		
		## Global attributes
		ncf.setncattr( "creation_date" , str(dt.datetime.utcnow())[:19] + " (UTC)" )
		ncf.setncattr( "BSAC_version"  , version )
		ncf.setncattr( "description" , f"Worst Possible Event" )
##}}}


## run_bsac_cmd_misc ##{{{
@log_start_end(logger)
def run_bsac_cmd_misc():
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the fit command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["wpe"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the fit command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "wpe":
		run_bsac_cmd_misc_wpe()
##}}}


