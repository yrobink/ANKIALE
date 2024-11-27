
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

import itertools as itt
import logging

from ..__logs import LINE
from ..__logs import log_start_end
from ..__logs import disable_warnings

from ..__BSACParams import bsacParams

from ..__sys import coords_samples

import numpy as np
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

## zdraw ##{{{

@disable_warnings
def zdraw( hpar , hcov , projF , projC , nslaw_class , n_samples ):
	
	## Coordinates
	nhpar = hpar.shape[-1]
	ssp   = hpar.shape[:-2]
	nper  = projF.shape[0]
	nname = projF.shape[1]
	ntime = projF.shape[-2]
	
	##
	projF = projF.reshape( projF.shape[-4:] )
	projC = projC.reshape( projF.shape[-4:] )
	nslaw = nslaw_class()
	
	## Build output
	ohpars  = np.zeros( ssp + (nper,n_samples,nhpar) ) + np.nan
	oXF     = np.zeros( ssp + (nper,nname,n_samples,ntime) ) + np.nan
	oXC     = np.zeros( ssp + (nper,nname,n_samples,ntime) ) + np.nan
	onsparF = [ np.zeros( ssp + (nper,n_samples,ntime) ) + np.nan for _ in range(nslaw.npar) ]
	onsparC = [ np.zeros( ssp + (nper,n_samples,ntime) ) + np.nan for _ in range(nslaw.npar) ]
	
	## Loop
	for idx in itt.product(*[range(s) for s in ssp]):
		
		ih = hpar[idx+(0,slice(None))]
		ic = hcov[idx+(0,slice(None),slice(None))]
		
		if not np.isfinite(ih).all() or not np.isfinite(ic).all():
			continue
		
		## Find hpars
		hpars = np.random.multivariate_normal( mean = ih , cov = ic , size = (n_samples,nper) )
		
		## Transform in XF: nper,name,samples,time
		dims   = ["period","name","sample","time"]
		coords = [range(nper),range(nname),range(n_samples),range(ntime)]
		XF     = xr.DataArray( np.array( [[hpars[:,i,:] @ projF[i,j,:,:].T for j in range(nname)] for i in range(nper)] ) , dims = dims , coords = coords )
		XC     = xr.DataArray( np.array( [[hpars[:,i,:] @ projC[i,j,:,:].T for j in range(nname)] for i in range(nper)] ) , dims = dims , coords = coords )
		
		## Find law parameters
		dims    = ["sample","period","hpar"]
		coords  = [range(n_samples),range(nper),list(nslaw.h_name)]
		nspars  = xr.DataArray( hpars[:,:,-nslaw.nhpar:] , dims = dims , coords = coords )
		kwargsF = nslaw.draw_params( XF[:,-1,:,:].drop_vars("name") , nspars )
		kwargsC = nslaw.draw_params( XC[:,-1,:,:].drop_vars("name") , nspars )
		
		## Transpose
		kwargsF = { key : kwargsF[key].transpose("period","time","sample") for key in kwargsF }
		kwargsC = { key : kwargsC[key].transpose("period","time","sample") for key in kwargsF }
		
		## Store
		idx0 = idx + tuple([slice(None) for _ in range(3)])
		idx1 = idx + tuple([slice(None) for _ in range(4)])
		ohpars[idx0] = hpars.transpose(1,0,2)[:]
		oXF[idx1] = XF[:]
		oXC[idx1] = XC[:]
		for ikey,key in enumerate(kwargsF):
			onsparF[ikey][idx0] = kwargsF[key].values.transpose(0,2,1)[:]
			onsparC[ikey][idx0] = kwargsC[key].values.transpose(0,2,1)[:]
	
	##
	out = [ohpars,oXF,oXC] + onsparF + onsparC
	
	return tuple(out)
##}}}

## run_bsac_cmd_draw ##{{{
@log_start_end(logger)
def run_bsac_cmd_draw():
	
	## Parameters
	clim        = bsacParams.clim
	n_samples   = bsacParams.n_samples
	nslaw_class = clim._nslaw_class
	time        = clim.time
	
	## Build projection operator for the covariable
	logger.info(" * Build projection operator")
	projF,projC = clim.projection()
	zprojF      = zr.ZXArray.from_xarray(projF)
	zprojC      = zr.ZXArray.from_xarray(projC)
	
	##
	zargs = [clim.hpar,clim.hcov,zprojF,zprojC]
	
	d_spatial  = clim.d_spatial
	c_spatial  = clim.c_spatial
	hpar_names = clim.hpar_names
	dpers   = clim.dpers
	samples = coords_samples(n_samples)
	time    = clim.time
	nslaw   = nslaw_class()
	npar    = nslaw.npar
	
	output_dims      = [ ("period","sample","hpar") + d_spatial ] + [ ("name","period","sample","time") + d_spatial for _ in range(2) ]  + [ ("period","sample","time") + d_spatial for _ in range(2 * npar) ]
	output_coords    = [ [dpers,samples,hpar_names] + [ c_spatial[d] for d in d_spatial ] ] + [ [clim.namesX,dpers,samples,time] + [ c_spatial[d] for d in d_spatial ] for _ in range(2) ] + [ [dpers,samples,time] + [ c_spatial[d] for d in d_spatial ] for _ in range(2*npar) ]
	output_dtypes    = [clim.hpar.dtype for _ in range( 3 + 2 * npar)]
	dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] , ["name","time","hpar"] , ["name","time","hpar"] ],
	                     "output_core_dims" : [ ["sample","hpar"] , ["name","sample","time"] , ["name","sample","time"] ] + [ ["sample","time"] for _ in range(2 * npar) ],
	                     "kwargs"           : { "nslaw_class" : nslaw_class , "n_samples" : n_samples } ,
	                     "dask"             : "parallelized",
	                     "output_dtypes"    : [clim.hpar.dtype for _ in range(3 + 2*npar)],
		                 "dask_gufunc_kwargs" : { "output_sizes"     : { "sample" : n_samples } }
	                    }
	
	##
#	xargs = [ K.dataarray for K in zargs ]
#	out = xr.apply_ufunc( zdraw , *xargs , **dask_kwargs )
	
	## Run with zxarray
	logger.info(" * Draw parameters")
	out = zr.apply_ufunc( zdraw , *zargs,
	                      bdims         = ("period",) + clim.d_spatial,
	                      max_mem       = bsacParams.total_memory,
	                      output_dims   = output_dims,
	                      output_coords = output_coords,
	                      output_dtypes = output_dtypes,
	                      dask_kwargs   = dask_kwargs,
	                    )
	
	## Transform in dict with names
	keys = ["hpars","XF","XC"] + [ f"{n}F" for n in nslaw.p_name ] + [ f"{n}C" for n in nslaw.p_name ]
	out  = { key : out[i] for i,key in enumerate(keys) }
	
	## Save in netcdf
	logger.info(" * Save in netCDF")
	with netCDF4.Dataset( bsacParams.output , "w" ) as ncf:
		
		## Define dimensions
		ncdims = {
		       "sample"   : ncf.createDimension( "sample" , len(samples) ),
		       "name"     : ncf.createDimension(   "name" , len(clim.namesX) ),
		       "time"     : ncf.createDimension(   "time" , len(time)    ),
		       "period"   : ncf.createDimension( "period" , len(dpers)   ),
		       "hyper_parameter" : ncf.createDimension( "hyper_parameter" , len(hpar_names)  ),
		}
		if clim.has_spatial is not None and not clim.spatial_is_fake:
			for d in d_spatial:
				ncdims[d] = ncf.createDimension( d , c_spatial[d].size )
		
		
		## Define variables
		ncvars = {
		       "sample" : ncf.createVariable( "sample" , str       , ("sample",) ),
		         "name" : ncf.createVariable(   "name" , str       , ("name",) ),
		       "period" : ncf.createVariable( "period" , str       , ("period",) ),
		       "time"   : ncf.createVariable(   "time" , "float32" ,   ("time",) ),
		       "hyper_parameter" : ncf.createVariable( "hyper_parameter" , str ,   ("hyper_parameter",) ),
		}
		
		ncvars["name"][:]   = np.array( clim.namesX , dtype = str )
		ncvars["sample"][:] = np.array( samples , dtype = str )
		ncvars["period"][:] = np.array(   dpers , dtype = str )
		ncvars["hyper_parameter"][:] = np.array( hpar_names , dtype = str )
		
		## Fill time axis
		calendar = "standard"
		units    = "days since 1750-01-01 00:00"
		ncvars["time"][:]  = cftime.date2num( [cftime.DatetimeGregorian( int(y) , 1 , 1 ) for y in time] , units = units , calendar = calendar )
		ncvars["time"].setncattr( "standard_name" , "time"      )
		ncvars["time"].setncattr( "long_name"     , "time_axis" )
		ncvars["time"].setncattr( "units"         , units       )
		ncvars["time"].setncattr( "calendar"      , calendar    )
		ncvars["time"].setncattr( "axis"          , "T"         )
		
		## Add spatial
		if clim.has_spatial and not clim.spatial_is_fake:
			for d in d_spatial:
				ncvars[d] = ncf.createVariable( d , "double" , (d,) )
				ncvars[d][:] = np.array(c_spatial[d]).ravel()
		
		replace_hpar = lambda n: "hyper_parameter" if n == "hpar" else n
		for key in out:
			dims = [ replace_hpar(d) for d in out[key].dims if not d == "fake" ]
			ncvars[key] = ncf.createVariable( key , out[key].dtype , dims , fill_value = np.nan , compression = "zlib" , complevel = 5 )
			idx = [slice(None) for _ in range(out[key].ndim)]
			if clim.spatial_is_fake:
				idx[-1] = 0
			ncvars[key][:] = out[key]._internal.zdata.get_orthogonal_selection(tuple(idx))
		
##}}}


