
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
import datetime as dt
import itertools as itt
from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams


import numpy  as np
import xarray as xr
import zarr
from ..__XZarr import XZarr
from ..__XZarr import random_zfile
from ..__sys import SizeOf
from ..__release import version
import netCDF4


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

def _find_bpc_parallel( *args , keys = None , p = None , side = None , nslaw_class = None ):##{{{
	
	## Extract
	kwargs = { key : arg for key,arg in zip(keys,args) }
	
	## Define a param to have 
	param   = kwargs[list(kwargs)[0]]
	mask    = np.isfinite(param[0,:])
	
	## Check if is finite
	if not np.isfinite(param).any():
		return np.zeros_like(param) + np.nan
	
	## Init eventL and eventH
	nslaw  = nslaw_class()
	eventL = nslaw.icdf_sf( min(     1e-6 ,         p  / 100 ) , side , **kwargs )
	eventH = nslaw.icdf_sf( max( 1 - 1e-6 ,1 - (1 - p) / 100 ) , side , **kwargs )
	if side == "right":
		eventL,eventH = eventH,eventL
	eventL[:] = np.nanmin( eventL , axis = 0 )
	eventH[:] = np.nanmax( eventH , axis = 0 )
	eventM = ( eventL + eventH ) / 2
	
	res = float(np.nanmax(np.abs(eventH - eventL)))
	
	## Loop for optimization
	res = float(np.nanmax(eventH)) - float(np.nanmin(eventL))
	eps = min( 0.01 * res , 1 ) * 1e-4
	while res > eps:
		
		## Probability of the central event
		pM = 1 - np.prod( 1 - nslaw.cdf_sf( eventM , side , **kwargs ) , axis = 0 )
		pM = np.where( mask , pM , np.nan )
		
		## Update
		eventH = np.where( pM > p , eventH , eventM )
		eventL = np.where( pM < p , eventL , eventM )
		eventM = ( eventL + eventH ) / 2
		
		##
		res = float(np.nanmax(np.abs(eventH - eventL)))
	
	return eventM
##}}}

## run_bsac_cmd_misc_bpc ##{{{
@log_start_end(logger)
def run_bsac_cmd_misc_bpc():
	
	## Parameters
	clim      = bsacParams.clim
	time      = clim.time
	n_samples = bsacParams.n_samples
	tmp       = bsacParams.tmp
	side      = bsacParams.config.get("side","right")
	nslaw     = clim._nslaw_class()
	sp_dims   = clim.d_spatial
	perbpc    = bsacParams.config.get("period")
	
	##
	if perbpc is None:
		perbpc = [dt.datetime.utcnow().year,int(time[-1])]
	else:
		perbpc = [int(y) for y in perbpc.split("/")]
	
	## Find the probabilities
	pbpc = bsacParams.config.get("pbpc","IPCC")
	if pbpc == "IPCC":
		pbpc = [0.01,0.1,0.33,0.5,0.66,0.9,0.99]
	else:
		pbpc = [float(s) for s in pbpc.split(":")]
	
	logger.info( " * Probabilities found: " + str(pbpc) )
	pbpc = np.array(pbpc).ravel()
	
	## Draw parameters
	logger.info( " * Draw parameters" )
	zdraw   = clim.rvsY(n_samples)
	ovars   = [key for key in zdraw if key not in ["XF","XC","XA"]]
	samples = np.array(zdraw[ovars[0]].coords[zdraw[ovars[0]].dims.index("sample")])
	periods = np.array([np.array(zdraw[ovars[0]].coords[zdraw[ovars[0]].dims.index("period")]).tolist() + ["cfactual"]]).ravel()
	
	## Init output
	logger.info( " * Init output" )
	dims   = ("probs","sample","period") + sp_dims
	coords = [xr.DataArray( pbpc , dims = ["pbpc"] , coords = [pbpc] ),samples,periods] + [clim.c_spatial[d] for d in sp_dims]
	shape  = [len(c) for c in coords]
	event  = XZarr.from_value( np.nan , shape, dims , coords , random_zfile( os.path.join( tmp , "eventF" ) ) )
	
	## Loop on probabilities
	logger.info( " * Find bpc" )
	for ip,p in enumerate(pbpc):
		
		logger.info( f"   => probability {p}" )
		
		## Loop on samples and spatial variables for parallelisation
		block = max( 0 , int( np.power( bsacParams.n_jobs , 1. / ( len(clim.s_spatial) + 1) ) ) ) + 1
		for idx in itt.product(*[range(0,s,block) for s in (n_samples,) + clim.s_spatial]):
			
			## Indexes
			sample_idx  = (slice(idx[0],idx[1]+block,1),)
			spatial_idx = tuple([slice(s,s+block,1) for s in idx[1:]])
			full_idx    = sample_idx + (slice(None),slice(None)) + spatial_idx
			
			## Find parameters
			kwargsF = { key : zdraw[key+"F"].get_orthogonal_selection(full_idx).sel( time = slice(*perbpc) ) for key in nslaw.coef_kind }
			kwargsC = { key : zdraw[key+"C"].get_orthogonal_selection(full_idx).sel( time = slice(*perbpc) ) for key in nslaw.coef_kind }
			
			## Find event
			eventF = xr.apply_ufunc( _find_bpc_parallel , *[kwargsF[key] for key in kwargsF] ,
			                    input_core_dims  = [["time","period"] for _ in range(len(kwargsF))],
			                    output_core_dims = [["time","period"]],
			                    output_dtypes    = [kwargsF[list(kwargsF)[0]].dtype],
			                    vectorize        = True,
			                    dask             = "parallelized",
			                    kwargs           = { "keys" : list(kwargsF) , "p" : p , "side" : side , "nslaw_class" : clim._nslaw_class }
			                    ).transpose(*(("sample","time","period")+clim.d_spatial)).compute()
			eventC = xr.apply_ufunc( _find_bpc_parallel , *[kwargsC[key] for key in kwargsC] ,
			                    input_core_dims  = [["time","period"] for _ in range(len(kwargsC))],
			                    output_core_dims = [["time","period"]],
			                    output_dtypes    = [kwargsC[list(kwargsC)[0]].dtype],
			                    vectorize        = True,
			                    dask             = "parallelized",
			                    kwargs           = { "keys" : list(kwargsC) , "p" : p , "side" : side , "nslaw_class" : clim._nslaw_class }
			                    ).transpose(*(("sample","time","period")+clim.d_spatial)).compute()
			
			## Save
			selF = (ip,) + sample_idx + tuple([[i for i in range(len(periods)-1)]]) + spatial_idx
			selC = (ip,) + sample_idx + tuple([                  len(periods)-1  ]) + spatial_idx
			event.set_orthogonal_selection( selF , eventF.mean( dim = "time" ).values )
			event.set_orthogonal_selection( selC , eventC.mean( dim = "time" ).sel( period = periods[0] ).values )
	
	## And save in netcdf
	logger.info( " * Save in netcdf" )
	with netCDF4.Dataset( bsacParams.output , "w" ) as ncf:
		
		## Define dimensions
		ncdims = {
		       "prob"   : ncf.createDimension( "prob"   , len(pbpc) ),
		       "sample" : ncf.createDimension( "sample" , n_samples ),
		       "period" : ncf.createDimension( "period" , len(periods) )
		}
		spatial = ()
		if clim._spatial is not None:
			for d in clim._spatial:
				ncdims[d] = ncf.createDimension( d , clim._spatial[d].size )
			spatial = tuple([d for d in clim._spatial])
		
		## Define variables
		ncvars = {
		       "prob"   : ncf.createVariable( "prob"   , "float32" , ("prob",) ),
		       "sample" : ncf.createVariable( "sample" , str       , ("sample",) ),
		       "period" : ncf.createVariable( "period" , str       , ("period",) )
		}
		ncvars["prob"][:]   = pbpc
		ncvars["sample"][:] = samples
		ncvars["period"][:] = periods
		if clim._spatial is not None:
			for d in clim._spatial:
				ncvars[d] = ncf.createVariable( d , "double" , (d,) )
				ncvars[d][:] = np.array(clim._spatial[d]).ravel()
		
		## Variables
		ncvars = ncf.createVariable( "bpc" , "float32" , ("prob","sample","period") + spatial , compression = "zlib" , complevel = 5 , chunksizes = (1,1,1) + clim.s_spatial )
		
		## Attributes
		ncvars.setncattr( "description" , "Best Possible Case of the Probability prob" )
		
		## Find the blocks to write netcdf
		blocks  = [1,1,1]
		sizes   = [len(pbpc),n_samples,len(periods)]
		nsizes  = [len(pbpc),n_samples,len(periods)]
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
		bias   = clim.bias[clim.names[-1]]
		idx_sp = tuple([slice(None) for _ in range(len(clim._spatial))])
		for idx in itt.product(*[range(0,s,block) for s,block in zip(sizes,blocks)]):
			
			s_idx = tuple([slice(s,s+block,1) for s,block in zip(idx,blocks)])
			idxs = s_idx + idx_sp
			
			xdata = event.get_orthogonal_selection(idxs)
			ncvars[idxs] = ( xdata + bias ).values
		
		## Global attributes
		ncf.setncattr( "creation_date" , str(dt.datetime.utcnow())[:19] + " (UTC)" )
		ncf.setncattr( "BSAC_version"  , version )
		ncf.setncattr( "description" , f"Best Possible Case" )
##}}}

## run_bsac_cmd_misc ##{{{
@log_start_end(logger)
def run_bsac_cmd_misc():
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the fit command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["bpc"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the fit command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "bpc":
		run_bsac_cmd_misc_bpc()
##}}}


