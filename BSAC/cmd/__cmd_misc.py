
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
import gc
import logging
import datetime as dt
import itertools as itt
from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams
from ..__climatology import Climatology

import numpy  as np
import xarray as xr
import zarr
from ..__XZarr import XZarr
from ..__XZarr import random_zfile
from ..__sys import SizeOf
from ..__release import version
from ..__sys import coords_samples
import netCDF4

from ..stats.__rvs import rvs_multivariate_normal

##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############


def _find_wpe_parallel( hpar , hcov , n_samples = 1 , pwpe = None , side = None , perwpe = None , clim = None ):##{{{
	
	## Extract parameters from clim
	_,_,designF_,designC_ = clim.build_design_XFC()
	nslaw = clim._nslaw_class()
	npers = len(clim.dpers) + 1
	
	## Probabilities
	pwpe = np.array([pwpe]).ravel()
	nwpe = pwpe.size
	
	## Init output
	output = np.zeros( (n_samples,npers,nwpe) ) + np.nan
	
	## If nan, return
	if not np.isfinite(hpar).all():
		return output
	
	## Draw parameters
	hpars = xr.DataArray( rvs_multivariate_normal( n_samples , hpar , hcov ) , dims = ["sample","hpar"] , coords = [range(n_samples),clim.hpar_names] )
	
	## Design
	_,_,designF_,designC_ = clim.build_design_XFC()
	hpar_coords = designF_.hpar.values.tolist()
	name        = clim.namesX[-1]
	
	designF_ = designF_.sel( time = slice(*perwpe) )
	designC_ = designC_.sel( time = slice(*perwpe) )
	
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
	    ).assign_coords( period = ["cfactual"] + [f"cfactual{i}" for i in range(len(clim.dpers)-1)] ).sel( period = "cfactual" )
	XFC = xr.concat( (XC,XF) , dim = "period" )
	
	## Build params
	dpars = nslaw.draw_params( XFC , hpars )
	
	## Loop on probs
	for ip,p in enumerate(pwpe):
		
		## Init eventL and eventH
		eventL = nslaw.icdf_sf( min(     1e-6 ,         pwpe.min()  / 100 ) , side , **dpars )
		eventH = nslaw.icdf_sf( max( 1 - 1e-6 ,1 - (1 - pwpe.max()) / 100 ) , side , **dpars )
		if side == "right":
			eventL,eventH = eventH,eventL
		eventL[:] = np.nanmin( eventL , axis = 0 )
		eventH[:] = np.nanmax( eventH , axis = 0 )
		eventM = ( eventL + eventH ) / 2
		
		## Loop for optimization
		res = float(np.nanmax(eventH)) - float(np.nanmin(eventL))
		eps = min( 0.01 * res , 1 ) * 1e-4
		while res > eps:
			
			## Probability of the central event
			pM = 1 - np.prod( 1 - nslaw.cdf_sf( eventM , side , **dpars ) , axis = 1 )
			
			## Update
			eventH = np.where( pM.reshape(n_samples,1,-1) > p , eventH , eventM )
			eventL = np.where( pM.reshape(n_samples,1,-1) < p , eventL , eventM )
			eventM = ( eventL + eventH ) / 2
			
			##
			res = float(np.nanmax(np.abs(eventH - eventL)))
			
		output[:,:,ip] = eventM.mean( axis = 1 )
	
	return output
##}}}

## run_bsac_cmd_misc_wpe ##{{{
@log_start_end(logger)
def run_bsac_cmd_misc_wpe():
	
	## Parameters
	clim      = bsacParams.clim
	time      = clim.time
	n_samples = bsacParams.n_samples
	tmp       = bsacParams.tmp
	side      = bsacParams.config.get("side","right")
	nslaw     = clim._nslaw_class()
	sp_dims   = clim.d_spatial
	perwpe    = bsacParams.config.get("period")
	
	##
	if perwpe is None:
		perwpe = [dt.datetime.utcnow().year,int(time[-1])]
	else:
		perwpe = [int(y) for y in perwpe.split("/")]
	
	## Find the probabilities
	pwpe = bsacParams.config.get("pwpe","IPCC")
	if pwpe == "IPCC":
		pwpe = [0.01,0.1,0.33,0.5,0.66,0.9,0.99]
	else:
		pwpe = [float(s) for s in pwpe.split(":")]
	
	logger.info( " * Probabilities found: " + str(pwpe) )
	pwpe = np.array(pwpe).ravel()
	
	## Extract parameters
	d_spatial = clim.d_spatial
	c_spatial = clim.c_spatial
	chpar     = clim.hpar_names
	ihpar     = xr.DataArray( clim.mean_.copy() , dims = ("hpar",)         + d_spatial , coords = { **{ "hpar"  : chpar }                   , **c_spatial } )
	ihcov     = xr.DataArray( clim.cov_.copy()  , dims = ("hpar0","hpar1") + d_spatial , coords = { **{ "hpar0" : chpar , "hpar1" : chpar } , **c_spatial } )
	ohpar     = xr.zeros_like(ihpar) + np.nan
	ohcov     = xr.zeros_like(ihcov) + np.nan
	samples   = np.array(coords_samples(n_samples))
	periods   = clim.dpers + ["cfactual"]
	
	## Init output
	logger.info( " * Init output" )
	dims   = ("probs","sample","period") + sp_dims
	coords = [xr.DataArray( pwpe , dims = ["pwpe"] , coords = [pwpe] ),samples,periods] + [clim.c_spatial[d] for d in sp_dims]
	shape  = [len(c) for c in coords]
	event  = XZarr.from_value( np.nan , shape, dims , coords , random_zfile( os.path.join( tmp , "eventF" ) ) , zarr_kwargs = { "synchronizer" : zarr.ThreadSynchronizer() } )
	
	## Find the block size for parallelization
	logger.info( " * Find block size...")
	sizes   = list(clim.s_spatial + (n_samples,))
	nsizes  = list(clim.s_spatial + (n_samples,))
	blocks  = list(sizes)
	nfind   = [True,True,True]
	fmem_use = lambda b: 5 * np.prod(blocks) * (np.finfo('float64').bits // SizeOf(n = 0).bits_per_octet) * pwpe.size * (perwpe[1] - perwpe[0]+1) * ( len(clim.dpers) + 1 ) * SizeOf("1o")
	mem_use = fmem_use(blocks)
	
	while any(nfind):
		i = np.argmin(nsizes)
		while mem_use > bsacParams.total_memory:# or np.prod(blocks) > 10 * bsacParams.n_workers * bsacParams.threads_per_worker:
			mem_use = fmem_use(blocks)
			if blocks[i] < 2:
				blocks[i] = 1
				break
			blocks[i] = blocks[i] // 2
		nfind[i] = False
		nsizes[i] = np.inf
	logger.info( f"   => Block size: {blocks}" )
	logger.info( f"   => Memory: {mem_use} / {bsacParams.total_memory}" )
	
	## Build a 'small' climatology
	## climatology must be pass at each threads, but duplication of mean, cov bias implies a memory leak
	climB = Climatology.init_from_file( bsacParams.load_clim )
	del climB._mean
	del climB._cov
	del climB._bias
	
	## Loop on samples and spatial variables for parallelisation
	logger.info( " * Find wpe" )
	for idx in itt.product(*[range(0,s,b) for s,b in zip(clim.s_spatial + (n_samples,),blocks)]):
		
		## Indexes
		spatial_idx = tuple([slice(s,s+b,1) for s,b in zip(idx[:-1],blocks[:-1])])
		sample_idx  = (slice(idx[-1],idx[-1]+blocks[-1],1),)
		
		## Extract data
		idx1d = (slice(None),)            + spatial_idx
		idx2d = (slice(None),slice(None)) + spatial_idx
		shpar = ihpar[idx1d].chunk( { d : 1 for d in ihpar.dims[1:] } )
		shcov = ihcov[idx2d].chunk( { d : 1 for d in ihpar.dims[1:] } )
		ssamples = samples[sample_idx[0]]
		
		##
		isfin = np.all( np.isfinite(shpar[1:]) , axis = 0 ).values
		valid = 100 * isfin.sum() / isfin.size
		logger.info( f"   => {idx} + {blocks} / {clim.s_spatial + (n_samples,)} ({round( valid , 3 )}%)" )
		if not valid > 0:
			continue
		
		## Find event
		xevent = xr.apply_ufunc( _find_wpe_parallel , shpar , shcov ,
		                    input_core_dims  = [["hpar"],["hpar0","hpar1"]],
		                    output_core_dims = [["sample","period","pwpe"],],
		                    output_dtypes    = [shpar.dtype],
		                    vectorize        = True,
		                    dask             = "parallelized",
		                    kwargs           = { "n_samples" : len(ssamples) , "pwpe" : pwpe , "side" : side , "perwpe" : perwpe , "clim" : climB },
		                    dask_gufunc_kwargs = { "output_sizes" : { "pwpe" : pwpe.size , "sample" : len(ssamples) , "period" : len(periods) } }
		                    ).assign_coords( { "sample" : ssamples , "period" : periods , "pwpe" : pwpe } ).transpose( *(("pwpe","sample","period") + d_spatial) ).compute()
		
		## And add to zarr
		sel = (slice(None),) + sample_idx + (slice(None),) + spatial_idx
		event.set_orthogonal_selection( sel , xevent.values )
		
		## Clean memory
		del xevent
		gc.collect()
	
	## And save in netcdf
	logger.info( " * Save in netcdf" )
	with netCDF4.Dataset( bsacParams.output , "w" ) as ncf:
		
		## Define dimensions
		ncdims = {
		       "prob"   : ncf.createDimension( "prob"   , len(pwpe) ),
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
		ncvars["prob"][:]   = np.array([pwpe]).squeeze()
		ncvars["sample"][:] = np.array([samples]).squeeze()
		ncvars["period"][:] = np.array([periods]).squeeze()
		if clim._spatial is not None:
			for d in clim._spatial:
				ncvars[d] = ncf.createVariable( d , "double" , (d,) )
				ncvars[d][:] = np.array(clim._spatial[d]).ravel()
		
		## Variables
		ncvars = ncf.createVariable( "wpe" , "float32" , ("prob","sample","period") + spatial , compression = "zlib" , complevel = 5 , chunksizes = (1,1,1) + clim.s_spatial )
		
		## Attributes
		ncvars.setncattr( "description" , "Best Possible Case of the Probability prob" )
		
		## Find the blocks to write netcdf
		blocks  = [1,1,1]
		sizes   = [len(pwpe),n_samples,len(periods)]
		nsizes  = [len(pwpe),n_samples,len(periods)]
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
		ncf.setncattr( "description" , f"Worst Possible Event" )
##}}}


## run_bsac_cmd_misc ##{{{
@log_start_end(logger)
def run_bsac_cmd_misc():
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the fit command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["bpc","wpe"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the fit command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "bpc":
		run_bsac_cmd_misc_bpc()
	if bsacParams.arg[0] == "wpe":
		run_bsac_cmd_misc_wpe()
##}}}


