
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

##############
## Packages ##
##############


#############
## Imports ##
#############

import os
import itertools as itt
import logging
from ..__logs import LINE
from ..__logs import log_start_end

from ..__sys import coords_samples
from ..__sys import SizeOf

from ..__XZarr import XZarr
from ..__XZarr import random_zfile

import numpy  as np
import xarray as xr
import zarr


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#############
## Classes ##
#############



###############
## Functions ##
###############

def sqrtm( C ):##{{{
	
	def _sqrtm(c):
		if not np.isfinite(c).all():
			return np.zeros_like(c) + np.nan
		u,s,v = np.linalg.svd(c)
		return u @ np.sqrt(np.diag(s)) @ v.T
	
	if C.ndim == 2:
		return _sqrtm(C)
	
	shape_nd = C.shape
	shape_1d = C.shape[:2] + (-1,)
	C = C.reshape(shape_1d)
	S = C.copy() + np.nan
	for i in range(C.shape[-1]):
		S[:,:,i] = _sqrtm(C[:,:,i])
	
	return S.reshape(shape_nd)
##}}}

def rvs_multivariate_normal( size , mean , cov , zfile = None ):##{{{
	
	## Transform in array
	mean_ = mean.values if isinstance(mean,xr.DataArray) else mean
	cov_  = cov.values  if isinstance( cov,xr.DataArray) else cov
	
	## Compute standard deviation
	std_ = sqrtm(cov_)
	
	## Output
	if zfile is None:
		out = np.zeros( (size,) + mean_.shape )
	else:
		out = zarr.open( zfile , mode = "w" , shape = (size,) + mean_.shape , dtype = "float32" , compressor = None )
	
	for idx in itt.product(*[range(s) for s in mean_.shape[1:]]):
		idx1d = (slice(None),) + idx
		idx2d = (slice(None),slice(None)) + idx
		draw = np.random.normal( loc = 0 , scale = 1 , size = mean_.shape[0] * size ).reshape(mean_.shape[0],size)
		draw = std_[idx2d] @ draw + mean_[idx1d].reshape(-1,1)
		try:
			out.set_orthogonal_selection( (slice(None),slice(None),) + idx , draw.T )
		except:
			out[idx2d] = draw.T
	
	return out
##}}}

def _rvs_climatology_parallel( m , c , clim , n_samples , BE ):##{{{
	
	## Draw hyper parameters
	hpars = xr.DataArray( rvs_multivariate_normal( mean = m , cov = c , size = n_samples ) , dims = ["sample","hpar"] , coords = [range(n_samples),clim.hpar_names] )
	if BE: hpars[0,:] = m
	
	## Design
	_,_,designF_,designC_ = clim.build_design_XFC()
	hpar_coords = designF_.hpar.values.tolist()
	
	## Build XF and XC
	XF =  xr.concat(
	    [
	     xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designF_
	         for per in clim.dpers
	        ],
	        dim = "period"
	        )
	     for name in clim.namesX
	    ],
	    dim = "name"
	    ).assign_coords( { "period" : clim.dpers , "name" : clim.namesX } ).transpose("sample","time","period","name")
	XC = xr.concat(
	    [
	     xr.concat(
	        [
	         xr.concat( [hpars[:,clim.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designC_
	         for per in clim.dpers
	        ],
	        dim = "period"
	        )
	     for name in clim.namesX
	    ],
	    dim = "name"
	    ).assign_coords( { "period" : clim.dpers , "name" : clim.namesX } ).transpose("sample","time","period","name")
	
	## Draw parameters
	nslaw = clim._nslaw_class()
	parF  = nslaw.draw_params( XF.sel( name = clim.namesX[-1] ) , hpars )
	parC  = nslaw.draw_params( XC.sel( name = clim.namesX[-1] ) , hpars )
	
	## Output
	out = [hpars.values,XF.values,XC.values] + [parF[k] for k in parF] + [parC[k] for k in parC]
	
	return tuple(out)
##}}}

def rvs_climatology( clim , n_samples , tmp , add_BE = False , n_jobs = 1 , mem_limit = None ):##{{{
	
	## Parameters
	time       = clim.time
	samples    = coords_samples(n_samples)
	nslaw      = clim._nslaw_class()
	spatial    = clim._spatial
	d_spatial  = tuple(list(spatial))
	hpar_names = clim.hpar_names
	periods    = clim.dpers
	
	## Update size if BE
	if add_BE:
		n_samples = n_samples + 1
		samples = ["BE"] + samples
	
	## Start by define output of hyper parameters
	zdims   = ["sample","hpar"] + list(spatial)
	zcoords = [samples,hpar_names] + [spatial[s] for s in spatial]
	zshape  = [len(c) for c in zcoords]
	zfile   = random_zfile( prefix = os.path.join( tmp , "rvsHPAR" ) )
	zhpar   = XZarr.from_value( np.nan , zshape , zdims , zcoords , zfile )
	
	## Output of XF and XC
	zdims   = ["sample","time","period","name"] + list(spatial)
	zcoords = [samples,time,periods,clim.namesX] + [spatial[s] for s in spatial]
	zshape  = [len(c) for c in zcoords]
	zfileF  = random_zfile( prefix = os.path.join( tmp , "XF" ) )
	zfileC  = random_zfile( prefix = os.path.join( tmp , "XC" ) )
	zXF     = XZarr.from_value( np.nan , zshape , zdims , zcoords , zfileF )
	zXC     = XZarr.from_value( np.nan , zshape , zdims , zcoords , zfileC )
	
	## And output of parameters
	znsp    = {}
	zdims   = ["sample","time","period"] + list(spatial)
	zcoords = [samples,time,periods] + [spatial[s] for s in spatial]
	zshape  = [len(c) for c in zcoords]
	n_coef  = len(nslaw.coef_kind)
	for c,K in itt.product( nslaw.coef_kind , ["F","C"] ):
		zfile     = random_zfile( prefix = os.path.join( tmp , c + K ) )
		znsp[c+K] = XZarr.from_value( np.nan , zshape , zdims , zcoords , zfile )
	
	## Find the block size for parallelization
	logger.info( " * Find block size...")
	sizes   = list(clim.s_spatial + (n_samples,))
	nsizes  = list(clim.s_spatial + (n_samples,))
	blocks  = list(sizes)
	nfind   = [True,True,True]
	mem_cst = (np.finfo('float64').bits // SizeOf(n = 0).bits_per_octet) * SizeOf("1o")
	fmem_use = lambda b: 5 * np.prod(blocks) * mem_cst * ( len(hpar_names) + time.size * len(periods) * len(clim.namesX) * 2 + time.size * len(periods) * len(znsp) )
	mem_use = fmem_use(blocks)
	
	while any(nfind):
		i = np.argmin(nsizes)
		while mem_use > mem_limit:# or np.prod(blocks) > 10 * bsacParams.n_workers * bsacParams.threads_per_worker:
			mem_use = fmem_use(blocks)
			if blocks[i] < 2:
				blocks[i] = 1
				break
			blocks[i] = blocks[i] // 2
		nfind[i] = False
		nsizes[i] = np.inf
	logger.info( f"   => Block size: {blocks}" )
	logger.info( f"   => Memory: {mem_use} / {mem_limit}" )
	
	## Clean
	mean_ = clim._mean.copy()
	cov_  = clim._cov.copy()
	bias_ = clim._bias.copy()
	
	del clim._mean
	del clim._cov
	del clim._bias
	
	## Now loop
	for idx in itt.product(*[range(0,s,b) for s,b in zip(clim.s_spatial + (n_samples,),blocks)]):
		
		## Indexes
		spatial_idx = tuple([slice(s,s+b,1) for s,b in zip(idx[:-1],blocks[:-1])])
		sample_idx  = (slice(idx[-1],idx[-1]+blocks[-1],1),)
		
		idx_m = (slice(None),) + spatial_idx
		idx_c = (slice(None),slice(None)) + spatial_idx
		
		ssamples   = np.array(samples)[sample_idx]
		n_ssamples = ssamples.size
		m          = xr.DataArray( mean_[idx_m] , dims = ["hpar"         ] + list(spatial) , coords = [hpar_names]                   + [spatial[s][ssp] for s,ssp in zip(spatial,spatial_idx)] )
		c          = xr.DataArray( cov_[idx_c]  , dims = ["hpar0","hpar1"] + list(spatial) , coords = [hpar_names for _ in range(2)] + [spatial[s][ssp] for s,ssp in zip(spatial,spatial_idx)] )
		
		## Parallelization
		res = xr.apply_ufunc( _rvs_climatology_parallel , m , c ,
		                    input_core_dims  = [["hpar"],["hpar0","hpar1"]],
		                    output_core_dims = [["sample","hpar"]] + [["sample","time","period","name"] for _ in range(2)] + [["sample","time","period"] for _ in range(2*n_coef)],
		                    output_dtypes    = [m.dtype for _ in range(1+2+2*n_coef)],
		                    vectorize        = True,
		                    dask             = "parallelized",
		                    kwargs           = { "clim" : clim , "n_samples" : n_ssamples , "BE" : "BE" in ssamples },
		                    dask_gufunc_kwargs = { "output_sizes" : { "sample" : n_ssamples , "time" : len(time) , "period" : len(periods) , "name" : len(clim.namesX) } }
		                    )
		
		## Good coordinates
		hpar = res[0].assign_coords( sample = ssamples ).transpose( *( ("sample","hpar") + d_spatial) ).compute()
		XF   = res[1].assign_coords( { "sample" : ssamples , "time" : time , "period" : periods , "name" : clim.namesX } ).transpose( *(("sample","time","period","name") + d_spatial) ).compute()
		XC   = res[2].assign_coords( { "sample" : ssamples , "time" : time , "period" : periods , "name" : clim.namesX } ).transpose( *(("sample","time","period","name") + d_spatial) ).compute()
		
		nsp = {}
		for i,(K,c) in enumerate(itt.product(["F","C"],nslaw.coef_kind)):
			nsp[c+K] = res[i+3].assign_coords( { "sample" : ssamples , "time" : time , "period" : periods } ).transpose( *(("sample","time","period") + d_spatial) ).compute()
		
		## Add to zarr file
		zhpar.set_orthogonal_selection( sample_idx + (slice(None),) + spatial_idx , hpar.values )
		zXF.set_orthogonal_selection( sample_idx + tuple([slice(None) for _ in range(3)]) + spatial_idx , XF.values )
		zXC.set_orthogonal_selection( sample_idx + tuple([slice(None) for _ in range(3)]) + spatial_idx , XC.values )
		for key in nsp:
			znsp[key].set_orthogonal_selection( sample_idx + tuple([slice(None) for _ in range(2)]) + spatial_idx , nsp[key].values )
	
	## End
	clim._mean = mean_
	clim._cov  = cov_ 
	clim._bias = bias_
	draw = { **{ "hpar" : zhpar , "XF" : zXF , "XC" : zXC } , **znsp }
	
	return draw
##}}}

