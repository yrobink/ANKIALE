
## Copyright(c) 2023 / 2025 Yoann Robin
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
import scipy.stats as sc

import netCDF4

from ..__sys     import coords_samples
from ..stats.__tools import nslawid_to_class

from ..stats.__constraint import gaussian_conditionning_independent
from ..stats.__constraint import gaussian_conditionning_kcc
from ..stats.__constraint import mcmc

from ..__linalg import mean_cov_hpars


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

def zgaussian_conditionning( *args , A = None , kcc = False , timeXo = None ):##{{{
	
	gaussian_conditionning = gaussian_conditionning_independent
	if kcc:
		gaussian_conditionning = gaussian_conditionning_kcc
	
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
		
		oh,oc = gaussian_conditionning( *iargs , A = A , timeXo = timeXo )
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
	
	## Check if KCC can be used
	if bsacParams.use_KCC and len(zXo) > 2:
		raise ValueError("KCC can not be used for a constraint with more than 2 covariates ({len(zXo)} required)")
	
	##
	time = clim.time
	spl,lin,_,_ = clim.build_design_XFC()
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
	                     "kwargs" : { "A" : A , "kcc" : bsacParams.use_KCC , "timeXo" : [ zXo[name][f"time{iname}"] for iname,name in enumerate(zXo) ] } ,
	                     "dask" : "parallelized",
	                     "output_dtypes"  : [clim.hpar.dtype,clim.hcov.dtype]
	                    }
	
	## Block memory function
	nhpar = len(clim.hpar_names)
	block_memory = lambda x : 2 * ( nhpar + nhpar**2 +  len(zXo) * time.size + nhpar + nhpar**2 ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
	
	##
	with bsacParams.get_cluster() as cluster:
		hpar,hcov = zr.apply_ufunc( zgaussian_conditionning , *args ,
		                            block_dims         = d_spatial,
		                            total_memory       = bsacParams.total_memory,
		                            block_memory       = block_memory,
		                            output_coords      = output_coords,
		                            output_dims        = output_dims,
		                            output_dtypes      = output_dtypes,
		                            dask_kwargs        = dask_kwargs,
		                            n_workers          = bsacParams.n_workers,
		                            threads_per_worker = bsacParams.threads_per_worker,
			                        cluster            = cluster,
		                         )
	
	## Save
	clim.hpar = hpar
	clim.hcov = hcov
	bsacParams.clim = clim
##}}}


def zmcmc( ihpar , ihcov , Yo , samples , A , size_chain , nslaw_class , use_STAN ):##{{{
	
	ssp    = ihpar.shape[:-2]
	nhpar  = ihpar.shape[-1]
	hpars  = np.zeros( ssp + (samples.size,nhpar,size_chain) ) + np.nan
	
	for idx in itt.product(*[range(s) for s in ssp]):
		for s in range(samples.size):
			
			## Extract
			idx1d = idx + (0,) + tuple([slice(None) for _ in range(1)])
			idx2d = idx + (0,) + tuple([slice(None) for _ in range(2)])
			ih    = ihpar[idx1d]
			ic    = ihcov[idx2d]
			iYo   = Yo[idx1d]
			
			if np.any(~np.isfinite(ih)):
				continue
			
			## MCMC
			oh = mcmc( ih , ic , iYo , A , size_chain , nslaw_class , use_STAN , bsacParams.tmp_stan )
			
			## Store
			idx2d = idx + (s,) + tuple([slice(None) for _ in range(2)])
			hpars[idx2d] = oh
	
	return hpars
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
	if clim.spatial_is_fake:
		Yo = xr.DataArray( Yo.values.reshape(-1,1) , dims = ("time",) + clim.d_spatial , coords = { **{ "time" : Yo.time.dt.year.values } , **clim.c_spatial } )
	else:
		Yo = xr.DataArray( Yo.values , dims = Yo.dims , coords = [Yo.time.dt.year.values] + [Yo.coords[d] for d in Yo.dims[1:]] )
	
	## Bias
	bias = Yo.sel( time = slice(*[str(y) for y in clim.bper]) ).mean( dim = "time" )
	Yo   = Yo - bias
	try:
		bias = float(bias)
	except:
		pass
	if clim.spatial_is_fake:
		bias = xr.DataArray( [bias] , dims = clim.d_spatial , coords = clim.c_spatial )
	clim._bias[clim.vname] = bias
	
	## Transform in ZXArray
	zYo = zr.ZXArray.from_xarray(Yo)
	
	## Extract parameters
	use_STAN   = not bsacParams.no_STAN
	d_spatial  = clim.d_spatial
	c_spatial  = clim.c_spatial
	hpar_names = clim.hpar_names
	ihpar      = clim.hpar
	ihcov      = clim.hcov
	
	## Samples
	n_samples = bsacParams.n_samples
	samples   = coords_samples(n_samples)
	zsamples  = zr.ZXArray.from_xarray( xr.DataArray( range(n_samples) , dims = ["sample"] , coords = [samples] ).astype(float) )
	
	##
	dpers  = clim.dpers
	chains = range(size_chain)
	
	## Build projection operator for the covariable
	time = clim.time
	spl,lin,_,_ = clim.build_design_XFC()
	nper = len(clim.dpers)
	
	design_ = []
	for nameX in clim.namesX:
		if nameX == clim.cname:
			design_ = design_ + [spl for _ in range(nper)] + [nper * lin]
		else:
			design_ = design_ + [np.zeros_like(spl) for _ in range(nper)] + [np.zeros_like(lin)]
	design_ = design_ + [np.zeros( (time.size,clim.sizeY) )]
	design_ = np.hstack(design_)
	
	T = xr.DataArray( np.identity(design_.shape[0]) , dims = ["timeA","timeB"] , coords = [time,time] ).loc[Yo.time,time].values
	A = T @ design_ / nper
	
	##
	nslaw_class = clim._nslaw_class
	
	## Init stan
	if use_STAN:
		logger.info(" * STAN compilation...")
		nslaw_class().init_stan( tmp = bsacParams.tmp_stan , force_compile = True )
		logger.info(" * STAN compilation... Done.")
	
	## Apply parameters
	output_dims      = [ ("hpar","sample","chain") + d_spatial   ]
	output_coords    = [ [hpar_names,samples,chains] + [ c_spatial[d] for d in d_spatial ] ]
	output_dtypes    = [float]
	dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] , ["time"] , [] ],
	                     "output_core_dims" : [ ["hpar","chain"] ],
	                     "kwargs" : { "A" : A , "size_chain" : size_chain , "nslaw_class" : nslaw_class , "use_STAN" : use_STAN } ,
	                     "dask" : "parallelized",
	                     "dask_gufunc_kwargs" : { "output_sizes" : { "chain" : size_chain } },
	                     "output_dtypes"  : [clim.hpar.dtype]
	                    }
	
	## Block memory function
	nhpar = len(clim.hpar_names)
	block_memory = lambda x : 5 * ( nhpar + nhpar**2 + time.size + nhpar * time.size + 1 + nhpar * size_chain ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
	
	## Draw samples
	logger.info(" * Draw samples")
	with bsacParams.get_cluster() as cluster:
		ohpars = zr.apply_ufunc( zmcmc , ihpar , ihcov , zYo , zsamples ,
		                         block_dims         = d_spatial + ("sample",),
		                         total_memory       = bsacParams.total_memory,
		                         block_memory       = block_memory,
		                         output_coords      = output_coords,
		                         output_dims        = output_dims,
		                         output_dtypes      = output_dtypes,
		                         dask_kwargs        = dask_kwargs,
		                         n_workers          = bsacParams.n_workers,
		                         threads_per_worker = bsacParams.threads_per_worker,
		                         cluster            = cluster,
		                        )
	
	## Clean memory
	gc.collect()
	
	## And find parameters of the distribution
	logger.info(" * Compute mean and covariance of parameters")
	output_dims      = [ ("hpar",) + d_spatial , ("hpar0","hpar1") + d_spatial ]
	output_coords    = [ [hpar_names] + [ c_spatial[d] for d in d_spatial ] , [hpar_names,hpar_names] + [ c_spatial[d] for d in d_spatial ] ]
	output_dtypes    = [float,float]
	dask_kwargs      = { "input_core_dims"  : [ ["hpar","sample","chain"]],
	                     "output_core_dims" : [ ["hpar"] , ["hpar0","hpar1"] ],
	                     "kwargs" : {},
	                     "dask" : "parallelized",
	                     "dask_gufunc_kwargs" : { "output_sizes" : { "hpar0" : len(hpar_names) , "hpar1" : len(hpar_names) } },
	                     "output_dtypes"  : [ohpars.dtype,ohpars.dtype]
	                    }
	
	## Block memory function
	block_memory = lambda x : 10 * ( (nhpar + nhpar**2) * n_samples * size_chain + nhpar + nhpar**2 ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
	
	## Apply
	with bsacParams.get_cluster() as cluster:
		hpar,hcov = zr.apply_ufunc( mean_cov_hpars , ohpars,
		                            block_dims         = d_spatial,
		                            total_memory       = bsacParams.total_memory,
		                            block_memory       = block_memory,
		                            output_dims        = output_dims,
		                            output_coords      = output_coords,
		                            output_dtypes      = output_dtypes,
		                            dask_kwargs        = dask_kwargs,
		                            n_workers          = bsacParams.n_workers,
		                            threads_per_worker = bsacParams.threads_per_worker,
		                            cluster            = cluster,
		                            chunks             = { d : 1 for d in d_spatial },
		                            )
	
	## Clean memory
	gc.collect()
	
	## Store (or not) the samples
	if bsacParams.output is not None:
		logger.info(" * Store samples on the disk")
		
		names = lambda n : "hyper_parameter" if n == "hpar" else n
		with netCDF4.Dataset( bsacParams.output , "w" ) as oncf:
			
			odims  = []
			oshape = []
			for d,s in zip(ohpars.dims,ohpars.shape):
				if not d == "fake":
					odims.append(d)
					oshape.append(s)
			
			## Create dimensions
			ncdims = { names(d) : oncf.createDimension( names(d) , s ) for d,s in zip(odims,oshape) }
			
			## Create variables
			ncvars = { names(d) : oncf.createVariable( names(d) , ohpars.coords[d].dtype , (names(d),) ) for d in odims }
			ncvars["hpars"] = oncf.createVariable( "hpars" , ohpars.dtype , [names(d) for d in odims] )
			
			## And fill
			for d in odims:
				ncvars[names(d)][:] = ohpars.coords[d].values[:]
			idx = [slice(None) for _ in range(len(odims))]
			if clim.spatial_is_fake:
				idx.append(0)
			idx = tuple(idx)
			ncvars["hpars"][:] = ohpars._internal.zdata.get_orthogonal_selection(idx)
	
	## And save
	clim.hpar = hpar
	clim.hcov = hcov
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

