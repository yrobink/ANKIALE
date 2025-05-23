
## Copyright(c) 2023 / 2025 Yoann Robin
## 
## This file is part of ANKIALE.
## 
## ANKIALE is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## ANKIALE is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with ANKIALE.  If not, see <https://www.gnu.org/licenses/>.

#############
## Imports ##
#############

import os
import logging
from ..__logs import log_start_end

from ..__ANKParams import ankParams

from ..stats.__MultiGAM import mgam_multiple_fit
from ..stats.__tools    import nslawid_to_class
from ..stats.__NSLawMLEFit import nslaw_fit
from ..__sys import coords_samples

from ..__linalg import mean_cov_hpars

import numpy  as np
import xarray as xr
import zxarray as zr


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## run_ank_cmd_fit_X ##{{{
@log_start_end(logger)
def run_ank_cmd_fit_X():
	
	## Check the inputs
	logger.info("Check inputs")
	n_X   = len(ankParams.input)
	names = []
	if n_X == 0:
		raise ValueError("Fit asked, but no input given, abort.")
	inputs = { inp.split(",")[0] : inp.split(",")[1] for inp in ankParams.input }
	for name in inputs:
		if not os.path.isfile(inputs[name]):
			raise FileNotFoundError(f"File '{inputs[name]}' is not found, abort.")
		else:
			logger.info( f" * covariate {name} detected" )
			names.append(name)
	ankParams.clim.names = names
	
	## Now open the data
	logger.info("Open the data")
	X = {}
	for name in ankParams.clim.names:
		idata   = xr.open_dataset( inputs[name] )[name].mean( dim = "run" )
		periods = list(set(idata.period.values.tolist()) & set(ankParams.dpers))
		periods.sort()
		X[name] = { p : idata.sel( period = ankParams.cper + [p] ).mean( dim = "period" ).dropna( dim = "time" ) for p in periods }
	ankParams.clim.dpers = periods
	
	## Restrict time axis
	time = X[name][periods[0]].time.values.tolist()
	for name in X:
		for p in X[name]:
			time = list(set(time) & set(X[name][p].time.values.tolist()))
	time = sorted(time)
	for name in X:
		for p in X[name]:
			X[name][p] = X[name][p].sel( time = time )
	time = np.array(time)
	ankParams.clim._time = time
	
	## Find the bias
	logger.info( "Build bias:" )
	bias = { name : 0 for name in X }
	for name in X:
		for p in X[name]:
			bias[name] = float(X[name][p].sel( time = slice(*ankParams.clim.bper) ).mean( dim = "time" ).values)
			X[name][p]   -= bias[name]
		logger.info( f" * {name}: {bias[name]}" )
	ankParams.clim._bias = bias
	
	## Find natural forcings version
	ankParams.clim._vXN = ankParams.XN_version
	
	## Build the natural forcings
	logger.info( "Build XN" )
	XN   = { name : { p : ankParams.clim.XN.loc[X[name][p].time] for p in X[name] } for name in X }
	
	## Check GAM parameters
	dof    = ankParams.clim.GAM_dof
	sbasis = ankParams.clim.GAM_sbasis
	degree = ankParams.clim.GAM_degree
	
	if dof is None or sbasis is None or degree is None:
		dof    = {}
		sbasis = {}
		degree = {}
		for name in X:
			dof[name]    = {}
			sbasis[name] = {}
			degree[name] = {}
			for per in periods:
				dof[name][per]    = 6
				sbasis[name][per] = 6
				degree[name][per] = 3
		ankParams.clim._Xconfig["dof"]    = dof
		ankParams.clim._Xconfig["sbasis"] = sbasis
		ankParams.clim._Xconfig["degree"] = degree
	
	## Fit MultiGAM model with bootstrap
	if len(X) > 1:
		logger.info( f"Fit the MultiGAM model (number of bootstrap: {ankParams.n_samples})" )
	else:
		logger.info( f"Fit the MultiGAM model" )
	hpar,hcov = mgam_multiple_fit( X , XN ,
	                               dof    = dof,
	                               sbasis = sbasis,
	                               degree = degree,
	                            n_samples = ankParams.n_samples,
	                           cluster    = ankParams.get_cluster()
	                               )
	
	ankParams.clim.hpar = hpar
	ankParams.clim.hcov = hcov
	
##}}}

## run_ank_cmd_fit_Y ##{{{
@log_start_end(logger)
def run_ank_cmd_fit_Y():
	
	## The current climatology
	clim = ankParams.clim
	## Name of the variable to fit
	name = ankParams.config["name"]
	
	## Spatial dimensions
	d_spatial = ankParams.config.get("spatial")
	if d_spatial is not None:
		d_spatial = d_spatial.split(":")
	else:
		d_spatial = []
	d_spatial = tuple(d_spatial)
	
	## Set the covariate
	cname = ankParams.config.get("cname",clim.names[-1])
	
	## Open the data
	ifile = ankParams.input[0]
	idata = xr.open_dataset(ifile).load()
	
	## Check if variables in idata
	for v in (name,) + d_spatial:
		if v not in idata:
			raise ValueError( f"Variable '{v}' not in input data" )
	
	## Spatial coordinates
	c_spatial = { d : idata[d] for d in d_spatial }
	
	## Find the nslaw
	nslawid = ankParams.config.get("nslaw")
	if nslawid is None:
		raise ValueError( "nslaw must be set" )
	
	## Find the bias, and remove it
	Y     = idata[name]
	biasY = Y.sel( time = slice(*clim.bias_period) ).mean( dim = [d for d in Y.dims if d not in d_spatial] )
	Y     = Y - biasY
	try:
		biasY = float(biasY)
	except Exception:
		pass
	
	## Force to add a spatial dimension
	if len(d_spatial) == 0:
		d_spatial = ("fake",)
		c_spatial = { "fake" : xr.DataArray( [0] , dims = ["fake"] , coords = [[0]] ) }
		Y         = xr.DataArray( Y.values.reshape( Y.shape + (1,) ) , dims = Y.dims + ("fake",) , coords = { **c_spatial , **dict(Y.coords) } )
		biasY     = xr.DataArray( [biasY] , dims = ("fake",) , coords = c_spatial )
	
	## Transform in ZXArray
	zY = zr.ZXArray.from_xarray(Y)
	
	## Check periods
	periods = list(set(clim.dpers) & set(Y.period.values.tolist()))
	periods.sort()
	
	## And restrict its
	clim = clim.restrict_dpers(periods)
	
	## Find the nslaw
	nslaw_class = nslawid_to_class(nslawid)
	nslaw       = nslaw_class()
	hpar_namesY = clim.hpar_names + list(nslaw.h_name)
	
	## Design matrix of the covariate
	proj,_ = clim.projection()
	
	## Time axis
	time = sorted( list( set(clim.time.tolist()) & set(Y.time.values.tolist()) ) )
	zY   = Y.sel( time = time )
	
	## Samples
	n_samples = ankParams.n_samples
	samples   = coords_samples(n_samples)
	samples   = zr.ZXArray.from_xarray( xr.DataArray( range(n_samples) , dims = ["sample"] , coords = [samples] ).astype(float) )
	
	## zxarray.apply_ufunc parameters
	output_dims      = [ ("hparY","dperiod","sample") + d_spatial ]
	output_coords    = [ [hpar_namesY,clim.dpers,samples.dataarray] + [ c_spatial[d] for d in d_spatial ] ]
	output_dtypes    = [ float ]
	dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] , ["time","period","run"] , []],
	                     "output_core_dims" : [ ["hparY","dperiod"] ],
	                     "kwargs" : { "nslaw_class" : nslaw_class , "proj" : proj , "cname" : cname },
	                     "dask" : "parallelized",
	                     "dask_gufunc_kwargs" : { "output_sizes" : { "hparY" : len(hpar_namesY) , "dperiod" : len(clim.dpers) } },
	                     "output_dtypes"  : [clim.hpar.dtype]
	                    }
	
	## Block memory function
	nhpar = len(clim.hpar_names)
	block_memory = lambda x : 2 * ( nhpar + nhpar**2 + len(time) * (len(clim.dpers)+1) * Y["run"].size + len(hpar_namesY) * len(clim.dpers) ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
	
	## Fit samples of parameters
	hpar  = clim.hpar
	hcov  = clim.hcov
	with ankParams.get_cluster() as cluster:
		hpars = zr.apply_ufunc( nslaw_fit , hpar , hcov , zY, samples,
		                        block_dims         = ("sample",) + d_spatial,
		                        block_memory       = block_memory,
		                        total_memory       = ankParams.total_memory,
		                        output_dims        = output_dims,
		                        output_coords      = output_coords,
		                        output_dtypes      = output_dtypes,
		                        dask_kwargs        = dask_kwargs,
		                        n_workers          = ankParams.n_workers,
		                        threads_per_worker = ankParams.threads_per_worker,
		                        cluster            = cluster,
		                        )
	
	## And find parameters of the distribution
	output_dims      = [ ("hpar",) + d_spatial , ("hpar0","hpar1") + d_spatial ]
	output_coords    = [ [hpar_namesY] + [ c_spatial[d] for d in d_spatial ] , [hpar_namesY,hpar_namesY] + [ c_spatial[d] for d in d_spatial ] ]
	output_dtypes    = [float,float]
	dask_kwargs      = { "input_core_dims"  : [ ["hparY","dperiod","sample"]],
	                     "output_core_dims" : [ ["hpar"] , ["hpar0","hpar1"] ],
	                     "kwargs" : {},
	                     "dask" : "parallelized",
	                     "dask_gufunc_kwargs" : { "output_sizes" : { "hpar" : len(hpar_namesY) , "hpar0" : len(hpar_namesY) , "hpar1" : len(hpar_namesY) } },
	                     "output_dtypes"  : [hpars.dtype,hpars.dtype]
	                    }
	
	## Block memory function
	block_memory = lambda x : 2 * ( nhpar * len(clim.dpers) * n_samples + nhpar + nhpar**2 ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
	
	## Apply
	with ankParams.get_cluster() as cluster:
		hpar,hcov = zr.apply_ufunc( mean_cov_hpars , hpars,
		                            block_dims         = d_spatial,
		                            total_memory       = ankParams.total_memory,
		                            block_memory       = block_memory,
		                            output_dims        = output_dims,
		                            output_coords      = output_coords,
		                            output_dtypes      = output_dtypes,
		                            dask_kwargs        = dask_kwargs,
		                            n_workers          = ankParams.n_workers,
		                            threads_per_worker = ankParams.threads_per_worker,
		                            cluster            = cluster,
		                            )
	
	## Update the climatology
	clim.hpar  = hpar
	clim.hcov  = hcov
	clim._names.append(name)
	clim._bias[name]  = biasY
	clim._nslawid     = nslawid
	clim._nslaw_class = nslaw_class
	clim.cname    = cname
	clim._spatial = c_spatial
	
##}}}

## run_ank_cmd_fit ##{{{
@log_start_end(logger)
def run_ank_cmd_fit():
	
	## Check the command
	if not len(ankParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the fit command: {', '.join(ankParams.arg)}")
	
	available_commands = ["X","Y"]
	if ankParams.arg[0] not in available_commands:
		raise ValueError(f"Bad argument of the fit command ({ankParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if ankParams.arg[0] == "X":
		run_ank_cmd_fit_X()
	if ankParams.arg[0] == "Y":
		run_ank_cmd_fit_Y()
##}}}


