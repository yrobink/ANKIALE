
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
import logging
import itertools as itt
from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams

from ..stats.__MultiGAM import MultiGAM
from ..stats.__MultiGAM import mgam_multiple_fit_bootstrap
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

## run_bsac_cmd_fit_X ##{{{
@log_start_end(logger)
def run_bsac_cmd_fit_X():
	
	## Check the inputs
	logger.info("Check inputs")
	n_X   = len(bsacParams.input)
	names = []
	if n_X == 0:
		raise ValueError("Fit asked, but no input given, abort.")
	inputs = { inp.split(",")[0] : inp.split(",")[1] for inp in bsacParams.input }
	for name in inputs:
		if not os.path.isfile(inputs[name]):
			raise FileNotFoundError(f"File '{inputs[name]}' is not found, abort.")
		else:
			logger.info( f" * covariate {name} detected" )
			names.append(name)
	bsacParams.clim.names = names
	
	## Now open the data
	logger.info("Open the data")
	X = {}
	for name in bsacParams.clim.names:
		idata   = xr.open_dataset( inputs[name] )[name].mean( dim = "run" )
		periods = list(set(idata.period.values.tolist()) & set(bsacParams.dpers))
		periods.sort()
		X[name] = { p : idata.sel( period = bsacParams.cper + [p] ).mean( dim = "period" ).dropna( dim = "time" ) for p in periods }
	bsacParams.clim.dpers = periods
	
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
	bsacParams.clim._time = time
	
	## Find the bias
	logger.info( "Build bias:" )
	bias = { name : 0 for name in X }
	for name in X:
		for p in X[name]:
			bias[name] = float(X[name][p].sel( time = slice(*bsacParams.clim.bper) ).mean( dim = "time" ).values)
			X[name][p]   -= bias[name]
		logger.info( f" * {name}: {bias[name]}" )
	bsacParams.clim._bias = bias
	
	## Build the natural forcings
	logger.info( "Build XN" )
	XN   = { name : { p : bsacParams.clim.XN.loc[X[name][p].time] for p in X[name] } for name in X }
	
	## Fit MultiGAM model with bootstrap
	logger.info( f"Fit the MultiGAM model (number of bootstrap: {bsacParams.n_samples})" )
	hpar,hcov = mgam_multiple_fit_bootstrap( X , XN ,
	                                         n_bootstrap = bsacParams.n_samples,
	                                         names  = bsacParams.clim.names,
	                                         dof    = bsacParams.clim.GAM_dof,
	                                         degree = bsacParams.clim.GAM_degree,
	                                         n_jobs = bsacParams.n_jobs
	                                         )
	
	bsacParams.clim.hpar = hpar
	bsacParams.clim.hcov = hcov
	
##}}}

## run_bsac_cmd_fit_Y ##{{{
@log_start_end(logger)
def run_bsac_cmd_fit_Y():
	
	## The current climatology
	clim = bsacParams.clim
	## Name of the variable to fit
	name = bsacParams.config["name"]
	
	## Spatial dimensions
	d_spatial = bsacParams.config.get("spatial")
	if d_spatial is not None:
		d_spatial = d_spatial.split(":")
	else:
		d_spatial = []
	d_spatial = tuple(d_spatial)
	
	## Set the covariate
	cname = bsacParams.config.get("cname")
	if cname is None:
		cname = clim.names[-1]
	
	## Open the data
	ifile = bsacParams.input[0]
	idata = xr.open_dataset(ifile).load()
	
	## Check if variables in idata
	for v in (name,) + d_spatial:
		if not v in idata:
			raise ValueError( f"Variable '{v}' not in input data" )
	
	## Spatial coordinates
	c_spatial = { d : idata[d] for d in d_spatial }
	
	## Find the nslaw
	nslawid = bsacParams.config.get("nslaw")
	if nslawid is None:
		raise ValueError( f"nslaw must be set" )
	
	## Find the bias, and remove it
	Y     = idata[name]
	biasY = Y.sel( time = slice(*clim.bias_period) ).mean( dim = [d for d in Y.dims if d not in d_spatial] )
	Y     = Y - biasY
	try:
		biasY = float(biasY)
	except:
		pass
	
	## Transform in ZXArray
	zY = zr.ZXArray.from_xarray(Y)
	
	## Check periods
	periods = list(set(clim.dpers) & set(Y.period.values.tolist()))
	periods.sort()
	
	## And restrict its
	clim = clim.restrict_dpers(periods)
	
	## Find the nslaw
	nslaw_class = nslawid_to_class(nslawid)
	hpar_namesY = clim.hpar_names + nslaw_class().coef_name
	
	## Design matrix of the covariate
	_,_,design,_ = clim.build_design_XFC()
	
	## Time axis
	time = sorted( list( set(clim.time.tolist()) & set(Y.time.values.tolist()) ) )
	zY   = Y.sel( time = time )
	
	## Samples
	n_samples = bsacParams.n_samples
	samples   = coords_samples(n_samples)
	samples   = zr.ZXArray.from_xarray( xr.DataArray( range(n_samples) , dims = ["sample"] , coords = [samples] ).astype(float) )
	
	## zxarray.apply_ufunc parameters
	output_dims      = [ ("hparY","dperiod","sample") + d_spatial ]
	output_coords    = [ [hpar_namesY,clim.dpers,samples.dataarray] + [ c_spatial[d] for d in d_spatial ] ]
	output_dtypes    = [ float ]
	dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] , ["time","period","run"] , []],
	                     "output_core_dims" : [ ["hparY","dperiod"] ],
	                     "kwargs" : { "nslaw_class" : nslaw_class , "design" : design , "hpar_names" : clim.hpar_names , "cname" : cname , "dpers" : clim.dpers , "time" : time },
	                     "dask" : "parallelized",
	                     "dask_gufunc_kwargs" : { "output_sizes" : { "hparY" : len(hpar_namesY) , "dperiod" : len(clim.dpers) } },
	                     "output_dtypes"  : [clim.hpar.dtype]
	                    }
	
	## Fit samples of parameters
	hpar  = clim.hpar
	hcov  = clim.hcov
	hpars = zr.apply_ufunc( nslaw_fit , hpar , hcov , zY, samples,
	                        bdims         = ("sample",) + d_spatial,
	                        max_mem       = bsacParams.total_memory,
	                        output_dims   = output_dims,
	                        output_coords = output_coords,
	                        output_dtypes = output_dtypes,
	                        dask_kwargs   = dask_kwargs )
	
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
	hpar,hcov = zr.apply_ufunc( mean_cov_hpars , hpars,
	                            bdims         = d_spatial,
	                            max_mem       = bsacParams.total_memory,
	                            output_dims   = output_dims,
	                            output_coords = output_coords,
	                            output_dtypes = output_dtypes,
	                            dask_kwargs   = dask_kwargs )
	
	## Update the climatology
	clim.hpar = hpar
	clim.hcov = hcov
	clim._names.append(name)
	clim._bias[name]  = biasY
	clim._nslawid     = nslawid
	clim._nslaw_class = nslaw_class
	clim._spatial = c_spatial
	
##}}}

## run_bsac_cmd_fit ##{{{
@log_start_end(logger)
def run_bsac_cmd_fit():
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the fit command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["X","Y"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the fit command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "X":
		run_bsac_cmd_fit_X()
	if bsacParams.arg[0] == "Y":
		run_bsac_cmd_fit_Y()
##}}}


