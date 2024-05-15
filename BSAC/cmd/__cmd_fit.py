
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
import zarr
from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams

from ..stats.__MultiGAM import MultiGAM
from ..stats.__MultiGAM import mgam_multiple_fit_bootstrap
from ..stats.__tools    import nslawid_to_class
from ..stats.__NSLawMLEFit import nslaw_fit_bootstrap

import numpy  as np
import xarray as xr


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
		X[name] = { p : idata.sel( period = bsacParams.cper + [p] ).mean( dim = "period" ) for p in periods }
	bsacParams.clim.dpers = periods
	
	## Find the bias
	logger.info( "Build bias:" )
	bias = { name : 0 for name in X }
	time    = []
	for name in X:
		for p in X[name]:
			bias[name] = float(X[name][p].sel( time = slice(*bsacParams.clim.bper) ).mean( dim = "time" ).values)
			X[name][p]   -= bias[name]
			time          = time + X[name][p]["time"].values.tolist()
		logger.info( f" * {name}: {bias[name]}" )
	time = list(set(time))
	time.sort()
	time = np.array(time)
	bsacParams.clim._bias = bias
	bsacParams.clim._time = time
	
	## Build the natural forcings
	logger.info( "Build XN" )
	XN   = { name : { p : bsacParams.clim.XN.loc[X[name][p].time] for p in X[name] } for name in X }
	
	## Fit MultiGAM model with bootstrap
	logger.info( f"Fit the MultiGAM model (number of bootstrap: {bsacParams.n_samples})" )
	coef_,cov_ = mgam_multiple_fit_bootstrap( X , XN ,
	                                          n_bootstrap = bsacParams.n_samples,
	                                          names  = bsacParams.clim.names,
	                                          dof    = bsacParams.clim.GAM_dof,
	                                          degree = bsacParams.clim.GAM_degree,
	                                          n_jobs = bsacParams.n_jobs
	                                          )
	
	bsacParams.clim.mean_ = coef_
	bsacParams.clim.cov_  = cov_
	
##}}}

## run_bsac_cmd_fit_Y ##{{{
@log_start_end(logger)
def run_bsac_cmd_fit_Y():
	
	## The current climatology
	clim = bsacParams.clim
	## Name of the variable to fit
	name = bsacParams.config["name"]
	
	## Spatial coordinates
	spatial = bsacParams.config.get("spatial")
	if spatial is not None:
		spatial = spatial.split(":")
	else:
		spatial = []
	
	## Set the covariate
	cname = bsacParams.config.get("cname")
	if cname is None:
		cname = clim.names[-1]
	
	## Open the data
	ifile = bsacParams.input[0]
	idata = xr.open_dataset(ifile).load()
	
	## Check if variables in idata
	for v in [name] + spatial:
		if not v in idata:
			raise ValueError( f"Variable '{v}' not in input data" )
	
	## Find the nslaw
	nslawid = bsacParams.config.get("nslaw")
	if nslawid is None:
		raise ValueError( f"nslaw must be set" )
	
	## Find the bias, and remove it
	Y     = idata[name]
	biasY = Y.sel( time = slice(*clim.bias_period) ).mean( dim = [d for d in Y.dims if d not in spatial] )
	Y     = Y - biasY
	try:
		biasY = float(biasY)
	except:
		pass
	
	## Check periods
	periods = list(set(clim.dpers) & set(Y.period.values.tolist()))
	periods.sort()
	
	## And restrict its
	clim = clim.restrict_dpers(periods)
	
	## Draw X
	X,hparX  = clim.rvsX( bsacParams.n_samples , add_BE = True , return_hpar = True )
	XF = X.XF.loc[:,cname,:,:]
	
	## Restrict the time axis of Y
	ctime = list( set(X.time.values.ravel().tolist()) & set(Y.time.values.ravel().tolist()) )
	ctime.sort()
	ctime = xr.DataArray( ctime , dims = ["time"] , coords = [ctime] )
	Y = Y.sel( time = ctime )
	
	##
	nslaw_class = nslawid_to_class(nslawid)
	nslaw       = nslaw_class()
	if Y.ndim == 3:
		hparYshape = [nslaw.coef_name,periods,X.sample]
	else:
		hparYshape = [nslaw.coef_name,periods,X.sample] + [Y.coords[d] for d in Y.dims[3:]]
	hparYshape = [len(s) for s in hparYshape]
	
	## Create the zarr file of hyperparameter of Y
	hparY = zarr.open( os.path.join( bsacParams.tmp , "hparY.zarr" ) , mode = "w" , shape = hparYshape , dtype = "float32" , compressor = None )
	
	## Now the fit
	nslaw_fit_bootstrap( Y = Y , X = XF , hparY = hparY , nslawid = nslawid , n_bootstrap = bsacParams.n_samples , n_jobs = bsacParams.n_jobs )
	
	## Mean and cov for new climatology
	meanX = clim.mean_.copy()
	covX  = clim.cov_.copy()
	
	if Y.ndim == 3:
		mean_ = xr.DataArray( np.zeros( (meanX.size + hparYshape[0],) ) , dims = ["hpar"] , coords = [hparX.hpar.values.tolist()+nslaw.coef_name] )
		cov_  = xr.DataArray( np.zeros( [meanX.size + hparYshape[0] for _ in range(2)] ) , dims = ["hpar0","hpar1"] , coords = [hparX.hpar.values.tolist()+nslaw.coef_name for _ in range(2)] )
	else:
		mean_ = xr.DataArray( np.zeros( [meanX.size + hparYshape[0]] + hparYshape[3:] ) , dims = ["hpar"] + list(Y.dims[3:]) , coords = [hparX.hpar.values.tolist()+nslaw.coef_name] + [Y.coords[d] for d in Y.dims[3:]] )
		cov_  = xr.DataArray( np.zeros( [meanX.size + hparYshape[0] for _ in range(2)] + hparYshape[3:] ) , dims = ["hpar0","hpar1"] + list(Y.dims[3:]) , coords = [hparX.hpar.values.tolist()+nslaw.coef_name for _ in range(2)] + [Y.coords[d] for d in Y.dims[3:]] )
	
	## Variables for a loop on spatial dimension (if exists)
	if Y.ndim == 3:
		spatial = [1]
	else:
		spatial = [Y[d].size for d in Y.dims[3:]]
	
	## Loop on spatial dimension to build mean and cov
	xhpar = np.zeros( (hparX.shape[1]+hparYshape[0],len(periods),X.sample.size) )
	xhpar[:hparX.shape[1],:,:] = hparX.values.T.reshape(hparX.shape[1],1,X.sample.size)
	for spatial_idx in itt.product(*[range(s) for s in spatial]):
		
		## Extract the spatial point for hparY
		idx    = (slice(None),slice(None),slice(None))
		idx1d  = (slice(None),)
		idx2d  = (slice(None),slice(None))
		if Y.ndim > 3:
			idx   = idx + spatial_idx
			idx1d = idx1d + spatial_idx
			idx2d = idx2d + spatial_idx
		xhpar[hparX.shape[1]:,:,:] = hparY.get_orthogonal_selection(idx)
		mean_[idx1d] = xhpar.reshape(xhpar.shape[0],-1).mean( axis = 1 )
		cov_[idx2d]  = np.cov( xhpar.reshape(xhpar.shape[0],-1) )
	
	## Mask
	try:
		mean_ = mean_.where( np.isfinite(biasY) , np.nan )
		cov_  =  cov_.where( np.isfinite(biasY) , np.nan )
	except:
		pass
	
	## Update the climatology
	clim.mean_ = mean_.values
	clim.cov_  = cov_.values
	clim._names.append(name)
	clim._bias[name]  = biasY
	clim._nslawid     = nslawid
	clim._nslaw_class = nslaw_class
	
	if Y.ndim > 3:
		clim._spatial = { d : Y[d] for d in Y.dims[3:] }
	
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


