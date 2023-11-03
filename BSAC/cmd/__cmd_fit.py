
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
from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams

from ..stats.__MultiGAM import MultiGAM
from ..stats.__MultiGAM import mgam_multiple_fit_bootstrap

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
	coef_,cov_ = mgam_multiple_fit_bootstrap( X , XN , n_bootstrap = bsacParams.n_samples )
	
	bsacParams.clim.mean_ = coef_
	bsacParams.clim.cov_  = cov_
	
##}}}

## run_bsac_cmd_fit_Y ##{{{
@log_start_end(logger)
def run_bsac_cmd_fit_Y():
	raise NotImplementedError
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


