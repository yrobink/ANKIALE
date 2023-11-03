
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

import logging
from ..__logs import LINE
from ..__logs import log_start_end

import numpy as np
import xarray as xr

from ..__BSACParams import bsacParams

from ..plot.__ebm        import plot_EBM
from ..plot.__covariates import plot_covariates


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## run_bsac_cmd_show_ebm ##{{{
@log_start_end(logger)
def run_bsac_cmd_show_ebm():
	plot_EBM( bsacParams.output )
##}}}

## run_bsac_cmd_show_X ##{{{
@log_start_end(logger)
def run_bsac_cmd_show_X():
	
	## Read the fit parameters
	for ifile in bsacParams.input:
		if "," in ifile:
			continue
	
	## Draw data
	XFC = bsacParams.clim.rvsX( size = bsacParams.n_samples )
	
	## And plot it
	plot_covariates( XFC , ofile = bsacParams.output )
	
##}}}

## run_bsac_cmd_show ##{{{
@log_start_end(logger)
def run_bsac_cmd_show():
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the show command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["EBM","X"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the show command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "EBM":
		run_bsac_cmd_show_ebm()
	if bsacParams.arg[0] == "X":
		run_bsac_cmd_show_X()
	
##}}}


