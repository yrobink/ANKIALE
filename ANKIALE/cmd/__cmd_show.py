
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

import logging
from ..__logs import log_start_end

import xarray as xr

from ..__ANKParams import ankParams
from ..__climatology import Climatology

from ..plot.__XN         import plot_XN
from ..plot.__covariates import plot_covariates
from ..plot.__covariates import plot_constrain_CX


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## run_ank_cmd_show_XN ##{{{
@log_start_end(logger)
def run_ank_cmd_show_XN():
	plot_XN( ankParams.output )
##}}}

## run_ank_cmd_show_X ##{{{
@log_start_end(logger)
def run_ank_cmd_show_X():
	
	## Draw data
	XFC = ankParams.clim.rvsX( size = ankParams.n_samples )
	
	## And plot it
	plot_covariates( XFC , ofile = ankParams.output )
	
##}}}

## run_ank_cmd_show_CX ##{{{
@log_start_end(logger)
def run_ank_cmd_show_CX():
	
	## Read climatology for comparison
	ifileS = ankParams.input[0]
	climS  = Climatology.init_from_file(ifileS)
	
	## Read observations
	Xo = {}
	for inp in ankParams.input[1:]:
		key     = inp.split(",")[0]
		ifile   = inp.split(",")[1]
		Xo[key] = xr.open_dataset(ifile)[key]
		Xo[key] = Xo[key] - Xo[key].sel( time = slice(*[str(y) for y in ankParams.bias_period]) ).mean()
	
	## Draw data
	XFC = ankParams.clim.rvsX( size = ankParams.n_samples )
	SFC = climS.rvsX( size = ankParams.n_samples )
	
	## And plot it
	plot_constrain_CX( XFC , SFC , Xo , ofile = ankParams.output )
	
##}}}

## run_ank_cmd_show_Y ##{{{
@log_start_end(logger)
def run_ank_cmd_show_Y():
	raise NotImplementedError
##}}}

## run_ank_cmd_show ##{{{
@log_start_end(logger)
def run_ank_cmd_show():
	
	## Check the command
	if not len(ankParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the show command: {', '.join(ankParams.arg)}")
	
	available_commands = ["XN","X","Y","CX"]
	if not ankParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the show command ({ankParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if ankParams.arg[0] == "XN":
		run_ank_cmd_show_XN()
	if ankParams.arg[0] == "X":
		run_ank_cmd_show_X()
	if ankParams.arg[0] == "CX":
		run_ank_cmd_show_CX()
	if ankParams.arg[0] == "Y":
		run_ank_cmd_show_Y()
	
##}}}


