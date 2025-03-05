
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

import logging
from ..__logs import LINE
from ..__logs import log_start_end

import numpy as np
import xarray as xr

from ..__BSACParams import bsacParams
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

## run_bsac_cmd_show_XN ##{{{
@log_start_end(logger)
def run_bsac_cmd_show_XN():
	plot_XN( bsacParams.output )
##}}}

## run_bsac_cmd_show_X ##{{{
@log_start_end(logger)
def run_bsac_cmd_show_X():
	
	## Draw data
	XFC = bsacParams.clim.rvsX( size = bsacParams.n_samples )
	
	## And plot it
	plot_covariates( XFC , ofile = bsacParams.output )
	
##}}}

## run_bsac_cmd_show_CX ##{{{
@log_start_end(logger)
def run_bsac_cmd_show_CX():
	
	## Read climatology for comparison
	ifileS = bsacParams.input[0]
	climS  = Climatology.init_from_file(ifileS)
	
	## Read observations
	Xo = {}
	for inp in bsacParams.input[1:]:
		key     = inp.split(",")[0]
		ifile   = inp.split(",")[1]
		Xo[key] = xr.open_dataset(ifile)[key]
		Xo[key] = Xo[key] - Xo[key].sel( time = slice(*[str(y) for y in bsacParams.bias_period]) ).mean()
	
	## Draw data
	XFC = bsacParams.clim.rvsX( size = bsacParams.n_samples )
	SFC = climS.rvsX( size = bsacParams.n_samples )
	
	## And plot it
	plot_constrain_CX( XFC , SFC , Xo , ofile = bsacParams.output )
	
##}}}

## run_bsac_cmd_show_Y ##{{{
@log_start_end(logger)
def run_bsac_cmd_show_Y():
	raise NotImplementedError
##}}}

## run_bsac_cmd_show ##{{{
@log_start_end(logger)
def run_bsac_cmd_show():
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the show command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["XN","X","Y","CX"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the show command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "XN":
		run_bsac_cmd_show_XN()
	if bsacParams.arg[0] == "X":
		run_bsac_cmd_show_X()
	if bsacParams.arg[0] == "CX":
		run_bsac_cmd_show_CX()
	if bsacParams.arg[0] == "Y":
		run_bsac_cmd_show_Y()
	
##}}}


