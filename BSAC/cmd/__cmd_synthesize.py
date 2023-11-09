
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

import xarray as xr

from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams

from ..__XZarr import XZarr


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## run_bsac_cmd_synthesize ##{{{
@log_start_end(logger)
def run_bsac_cmd_synthesize():
	
	## Read the grid
	grid = xr.open_dataset(bsacParams.config["grid"])
	grid_name = bsacParams.config["grid_name"]
	
	## Temporary files
	
	logger.info(grid)
	##
	for ifile in bsacParams.input:
		logger.info( f" * {os.path.basename(ifile)}" )
	
	##
	raise Exception("Stop for dev")
##}}}


