
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

from ..__BSACParams import bsacParams


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## run_bsac_cmd_draw_Y ##{{{
@log_start_end(logger)
def run_bsac_cmd_draw_Y():
	
	## Draw data
	hpars = bsacParams.clim.rvsY( size = bsacParams.n_samples , add_BE = True )
	
	## And save
	raise NotImplementedError
##}}}

## run_bsac_cmd_draw ##{{{
@log_start_end(logger)
def run_bsac_cmd_draw():
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the show command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["Y"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the show command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "Y":
		run_bsac_cmd_draw_Y()
##}}}


