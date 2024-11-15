
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
import tarfile

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

## run_bsac_cmd_example_GMST ##{{{
@log_start_end(logger)
def run_bsac_cmd_example_GMST():
	
	## Find output folder
	iopath = os.path.abspath(bsacParams.output)
	if not os.path.isdir(iopath):
		raise NotADirectoryError( f"{iopath} is not a path" )
	logger.info( f" * Output path found: {iopath}" )
	
	## Copy data
	logger.info( f" * Copy data" )
	cpath = os.path.dirname(os.path.abspath(__file__))
	idata = os.path.join( cpath , ".." , "data" , "GMST.tar.gz" )
	with tarfile.open( idata , mode = "r" ) as ifile:
		ifile.extractall( os.path.join( iopath , "INPUT" ) )
	
	## Copy script
	logger.info( f" * Copy script" )
	
	## Parameters
	sh = "\n".join( ["#!/bin/bash","",
	"## Parameters",
	f"N_WORKERS={bsacParams.n_workers}",
	f"THREADS_PER_WORKER={bsacParams.threads_per_worker}",
	f"TOTAL_MEMORY={bsacParams.total_memory}",
	f"WDIR={iopath}",
	f"N_SAMPLES={bsacParams.n_samples}",
	f"BIAS_PERIOD='1900/1950'",
	f"GAM_DOF=7",
	f"GAM_DEGREE=3",""] )
	
	## Add common part of the script
	with open( os.path.join( cpath , ".." , "data" , "EXAMPLE_GMST.txt" ) , "r" ) as ish:
		sh = sh + "".join( ish.readlines() )
	
	## And save the script
	with open( os.path.join( iopath , "RUN_BSAC_EXAMPLE_GMST.sh" ) , "w" ) as ofile:
		ofile.write(sh)
	
##}}}

## run_bsac_cmd_example ##{{{
@log_start_end(logger)
def run_bsac_cmd_example():
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the example command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["GMST"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the fit command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "GMST":
		run_bsac_cmd_example_GMST()
##}}}



