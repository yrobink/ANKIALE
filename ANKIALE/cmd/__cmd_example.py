
## Copyright(c) 2024 Yoann Robin
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

def find_path( output ):##{{{
	
	## Find output folder
	if "," in output:
		hiopath,diopath = bsacParams.output.split(",")
		hiopath = os.path.abspath(hiopath)
		diopath = os.path.abspath(diopath)
		for p in [hiopath,diopath]:
			if not os.path.isdir(p):
				raise NotADirectoryError( f"{p} is not a path" )
	else:
		iopath = os.path.abspath(output)
		if not os.path.isdir(iopath):
			raise NotADirectoryError( f"{iopath} is not a path" )
		hiopath = os.path.join( iopath , "home" )
		diopath = os.path.join( iopath , "data" )
		for p in [hiopath,diopath]:
			if not os.path.isdir(p):
				os.makedirs(p)
	logger.info( f" * Home path found: {hiopath}" )
	logger.info( f" * Data path found: {diopath}" )
	
	return hiopath,diopath
##}}}

## run_bsac_cmd_example ##{{{
@log_start_end(logger)
def run_bsac_cmd_example():
	
	cpath = os.path.dirname(os.path.abspath(__file__))
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the example command: {', '.join(bsacParams.arg)}")
	
	## Find available commands
	available_commands = sorted( list( set( [ "_".join( f.split(".")[0].split("_")[1:] ) for f in os.listdir( os.path.join( cpath , ".." , "data" ) ) if "EXAMPLE_" in f ] ) ) )
	cmd = bsacParams.arg[0].upper()
	if not cmd in available_commands:
		raise ValueError(f"Bad argument of the fit command ({cmd}), must be: {', '.join(available_commands)}")
	
	## Path
	hiopath,diopath = find_path( bsacParams.output )
	
	## Copy data
	logger.info( f" * Copy data" )
	idata = os.path.join( cpath , ".." , "data" , f"EXAMPLE_{cmd}.tar.gz" )
	with tarfile.open( idata , mode = "r" ) as ifile:
		ifile.extractall( os.path.join( diopath , "INPUT" ) )
	
	## Copy script
	logger.info( f" * Copy script" )
	
	## Parameters
	sh = "\n".join( ["#!/bin/bash","",
	"## Parameters",
	f"N_WORKERS={bsacParams.n_workers}",
	f"THREADS_PER_WORKER={bsacParams.threads_per_worker}",
	f"TOTAL_MEMORY={bsacParams.total_memory}",
	f"HPATH={hiopath}",
	f"DPATH={diopath}",
	f"N_SAMPLES={bsacParams.n_samples}",""] )
	
	## Add common part of the script
	with open( os.path.join( cpath , ".." , "data" , f"EXAMPLE_{cmd}.txt" ) , "r" ) as ish:
		sh = sh + "".join( ish.readlines() )
	
	## And save the script
	with open( os.path.join( hiopath , f"RUN_BSAC_EXAMPLE_{cmd}.sh" ) , "w" ) as ofile:
		ofile.write(sh)
##}}}



