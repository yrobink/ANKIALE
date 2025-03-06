
## Copyright(c) 2024 Yoann Robin
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

import os
import logging
import tarfile

from ..__logs import log_start_end

from ..__ANKParams import ankParams



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
		hiopath,diopath = ankParams.output.split(",")
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

## run_ank_cmd_example ##{{{
@log_start_end(logger)
def run_ank_cmd_example():
	
	cpath = os.path.dirname(os.path.abspath(__file__))
	
	## Check the command
	if not len(ankParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the example command: {', '.join(ankParams.arg)}")
	
	## Find available commands
	available_commands = sorted( list( set( [ "_".join( f.split(".")[0].split("_")[1:] ) for f in os.listdir( os.path.join( cpath , ".." , "data" ) ) if "EXAMPLE_" in f ] ) ) )
	cmd = ankParams.arg[0].upper()
	if cmd not in available_commands:
		raise ValueError(f"Bad argument of the fit command ({cmd}), must be: {', '.join(available_commands)}")
	
	## Path
	hiopath,diopath = find_path( ankParams.output )
	
	## Copy data
	logger.info( " * Copy data" )
	idata = os.path.join( cpath , ".." , "data" , f"EXAMPLE_{cmd}.tar.gz" )
	with tarfile.open( idata , mode = "r" ) as ifile:
		ifile.extractall( os.path.join( diopath , "INPUT" ) )
	
	## Copy script
	logger.info( " * Copy script" )
	
	## Begin
	beg = "#!/bin/bash\n"
	
	## Parameters
	sh = "\n".join( ["## Parameters",
	f"N_WORKERS={ankParams.n_workers}",
	f"THREADS_PER_WORKER={ankParams.threads_per_worker}",
	f"TOTAL_MEMORY={ankParams.total_memory}",
	f"HPATH={hiopath}",
	f"DPATH={diopath}",
	f"N_SAMPLES={ankParams.n_samples}","\n"] )
	
	## Open common part of the script
	with open( os.path.join( cpath , ".." , "data" , f"EXAMPLE_{cmd}.txt" ) , "r" ) as ish:
		csh =  ish.readlines()
	
	## Split in licence and core part
	lic = "".join(csh[:18])
	csh = "".join(csh[18:])
	
	## Merge
	sh = beg + lic + sh + csh
	
	## And save the script
	with open( os.path.join( hiopath , f"RUN_ANKIALE_EXAMPLE_{cmd}.sh" ) , "w" ) as ofile:
		ofile.write(sh)
##}}}



