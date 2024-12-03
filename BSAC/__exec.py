
## Copyright(c) 2023, 2024 Yoann Robin
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

##############
## Packages ##
##############

import sys
import os
import logging
import traceback
import datetime as dt

import numpy   as np
import pandas  as pd
import xarray  as xr
import zxarray as zr
import dask
import distributed
import zarr
import netCDF4
import SDFC


#############
## Imports ##
#############

from .__BSACParams  import bsacParams
from .__exceptions  import AbortForHelpException
from .__exceptions  import NoUserInputException

from .__logs import LINE
from .__logs import log_start_end

from .__release    import version
from .__curses_doc import print_doc

from .cmd.__cmd_attribute  import run_bsac_cmd_attribute
from .cmd.__cmd_constrain  import run_bsac_cmd_constrain 
from .cmd.__cmd_fit        import run_bsac_cmd_fit       
from .cmd.__cmd_draw       import run_bsac_cmd_draw       
from .cmd.__cmd_show       import run_bsac_cmd_show      
from .cmd.__cmd_synthesize import run_bsac_cmd_synthesize
from .cmd.__cmd_misc       import run_bsac_cmd_misc
from .cmd.__cmd_example    import run_bsac_cmd_example


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## run_bsac ##{{{

@log_start_end(logger)
def run_bsac():
	"""
	BSAC.run_bsac
	=============
	
	Main execution, after the control of user input.
	
	"""
	
	bsacParams.init_dask()
	try:
		
		## Init clim
		bsacParams.init_clim()
		logger.info(LINE)
		logger.info("Summary of the climatology")
		logger.info(bsacParams.clim)
		logger.info(LINE)
		
		## Run command
		cmd = bsacParams.cmd
		if cmd.lower() == "show":
			run_bsac_cmd_show()
		elif cmd.lower() == "fit":
			run_bsac_cmd_fit()
		elif cmd.lower() == "draw":
			run_bsac_cmd_draw()
		elif cmd.lower() == "synthesize":
			run_bsac_cmd_synthesize()
		elif cmd.lower() == "constrain":
			run_bsac_cmd_constrain()
		elif cmd.lower() == "attribute":
			run_bsac_cmd_attribute()
		elif cmd.lower() == "misc":
			run_bsac_cmd_misc()
		elif cmd.lower() == "example":
			run_bsac_cmd_example()
		
		## And save clim ?
		logger.info(LINE)
		logger.info("Summary of the climatology")
		logger.info(bsacParams.clim)
		logger.info(LINE)
		if bsacParams.save_clim is not None:
			bsacParams.clim.save( bsacParams.save_clim )
			logger.info(LINE)
		
	finally:
		bsacParams.stop_dask()
	
##}}}

def start_bsac(*argv):##{{{
	"""
	BSAC.start_bsac
	===============
	
	Starting point of 'bsac'.
	
	"""
	## Time counter
	walltime0 = dt.datetime.now(dt.UTC)
	## Read input
	try:
		bsacParams.init_from_user_input(*argv)
	except NoUserInputException as e:
		print(e)
		return
	
	## Init logs
	bsacParams.init_logging()
	
	## Logging
	logger.info(LINE)
	logger.info( "Start: {}".format( str(walltime0)[:19] + " (UTC)") )
	logger.info(LINE)
	logger.info( r" ____   _____         _____ " )
	logger.info( r"|  _ \ / ____|  /\   / ____|" )
	logger.info( r"| |_) | (___   /  \ | |     " )
	logger.info( r"|  _ < \___ \ / /\ \| |     " )
	logger.info( r"| |_) |____) / ____ \ |____ " )
	logger.info( r"|____/|_____/_/    \_\_____|" )
	logger.info( r"                            " )
	logger.info(LINE)
	
	
	## Package version
	pkgs = [
	        ("numpy"      , np ),
	        ("pandas"     , pd ),
	        ("xarray"     , xr ),
	        ("zxarray"    , zr ),
	        ("dask"       , dask ),
	        ("distributed", distributed ),
	        ("zarr"       , zarr ),
	        ("netCDF4"    , netCDF4 ),
	        ("SDFC"       , SDFC )
	       ]
	
	logger.info( "Packages version:" )
	logger.info( " * {:{fill}{align}{n}}".format( "BSAC" , fill = " " , align = "<" , n = 12 ) + f"version {version}" )
	for name_pkg,pkg in pkgs:
		logger.info( " * {:{fill}{align}{n}}".format( name_pkg , fill = " " , align = "<" , n = 12 ) +  f"version {pkg.__version__}" )
	logger.info(LINE)
	
	## Set (or not) the seed
	if bsacParams.set_seed is not None:
		np.random.seed(int(bsacParams.set_seed))
	
	## Serious functions start here
	try:
		
		## Check inputs
		bsacParams.check()
		
		## Init temporary
		bsacParams.init_tmp()
		zr.zxParams.tmp_folder = bsacParams.tmp
		
		## List of all input
		logger.info("Input parameters:")
		for key in bsacParams.keys():
			logger.info( " * {:{fill}{align}{n}}".format( key , fill = " ",align = "<" , n = 10 ) + ": {}".format(bsacParams[key]) )
		logger.info(LINE)
		
		## User asks help
		if bsacParams.help:
			print_doc()
		
		## In case of abort, raise Exception
		if bsacParams.abort:
			raise bsacParams.error
		
		## Go!
		run_bsac()
		
	except AbortForHelpException:
		pass
#	except Exception as e:
#		logger.error(LINE)
#		logger.error( traceback.print_tb( sys.exc_info()[2] ) )
#		logger.error( f"Error: {e}" )
#		logger.error(LINE)
	
	## End
	walltime1 = dt.datetime.now(dt.UTC)
	logger.info(LINE)
	logger.info( "End: {}".format(str(walltime1)[:19] + " (UTC)") )
	logger.info( "Wall time: {}".format(walltime1 - walltime0) )
	logger.info(LINE)
##}}}

