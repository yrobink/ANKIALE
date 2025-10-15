
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

##############
## Packages ##
##############

import sys
import traceback
import logging
import datetime as dt
from typing import Any

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

from .__ANKParams  import ankParams
from .__exceptions import AbortForHelpException
from .__exceptions import NoUserInputException
from .__exceptions import DevException

from .__logs import LINE
from .__logs import log_start_end

from .__release    import version
from .__curses_doc import print_doc

from .cmd.__cmd_attribute  import run_ank_cmd_attribute
from .cmd.__cmd_constrain  import run_ank_cmd_constrain 
from .cmd.__cmd_fit        import run_ank_cmd_fit       
from .cmd.__cmd_draw       import run_ank_cmd_draw       
from .cmd.__cmd_show       import run_ank_cmd_show      
from .cmd.__cmd_synthesize import run_ank_cmd_synthesize
from .cmd.__cmd_misc       import run_ank_cmd_misc
from .cmd.__cmd_example    import run_ank_cmd_example


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## run_ank ##{{{

@log_start_end(logger)
def run_ank() -> None:
    """
    ANKIALE.run_ank
    ===============
    
    Main execution, after the control of user input.
    
    """
    
    ankParams.init_dask()
    try:
        
        ## Init clim
        ankParams.init_clim()
        logger.info(LINE)
        logger.info("Summary of the climatology")
        logger.info(ankParams.clim)
        logger.info(LINE)
        
        ## Run command
        cmd = ankParams.cmd
        match cmd.lower():
            case "show":
                run_ank_cmd_show()
            case "fit":
                run_ank_cmd_fit()
            case "draw":
                run_ank_cmd_draw()
            case "synthesize":
                run_ank_cmd_synthesize()
            case "constrain":
                run_ank_cmd_constrain()
            case "attribute":
                run_ank_cmd_attribute()
            case "misc":
                run_ank_cmd_misc()
            case "example":
                run_ank_cmd_example(False)
            case "sexample":
                run_ank_cmd_example(True)
        
        ## And save clim ?
        logger.info(LINE)
        logger.info("Summary of the climatology")
        logger.info(ankParams.clim)
        logger.info(LINE)
        if ankParams.save_clim is not None:
            ankParams.clim.save( ankParams.save_clim )
            logger.info(LINE)
    except DevException as e:
        logger.info(LINE)
        logger.info("STOP FOR DEVELOPMENT")
        logger.info(e)
        logger.info(LINE)
    finally:
        ankParams.stop_dask()
    
##}}}

def start_ank( *argv: Any ) -> None:##{{{
    """
    ANKIALE.start_ank
    =================
    
    Starting point of 'ank'.
    
    """
    ## Time counter
    walltime0 = dt.datetime.now(dt.UTC)
    ## Read input
    try:
        ankParams.init_from_user_input(*argv)
    except NoUserInputException as e:
        print(e)
        return
    
    ## Init logs
    ankParams.init_logging()
    
    ## Logging
    logger.info(LINE)
    logger.info( "Start: {}".format( str(walltime0)[:19].replace(' ','T') + 'Z' ) )
    logger.info(LINE)
    logger.info( r"           _   _ _  _______          _      ______ " )
    logger.info( r"     /\   | \ | | |/ /_   _|   /\   | |    |  ____|" )
    logger.info( r"    /  \  |  \| | ' /  | |    /  \  | |    | |__   " )
    logger.info( r"   / /\ \ | . ` |  <   | |   / /\ \ | |    |  __|  " )
    logger.info( r"  / ____ \| |\  | . \ _| |_ / ____ \| |____| |____ " )
    logger.info( r" /_/    \_\_| \_|_|\_\_____/_/    \_\______|______|" )
    logger.info( r"                                                   " )
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
    
    logger.info( f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ({sys.version_info.releaselevel})" )
    logger.info( "Packages version:" )
    logger.info( " * {:{fill}{align}{n}}".format( "ANKIALE" , fill = " " , align = "<" , n = 12 ) + f"version {version}" )
    for name_pkg,pkg in pkgs:
        logger.info( " * {:{fill}{align}{n}}".format( name_pkg , fill = " " , align = "<" , n = 12 ) +  f"version {pkg.__version__}" )
    logger.info(LINE)
    
    ## Set (or not) the seed
    if ankParams.set_seed is not None:
        np.random.seed(int(ankParams.set_seed))
    
    ## Serious functions start here
    try:
        
        ## Check inputs
        ankParams.check()
        
        ## Init temporary
        ankParams.init_tmp()
        zr.zxParams.tmp_folder = ankParams.tmp
        
        ## List of all input
        logger.info("Input parameters:")
        for key in ankParams.keys():
            logger.info( " * {:{fill}{align}{n}}".format( key , fill = " ",align = "<" , n = 10 ) + ": {}".format(ankParams[key]) )
        logger.info(LINE)
        
        ## User asks help
        if ankParams.help:
            print_doc()
        
        ## User asks help
        if ankParams.version:
            print(version)
        
        ## In case of abort, raise Exception
        if ankParams.abort:
            raise ankParams.error
        
        ## Go!
        run_ank()
        
    except AbortForHelpException:
        pass
    except Exception as e:
        logger.error(LINE)
        logger.error( traceback.print_tb( sys.exc_info()[2] ) )
        logger.error( f"Error: {e}" )
        logger.error(LINE)
    finally:
        ankParams.clean_tmp()
    
    ## End
    walltime1 = dt.datetime.now(dt.UTC)
    logger.info(LINE)
    logger.info( "End: {}".format( str(walltime1)[:19].replace(' ','T') + 'Z' ) )
    logger.info( "Wall time: {}".format(walltime1 - walltime0) )
    logger.info(LINE)
##}}}

