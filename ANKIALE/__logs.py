
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

import functools
import warnings
import logging
from typing import Any
from typing import Callable

import datetime as dt

###############
## Functions ##
###############

LINE = "=" * 80

def log_start_end( plog: logging.Logger ) -> Callable:##{{{
    """Decorator to add to the log the start / end of a function, and a walltime
    
    Parameters
    ----------
    plog:
        A logger from logging
    
    """
    def _decorator( f: Callable ) -> Callable:
        
        @functools.wraps(f)
        def f_decor( *args: Any , **kwargs: Any ) -> Any:
            plog.info(f"ANKIALE:{f.__name__}:start")
            time0 = dt.datetime.now(dt.UTC)
            out = f(*args,**kwargs)
            time1 = dt.datetime.now(dt.UTC)
            plog.info(f"ANKIALE:{f.__name__}:walltime:{time1-time0}")
            plog.info(f"ANKIALE:{f.__name__}:end")
            return out
        
        return f_decor
    
    return _decorator
##}}}

def disable_warnings( fun: Callable ) -> Callable:##{{{
    """Decorator to supress warnings
    """
    def fun_without_warnings( *args: Any , **kwargs: Any ) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fun( *args , **kwargs )
        return res
    
    return fun_without_warnings
##}}}

