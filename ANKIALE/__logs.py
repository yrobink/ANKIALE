
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

#############
## Imports ##
#############

import functools
import logging
import warnings

import datetime as dt

###############
## Functions ##
###############

LINE = "=" * 80

def log_start_end(plog):##{{{
	"""
	BSAC.log_start_end
	==================
	
	Decorator to add to the log the start / end of a function, and a walltime
	
	Parameters
	----------
	plog:
		A logger from logging
	
	"""
	def _decorator(f):
		
		@functools.wraps(f)
		def f_decor(*args,**kwargs):
			plog.info(f"BSAC:{f.__name__}:start")
			time0 = dt.datetime.now(dt.UTC)
			out = f(*args,**kwargs)
			time1 = dt.datetime.now(dt.UTC)
			plog.info(f"BSAC:{f.__name__}:walltime:{time1-time0}")
			plog.info(f"BSAC:{f.__name__}:end")
			return out
		
		return f_decor
	
	return _decorator
##}}}

def disable_warnings( fun ):##{{{
	"""
	BSAC.disable_warnings
	=====================
	
	Decorator to supress warnings
	"""
	def fun_without_warnings( *args , **kwargs ):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			res = fun( *args , **kwargs )
		return res
	
	return fun_without_warnings
##}}}

