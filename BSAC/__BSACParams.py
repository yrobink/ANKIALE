
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

from __future__ import annotations

import sys
import os
import datetime as dt
import argparse
import tempfile
import psutil
import logging
import dataclasses

import dask
import distributed

import numpy as np

from .__exceptions  import AbortForHelpException
from .__exceptions  import NoUserInputException
from .__sys import SizeOf
from .__climatology import Climatology

###############
## Variables ##
###############


@dataclasses.dataclass
class BSACParams:
	
	abort : bool                = False
	error : Exception | None    = None
	help  : bool                = False
	log   : tuple[str|int,str|None] = ("WARNING",None)
	
	cmd : str  | None = None
	arg : list | None = None
	
	n_workers            : int                = 1
	threads_per_worker   : int                = 1
	memory_per_worker    : str                = "auto"
	frac_memory_per_array: float              = 0.2
	total_memory         : str                = "auto"
	client               : distributed.client.Client | None = None
	disable_dask         : bool               = False
	
	tmp_base    : str | None         = None
	tmp_gen     : tempfile.TemporaryDirectory | None = None
	tmp         : str | None         = None
	tmp_gen_dask: tempfile.TemporaryDirectory | None = None
	tmp_dask    : str | None         = None
	
	config : str | None = None
	
	input  : list | None = None
	output : str  | None = None
	common_period     : list | None = None
	different_periods : list | None = None
	bias_period       : tuple[int,int] | str = "1961/1990"
	n_samples         : int = 10000
	
	clim      : Climatology | None = None
	load_clim :         str | None = None
	save_clim :         str | None = None
	set_seed  :         int | None = None
	
	def init_from_user_input( self , *argv ):##{{{
		
		if len(argv) == 0:
			raise NoUserInputException("No arguments given, abort.\nRead the documentation with 'xsbck --help' ?")
		
		## Parser for user input
		parser = argparse.ArgumentParser( add_help = False )
		parser.add_argument( "CMD" , nargs = '*' )
		
		parser.add_argument( "-h" , "--help" , action = "store_const" , const = True , default = False )
		parser.add_argument( "--log" , nargs = '*' , default = ("WARNING",None) )
		
		parser.add_argument( "--tmp"                    , default = None )
		parser.add_argument( "--n-workers"              , default = 1 , type = int )
		parser.add_argument( "--threads-per-worker"     , default = 1 , type = int )
		parser.add_argument( "--memory-per-worker"      , default = "auto" )
		parser.add_argument( "--frac-memory-per-array"  , default = 0.15 , type = float )
		parser.add_argument( "--total-memory"           , default = "auto" )
		parser.add_argument( "--disable-dask" , action = "store_const" , const = True , default = False )
		
		parser.add_argument( "--config" , default = None )
		
		parser.add_argument( "--input" , nargs = "+" , action = "extend" )
		parser.add_argument( "--output" , default = None )
		
		parser.add_argument( "--load-clim" , default = None )
		parser.add_argument( "--save-clim" , default = None )
		parser.add_argument( "--set-seed"  , default = None )
		
		parser.add_argument( "--common-period"     , default = None )
		parser.add_argument( "--different-periods" , default = None )
		parser.add_argument( "--bias-period"       , default = "1961/1990" )
		parser.add_argument( "--n-samples"         , default = 10000 , type = int )
		
		## Transform in dict
		kwargs = vars(parser.parse_args(argv))
		
		## And store in the class
		for key in kwargs:
			if key in ["CMD"]:
				try:
					self.cmd = kwargs[key][0]
					self.arg = kwargs[key][1:]
				except:
					pass
				continue
			
			if key not in self.__dict__:
				raise Exception(f"Parameter '{key}' not present in the class")
			self.__dict__[key] = kwargs[key]
			
		
	##}}}
	
	def init_tmp(self):##{{{
		
		if self.tmp is None:
			self.tmp_base = tempfile.gettempdir()
		else:
			self.tmp_base     = self.tmp
		
		now               = str(dt.datetime.utcnow())[:19].replace("-","").replace(":","").replace(" ","-")
		prefix            = f"BSAC_{now}_"
		self.tmp_gen      = tempfile.TemporaryDirectory( dir = self.tmp_base , prefix = prefix )
		self.tmp          = self.tmp_gen.name
		self.tmp_gen_dask = tempfile.TemporaryDirectory( dir = self.tmp_base , prefix = prefix + "DASK_" )
		self.tmp_dask     = self.tmp_gen_dask.name
	##}}}
	
	def init_logging(self):##{{{
		
		if len(self.log) == 0:
			self.log = ("INFO",None)
		elif len(self.log) == 1:
			
			try:
				level = int(self.log[0])
				lfile = None
			except:
				try:
					level = getattr( logging , self.log[0].upper() , None )
					lfile = None
				except:
					level = "INFO"
					lfile = self.log[0]
			self.log = (level,lfile)
		
		level,lfile = self.log
		
		## loglevel can be an integet
		try:
			level = int(level)
		except:
			level = getattr( logging , level.upper() , None )
		
		## If it is not an integer, raise an error
		if not isinstance( level , int ): 
			raise UserDefinedLoggingLevelError( f"Invalid log level: {level}; nothing, an integer, 'debug', 'info', 'warning', 'error' or 'critical' expected" )
		
		##
		log_kwargs = {
			"format" : '%(message)s',
			"level"  : level
			}
		
		if lfile is not None:
			log_kwargs["filename"] = lfile
		
		logging.basicConfig(**log_kwargs)
		logging.captureWarnings(True)
	##}}}
	
	def init_dask(self):##{{{
		
		if self.disable_dask:
			return
		
		dask_config  = { "temporary_directory" : self.tmp_dask } #, "array.slicing.split_large_chunks" : False }
		client_config = { "n_workers"          :self.n_workers ,
		                  "threads_per_worker" :self.threads_per_worker ,
		                  "memory_limit"       : f"{self.memory_per_worker.B}B" }
		
		dask.config.set(**dask_config)
		self.client = distributed.Client(**client_config)
	##}}}
	
	def init_clim(self):##{{{
		
		## Load from file
		if self.load_clim is not None:
			self.clim = Climatology.init_from_file( self.load_clim )
			self.clim._tmp = self.tmp
			return
		
		## Init from scratch
		self.clim = Climatology()
		
		self.clim._Xconfig["GAM_dof"]    = int( self.config.get( "GAM_dof"    , 7 ) )
		self.clim._Xconfig["GAM_degree"] = int( self.config.get( "GAM_degree" , 3 ) )
		
		self.clim.bper  = self.bias_period
		self.clim.cper  = self.common_period
		self.clim.dpers = self.different_periods
		self.clim._tmp  = self.tmp
		
	##}}}
	
	def stop_dask(self):##{{{
		if self.disable_dask:
			return
		
		self.client.close()
		del self.client
		self.client = None
	##}}}
	
	def check( self ): ##{{{
		
		try:
			if self.help:
				raise AbortForHelpException
			
			## Check the CMD
			list_cmd = ["show","fit","draw","synthesize","constrain","attribute","misc"]
			if self.cmd is None or not self.cmd.lower() in list_cmd:
				raise Exception(f"Bad command arguments, must be one of {', '.join(list_cmd)}")
			
			## Check and set the memory
			if self.memory_per_worker == "auto":
				if self.total_memory == "auto":
					self.total_memory = SizeOf( n = int(0.8 * psutil.virtual_memory().total) , unit = "B" )
				else:
					self.total_memory = SizeOf(self.total_memory)
				self.memory_per_worker = self.total_memory // self.n_workers
			else:
				self.memory_per_worker = SizeOf(self.memory_per_worker)
				self.total_memory      = self.memory_per_worker * self.n_workers
			
			## Change periods format
			try:
				self.common_period     = self.common_period.split(",")
				self.different_periods = self.different_periods.split(",")
			except:
				pass
			
			self.bias_period = tuple([int(s) for s in self.bias_period.split("/")])
			
			##
			if self.config is not None:
				self.config     = { c.split("=")[0] : c.split("=")[1] for c in self.config.split(",") }
			else:
				self.config = {}
			
		except Exception as e:
			self.abort = True
			self.error = e
		
		
	##}}}
	
	def keys(self):##{{{
		keys = [key for key in self.__dict__]
		keys.sort()
		return keys
	##}}}
	
	def __getitem__( self , key ):##{{{
		return self.__dict__.get(key)
	##}}}
	
	## Properties ## {{{
	
	@property
	def cper(self):
		return self.common_period
	
	@property
	def dpers(self):
		return self.different_periods
	
	@property
	def n_jobs(self):
		return self.n_workers * self.threads_per_worker
	
	##}}}

bsacParams = BSACParams()


