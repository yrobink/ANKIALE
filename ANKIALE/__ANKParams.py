
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

from __future__ import annotations

import os
import toml
import datetime as dt
import argparse
import tempfile
import psutil
import logging
import dataclasses

import dask
import distributed
import numpy as np
import xarray as xr
import zxarray as zr

from .__exceptions  import AbortForHelpException
from .__exceptions  import NoUserInputException
from .__climatology import Climatology
from .__climatology import CoVarConfig
from .__climatology import VarConfig

from typing import Any
from typing import Sequence


###############
## Variables ##
###############


@dataclasses.dataclass
class ANKParams:
    
    abort     : bool             = False
    error     : Exception | None = None
    help      : bool             = False
    version   : bool             = False
    log_level : str | int        = "WARNING"
    log_file  : str | None       = None
    verbose   : bool             = False
    debug     : bool             = False
    
    cmd : str  | None = None
    arg : list | None = None
    
    n_workers            : int = 1
    threads_per_worker   : int = 1
    memory_per_worker    : str = "auto"
    total_memory         : str = "auto"
    cluster              : str = "THREADING"
    
    tmp_base    : str | None         = None
    tmp_gen     : tempfile.TemporaryDirectory | None = None
    tmp         : str | None         = None
    tmp_gen_dask: tempfile.TemporaryDirectory | None = None
    tmp_dask    : str | None         = None
    tmp_gen_stan: tempfile.TemporaryDirectory | None = None
    tmp_stan    : str | None         = None
    
    covar_config : list | None = None
    config : str | None = None
    no_STAN: bool = False
    
    input  : list | None = None
    output : str  | None = None
    common_period     : list | None = None
    different_periods : list | None = None
    bias_period       : tuple[int,int] | str = "1961/1990"
    n_samples         : int = 10
    XN_version        : str = "CMIP6"
    names: list | None = None
    cname: str | None = None
    vname: str | None = None
    time: str = "1850/2014/2100"
    nslaw: str | None = None
    spatial: str | None = None

    clim      : Climatology | None = None
    load_clim :         str | None = None
    save_clim :         str | None = None
    set_seed  :         int | None = None
    
    def init_from_user_input( self , *argv: Any ) -> None:##{{{
        
        if len(argv) == 0:
            raise NoUserInputException("No arguments given, abort.\nRead the documentation with 'ank --help' ?")
        
        ## Parser for user input
        parser = argparse.ArgumentParser( add_help = False )
        parser.add_argument( "CMD" , nargs = '*' )
        
        parser.add_argument( "-h" , "--help" , action = "store_const" , const = True , default = False )
        parser.add_argument( "-V" , "--version" , action = "store_const" , const = True , default = False )
        parser.add_argument( "--log-level" , default = "WARNING" )
        parser.add_argument( "--log-file"  , default = None )
        parser.add_argument( "-v" , "--verbose" , action = "store_const" , const = True , default = False )
        parser.add_argument( "-d" , "--debug"   , action = "store_const" , const = True , default = False )
        
        parser.add_argument( "--tmp"                    , default = None )
        parser.add_argument( "--n-workers"              , default = 1 , type = int )
        parser.add_argument( "--threads-per-worker"     , default = 1 , type = int )
        parser.add_argument( "--memory-per-worker"      , default = "auto" )
        parser.add_argument( "--total-memory"           , default = "auto" )
        parser.add_argument( "--cluster"                , default = "PROCESS" )
        
        parser.add_argument( "--config" , default = None )
        parser.add_argument( "--covar-config" , nargs = "+" , action = "extend" )
        parser.add_argument( "--no-STAN" , action = "store_const" , const = True , default = False )
        
        parser.add_argument( "--input" , nargs = "+" , action = "extend" )
        parser.add_argument( "--output" , default = None )
        
        parser.add_argument( "--load-clim" , default = None )
        parser.add_argument( "--save-clim" , default = None )
        parser.add_argument( "--set-seed"  , default = None )
        
        parser.add_argument( "--common-period"     , default = None )
        parser.add_argument( "--different-periods" , default = None )
        parser.add_argument( "--bias-period"       , default = "1961/1990" )
        
        parser.add_argument( "--names" , nargs = "+" , action = "extend" )
        parser.add_argument( "--cname" , default = None )
        parser.add_argument( "--vname" , default = None )
        parser.add_argument( "--time"  , default = "1850/2014/2100" , type = str )
        parser.add_argument( "--nslaw"  , default = None )
        parser.add_argument( "--spatial"  , default = None )
        
        parser.add_argument( "--n-samples"         , default = 10 , type = int )
        parser.add_argument( "--XN-version"        , default = "CMIP6" , type = str )
        
        ## Transform in dict
        kwargs = vars(parser.parse_args(argv))
        
        ## And store in the class
        for key in kwargs:
            if key in ["CMD"]:
                try:
                    self.cmd = kwargs[key][0]
                    self.arg = kwargs[key][1:]
                except Exception:
                    pass
                continue
            
            if key not in self.__dict__:
                raise Exception(f"Parameter '{key}' not present in the class")
            self.__dict__[key] = kwargs[key]
            
        
    ##}}}
    
    def init_tmp(self) -> None:##{{{
        
        if self.tmp is None:
            self.tmp_base = tempfile.gettempdir()
        else:
            self.tmp_base     = self.tmp
        
        now               = str(dt.datetime.now(dt.UTC))[:19].replace("-","").replace(":","").replace(" ","-")
        prefix            = f"ANK_{now}_"
        self.tmp_gen      = tempfile.TemporaryDirectory( dir = self.tmp_base , prefix = prefix )
        self.tmp          = self.tmp_gen.name
        self.tmp_gen_dask = tempfile.TemporaryDirectory( dir = self.tmp_base , prefix = prefix + "DASK_" )
        self.tmp_dask     = self.tmp_gen_dask.name
        self.tmp_gen_stan = tempfile.TemporaryDirectory( dir = self.tmp_base , prefix = prefix + "STAN_" )
        self.tmp_stan     = self.tmp_gen_stan.name
    ##}}}
    
    def clean_tmp(self) -> None:##{{{
        self.tmp_gen.cleanup()
#        self.tmp_gen_dask.cleanup()
        self.tmp_gen_stan.cleanup()
    ##}}}
    
    def init_logging(self) -> None:##{{{
        
        if self.verbose:
            self.log_level = "INFO"
        if self.debug:
            self.log_level = "DEBUG"
        
        if isinstance( self.log_level , str ):
            self.log_level = getattr( logging , self.log_level.upper() , None )
        
        ## If it is not an integer, raise an error
        if not isinstance( self.log_level , int ): 
            raise ValueError( f"Invalid log level: {self.log_level}; nothing, an integer, 'debug', 'info', 'warning', 'error' or 'critical' expected" )
        
        ##
        log_kwargs = {
            "format" : '%(message)s',
            "level"  : self.log_level
            }
        
        if self.log_file is not None:
            log_kwargs["filename"] = self.log_file
        
        logging.basicConfig(**log_kwargs)
        logging.captureWarnings(True)
    ##}}}
    
    def init_dask(self) -> None:##{{{
        
        dask_config  = { "temporary_directory" : self.tmp_dask ,
                         "logging.distributed" : "error" }
        
        dask.config.set(**dask_config)
    ##}}}
    
    def init_GAM_config(self) -> None:##{{{
        ## Create GAM configuration, if given
        if "X_degree" not in self.config:
            self.config["X_degree"] = 3
        cnames = self.names
        if self.vname is not None:
            if self.vname in cnames:
                cnames.remove(self.vname)

        self.config["X_dof"] = xr.DataArray( 8. , dims = ["name","period"] , coords = [cnames,self.dpers] )
        if self.covar_config is not None:
            for f in self.covar_config:
                cname,dper,dof = f.split(":")
                self.config["X_dof"].loc[cname,dper] = int(dof)
    ##}}}

    def _init_clim_example(self) -> None:##{{{
        
        ## Open configuration
        cpath = os.path.dirname(os.path.abspath(__file__))
        epath = os.path.join( cpath , "data" , "EXAMPLE" )
        fconfig = toml.load( os.path.join( epath , "CONFIGURATION.toml" ) )
        cmd = self.arg[0].upper()
        if cmd not in fconfig:
            raise ValueError(f"Bad argument of the fit command ({cmd}), must be: {', '.join(list(fconfig))}")
        config = fconfig[cmd]
        if self.bias_period is None:
            self.bias_period = config["bper"]
        if self.common_period is None:
            self.common_period = [config["cper"]]
        if self.different_periods is None:
            self.different_periods = config["dpers"].split(",")
        if self.names is None:
            self.names = config["names"].split(" ")
        for v in ["time","cname","vname","nslaw","spatial"]:
            if getattr( self , v , None ) is None and v in config:
                setattr( self , v , config[v] )
    ##}}}

    def init_clim(self) -> None:##{{{
        
        ## Special case 1: load from file
        if self.load_clim is not None:
            self.clim = Climatology.init_from_file( self.load_clim )
            self.clim._tmp = self.tmp
            return
        
        ## Special case 2: this is the show command for XN, so no parameters
        ## are needed
        if self.cmd == "show":
            if len(self.arg) > 0 and self.arg[0] == "XN":
                return
        
        ## Special case 3: init from an example
        if self.cmd in ["example","sexample"]:
            self._init_clim_example()
        
        ## Global case: init from scratch
        self.init_GAM_config()
        self.clim = Climatology()
        
        ## Period
        self.clim.bper  = self.bias_period
        self.clim.cper  = self.common_period
        self.clim.dpers = self.different_periods
        
        ## Time axis
        t0,t1,t2 = [ int(s) for s in self.time.split("/") ]
        self.clim._time  = np.arange( t0     , t2 + 1 , 1 , dtype = int )
        self.clim._ctime = np.arange( t0     , t1 + 1 , 1 , dtype = int )
        self.clim._dtime = np.arange( t1 + 1 , t2 + 1 , 1 , dtype = int )
        
        ## Covariate configuration
        self.clim.cconfig = CoVarConfig( self.config.get("X_dof") , self.config.get("X_degree") )
        self.clim.vconfig = VarConfig( self.cname , self.vname , self.nslaw )

    ##}}}
    
    def get_cluster(self) -> distributed.deploy.local.LocalCluster:##{{{
        match self.cluster.upper():
            case "PROCESS":
                cluster = distributed.LocalCluster( n_workers  = self.n_workers , threads_per_worker = self.threads_per_worker , memory_limit = f"{self.memory_per_worker.B}B" , processes = True ) 
            case _:
                cluster = distributed.LocalCluster( n_workers  = self.n_workers , threads_per_worker = self.threads_per_worker , memory_limit = f"{self.memory_per_worker.B}B" , processes = False ) 
        
        return cluster
    ##}}}
    
    def stop_dask(self):##{{{
        pass
    ##}}}
    
    def check( self ) -> None: ##{{{
        
        try:
            if self.help or self.version:
                raise AbortForHelpException
            
            ## Check the CMD
            list_cmd = ["show","fit","draw","synthesize","constrain","attribute","misc","example","sexample"]
            if self.cmd is None or self.cmd.lower() not in list_cmd:
                raise Exception(f"Bad command arguments, must be one of {', '.join(list_cmd)}")
            
            ## Check and set the memory
            if self.memory_per_worker == "auto":
                if self.total_memory == "auto":
                    self.total_memory = zr.DMUnit( n = int(0.8 * psutil.virtual_memory().total) , unit = "B" )
                else:
                    self.total_memory = zr.DMUnit(self.total_memory)
                self.memory_per_worker = self.total_memory // self.n_workers
            else:
                self.memory_per_worker = zr.DMUnit(self.memory_per_worker)
                self.total_memory      = self.memory_per_worker * self.n_workers
            if self.cluster.upper() not in ["THREADING","PROCESS"]:
                raise ValueError(f"Cluster {self.cluster.upper()} is not supported" )
            
            ## Check time axis
            try:
                t0,t1,t2 = [ int(s) for s in self.time.split("/") ]
            except:
                raise ValueError("Invalid format for time axis")
            if not ( (t0 < t1) and (t1 < t2) ):
                raise ValueError("Invalid format for time axis")

            ## Change periods format
            self.common_period = [self.common_period]
            try:
                self.different_periods = self.different_periods.split(",")
            except:
                pass
            
            self.bias_period = tuple([int(s) for s in self.bias_period.split("/")])

            ## Others configurations
            if self.config is not None:
                self.config     = { c.split("=")[0] : c.split("=")[1] for c in self.config.split(",") }
            else:
                self.config = {}
            
        except Exception as e:
            self.abort = True
            self.error = e
        
        
    ##}}}
    
    def keys(self) -> str:##{{{
        keys = [key for key in self.__dict__]
        keys.sort()
        return keys
    ##}}}
    
    def __getitem__( self , key ) -> Any:##{{{
        return self.__dict__.get(key)
    ##}}}
    
    ## Properties ## {{{
    
    @property
    def cper(self) -> str:
        return self.common_period
    
    @property
    def dpers(self) -> Sequence[str]:
        return self.different_periods
    
    @property
    def n_jobs(self) -> int:
        return self.n_workers * self.threads_per_worker
    
    ##}}}

ankParams = ANKParams()


