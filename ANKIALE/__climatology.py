
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

## Packages
###########

import logging
import datetime as dt
import itertools as itt

import numpy as np
import xarray as xr
import netCDF4
import cftime

from typing import Sequence
from typing import Any
from typing import Self

import zxarray as zr

from .__release import version
from .__sys     import as_list
from .__sys     import coords_samples
from .__natural import get_XN
from .__exceptions import DevException
from .stats.__tools import nslawid_to_class
from .stats.models.__AbstractModel import AbstractModel
from .stats.__MultiGAM import MPeriodSmoother


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#############
## Classes ##
#############

class CoVarConfig:##{{{
    degree: int
    dof: dict[str,int]
    nknot: int
    vXN: str

    def __init__( self , dof: dict[str,int] , degree: int = 3 , vXN = "CMIP6" ):##{{{
        self.degree = degree
        self.dof    = dof
        self.vXN    = vXN
        self.find_nknot()
    ##}}}
    
    def copy(self) -> Self:##{{{
        return CoVarConfig( self.dof , self.degree , self.vXN )
    ##}}}

    def find_nknot(self) -> None:##{{{
        self.nknot  = 0
        for key in self.dof:
            self.nknot = max( self.nknot , self.dof[key] )
        self.nknot += self.degree + 1
    ##}}}
    
    ## Properties ##{{{

    @property
    def spl_config(self) -> dict:
        return { **self.dof , **{ 'degree' : self.degree , 'nknot' : self.nknot} }
    ##}}}    

##}}}

class VarConfig:##{{{
    _cname: str | None
    _vname: str | None
    _idnslaw: str | None
    cnslaw: AbstractModel | None = None
    
    def __init__( self , cname: str | None = None,
                         vname: str | None = None,
                       idnslaw: str | None = None ) -> None:
        self._cname = cname
        self._vname = vname
        self.idnslaw = idnslaw
    
    def copy(self) -> Self:
        return VarConfig( self._cname , self._vname , self.idnslaw )
    
    @property
    def idnslaw(self) -> str:
        return self._idnslaw
    
    @idnslaw.setter
    def idnslaw( self , value: str | None ) -> None:
        if value is not None:
            self._idnslaw = value
            self.cnslaw = nslawid_to_class(self.idnslaw)

    @property
    def is_init(self) -> bool:
        return self._cname is not None
    
    @property
    def cname(self) -> str:
        if self._cname is None:
            raise ValueError("Variable is not initialized, cname is not set")
        return self._cname
    
    @property
    def vname(self) -> str:
        if self._vname is None:
            raise ValueError("Variable is not initialized, vname is not set")
        return self._vname
    
    @property
    def vsize(self) -> int:
        if not self.is_init:
            return 0
        return self.cnslaw().nhpar
##}}}


class Climatology:##{{{
    
    ## Attributes ##{{{
    _names: Sequence[str]
    _cper: str
    _bper: tuple[int,int]
    _dpers = Sequence[str]
    
    _hpar : zr.ZXArray | None = None
    _hcov : zr.ZXArray | None = None
    _bias : dict[Any]
    _time : np.ndarray
    _XN   : xr.DataArray | None = None
    
    _spatial     = None
    
    cconfig: CoVarConfig
    vconfig: VarConfig = VarConfig()
    
    ##}}}

    ## Input / output ##{{{
    
    def __init__( self ) -> None:##{{{
        pass
    ##}}}
    
    def __str__(self) -> str:##{{{
        
        out = "<class ANKIALE.Climatology>"
        def not_defined( object_: object, attr: str , joiner = ", " , max_size = 60 ):
            try:
                out = getattr( object_ , attr )
                if isinstance( out , (list,tuple) ):
                    out = joiner.join( [str(o) for o in out] )
                out = str(out)
            except Exception:
                out = "Not defined"
            if len(out) > max_size:
                dsize = (max_size // 2) - 5
                out = out[:dsize] + " ... " + out[-dsize:]
            return out
        
        ## Build strings
        only_covar = not_defined( self , "only_covar" )
        cname = not_defined( self , "cname" )
        vname = not_defined( self , "vname" )
        names = not_defined( self , "names" )
        nslaw = not_defined( self , "idnslaw" )
        hpar  = not_defined( self , "hpar_names" )
        bper  = not_defined( self , "bper" , joiner = " / " )
        cper  = not_defined( self , "cper" )
        dpers = not_defined( self , "dpers" )
        time  = not_defined( self , "time" )
        mshape  = "Not defined" if self._hpar is None else str(self.hpar.shape)
        cshape  = "Not defined" if self._hcov is None else str(self.hcov.shape)
        spatial = "Not defined" if self._spatial is None else ", ".join(self._spatial)
        
        sns = [
               "only_covar",
               "names",
               "cname",
               "vname",
               "nslaw",
               "hyper_parameter",
               "bias_period",
               "common_period",
               "different_periods",
               "time",
               "hpar_shape",
               "hcov_shape",
               "spatial"
              ]
        ss  = [
               only_covar,
               names,
               cname,
               vname,
               nslaw,
               hpar,
               bper,
               cper,
               dpers,
               time,
               mshape,
               cshape,
               spatial
              ]
        
        ## Output
        for sn,s in zip(sns,ss):
            out = out + "\n" + " * " + "{:{fill}{align}{n}}".format( sn , fill = " " , align = "<" , n = 15 ) + ": " + s
        
        return out
    ##}}}
    
    def __repr__(self) -> str:##{{{
        return self.__str__()
    ##}}}
    
    def copy(self) -> Self:##{{{
        
        ##
        oclim = Climatology()
        
        ##
        oclim._names = self.names
        oclim._cper  = self.cper
        oclim._bper  = self.bper
        oclim._dpers = self.dpers
        
        oclim.hpar = self.hpar.copy()
        oclim.hcov = self.hcov.copy()
        oclim._bias = {}
        for key in self._bias:
            b = self._bias[key]
            if isinstance( b , (np.ndarray,xr.DataArray) ):
                oclim._bias[key] = b.copy()
            else:
                oclim._bias[key] = float(b)
        oclim.time = self.time
        
        oclim.cconfig = self.cconfig.copy()
        oclim.vconfig = self.vconfig.copy()
        
        if self._spatial is not None:
            oclim._spatial = {}
            for key in self._spatial:
                oclim._spatial[key] = self._spatial[key].copy()
        
        return oclim
    ##}}}
    
    ## staticmethod.init_from_file ##{{{
    @staticmethod
    def init_from_file( ifile: str ) -> Self:
        
        clim = Climatology()
        
        with netCDF4.Dataset( ifile , "r" ) as incf:
            
            if "ANK_version" in incf.ncattrs():
                incf_version = incf.getncattr("ANK_version")
            elif "BSAC_version" in incf.ncattrs():
                incf_version = incf.getncattr("BSAC_version")
            else:
                raise ValueError("Impossible to find the ANKIALE version of the file")
            
            if incf_version < "1.1.0":
                raise ValueError( f"Input file from ANKIALE / BSAC version {incf_version} < 1.1.0 can not be read by ANKIALE version {version} >= 1.1.0, abort." )

            clim._names     = incf.variables["names"][:].tolist()
            clim.cper       = incf.variables["common_period"][:].tolist()
            clim.dpers      = incf.variables["different_periods"][:].tolist()
            
            clim._bias = {}
            for name in clim.names:
                if len(incf.variables[f"bias_{name}"].shape) == 0:
                    clim._bias[name] = float(incf.variables[f"bias_{name}"][:])
                else:
                    clim._bias[name] = np.array(incf.variables[f"bias_{name}"][:])
                clim.bper = str(incf.variables[f"bias_{name}"].getncattr("period")).split("/")
            
            nctime    = incf.variables["time"]
            units     = nctime.getncattr( "units"    )
            calendar  = nctime.getncattr( "calendar" )
            clim.time = [ t.year for t in cftime.num2date( nctime[:] , units , calendar ) ]
            
            ## Y config
            try:
                cname   = str(incf.variables["Y"].getncattr("cname"))
                vname   = str(incf.variables["Y"].getncattr("vname"))
                idnslaw = str(incf.variables["Y"].getncattr("idnslaw"))
                clim.vconfig = VarConfig( cname = cname , vname = vname , idnslaw = idnslaw )
            except Exception:
                vname = ""
            
            ## X config
            vXN    = str(incf.variables["X"].getncattr("XN_version"))
            cnames = [ name for name in clim._names if not name == vname ]

            dof    = { f"{cname}_{per}" : int(incf.variables["X_dof"][icname,iper]) for (icname,cname),(iper,per) in itt.product(enumerate(cnames),enumerate(clim.dpers)) }
            degree = int(incf.variables["X_degree"][:])
            clim.cconfig = CoVarConfig( dof = dof , degree = degree , vXN = vXN )
            clim.cconfig.nknot = int(incf.variables["X_nknot"][:])
            
            ## And spatial
            spatial_is_fake = False
            try:
                spatial = str(incf.variables["Y"].getncattr("spatial"))
                if ":" in spatial:
                    spatial = spatial.split(":")
                else:
                    spatial_is_fake = (spatial == "fake")
                    spatial = [spatial]
                
                if not spatial_is_fake:
                    clim._spatial = { s : xr.DataArray( incf.variables[s][:] , dims = [s] , coords = [incf.variables[s][:]] ) for s in spatial }
                else:
                    clim._spatial = { "fake" : xr.DataArray( [0] , dims = ["fake"] , coords = [[0]] ) }
            except Exception:
                pass
            
            if clim._spatial is not None:
                for name in clim.names:
                    if isinstance(clim._bias[name],np.ndarray) or (name == clim.vname and spatial_is_fake):
                        clim._bias[name] = xr.DataArray( clim._bias[name] , dims = list(clim._spatial) , coords = clim._spatial )
            
            hpar = np.array(incf.variables["hpar"][:])
            hcov = np.array(incf.variables["hcov"][:])
            if spatial_is_fake:
                hpar = hpar.reshape( hpar.shape + (1,) )
                hcov = hcov.reshape( hcov.shape + (1,) )
            clim.hpar = hpar
            clim.hcov = hcov
            
        return clim
    ##}}}
    
    def save( self , ofile: str ) -> None:##{{{
        
        ##
        logger.info( f"Save clim in {ofile}" )
        with netCDF4.Dataset( ofile , "w" ) as ncf:
            
            ## Define dimensions
            logger.info(" * Define dimensions")
            ncdims = {
                   "hyper_parameter"   : ncf.createDimension(   "hyper_parameter" , self.hpar.shape[0] ),
                   "names"             : ncf.createDimension(             "names" , len(self.names) ),
                   "common_period"     : ncf.createDimension(     "common_period" , len(self.cper)  ),
                   "different_periods" : ncf.createDimension( "different_periods" , len(self.dpers) ),
                   "time"              : ncf.createDimension(              "time" , len(self.time)  ),
            }
            spatial = tuple()
            if self._spatial is not None and not self.spatial_is_fake:
                for d in self._spatial:
                    ncdims[d] = ncf.createDimension( d , self._spatial[d].size )
                spatial = tuple([d for d in self._spatial])
            logger.info( f"   => spatial: {spatial}" )
            
            ## Define variables
            logger.info(" * Define variables")
            ncvars = {
                   "hyper_parameter"   : ncf.createVariable(   "hyper_parameter" , str       ,          ("hyper_parameter",) ),
                   "names"             : ncf.createVariable(             "names" , str       ,                    ("names",) ),
                   "common_period"     : ncf.createVariable(     "common_period" , str       ,            ("common_period",) ),
                   "different_periods" : ncf.createVariable( "different_periods" , str       ,        ("different_periods",) ),
                   "time"              : ncf.createVariable(              "time" , "float32" ,                     ("time",) ),
                     "X_dof"           : ncf.createVariable(         "X_dof"     , "int32"   , ("names","different_periods") ),
                     "X_nknot"         : ncf.createVariable(         "X_nknot"   , "int32"                                   ),
                     "X_degree"        : ncf.createVariable(         "X_degree"  , "int32"                                   ),
                     "X"               : ncf.createVariable(                 "X" , "int32"                                   ),
                   "hpar"              : ncf.createVariable(              "hpar" , "float32" ,   ("hyper_parameter",) + spatial ),
                   "hcov"              : ncf.createVariable(              "hcov" , "float32" ,   ("hyper_parameter","hyper_parameter") + spatial ),
            }
            for name in self.names:
                B = self.bias[name]
                try:
                    B = float(B)
                except Exception:
                    pass
                if isinstance(B,float):
                    ncvars[f"bias_{name}"]    = ncf.createVariable( f"bias_{name}" , "float32" )
                    ncvars[f"bias_{name}"][:] = B
                else:
                    ncvars[f"bias_{name}"]    = ncf.createVariable( f"bias_{name}" , "float32" , spatial )
                    ncvars[f"bias_{name}"][:] = B[:]
                ncvars[f"bias_{name}"].setncattr( "period" , "{}/{}".format(*self.bper) )
            if self._spatial is not None and not self.spatial_is_fake:
                for d in self._spatial:
                    ncvars[d] = ncf.createVariable( d , "double" , (d,) )
                    ncvars[d][:] = np.array(self._spatial[d]).ravel()
            
            if not self.only_covar:
                ncvars["Y"] = ncf.createVariable( "Y" , "int32" )
            
            ## Fill variables of dimension
            logger.info(" * Fill variables of dimensions")
            ncvars[  "hyper_parameter"][:] = np.array( self.hpar_names , dtype = str )
            ncvars[            "names"][:] = np.array( self.names      , dtype = str )
            ncvars[    "common_period"][:] = np.array( self.cper       , dtype = str )
            ncvars["different_periods"][:] = np.array( self.dpers      , dtype = str )
            
            ## Fill time axis
            logger.info(" * Fill time axis")
            calendar = "standard"
            units    = "days since 1750-01-01 00:00"
            ncvars["time"][:]  = cftime.date2num( [cftime.DatetimeGregorian( int(y) , 1 , 1 ) for y in self.time] , units = units , calendar = calendar )
            ncvars["time"].setncattr( "standard_name" , "time"      )
            ncvars["time"].setncattr( "long_name"     , "time_axis" )
            ncvars["time"].setncattr( "units"         , units       )
            ncvars["time"].setncattr( "calendar"      , calendar    )
            ncvars["time"].setncattr( "axis"          , "T"         )
            
            ## Fill variables
            logger.info(" * Fill variables")
            if self.spatial_is_fake:
                idx1d = tuple([slice(None) for _ in range(self.hpar.ndim-1)]) + (0,)
                idx2d = tuple([slice(None) for _ in range(self.hcov.ndim-1)]) + (0,)
            else:
                idx1d = tuple([slice(None) for _ in range(self.hpar.ndim)])
                idx2d = tuple([slice(None) for _ in range(self.hcov.ndim)])
            ncvars["hpar"][:] = self.hpar._internal.zdata.get_orthogonal_selection(idx1d)
            ncvars["hcov"][:] = self.hcov._internal.zdata.get_orthogonal_selection(idx2d)
            
            ## Fill GAM basis
            logger.info( " * Fill GAM basis" )
            ncvars["X_nknot"][:]  = self.nknot
            ncvars["X_degree"][:] = self.degree
            ncvars["X_dof"][:] = 0
            for icname,cname in enumerate(self.cnames):
                for idper,dper in enumerate(self.dpers):
                    ncvars["X_dof"][icname,idper]    = self.dof[f"{cname}_{dper}"]
            
            ## Fill informations variables
            logger.info(" * Fill informations variables")
            ncvars["X"][:] = 1
            ncvars["X"].setncattr( "XN_version" , self.cconfig.vXN )
            
            if not self.only_covar:
                ncvars["Y"][:] = 1
                ncvars["Y"].setncattr( "idnslaw" , self.idnslaw )
                ncvars["Y"].setncattr( "cname"   , self.cname    )
                ncvars["Y"].setncattr( "vname"   , self.vname    )
                if self._spatial is not None:
                    if self.spatial_is_fake:
                        ncvars["Y"].setncattr( "spatial" , "fake" )
                    else:
                        ncvars["Y"].setncattr( "spatial" , ":".join(self._spatial) )
            
            ## Global attributes
            logger.info(" * Add global attributes")
            ncf.setncattr( "creation_date" , str(dt.datetime.now(dt.UTC))[:19].replace(" ","T") + "Z" )
            ncf.setncattr( "ANK_version"  , version )
    ##}}}
    
    ##}}}
    
    def restrict_spatial( self , coords: dict[str,xr.DataArray] , drop: bool = False ) -> Self:##{{{
        clim = Climatology()
        
        clim._hpar = self.hpar.zsel( drop = drop , **coords )
        clim._hcov = self.hcov.zsel( drop = drop , **coords )
        clim._bias = {}
        for key in self._bias:
            if isinstance( self._bias[key] , xr.DataArray ):
                clim._bias[key] = self._bias[key].sel( coords )
            else:
                clim._bias[key] = self._bias[key]
        clim._spatial = coords
        
        if clim._hpar.ndim > 1:
            if np.prod(clim._hpar.shape[1:]) == 1:
                fakes = xr.DataArray( [0] , dims = ["fake"] , coords = [[0]] )
                hpar_names = self.hpar_names
                nhpar = len(hpar_names)
                hpar  = zr.ZXArray( clim._hpar.dataarray.values.reshape(nhpar,1)       , dims = ["hpar",         "fake"] , coords = [hpar_names,fakes] )
                hcov  = zr.ZXArray( clim._hcov.dataarray.values.reshape(nhpar,nhpar,1) , dims = ["hpar0","hpar1","fake"] , coords = [hpar_names,hpar_names,fakes] )
                clim.hpar = hpar
                clim.hcov = hcov
                clim._spatial = { "fake" : fakes }
                if not self.only_covar:
                    clim._bias[self.vname] = xr.DataArray( [float(clim._bias[self.vname])] , dims = ["fake"] , coords = [fakes] )
        
        clim._names = self._names
        clim._cper  = self._cper
        clim._bper  = self._bper
        clim._dpers = self._dpers
        
        clim._time = self._time
        
        clim._cconfig = self._cconfig
        clim._vconfig = self._vconfig
        
        return clim
    ##}}}
    
    def restrict_dpers( self , periods: Sequence[str] ) -> Self:##{{{
        hpar_names = self._hpar.coords["hpar"].values.tolist()
        idx = []
        for ih,h in enumerate(hpar_names):
            if "X0" in h or "XN" in h:
                idx.append(ih)
            else:
                _,_,p = h.split("_")
                if p in periods:
                    idx.append(ih)
        
        self._hpar  = self._hpar.zisel( **{  "hpar" : idx } )
        self._hcov  = self._hcov.zisel( **{ "hpar0" : idx , "hpar1" : idx } )
        self._dpers = periods
        
        return self
    ##}}}
    
    def projection(self) -> tuple[xr.DataArray,xr.DataArray]:##{{{
        
        mps = MPeriodSmoother( self.XN , self.cnames , self.dpers , self.cconfig.spl_config , init_smoother = False )
        chpar_names = self.chpar_names
        
        ## Create projF, for X
        cprojF = xr.DataArray( mps.D0.reshape(self.ncnames,self.ndpers,self.time.size,-1),
                             dims = ["name","period","time","hpar"],
                             coords = [self.cnames,self.dpers,self.time,chpar_names]
                             )
        
        ## Add variable part
        if self.only_covar:
            projF = cprojF
        else:
            vhpar_names = self.vhpar_names
            vprojF = xr.DataArray( 0. ,
                                  dims = ["name","period","time","hpar"],
                                  coords = [self.cnames,self.dpers,self.time,vhpar_names]
                                  )
            projF = xr.concat( (cprojF,vprojF) , dim = "hpar" )
        
        ## Create projC from projF
        idx   = [h for h in chpar_names if not "X0" in h and not "XN" in h]
        projC = projF.copy()
        projC.loc[:,:,:,idx] = 0.

        return projF,projC
    ##}}}
    
    def crvs( self , size: int , add_BE: bool = False ) -> xr.Dataset:##{{{
        
        ## Extract parameters of the distribution
        if not self.only_covar:
            hpar = np.nanmean( self.hpar.dataarray.values , axis = tuple([i+1 for i in range(self.hpar.ndim-1)]) )
            hcov = np.nanmean( self.hcov.dataarray.values , axis = tuple([i+2 for i in range(self.hpar.ndim-1)]) )
        else:
            hpar = self.hpar.dataarray.values
            hcov = self.hcov.dataarray.values
        
        ## if add BE
        samples = coords_samples(size)
        if add_BE:
            size    = size + 1
            samples = ["BE"] + samples
        
        hpars = xr.DataArray( np.random.multivariate_normal( mean = hpar , cov = hcov , size = size ) , dims = ["sample","hpar"] , coords = [samples,self.hpar_names] )
        if add_BE:
            hpars[0,:] = hpar

        ## Build the design matrix
        projF,projC = self.projection()
        
        ## Compute covariates
        XF = projF @ hpars
        XC = projC @ hpars
        XA = XF - XC
        
        out = xr.Dataset( { "XF" : XF , "XC" : XC , "XA" : XA , "hpars" : hpars } )
        
        return out
    ##}}}
    
    ## Properties ##{{{
    
    ## Attributes ## {{{
    
    @property
    def time(self) -> np.ndarray:
        return self._time
    
    @time.setter
    def time( self , value: np.ndarray ) -> None:
        self._time = np.array(value).ravel()
    
    @property
    def bias(self) -> dict:
        return self._bias
    
    @property
    def hpar(self) -> zr.ZXArray:
        return self._hpar
    
    @hpar.setter
    def hpar( self , value: zr.ZXArray | np.ndarray ) -> None:
        if isinstance(value,zr.ZXArray):
            self._hpar = value
        else:
            zcoords = { **{ "hpar" : self.hpar_names } , **self.c_spatial }
            self._hpar = zr.ZXArray( data = value , coords = zcoords )
    
    @property
    def hcov(self) -> zr.ZXArray:
        return self._hcov
    
    @hcov.setter
    def hcov( self , value: zr.ZXArray | np.ndarray ) -> None:
        if isinstance(value,zr.ZXArray):
            self._hcov = value
        else:
            zcoords = { **{ "hpar0" : self.hpar_names , "hpar1" : self.hpar_names } , **self.c_spatial }
            self._hcov = zr.ZXArray( data = value , coords = zcoords )
    
    @property
    def names(self) -> Sequence[str]:
        return self._names
    
    @property
    def cnames(self) -> Sequence[str]:
        return self.names if self.only_covar else [cname for cname in self.names if not cname == self.vconfig.vname]
    
    @property
    def ncnames(self) -> int:
        return len(self.cnames)
    
    @property
    def csize(self) -> int:
        return self.nknot * self.ndpers + 2
    
    @property
    def vsize(self) -> int:
        return self.vconfig.vsize
    
    @property
    def size(self) -> int:
        return self.csize * self.ncnames + self.vsize
    
    @property
    def chpar_names(self) -> Sequence[str]:
        _hpar_names = []
        for _cname in self.cnames:
            _hpar_names.extend( [f"X0_{_cname}",f"XN_{_cname}"] )
            for _per in self.dpers:
                _hpar_names.extend(
                    [f"XS{i}_{_cname}_{_per}" for i in range(self.nknot)]
                )
        return _hpar_names
    
    @property
    def vhpar_names(self) -> Sequence[str]:
        if self.only_covar:
            return []
        return list(self.vconfig.cnslaw().h_name)

    @property
    def hpar_names(self) -> Sequence[str]:
        return self.chpar_names + self.vhpar_names

    ##}}}
    
    ## About spatial ##{{{
    
    @property
    def has_spatial(self) -> bool:
        return self._spatial is not None
    
    @property
    def spatial_is_fake(self) -> bool:
        try:
            d = self.d_spatial[0]
            test_ndim = len(self.d_spatial) == 1
            test_sdim = len(self.c_spatial[d]) == 1
            test_name = (d == "fake")
            test = test_ndim and test_sdim and test_name
        except Exception:
            test = False
        return test
    
    @property
    def d_spatial(self) -> Sequence[str]:
        if self._spatial is None:
            return ()
        else:
            return tuple(list(self._spatial))
    
    @property
    def c_spatial(self) -> dict:
        if self._spatial is None:
            return {}
        else:
            return self._spatial
    
    @property
    def s_spatial(self) -> Sequence[int]:
        if self._spatial is None:
            return ()
        else:
            return tuple([self._spatial[d].size for d in self.d_spatial])

    ##}}}
    
    ## About periods ##{{{
    
    @property
    def cper(self) -> str:
        return self._cper
    
    @property
    def common_period(self) -> str:
        return self._cper
    
    @cper.setter
    def cper( self , value: str ) -> None:
        self._cper = value
    
    @property
    def bper(self) -> tuple[int,int]:
        return self._bper
    
    @property
    def bias_period(self) -> tuple[int,int]:
        return self._bper
    
    @bper.setter
    def bper( self , value: Sequence[str|int] ) -> None:
        self._bper = tuple([int(v) for v in value])
        if not len(self._bper) == 2:
            raise ValueError("Invalid bias period")
    
    
    @property
    def different_periods(self) -> Sequence[str]:
        return self._dpers
    
    @property
    def dpers(self) -> Sequence[str]:
        return self._dpers
    
    @dpers.setter
    def dpers( self , value: Sequence[str] ):
        self._dpers = as_list(value)
    
    @property
    def ndpers( self ) -> int:
        return len(self.dpers)

    ##}}}
    
    ## About covariates X ##{{{
    
    @property
    def XN(self) -> xr.DataArray:
        if self._XN is None:
            self._XN = get_XN( time = self.time , version = self.cconfig.vXN )
        return self._XN
    
    @property
    def only_covar(self) -> bool:
        return not self.vconfig.is_init
    
    @property
    def has_var(self) -> bool:
        return not self.only_covar

    @property
    def dof(self) -> dict:
        return self.cconfig.dof
    
    @property
    def nknot(self) -> int:
        return self.cconfig.nknot
    
    @property
    def degree(self) -> int:
        return self.cconfig.degree

    ##}}}
    
    ## About variable Y ## {{{
    
    @property
    def idnslaw(self) -> str:
        return self.vconfig.idnslaw
    
    @property
    def cnslaw(self) -> AbstractModel:
        return self.vconfig.cnslaw

    @property
    def cname(self) -> str:
        return self.vconfig.cname
    
    @property
    def vname(self) -> str:
        return self.vconfig.vname

    ##}}}
    
    ##}}}
    
##}}}


