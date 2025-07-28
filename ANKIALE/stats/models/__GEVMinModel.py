
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

import numpy as np
import xarray as xr
import scipy.stats as sc
import SDFC  as sd

from typing import Any

from .__AbstractModel import ValueType
from .__AbstractModel import AbstractModel


## Classes
##########

class GEVMinModel(AbstractModel):##{{{
    
    def __init__( self ) -> None:##{{{
        
        AbstractModel.__init__( self ,
                                p_name    = ("loc","scale","shape"),
                                h_name    = ("loc0","loc1","scale0","scale1","shape0"),
                                sdlaw     = sd.GEV,
                                sclaw     = sc.genextreme,
                                stan_file = "STAN_GEVMODEL_PRIOR-NORMAL.stan"
                                )
        
    ##}}}
    
    def __repr__(self) -> str:##{{{
        return self.__str__()
    ##}}}
    
    def __str__(self) -> str:##{{{
        return "ANKIALE.stats.GEVMinModel"
    ##}}}
    
    def _map_sdfit( self , Y: np.ndarray , X: np.ndarray ) -> tuple[np.ndarray,dict[str,Any]]:##{{{
        return (-Y,),{ "c_loc" : -X , "c_scale" : -X , "l_scale" : sd.link.ULExponential() }
    ##}}}
    
    def _map_scpar( self , **kwargs: dict[str,np.ndarray | xr.DataArray] ) -> dict[str,np.ndarray | xr.DataArray]:##{{{
        return { "loc" : kwargs["loc"] , "scale" : kwargs["scale"] , "c" : - kwargs["shape"] }
    ##}}}
    
    def _map_stanpar( self , Y: np.ndarray , X: np.ndarray ) -> tuple[np.ndarray,np.ndarray]:##{{{
        return -Y,-X
    ##}}}
    
    def draw_params( self , X: xr.DataArray , hpar: xr.DataArray ) -> dict[str,xr.DataArray]:##{{{
        
        loc   = hpar.sel( hpar = "loc0" ) - hpar.sel( hpar = "loc1" ) * X
        scale = np.exp( hpar.sel( hpar = "scale0" ) - hpar.sel( hpar = "scale1" ) * X )
        shape = hpar.sel( hpar = "shape0" ) - 0 * X
        
        return { "loc" : loc , "scale" : scale , "shape" : shape }
    ##}}}
    
    def _cdf_sf( self , x: ValueType , side: str , **kwargs: Any ) -> ValueType:##{{{
        
        sckwargs = self._map_scpar(**kwargs)
        
        if side == "right":
            return self.sclaw.cdf( -x , **sckwargs )
        else:
            return self.sclaw.sf( -x , **sckwargs )
    ##}}}
    
    def _icdf_sf( self , p: ValueType , side: str , **kwargs: Any ) -> ValueType:##{{{
        
        sckwargs = self._map_scpar(**kwargs)
        
        if side == "right":
            return -self.sclaw.ppf( p , **sckwargs )
        else:
            return -self.sclaw.isf( p , **sckwargs )
    ##}}}
    
##}}}

