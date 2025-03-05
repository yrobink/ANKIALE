
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

## Packages
###########

import os
import numpy as np
import scipy.stats as sc
import SDFC  as sd

from .__AbstractModel import AbstractModel


## Classes
##########

class GEVMinModel(AbstractModel):##{{{
	
	def __init__( self ):##{{{
		
		AbstractModel.__init__( self ,
		                        p_name    = ("loc","scale","shape"),
		                        h_name    = ("loc0","loc1","scale0","scale1","shape0"),
		                        sdlaw     = sd.GEV,
		                        sclaw     = sc.genextreme,
		                        stan_file = "STAN_GEVMODEL_PRIOR-NORMAL.stan"
		                        )
		
	##}}}
	
	def __repr__(self):##{{{
		return self.__str__()
	##}}}
	
	def __str__(self):##{{{
		return "BSAC.stats.GEVMinModel"
	##}}}
	
	def _map_sdfit( self , Y , X ):##{{{
		return (-Y,),{ "c_loc" : -X , "c_scale" : -X , "l_scale" : sd.link.ULExponential() }
	##}}}
	
	def _map_scpar( self , **kwargs ):##{{{
		return { "loc" : kwargs["loc"] , "scale" : kwargs["scale"] , "c" : - kwargs["shape"] }
	##}}}
	
	def _map_stanpar( self , Y , X ):##{{{
		return -Y,-X
	##}}}
	
	def draw_params( self , X , hpar ):##{{{
		
		loc   = hpar.sel( hpar = "loc0" ) - hpar.sel( hpar = "loc1" ) * X
		scale = np.exp( hpar.sel( hpar = "scale0" ) - hpar.sel( hpar = "scale1" ) * X )
		shape = hpar.sel( hpar = "shape0" ) - 0 * X
		
		return { "loc" : loc , "scale" : scale , "shape" : shape }
	##}}}
	
	def _cdf_sf( self , x , side , **kwargs ):##{{{
		
		sckwargs = self._map_scpar(**kwargs)
		
		if side == "right":
			return self.sclaw.cdf( -x , **sckwargs )
		else:
			return self.sclaw.sf( -x , **sckwargs )
	##}}}
	
	def _icdf_sf( self , p , side , **kwargs ):##{{{
		
		sckwargs = self._map_scpar(**kwargs)
		
		if side == "right":
			return -self.sclaw.ppf( p , **sckwargs )
		else:
			return -self.sclaw.isf( p , **sckwargs )
	##}}}
	
##}}}

