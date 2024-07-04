
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

## Packages
###########

import warnings
import numpy as np
import SDFC  as sd

import scipy.stats as sc


## Classes
##########

class AbstractModel:##{{{
	
	def __init__( self , n_coef ):##{{{
		self.sd     = None
		self.law    = None
		self.n_coef = n_coef
		self.coef_  = None
		self.coef_kind = []
		self.coef_name = []
	##}}}
	
	## Properties ##{{{
	
	##}}}
	
	def cdf_sf( self , x , side , **kwargs ):##{{{
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			return self._cdf_sf( x , side , **kwargs )
	##}}}
	
	def icdf_sf( self , p , side , **kwargs ):##{{{
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			return self._icdf_sf( p , side , **kwargs )
	##}}}
	
##}}}


