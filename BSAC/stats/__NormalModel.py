
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

from .__AbstractModel import AbstractModel


## Classes
##########

class NormalModel(AbstractModel):##{{{
	
	def __init__( self ):##{{{
		
		AbstractModel.__init__( self , 4 )
		self.sd = sd.Normal
		self.coef_kind = ["loc","scale"]
		self.coef_name = ["loc0","loc1","scale0","scale1"]
		
	##}}}
	
	def fit_mle( self , Y , X , **kwargs ):##{{{
		self.law = self.sd( method = "mle" , **kwargs )
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.law.fit( Y , c_loc = X , c_scale = X , l_scale = sd.link.ULExponential() )
		self.coef_ = self.law.coef_
	##}}}
	
	def fit_bayesian( self , Y , X , prior , n_mcmc_drawn ):##{{{
		self.law = self.sd( method = "bayesian" )
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.law.fit( Y , c_loc = X , c_scale = X , l_scale = sd.link.ULExponential() , prior = prior , n_mcmc_drawn = n_mcmc_drawn )
		self.coef_ = self.law.info_.draw[-1,:]
	##}}}
	
	def draw_params( self , X , coef ):##{{{
		
		loc   = coef.loc[:,"loc0"] + coef.loc[:,"loc1"] * X
		scale = np.exp( coef.loc[:,"scale0"] + coef.loc[:,"scale1"] * X )
		
		return { "loc" : loc , "scale" : scale }
	##}}}
	
	def cdf_sf( self , x , side , **kwargs ):##{{{
		
		if side == "right":
			return sc.norm.sf( x , **kwargs )
		else:
			return sc.norm.cdf( x , **kwargs )
	##}}}
	
	def icdf_sf( self , p , side , **kwargs ):##{{{
		
		if side == "right":
			return sc.norm.isf( p , **kwargs )
		else:
			return sc.norm.icdf( p , **kwargs )
	##}}}
	
##}}}


