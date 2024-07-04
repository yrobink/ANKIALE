
## Copyright(c) 2023 / 2024 Yoann Robin
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
import warnings
import numpy as np
import SDFC  as sd

import scipy.stats as sc

from .__AbstractModel import AbstractModel

import cmdstanpy as stan

import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True



## Classes
##########

class GEVModel(AbstractModel):##{{{
	
	def __init__( self ):##{{{
		
		AbstractModel.__init__( self , 5 )
		self.sd = sd.GEV
		self.coef_kind = ["loc","scale","shape"]
		self.coef_name = ["loc0","loc1","scale0","scale1","shape0"]
		
	##}}}
	
	def __repr__(self):##{{{
		return self.__str__()
	##}}}
	
	def __str__(self):##{{{
		return "BSAC.stats.GEVModel"
	##}}}
	
	def fit_mle( self , Y , X , **kwargs ):##{{{
		self.law = self.sd( method = "mle" )
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.law.fit( Y , c_loc = X , c_scale = X , l_scale = sd.link.ULExponential() , **kwargs )
		self.coef_ = self.law.coef_
	##}}}
	
	## staticmethod@init_stan ##{{{
	
	@staticmethod
	def init_stan( force_compile = False ):
		### Define stan model
		stan_path  = os.path.join( os.path.dirname(os.path.abspath(__file__)) , ".." , ".." , "data" )
		stan_ifile = os.path.join( stan_path , "GEVModel.stan" )
		stan_exec  = os.path.join( stan_path , "GEVModel.exec" )
		stan_model = stan.CmdStanModel( stan_file = stan_ifile , force_compile = force_compile , stanc_options = { "O" : 3 } , cpp_options = { "O" : 3 } )
		
		return stan_model
	##}}}
	
	def fit_bayesian( self , Y , X , prior , n_mcmc_drawn ):##{{{
		
		## Load stan model
		stan_model = self.init_stan()
		
		## Fit the model
		idata  = {
			"nhpar" : 5,
			"prior_hpar" : prior.mean,
			"prior_hcov" : prior.cov,
			"nXY"        : Y.size,
			"X"          : X,
			"Y"          : Y,
		}
		fit = stan_model.sample( data = idata , chains = 1 , iter_sampling = n_mcmc_drawn , parallel_chains = 1 , threads_per_chain = 1 , show_progress = False )
		
		## Draw samples
		draw = fit.draws_xr("hpar")["hpar"].rename( hpar_dim_0 = "hpar" ).assign_coords( hpar = self.coef_name )
		return draw[0,:,:].values
#		###
#		self.law = self.sd( method = "bayesian" )
#		with warnings.catch_warnings():
#			warnings.simplefilter("ignore")
#			self.law.fit( Y , c_loc = X , c_scale = X , l_scale = sd.link.ULExponential() , prior = prior , n_mcmc_drawn = n_mcmc_drawn )
#		self.coef_ = self.law.info_.draw[-1,:]
#		return self.law.info_.draw
	##}}}
	
	def draw_params( self , X , coef ):##{{{
		
		loc   = coef.loc[:,"loc0"] + coef.loc[:,"loc1"] * X
		scale = np.exp( coef.loc[:,"scale0"] + coef.loc[:,"scale1"] * X )
		shape = coef.loc[:,"shape0"] + 0 * X
		
		return { "loc" : loc , "scale" : scale , "shape" : shape }
	##}}}
	
	def _cdf_sf( self , x , side , **kwargs ):##{{{
		
		sckwargs = { "loc" : kwargs["loc"] , "scale" : kwargs["scale"] , "c" : - kwargs["shape"] }
		
		if side == "right":
			return sc.genextreme.sf( x , **sckwargs )
		else:
			return sc.genextreme.cdf( x , **sckwargs )
	##}}}
	
	def _icdf_sf( self , p , side , **kwargs ):##{{{
		
		sckwargs = { "loc" : kwargs["loc"] , "scale" : kwargs["scale"] , "c" : - kwargs["shape"] }
		if side == "right":
			return sc.genextreme.isf( p , **sckwargs )
		else:
			return sc.genextreme.ppf( p , **sckwargs )
	##}}}
	
##}}}

