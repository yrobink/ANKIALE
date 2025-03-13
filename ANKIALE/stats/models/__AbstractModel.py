
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

import os
import warnings
import tempfile

import numpy as np

import cmdstanpy as stan
import logging

from ...__sys import copy_files
from ...__linalg import sqrtm

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

## Classes
##########

class StanError(Exception):
	def __init__( self , *args , **kwargs ):
		super().__init__( *args , **kwargs )

class AbstractModel:##{{{
	
	def __init__( self , p_name , h_name , sdlaw , sclaw , stan_file ):##{{{
		self.sdlaw     = sdlaw
		self.sclaw     = sclaw
		self.p_name    = p_name
		self.h_name    = h_name
		self.stan_file = stan_file
	##}}}
	
	## Properties ##{{{
	
	@property
	def npar(self):
		return len(self.p_name)
	
	@property
	def nhpar(self):
		return len(self.h_name)
	
	##}}}
	
	def _map_sdfit( self , Y , X ):##{{{
		raise NotImplementedError
	##}}}
	
	def _map_scpar( self , **kwargs ):##{{{
		raise NotImplementedError
	##}}}
	
	def _map_stanpar( self , Y , X ):##{{{
		return Y,X
	##}}}
	
	def fit_mle( self , Y , X , **kwargs ):##{{{
		law = self.sdlaw( method = "mle" )
		sdargs,sdkwargs = self._map_sdfit( Y , X )
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			law.fit( *sdargs , **{ **sdkwargs , **kwargs } )
		return law.coef_
	##}}}
	
	def init_stan( self , tmp , force_compile = False ):##{{{
		### Define stan model
		stan_path  = os.path.join( os.path.dirname(os.path.abspath(__file__)) , ".." , ".." , "data" )
		stan_ifile = os.path.join( stan_path , self.stan_file )
		stan_ofile = os.path.join(       tmp , self.stan_file )
		if not os.path.isfile(stan_ofile):
			copy_files( stan_ifile , stan_ofile )
		stan_model = stan.CmdStanModel( stan_file = stan_ofile , force_compile = force_compile )
		
		return stan_model
	##}}}
	
	def _fit_bayesian_ORIGIN( self , Y , X , prior , n_mcmc_drawn , n_try = 10 ):##{{{
		
		sdargs,sdkwargs = self._map_sdfit( Y , X )
		for _ in range(n_try):
			
			law = self.sdlaw( method = "bayesian" )
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				law.fit( *sdargs , prior = prior , n_mcmc_drawn = 20 * n_mcmc_drawn , burn = 1000 , **sdkwargs )
			rate = law.info_.rate_accept
			draw = law.info_.draw[law.info_.accept,:][::5,:][-n_mcmc_drawn:,:]
			
			if draw.shape[0] < n_mcmc_drawn or rate < 0.3:
				success = False
				continue
			else:
				success = True
				break
		
		if not success:
			draw = np.zeros( (n_mcmc_drawn,prior.mean.size) ) + np.nan
		return draw
	##}}}
	
	def _fit_bayesian_STAN( self , Y , X , prior , n_mcmc_drawn , tmp , n_try = 10 ):##{{{
		YY,XX = self._map_stanpar(Y,X)
		show_console = False
		for n in range(n_try):
			try:
#				if n > 0:
#					print( f"ntry:: {n}" )
#					show_console = True
				## Load stan model
				stan_model = self.init_stan( tmp )
				
				## Fit the model
				idata  = {
					"nhpar" : prior.mean.size,
					"prior_hpar" : prior.mean,
					"prior_hcov" : prior.cov,
					"prior_hstd" : sqrtm(prior.cov),
					"nXY"        : YY.size,
					"X"          : XX,
					"Y"          : YY,
				}
				with tempfile.TemporaryDirectory( dir = tmp ) as tmp_draw:
					try:
						fit  = stan_model.sample( data = idata , chains = 1 , iter_sampling = n_mcmc_drawn , output_dir = tmp_draw , parallel_chains = 1 , threads_per_chain = 1 , show_progress = False , show_console = show_console )
						draw = fit.draws_xr("hpar")["hpar"][0,:,:].values
					except Exception:
						raise StanError
				
				success = True
				break
			except StanError:
				success = False
		if not success:
			draw = self._fit_bayesian_ORIGIN( Y , X , prior , n_mcmc_drawn , n_try )
		
		return draw
	##}}}
	
	def fit_bayesian( self , Y , X , prior , n_mcmc_drawn , use_STAN , tmp , n_try = 10 ):##{{{
		if use_STAN:
			draw = self._fit_bayesian_STAN( Y , X , prior , n_mcmc_drawn , tmp , n_try )
		else:
			draw = self._fit_bayesian_ORIGIN( Y , X , prior , n_mcmc_drawn , n_try )
		
		return draw
	##}}}
	
	def _cdf_sf( self , x , side , **kwargs ):##{{{
		
		sckwargs = self._map_scpar(**kwargs)
		
		if side == "right":
			return self.sclaw.sf( x , **sckwargs )
		else:
			return self.sclaw.cdf( x , **sckwargs )
	##}}}
	
	def _icdf_sf( self , p , side , **kwargs ):##{{{
		
		sckwargs = self._map_scpar(**kwargs)
		
		if side == "right":
			return self.sclaw.isf( p , **sckwargs )
		else:
			return self.sclaw.ppf( p , **sckwargs )
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


