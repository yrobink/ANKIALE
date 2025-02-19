
## Copyright(c) 2024, 2025 Yoann Robin
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

##############
## Packages ##
##############


#############
## Imports ##
#############

import logging
from ..__logs import LINE
from ..__logs import log_start_end

import numpy as np
import scipy.stats as sc
import xarray as xr

from .__KCC import KCC
from .__KCC import MAR2


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#############
## Classes ##
#############


###############
## Functions ##
###############

def gaussian_conditionning_independent( *args , A = None , timeXo = None ):##{{{
	
	## Extract arguments
	hpar = args[0]
	hcov = args[1]
	lXo  = args[2:]
	gXo  = np.concatenate( args[2:] , axis = 0 )
	
	## Variance of obs
	R      = gXo - A @ hpar
	hcov_o = []
	i      = 0
	for Xo in lXo:
		s = Xo.size
		hcov_o.append( np.ones(s) * float(np.std(R[i:(i+s)]))**2 )
		i += s
	hcov_o = np.diag( np.hstack(hcov_o) )
	
	## Application
	K0 = A @ hcov
	K1 = ( hcov @ A.T ) @ np.linalg.inv( K0 @ A.T + hcov_o )
	hpar = hpar + K1 @ ( gXo.squeeze() - A @ hpar )
	hcov = hcov - K1 @ K0
	
	return hpar,hcov
##}}}

def _gaussian_conditionning_kcc_2covariates( *args , A = None , timeXo = None ):##{{{
	
	## Extract arguments
	hpar = args[0]
	hcov = args[1]
	lXo  = args[2:]
	gXo  = np.concatenate( args[2:] , axis = 0 )
	
	## Variance of obs
	R      = gXo - A @ hpar
	size0  = lXo[0].size
	size1  = lXo[1].size
	RXo0   = xr.DataArray( R[:size0] , dims = ["time"] , coords = [timeXo[0].values] )
	RXo1   = xr.DataArray( R[size0:] , dims = ["time"] , coords = [timeXo[1].values] )
	
	hcov_o_meas0 = RXo0.values.reshape(-1,1) @ RXo0.values.reshape(1,-1)
	hcov_o_meas1 = RXo1.values.reshape(-1,1) @ RXo1.values.reshape(1,-1)
	kcc          = KCC().fit( RXo0 , RXo1 )
	hcov_o_iv0   = kcc.cov_iv0
	hcov_o_iv1   = kcc.cov_iv1
	hcov_o_iv01  = kcc.cov_iv01
	hcov_o       = np.block( [ [hcov_o_meas0 + hcov_o_iv0 , hcov_o_iv01  ],
	                           [hcov_o_iv01.T , hcov_o_meas1 + hcov_o_iv1] ] )
	
	## Application
	K0 = A @ hcov
	K1 = ( hcov @ A.T ) @ np.linalg.inv( K0 @ A.T + hcov_o )
	hpar = hpar + K1 @ ( gXo.squeeze() - A @ hpar )
	hcov = hcov - K1 @ K0
	
	hcov_iv = np.block( [ [hcov_o_iv0    , hcov_o_iv01 ],
	                      [hcov_o_iv01.T , hcov_o_iv1] ] )
	
	return hpar,hcov,hcov_iv
##}}}

def _gaussian_conditionning_kcc_1covariate( *args , A = None , timeXo = None ):##{{{
	
	## Extract arguments
	hpar = args[0]
	hcov = args[1]
	gXo  = args[2]
	
	## Variance of obs
	R      = gXo - A @ hpar
	size0  = gXo.size
	RXo0   = xr.DataArray( R , dims = ["time"] , coords = [timeXo[0].values] )
	
	hcov_o_meas0 = RXo0.values.reshape(-1,1) @ RXo0.values.reshape(1,-1)
	mar2         = MAR2().fit( RXo0.values )
	hcov_o_iv0   = mar2._covariance_matrix()
	hcov_o       = hcov_o_meas0 + hcov_o_iv0
	
	## Application
	K0 = A @ hcov
	K1 = ( hcov @ A.T ) @ np.linalg.inv( K0 @ A.T + hcov_o )
	hpar = hpar + K1 @ ( gXo.squeeze() - A @ hpar )
	hcov = hcov - K1 @ K0
	
	return hpar,hcov,hcov_o_iv0
##}}}

def gaussian_conditionning_kcc( *args , A = None , timeXo = None ):##{{{
	
	args   = list(args)
	hpar   = args[0]
	
	if len(args[2:]) == 1:
		_gaussian_conditionning_kcc = _gaussian_conditionning_kcc_1covariate
	else:
		_gaussian_conditionning_kcc = _gaussian_conditionning_kcc_2covariates
	
	norm_p = 1e9
	for i in range(10):
		args[0] = hpar
		hpar,hcov,hcov_iv = _gaussian_conditionning_kcc( *args , A = A , timeXo = timeXo )
		norm_c = np.linalg.norm(hcov_iv)
		if np.abs( (norm_c - norm_p) / norm_p ) < 1e-2 and i > 0:
			break
		norm_p = norm_c
	
	return hpar,hcov
##}}}


def mcmc( hpar , hcov , Y , A , size_chain , nslaw_class , use_STAN , tmp_stan = None ):##{{{
	
	## Law
	nslaw   = nslaw_class()
	nnshpar = nslaw.nhpar
	
	## Prior
	prior_hpar = hpar[-nnshpar:]
	prior_hcov = hcov[-nnshpar:,:][:,-nnshpar:]
	prior      = sc.multivariate_normal( mean = prior_hpar , cov = prior_hcov , allow_singular = True )
	
	## Output
	hpars = np.zeros( hpar.shape + (size_chain,) ) + np.nan
	
	##
	chain_is_valid = False
	while not chain_is_valid:
		## Draw covariate parameters
		hpars[:] = np.random.multivariate_normal( mean = hpar , cov = hcov , size = 1 ).reshape(-1,1)
		
		## Build the covariable
		X = A @ hpars[:,0]
		
		## Apply constraint
		draw = nslaw.fit_bayesian( Y , X , prior , size_chain , use_STAN = use_STAN , tmp = tmp_stan )
		
		## Store
		hpars[-nnshpar:,:] = draw.T
		
		##
		chain_is_valid = np.isfinite(draw).all()
	
	return hpars
##}}}

