
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

def _gaussian_conditionning_kcc_2covariates( *args , A = None , timeXo = None , dep = 1 ):##{{{
	
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
	
	## Find observed variability
	hcov_o_meas0 = 0
	hcov_o_meas1 = 0
	kcc          = KCC().fit( RXo0 , RXo1 )
	hcov_o_iv0   = kcc.cov_iv0
	hcov_o_iv1   = kcc.cov_iv1
	hcov_o_iv01  = kcc.cov_iv01 * dep
	hcov_o       = np.block( [ [hcov_o_meas0 + hcov_o_iv0 , hcov_o_iv01  ],
	                           [hcov_o_iv01.T , hcov_o_meas1 + hcov_o_iv1] ] )
	
	## Application
	K0 = A @ hcov
	K1 = ( hcov @ A.T ) @ np.linalg.inv( K0 @ A.T + hcov_o )
	hpar = hpar + K1 @ ( gXo.squeeze() - A @ hpar )
	hcov = hcov - K1 @ K0
	
	return hpar,hcov,hcov_o
##}}}

def _gaussian_conditionning_kcc_1covariate( *args , A = None , timeXo = None , dep = 1 ):##{{{
	
	## Extract arguments
	hpar = args[0]
	hcov = args[1]
	gXo  = args[2]
	
	## Variance of obs
	R      = gXo - A @ hpar
	size0  = gXo.size
	RXo0   = xr.DataArray( R , dims = ["time"] , coords = [timeXo[0].values] )
	
	mar2   = MAR2().fit( RXo0.values )
	hcov_o = mar2._covariance_matrix()
	
	## Application
	K0 = A @ hcov
	K1 = ( hcov @ A.T ) @ np.linalg.inv( K0 @ A.T + hcov_o )
	hpar = hpar + K1 @ ( gXo.squeeze() - A @ hpar )
	hcov = hcov - K1 @ K0
	
	return hpar,hcov,hcov_o
##}}}

def gaussian_conditionning_KCC( *args , A = None , timeXo = None , dep = 1 ):##{{{
	
	args   = list(args)
	hpar   = args[0]
	hcov   = args[1]
	
	if len(args[2:]) == 1:
		_gaussian_conditionning_kcc = _gaussian_conditionning_kcc_1covariate
	else:
		_gaussian_conditionning_kcc = _gaussian_conditionning_kcc_2covariates
	
	hcov_o_p = 1e9
	hpar_p   = 1e9
	hcov_p   = 1e9
	err_p    = 1e9
	err      = 1e9
	rerr     = 1e9
	for i in range(10):
		args[0] = hpar
		args[1] = hcov
		hpar,hcov,hcov_o = _gaussian_conditionning_kcc( *args , A = A , timeXo = timeXo , dep = dep )
		
		err  = np.linalg.norm( hcov_o_p - hcov_o )
		rerr = err / np.linalg.norm(hcov_o_p)
		logger.debug( f"KCC error: {err}, rerr: {rerr}" )
		
		if err > err_p and i > 0:
			hpar = hpar_p
			hcov = hcov_p
			break
		if rerr < 1 and i > 0:
			break
		
		hcov_o_p = hcov_o.copy()
		hpar_p   = hpar.copy()
		hcov_p   = hcov.copy()
		err_p    = err
	
	logger.info( f"Numbers of KCC iterations: {i}" )
	
	return hpar,hcov
##}}}

def gaussian_conditionning_independent( *args , A = None , timeXo = None ):##{{{
	
	## Extract arguments
	hpar = args[0]
	hcov = args[1]
	lXo  = args[2:]
	gXo  = np.concatenate( args[2:] , axis = 0 )
	
	## Find observed variability
	R      = gXo - A @ hpar
	hcov_o = []
	i      = 0
	for Xo in lXo:
		s = Xo.size
		hcov_o.append( np.ones(s) * float(np.std(R[i:(i+s)]))**2 )
		i += s
	hcov_o = np.diag( np.hstack(hcov_o) )
	
	## Application
	K0    = A @ hcov
	K1    = ( hcov @ A.T ) @ np.linalg.inv( K0 @ A.T + hcov_o )
	hparC = hpar + K1 @ ( gXo.squeeze() - A @ hpar )
	hcovC = hcov - K1 @ K0
	
	return hparC,hcovC
##}}}

def gaussian_conditionning( *args , A = None , timeXo = None , method = None ):##{{{
	
	match method:
		case "KCC":
			try:
				hpar,hcov = gaussian_conditionning_KCC( *args , A = A , timeXo = timeXo , dep = 1 )
			except Exception as e:
				logger.warning(f"Fail to use KCC, use MAR2 method (reason {e})")
				hpar,hcov = gaussian_conditionning( *args , A = A , timeXo = timeXo , method = "MAR2" )
		case "MAR2":
			try:
				hpar,hcov = gaussian_conditionning_KCC( *args , A = A , timeXo = timeXo , dep = 0 )
			except Exception as e:
				logger.warning(f"Fail to use MAR2, use independent method (reason {e})")
				hpar,hcov = gaussian_conditionning( *args , A = A , timeXo = timeXo , method = "INDEPENDENT" )
		case _:
			hpar,hcov = gaussian_conditionning_independent( *args , A = A , timeXo = timeXo )
	
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

