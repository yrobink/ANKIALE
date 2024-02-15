
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

##############
## Packages ##
##############


#############
## Imports ##
#############

import logging
from ..__logs import LINE
from ..__logs import log_start_end

from ..__linalg import matrix_positive_part

import numpy  as np
import xarray as xr
import scipy.linalg as scl
import statsmodels.gam.api as smg


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#############
## Classes ##
#############

class MultiGAM:##{{{
	
	def __init__( self , dof = 7 , degree = 3 , find_cov = False , tol = 1e-3 , maxit = 500 ):##{{{
		self.dof      = dof
		self.degree   = degree
		self.find_cov = find_cov
		self.tol      = tol
		self.maxit    = maxit
	##}}}
	
	def fit( self , X , XN , X0 = None ):##{{{
		
		##
		names = list(X)
		
		## Init
		Xr  = { s : X[s].copy() for s in names }
		spl = { s : smg.BSplines( X[s].time.values , df = self.dof , degree = self.degree , include_intercept = False ) for s in names }
		s   = names[0]
		lin = np.stack( [np.ones(X[s].size),XN[s].values] ).T.copy()
		coefs_ = [np.zeros((self.dof-1) * len(names) + 2)+1e9]
		
		## If a starting point
		if X0 is not None:
			L   = lin @ X0[-2:]
			Xr = { s : X[s] - L for s in names }
			coefs_ = [X0]
		
		## Now the fit: backfitting algorithm
		diff   = 1e9
		nit    = 0
		while diff > self.tol:
			
			##
			coef_ = []
			
			## Spline part
			for s in names:
				gam   = smg.GLMGam( endog = Xr[s].values , smoother = spl[s] )
				res   = gam.fit()
				coef_ = coef_ + res.params.tolist()
				Xr[s] = X[s] - res.predict()
			
			## Linear part
			x_lin,_,_,_ = scl.lstsq( lin , np.mean( [Xr[s] for s in names] , axis = 0 ) )
			coef_ = coef_ + x_lin.tolist()
			for s in names:
				Xr[s] = X[s] - lin @ x_lin
			
			## Add the new
			coefs_.append( np.array(coef_) )
			
			## Compute the difference
			diff = np.linalg.norm(coefs_[-1] - coefs_[-2])
			
			## And iteration
			nit += 1
			if not nit < self.maxit:
				logger.warning( f"Max iterations reached during the fit of the Multi GAM model (max: {self.maxit}, diff: {diff})" )
				break
		self.coef_ = coefs_[-1]
		
		if self.find_cov:
			## Find the covariance matrix
			Xr = { s : X[s] - res.predict() - lin @ x_lin for s in names }
			H  = np.hstack( (spl[names[0]].basis,lin) )
			H  = np.linalg.inv( H.T @ H )
			S  = { s : H * float(Xr[s].std())**2 / self.dof for s in names }
			
			n_spl = spl[names[0]].basis.shape[1]
			self.cov_  = np.zeros( (self.coef_.size,self.coef_.size) )
			for i in range(len(names)):
				i0 =  i    * n_spl
				i1 = (i+1) * n_spl
				
				## Spline part
				self.cov_[i0:i1,i0:i1] = S[s][:-2,:-2]
				
				## Linear part
				self.cov_[-2:,i0:i1] = S[s][-2:,:n_spl]
				self.cov_[i0:i1,-2:] = S[s][:n_spl,-2:]
			self.cov_[-2:,-2:] = np.sum( [S[s][-2:,-2:] for s in names] , axis = 0 ) / len(names)**2
		
		
		return self
	##}}}
	
##}}}

def _mgam_multiple_fit_bootstrap( idx , X , XN , dof , degree , coef_be = None ):##{{{
	
	## Find parameters and time axis
	names = list(X)
	for name in names:
		periods = list(X[name])
		for p in periods:
			time = X[name][p].time.values
			break
		break
	
	##
	coefs = []
	for _ in range(idx.size):
		bs   = np.random.choice( time , time.size , replace = True )
		Xbs  = { name : { p : X[name][p].loc[bs]  for p in periods } for name in names }
		XNbs = { name : { p : XN[name][p].loc[bs] for p in periods } for name in names }
		coefs.append( np.hstack( [MultiGAM( dof = dof , degree = degree ).fit( Xbs[name] , XNbs[name] , X0 = coef_be[name] ).coef_ for name in names] ) )
	
	return np.array(coefs)
##}}}

def mgam_multiple_fit_bootstrap( X , XN , n_bootstrap , names , dof , degree , n_jobs ):##{{{
	
	## Fit the best estimate
	coef_be = {}
	cov_be  = {}
	for name in names:
		mgam = MultiGAM( dof = dof , degree = degree , find_cov = True )
		mgam.fit( X[name] , XN[name] )
		coef_be[name] = mgam.coef_
		cov_be[name]  = mgam.cov_
	
	## If one covariate, just return parameters fitted
	if len(names) == 1:
		logger.info( "Only one covariate, no bootstrap required" )
		return coef_be[name],cov_be[name]
	
	## Find dtype
	for name in names:
		for p in X[name]:
			dtype = X[name][p].dtype
			break
		break
	
	## Prepare dimension for parallelization
	idxs = xr.DataArray( range(n_bootstrap) , dims = ["bootstrap"] , coords = [range(n_bootstrap)] ).chunk( { "bootstrap" : n_bootstrap // n_jobs } )
	
	## Parallelization of the bootstrap
	coef_bs = xr.apply_ufunc(
	             _mgam_multiple_fit_bootstrap , idxs ,
	             kwargs             = { "X" : X , "XN" : XN , "dof" : dof , "degree" : degree , "coef_be" : coef_be },
	             input_core_dims    = [[]],
	             output_core_dims   = [["parameter"]],
			     output_dtypes      = dtype ,
			     vectorize          = True ,
			     dask               = "parallelized" ,
			     dask_gufunc_kwargs = { "output_sizes" : { "parameter" : sum([coef_be[name].size for name in names]) } }
	             ).compute()
	
	## Now find final coef and covariance matrix
	coef_ = np.hstack( [coef_be[name] for name in names] )
	cov_  = np.cov( coef_bs.values.T )
	
	## For the covariance matrix
	for i,name in enumerate(names):
		i0 =  i    * cov_be[name].shape[0]
		i1 = (i+1) * cov_be[name].shape[0]
		cov_[i0:i1,i0:i1] = cov_be[name]
	cov_ = matrix_positive_part(cov_)
	
	return coef_,cov_
##}}}


