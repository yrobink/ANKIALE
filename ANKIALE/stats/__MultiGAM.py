
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

##############
## Packages ##
##############


#############
## Imports ##
#############

import logging

from ..__linalg import matrix_positive_part

import numpy  as np
import xarray as xr
import scipy.linalg as scl
import statsmodels.gam.api as smg
import distributed

from ..__exceptions import DevException


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#############
## Classes ##
#############

class SplineSmoother:##{{{
	
	def __init__( self , x , sbasis , dof , degree , include_intercept = True ):##{{{
		self._spl = smg.BSplines( x , df = sbasis + int(not include_intercept) , degree = degree , include_intercept = include_intercept )
		self._gam = None
		self.dof  = dof
	##}}}
	
	def fit( self , X ):##{{{
		
		alpha = 1e-6
		edof  = self.sbasis + 1
		while edof > self.dof:
			alpha *= 10
			gam    = smg.GLMGam( endog = X , smoother = self._spl , alpha = alpha )
			res    = gam.fit()
			edof   = res.edf.sum()
		
		alphaL = alpha
		alphaR = alphaL / 10
		while np.abs(edof - self.dof) > 1e-2:
			alpha = ( alphaL + alphaR ) / 2
			gam    = smg.GLMGam( endog = X , smoother = self._spl , alpha = alpha )
			res    = gam.fit()
			edof   = res.edf.sum()
			if edof < self.dof:
				alphaL = alpha
			else:
				alphaR = alpha

		self._gam = gam
		self._res = res

		return self
	##}}}
	
	def predict( self , *args , **kwargs ):##{{{
		if self._res is not None:
			return self._res.predict( *args , **kwargs )
	##}}}
	
	## Properties ##{{{
	
	@property
	def degree(self):
		return int(self._spl.degree[0])
	
	@property
	def sbasis(self):
		return self._spl.dim_basis
	
	@property
	def include_intercept(self):
		return self._spl.include_intercept
	
	@property
	def basis(self):
		return self._spl.basis
	
	@property
	def edof(self):
		if self._res is not None:
			return self._res.edf.sum()
	
	@property
	def hpar(self):
		if self._res is not None:
			return self._res.params
	
	@property
	def hcov(self):
		if self._res is not None:
			return self._res.cov_params()
	

	##}}}

##}}}

class MultiGAM:##{{{
	
	def __init__( self , dof , sbasis , degree , infer_hcov = True , tol = 1e-3 , maxit = 500 ):##{{{
		self.dof      = dof
		self.sbasis   = sbasis
		self.degree   = degree
		self.design   = None
		self.infer_hcov = infer_hcov
		self.tol      = tol
		self.maxit    = maxit
		
		self.hpar     = None
		self.hcov     = None
	##}}}
	
	def fit( self , X , XN , hpar0 = None ):##{{{
		
		##
		dpers = list(X)
		
		## Init
		Xr    = { per : X[per].copy() for per in dpers }
		time  = X[dpers[0]].time.values
		lin   = np.stack( [np.ones(X[dpers[0]].size),XN[dpers[0]].values] ).T.copy()
		hpar  = np.zeros( np.sum( [self.sbasis[per] for per in dpers] ) + 2 ) + 1e9
		hpars = [hpar]
		ssm   = {}
		
		## If a starting point
		if hpar0 is not None:
			L   = lin @ hpar0[-2:]
			Xr = { per : X[per] - L for per in dpers }
			hpars = [hpar0]


		## Now the fit: backfitting algorithm
		diff   = 1e9
		nit    = 0
		while diff > self.tol:
			
			## Init new hpar
			hpar = []
			
			## Fit the spline part on residues Xr
			for per in dpers:
				ssm[per] = SplineSmoother( time , sbasis = self.sbasis[per] , dof = self.dof[per] , degree = self.degree[per] , include_intercept = False ).fit( Xr[per].values )
				hpar.extend( ssm[per].hpar )
				Xr[per] = X[per] - ssm[per].predict()
			
			## Fit the linear part on residues Xr
			x_lin,_,_,_ = scl.lstsq( lin , np.mean( [Xr[per] for per in dpers] , axis = 0 ) )
			hpar.extend( x_lin )
			for per in dpers:
				Xr[per] = X[per] - lin @ x_lin
			
			## Add the new
			hpars.append( np.array(hpar) )
			
			## Compute the difference
			diff = np.linalg.norm(hpars[-1] - hpars[-2])
			
			## And iteration
			nit += 1
			if not nit < self.maxit:
				logger.warning( f"Max iterations reached during the fit of the Multi GAM model (max: {self.maxit}, diff: {diff})" )
				break
		
		self.hpar = hpars[-1]
		
		if not self.infer_hcov:
			return self

#		## Create design matrix
#		self.design = np.hstack( [ SplineSmoother( time , sbasis = self.sbasis[per] , dof = self.dof[per] , degree = self.degree[per] , include_intercept = False ).basis for per in dpers ] + [len(dpers) * lin] )
#
#		## And find covariance matrix
#		sig = np.std( np.sum( [X[per] for per in dpers] , axis = 0 ) - self.design @ self.hpar )
#		self.hcov = np.linalg.inv( self.design.T @ self.design ) * sig**2 / ( np.sum( [self.dof[per] for per in dpers] ) + 2 )
		
		## Find the covariance matrix
		Xr = { per : X[per] - ssm[per].predict() - lin @ x_lin for per in dpers }
		
		self.hcov = np.zeros( (self.hpar.size,self.hpar.size) )
		for iper,per in enumerate(dpers):
			spl = ssm[per]
			n_spl = spl.sbasis
			i0 =  iper    * n_spl
			i1 = (iper+1) * n_spl
			H  = np.hstack( (spl.basis,lin) )
			H  = np.linalg.inv( H.T @ H )
			S  = H * Xr[per].values.std()**2 / (ssm[per].dof + 2)
			
			## Spline part
			self.hcov[i0:i1,i0:i1] = S[:-2,:-2]
			
			## Linear part
			self.hcov[-2:,i0:i1] = S[-2:,:n_spl]
			self.hcov[i0:i1,-2:] = S[:n_spl,-2:]
			self.hcov[-2:,-2:]   = self.hcov[-2:,-2:] + S[-2:,-2:]
		
		self.hcov[-2:,-2:] = self.hcov[-2:,-2:] / len(dpers)**2

		return self
	##}}}
	
##}}}

def _mgam_multiple_fit( idx , X , XN , dof , sbasis , degree , init ):##{{{
	
	##
	cnames    = list(init)
	dpers     = list(X[cnames[0]])
	n_hpar    = sum([init[cname].size for cname in cnames])
	n_samples = idx.size
	hpars     = np.zeros( (n_samples,n_hpar) ) + np.nan
	time      = X[cnames[0]][dpers[0]].time.values
	
	##
	for i in range(n_samples):
		bs    = np.random.choice( time , time.size , replace = True )
		Xbs   = { cname : { per : X[cname][per].loc[bs]  for per in dpers } for cname in cnames }
		XNbs  = { cname : { per : XN[cname][per].loc[bs] for per in dpers } for cname in cnames }
		mgams = { cname : MultiGAM( dof = dof[cname] , sbasis = sbasis[cname] , degree = degree[cname] , infer_hcov = False ).fit( Xbs[cname] , XNbs[cname] , hpar0 = init[cname] ) for cname in cnames }
		hpar  = np.hstack( [mgams[cname].hpar for cname in cnames] )
		hpars[i,:] = hpar

	return hpars
##}}}

def mgam_multiple_fit( X , XN , dof , sbasis , degree , n_samples , cluster ):##{{{
	
	## Fit multi gam for each covariate
	cnames = list(X)
	mgams  = { cname : MultiGAM( dof = dof[cname] , sbasis = sbasis[cname] , degree = degree[cname] ).fit( X[cname] , XN[cname] ) for cname in cnames }
	
	## If one covariate, just return parameters fitted
	if len(cnames) == 1:
		hpar = mgams[cnames[0]].hpar
		hcov = mgams[cnames[0]].hcov
		return hpar,hcov
	
	## Find dtype
	for name in cnames:
		for p in X[name]:
			dtype = X[name][p].dtype
			break
		break
	
	## Prepare dimension for parallelization
	idxs = xr.DataArray( range(n_samples) , dims = ["sample"] , coords = [range(n_samples)] ).chunk( { "sample" : max( 1 , n_samples // len(cluster.workers) ) } )
	init = { cname : mgams[cname].hpar for cname in cnames }
	
	
	## Parallelization of the bootstrap
	with distributed.Client(cluster) as client:
		hpars = xr.apply_ufunc(
		             _mgam_multiple_fit , idxs ,
		             kwargs             = { "X" : X , "XN" : XN , "dof" : dof , "sbasis" : sbasis , "degree" : degree , "init" : init },
		             input_core_dims    = [[]],
		             output_core_dims   = [["hyper_parameter"]],
		             output_dtypes      = dtype ,
		             vectorize          = False ,
		             dask               = "parallelized" ,
		             dask_gufunc_kwargs = { "output_sizes" : { "hyper_parameter" : sum([ mgams[cname].hpar.size for cname in cnames ]) } }
		             ).persist().compute()
	
	## Now find final hyper-parameters and covariance matrix
	hpar = np.hstack( [mgams[cname].hpar for cname in cnames] )
	hcov = np.cov( hpars.values.T )
	
	## For the covariance matrix
	for i,cname in enumerate(cnames):
		i0 =  i    * mgams[cname].hcov.shape[0]
		i1 = (i+1) * mgams[cname].hcov.shape[0]
		hcov[i0:i1,i0:i1] = mgams[cname].hcov
	hcov = matrix_positive_part(hcov)
	
	return hpar,hcov
##}}}

