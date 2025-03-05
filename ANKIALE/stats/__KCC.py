
## Copyright(c) 2025 Yoann Robin
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

import numpy as np
import xarray as xr
import scipy.stats as sc
import scipy.linalg as scl
import scipy.optimize as sco


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###########################
## Functions and classes ##
###########################

class AR1:##{{{
	
	def __init__( self , c , alpha , scale ):
		self.c     = c
		self.alpha = alpha
		self.scale = scale

	def __str__( self ):
		s = "AR1: X(t) = {:.3f} + {:.3f}X(t-1) + N(0,{:.3f})".format( self.c , self.alpha , self.scale )
		return s
	
	def __repr__( self ):
		return self.__str__()
	
	def mu( self ):
		return self.c / ( 1 - self.alpha )
	
	def cov( self , size ):
		C = self.scale**2 / ( 1 - self.alpha**2) * scl.toeplitz( self.alpha**np.arange( 0 , size , 1 ) )
		return C
	
	def rvs( self , size = 1 , samples = 1 , burn = None ):
		X    = np.zeros((size,samples))
		X[0,:] = np.random.normal( loc = self.c , scale = self.scale , size = samples )
		if burn is None:
			burn = int( 0.1 * size )
		for _ in range(burn):
			X[0,:] = self.c + self.alpha * X[0,:] + np.random.normal( loc = 0 , scale = self.scale , size = samples )
		
		for i in range(size-1):
			X[i+1,:] = self.c + self.alpha * X[i,:] + np.random.normal( loc = 0 , scale = self.scale , size = samples )

		if samples == 1:
			X = X.reshape(-1)
		
		return X

	@staticmethod
	def fit( X ):
		a,c,_,_,_ = sc.linregress( X[:-1] , X[1:] )
		s = np.std( X[1:] - c - a * X[:-1] )
		
		return AR1( c = c , alpha = a , scale = s )
	
##}}}

class MAR2:##{{{
	
	def __init__( self , alpha_f , alpha_s , scale_f , scale_s , c_f = 0 , c_s = 0 ):
		if abs(alpha_s) < abs(alpha_f):
			alpha_f,scale_f,c_f,alpha_s,scale_s,c_s = alpha_s,scale_s,c_s,alpha_f,scale_f,c_f
		self._ar_f = AR1( c_f , alpha_f , scale_f )
		self._ar_s = AR1( c_s , alpha_s , scale_s )

	def __repr__( self ):
		return self.__str__()

	def __str__( self ):
		sf = str(self._ar_f)
		ss = str(self._ar_s)
		return "\n".join( [sf,ss] )
	
	def cov( self , size ):
		return self._ar_f.cov(size) + self._ar_s.cov(size)
	
	def rvs( self , size = 1 , samples = 1 , burn = None ):
		Xf = self._ar_f.rvs( size = size , samples = samples , burn = burn )
		Xs = self._ar_s.rvs( size = size , samples = samples , burn = burn )
		return Xf + Xs

	@staticmethod
	def _fit_backfitting( X , maxit ):
		X = X - X.mean()
		R = X
		p = np.zeros(6) + 1e9
		c = np.zeros(6)
		for nit in range(maxit):
			arf = AR1.fit(R)
			arf = AR1( c = arf.c , alpha = 1 / arf.scale , scale = arf.scale )
			R = (X[1:] - arf.alpha * X[:-1] - arf.c)/ arf.scale
			
			ars = AR1.fit(R)
			ars = AR1( c = ars.c , alpha = 1 / ars.scale , scale = ars.scale )
			R = (X[1:] - ars.alpha * X[:-1] - ars.c)/ ars.scale

			c = np.array([arf.alpha , arf.scale , arf.c , ars.alpha , ars.scale , ars.c ])
			if np.linalg.norm( c - p ) / np.linalg.norm(c) < 1e3:
				break
			p = c
		
		return MAR2( arf.alpha , ars.alpha , arf.scale , ars.scale , arf.c , ars.c )

	@staticmethod
	def _fit_mle( X ):

		def logit( x , a = -1 , b = 1 ):
			return 1. / ( 1 + np.exp(-x) ) * (b - a) + a
	
		def ilogit( y , a = -1 , b = 1 ):
			return np.where( np.abs(y) < 1 , - np.log( (b-a) / (y - a) - 1 ) , np.sign(y) - np.sign(y) * 1e-3 )

		def _nlll( hpar , X ):
			arf  = AR1( hpar[0] , logit(hpar[1]) , np.exp(hpar[2]) )
			ars  = AR1( hpar[3] , logit(hpar[4]) , np.exp(hpar[5]) )
			size = X.size
			cov  = arf.cov(size) + ars.cov(size)
			return -sc.multivariate_normal.logpdf( X , mean = np.zeros(size) , cov = cov ).sum()

		hpar0 = MAR2.fit( X , method = "backfitting" ).hpar
		hpar0[[1,4]] = ilogit(hpar0[[1,4]])
		hpar0[[2,5]] = np.log(hpar0[[2,5]])
		res = sco.minimize( _nlll , x0 = hpar0 , args = (X,) , method = "BFGS" )
		hpar = res.x
		arf  = AR1( hpar[0] , logit(hpar[1]) , np.exp(hpar[2]) )
		ars  = AR1( hpar[3] , logit(hpar[4]) , np.exp(hpar[5]) )
		
		return MAR2( arf.alpha , ars.alpha , arf.scale , ars.scale , arf.c , ars.c )
	
	@staticmethod
	def fit( X , method = "mle" , maxit = 50 ):

		match method.lower():
			case "backfitting":
				mar2 = MAR2._fit_backfitting( X , maxit )
			case "mle":
				mar2 = MAR2._fit_mle( X )

		return mar2

	@property
	def alpha_f(self):
		return self._ar_f.alpha
	
	@property
	def alpha_s(self):
		return self._ar_s.alpha

	@property
	def scale_f(self):
		return self._ar_f.scale
	
	@property
	def scale_s(self):
		return self._ar_s.scale
	
	@property
	def c_f(self):
		return self._ar_f.c
	
	@property
	def c_s(self):
		return self._ar_s.c

	@property
	def hpar(self):
		return np.array( [self.c_f,self.alpha_f,self.scale_f,self.c_s,self.alpha_s,self.scale_s] )
##}}}

class KCC:##{{{
	
	def __init__( self ):##{{{
		self._size0    = None
		self._size1    = None
		self._mar2_0   = None
		self._mar2_1   = None
		self._L        = None
		self._cov_iv01 = None
	##}}}
	
	@staticmethod
	def _build_toeplitz_iv( alpha_0 , alpha_1 , size0 , size1 ):##{{{
		sizen  = min(size0,size1)
		sizex  = max(size0,size1)
		cov_ll = scl.toeplitz( alpha_0**np.arange( 0 , size0 , 1 ).astype(int) )
		cov_ur = scl.toeplitz( alpha_1**np.arange( 0 , size1 , 1 ).astype(int) )
		cov_ll[np.triu_indices(size0)] = 0
		cov_ur[np.tril_indices(size1)] = 0
		cov    = np.identity(sizex)[:size0,:size1]
		cov[:size0,:sizen] += cov_ll[:size0,:sizen]
		cov[:sizen,:size1] += cov_ur[:sizen,:size1]
		
		return cov
	##}}}
	
	@staticmethod
	def _build_cst_iv(  L , alpha_0 , alpha_1 , scale_0 , scale_1 ):##{{{
		
		n0  = L * scale_0 * scale_1
		d0  = np.sqrt( 1 - alpha_0**2 )
		d1  = np.sqrt( 1 - alpha_1**2 )
		d01 = 1 - alpha_0 * alpha_1
		
		cst = n0 / ( d0 * d1 * d01 )
		
		return cst
	##}}}
	
	def _build_lag_cov( self , h , L , alpha_0 , alpha_1 , scale_0 , scale_1 ):##{{{
		return alpha_0**h * self._build_cst_iv( L , alpha_0 , alpha_1 , scale_0 , scale_1 )
	##}}}
	
	def _find_L( self , R0 , R1 ):##{{{
		
		cf_max = self._build_cst_iv( 1. , self._mar2_0.alpha_f , self._mar2_1.alpha_f , self._mar2_0.scale_f , self._mar2_1.scale_f )
		cs_max = self._build_cst_iv( 1. , self._mar2_0.alpha_s , self._mar2_1.alpha_s , self._mar2_0.scale_s , self._mar2_1.scale_s )
		
		self._rho_res = float(xr.corr( R0 , R1 ))
		self._rho_mar = (cf_max + cs_max) / np.sqrt( (self._mar2_0.scale_f + self._mar2_0.scale_s) * (self._mar2_1.scale_f + self._mar2_1.scale_s) )
		
		self._L = max( min( self._rho_res / self._rho_mar , 1 ) , -1 )
	##}}}
	
	def fit( self , R0 , R1 ):##{{{
		
		## Store parameters
		self._size0 = R0.size
		self._size1 = R1.size

		## Fit the mixture of two AR1 processes
		self._mar2_0 = MAR2.fit(R0.values)
		self._mar2_1 = MAR2.fit(R1.values)
		
		## Find L
		self._find_L( R0 , R1 )
		
		## Build the toeplitz matrix
		cov_iv_f = self._build_toeplitz_iv( self._mar2_0.alpha_f , self._mar2_1.alpha_f , self._size0 , self._size1 )
		cov_iv_s = self._build_toeplitz_iv( self._mar2_0.alpha_s , self._mar2_1.alpha_s , self._size0 , self._size1 )
		
		## And build the cov_iv matrix
		cf = self._build_cst_iv( self.L , self._mar2_0.alpha_f , self._mar2_1.alpha_f , self._mar2_0.scale_f , self._mar2_1.scale_f )
		cs = self._build_cst_iv( self.L , self._mar2_0.alpha_s , self._mar2_1.alpha_s , self._mar2_0.scale_s , self._mar2_1.scale_s )
		
		self._cov_iv01 = cf * cov_iv_f + cs * cov_iv_s
		
		return self
	##}}}
	
	## Properties ##{{{
	
	@property
	def L(self):
		return self._L
	
	@property
	def cov_iv0(self):
		return self._mar2_0.cov(self._size0)
	
	@property
	def cov_iv1(self):
		return self._mar2_1.cov(self._size1)
	
	@property
	def cov_iv01(self):
		return self._cov_iv01
	
	##}}}
	
##}}}

