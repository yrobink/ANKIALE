
## Copyright(c) 2025 Yoann Robin
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

def covariance_matrix_ar1( alpha , scale , size ):##{{{
	
	C = scl.toeplitz( alpha**np.arange( 0 , size , 1 ).astype(int) )
	C = scale**2 / (1 - alpha**2) * C
	
	return C
##}}}

class MAR2:##{{{
	
	## I/O functions ##{{{
	
	def __init__( self ):
		self._thpar = None
		self._opt   = None
		self._size  = None
	
	def __repr__( self ):
		return self.__str__()
	
	def __str__( self ):
		s = "af: {:.3f}, as: {:.3f}, sf: {:.3f}, ss: {:.3f}".format( self.alpha_f , self.alpha_s , self.scale_f , self.scale_s )
		return s
	
	##}}}
	
	## Fit functions ##{{{
	
	def _covariance_matrix( self ):
		
		hpar   = self.hpar
		Cf_ar1 = covariance_matrix_ar1( self.alpha_f , self.scale_f , self.size )
		Cs_ar1 = covariance_matrix_ar1( self.alpha_s , self.scale_s , self.size )
		
		return Cf_ar1 + Cs_ar1
	
	def _nlll_multivariate_normal( self , Y , m , C ):
		mY   = Y - m
		iC   = np.linalg.pinv(C)
		ldet = self._size * np.log(C[0,0]) + np.log(np.abs(np.linalg.det(C / C[0,0] )))
		nlll = ldet + mY @ iC @ mY
		
		return nlll
	
	def _nlll_mar2( self , thpar , Y ):
		
		self._thpar = thpar
		C_mar2 = self._covariance_matrix()
		nlll = self._nlll_multivariate_normal( Y , Y.mean() , C_mar2 )
		
		return nlll
	
	def fit( self , Y ):
		
		## Set class parameters
		Y = Y.ravel()
		self._size = Y.size
		
		## Init point
		self.hpar = np.array( [ Y.std() / np.sqrt(2) , Y.std() / np.sqrt(2) , 0.4 , 0.8 ] )
		
		## Optimization
		self._opt   = sco.minimize( self._nlll_mar2 , x0 = self._thpar , args = (Y,) , method = "BFGS" )
		self._thpar = self._opt.x
		
		return self
	
	##}}}
	
	## Link function ##{{{
	
	def _transform( self , hpar ):
		thpar    = np.zeros_like(hpar)
		thpar[0] = np.log(hpar[0])
		thpar[1] = np.log(hpar[1])
		thpar[2] = np.tan(hpar[2] * (np.pi / 2) )
		thpar[3] = np.tan(hpar[3] * (np.pi / 2) )
		
		return thpar
	
	def _itransform( self , thpar ):
		hpar    = np.zeros_like(thpar)
		hpar[0] = np.exp( thpar[0])
		hpar[1] = np.exp( thpar[1])
		hpar[2] = np.arctan(thpar[2]) / (np.pi / 2)
		hpar[3] = np.arctan(thpar[3]) / (np.pi / 2)
		
		return hpar
	
	##}}}
	
	## Properties ##{{{
	
	@property
	def alpha_s(self):
		hpar = self.hpar
		alpha_s = hpar[2] if hpar[2] > hpar[3] else hpar[3]
		return alpha_s
	
	@property
	def alpha_f(self):
		hpar = self.hpar
		alpha_f = hpar[3] if hpar[2] > hpar[3] else hpar[2]
		return alpha_f
	
	@property
	def scale_s(self):
		hpar = self.hpar
		scale_s = hpar[0] if hpar[2] > hpar[3] else hpar[1]
		return scale_s
	
	@property
	def scale_f(self):
		hpar = self.hpar
		scale_f = hpar[1] if hpar[2] > hpar[3] else hpar[0]
		return scale_f
	
	@property
	def size(self):
		return self._size
	
	@property
	def hpar(self):
		return self._itransform(self._thpar)
	
	@hpar.setter
	def hpar( self , value ):
		self._thpar = self._transform(value)
	
	##}}}
	
##}}}

class KCC:##{{{
	
	def __init__( self ):##{{{
		self._mar2_0   = None
		self._mar2_1   = None
		self._L        = None
		self._cov_iv01 = None
	##}}}
	
	def _build_toeplitz_iv( self , alpha_0 , alpha_1 , size0 , size1 ):##{{{
		sizen       = min(size0,size1)
		sizex       = max(size0,size1)
		cov_ll = scl.toeplitz( alpha_0**np.arange( 0 , size0 , 1 ).astype(int) )
		cov_ur = scl.toeplitz( alpha_1**np.arange( 0 , size1 , 1 ).astype(int) )
		cov_ll[np.triu_indices(size0)] = 0
		cov_ur[np.tril_indices(size1)] = 0
		cov    = np.identity(sizex)[:size0,:size1]
		cov[:size0,:sizen] += cov_ll[:size0,:sizen]
		cov[:sizen,:size1] += cov_ur[:sizen,:size1]
		
		return cov
	##}}}
	
	def _build_cst_iv( self , L , alpha_0 , alpha_1 , scale_0 , scale_1 ):##{{{
		
		n0  = L * scale_0 * scale_1
		d0  = np.sqrt( 1 - alpha_0**2 )
		d1  = np.sqrt( 1 - alpha_1**2 )
		d01 = 1 - alpha_0 * alpha_1
		
		return n0 / ( d0 * d1 * d01 )
	##}}}
	
	def _build_lag_cov( self , h , L , alpha_0 , alpha_1 , scale_0 , scale_1 ):##{{{
		return alpha_0**h * self._build_cst_iv( L , alpha_0 , alpha_1 , scale_0 , scale_1 )
	##}}}
	
	def _find_L( self , R0 , R1 ):##{{{
		
		self._rho_res = float(xr.corr( R0 , R1 ))
		cf_max = self._mar2_0.scale_f * self._mar2_1.scale_f * np.sqrt( 1 - self._mar2_0.alpha_f**2 ) * np.sqrt( 1 - self._mar2_1.alpha_f**2 ) / ( 1 - self._mar2_0.alpha_f * self._mar2_1.alpha_f )
		cs_max = self._mar2_0.scale_s * self._mar2_1.scale_s * np.sqrt( 1 - self._mar2_0.alpha_s**2 ) * np.sqrt( 1 - self._mar2_1.alpha_s**2 ) / ( 1 - self._mar2_0.alpha_s * self._mar2_1.alpha_s )
		self._rho_mar = (cf_max + cs_max) / np.sqrt( (self._mar2_0.scale_f + self._mar2_0.scale_s) * (self._mar2_1.scale_f + self._mar2_1.scale_s) )
		ratio = self._rho_res / self._rho_mar
		
		if ratio < -1:
			self._L = -1
		elif ratio > 1:
			self._L = 1
		else:
			self._L = ratio
	##}}}
	
	def fit( self , R0 , R1 ):##{{{
		
		## Fit the mixture of two AR1 processes
		self._mar2_0 = MAR2().fit(R0.values)
		self._mar2_1 = MAR2().fit(R1.values)
		
		## Build the toeplitz matrix
		cov_iv_f = self._build_toeplitz_iv( self._mar2_0.alpha_f , self._mar2_1.alpha_f , self._mar2_0.size , self._mar2_1.size )
		cov_iv_s = self._build_toeplitz_iv( self._mar2_0.alpha_s , self._mar2_1.alpha_s , self._mar2_0.size , self._mar2_1.size )
		
		## Find L
		self._find_L( R0 , R1 )
		
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
		return self._mar2_0._covariance_matrix()
	
	@property
	def cov_iv1(self):
		return self._mar2_1._covariance_matrix()
	
	@property
	def cov_iv01(self):
		return self._cov_iv01
	
	##}}}
	
##}}}

