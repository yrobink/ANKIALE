
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

import itertools as itt
import numpy as np
import scipy.stats as sc

from .__logs import disable_warnings


###############
## Functions ##
###############


def matrix_positive_part(M):##{{{
	
	def _matrix_positive_part( M ):
		
		if not np.isfinite(M).all():
			return M + np.nan
		lbda,v = np.linalg.eig(M)
		lbda   = np.real(lbda)
		v      = np.real(v)
		lbda[lbda<0] = 0
		return v @ np.diag(lbda) @ v.T
	
	if M.ndim == 2:
		return _matrix_positive_part(M)
	else:
		shp = M.shape[2:]
		P = M.copy() + np.nan
		for idx in itt.product(*[range(s) for s in shp]):
			idx2d = (slice(None),slice(None)) + idx
			try:
				P[idx2d] = _matrix_positive_part(M[idx2d])
			except Exception:
				pass
		return P

##}}}

def nancov(X):##{{{
	if X.ndim == 2:
		return np.ma.cov( np.ma.masked_invalid(X) , rowvar = False ).filled(np.nan)
	else:
		shp = X.shape[2:]
		P = np.zeros( (X.shape[1],X.shape[1])  + shp )
		for idx in itt.product(*[range(s) for s in shp]):
			idx2d = (slice(None),slice(None)) + idx
			P[idx2d] = np.ma.cov( np.ma.masked_invalid(X[idx2d]) , rowvar = False ).filled(np.nan)
		
		return P
##}}}

def sqrtm( C ):##{{{
	
	def _sqrtm(c):
		if not np.isfinite(c).all():
			return np.zeros_like(c) + np.nan
		u,s,v = np.linalg.svd(c)
		return u @ np.sqrt(np.diag(s)) @ v.T
	
	if C.ndim == 2:
		return _sqrtm(C)
	
	shape_nd = C.shape
	shape_1d = C.shape[:2] + (-1,)
	C = C.reshape(shape_1d)
	S = C.copy() + np.nan
	for i in range(C.shape[-1]):
		S[:,:,i] = _sqrtm(C[:,:,i])
	
	return S.reshape(shape_nd)
##}}}

def robust_covariance( X , method = "empirical" , index = slice(None) ):##{{{
	
	if method == "norm-quantile":
		XX = X[:,index]
		loc   = np.quantile( XX , 0.50 , axis = 0 )
		scale = np.quantile( XX - loc.reshape(1,-1) , 0.8413447460685429 , axis = 0 )
		e     = 1 / XX.shape[0] / 2
		valid = np.ones( XX.shape[0] , dtype = bool )
		for i in range(XX.shape[1]):
			p     = sc.norm.cdf( XX[:,i] , loc = loc[i] , scale = scale[i] )
			valid = valid & (p > e) & (1 - p > e)
		C = np.cov( X[valid,:].T )
	else:
		C = np.ma.cov( np.ma.masked_invalid(X) , rowvar = True ).data
	
	return C
##}}}

## mean_cov_hpars ##{{{

@disable_warnings
def mean_cov_hpars( hpars ):
	
	nhpar = hpars.shape[-3]
	hpar  = np.nanmean( hpars , axis = (-2,-1) )
	hcov  = np.apply_along_axis( lambda x: robust_covariance( x.reshape(nhpar,-1).T ) , 1 , hpars.reshape(-1,np.prod(hpars.shape[-3:])) ).reshape( hpars.shape[:-3] + (nhpar,nhpar) )
	
	return hpar,hcov
##}}}


