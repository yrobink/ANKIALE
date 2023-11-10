
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

import itertools as itt
import numpy as np


###############
## Functions ##
###############

def _matrix_positive_part( M ):##{{{
	lbda,v = np.linalg.eig(M)
	lbda   = np.real(lbda)
	v      = np.real(v)
	lbda[lbda<0] = 0
	return v @ np.diag(lbda) @ v.T
##}}}

def matrix_positive_part(M):##{{{
	
	if M.ndim == 2:
		return _matrix_positive_part(M)
	else:
		shp = M.shape[2:]
		P = M.copy() + np.nan
		for idx in itt.product(*[range(s) for s in shp]):
			idx2d = (slice(None),slice(None)) + idx
			try:
				P[idx2d] = _matrix_positive_part(M[idx2d])
			except:
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



