
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


import itertools as itt
import logging
from ..__logs import LINE
from ..__logs import log_start_end

import numpy  as np
import xarray as xr
import zarr


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

def rvs_multivariate_normal( size , mean , cov , zfile = None ):##{{{
	
	## Transform in array
	mean_ = mean.values if isinstance(mean,xr.DataArray) else mean
	cov_  = cov.values  if isinstance( cov,xr.DataArray) else cov
	
	## Compute standard deviation
	std_ = sqrtm(cov_)
	
	## Output
	if zfile is None:
		out = np.zeros( (size,) + mean_.shape )
	else:
		out = zarr.open( zfile , mode = "w" , shape = (size,) + mean_.shape , dtype = "float32" , compressor = None )
	
	for idx in itt.product(*[range(s) for s in mean_.shape[1:]]):
		idx1d = (slice(None),) + idx
		idx2d = (slice(None),slice(None)) + idx
		draw = np.random.normal( loc = 0 , scale = 1 , size = mean_.shape[0] * size ).reshape(mean_.shape[0],size)
		draw = std_[idx2d] @ draw + mean_[idx1d].reshape(-1,1)
		try:
			out.set_orthogonal_selection( (slice(None),slice(None),) + idx , draw.T )
		except:
			out[idx2d] = draw.T
	
	return out
##}}}



