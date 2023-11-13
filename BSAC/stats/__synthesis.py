
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

from ..__XZarr import XZarr

from ..__linalg import matrix_positive_part
from ..__linalg import nancov

from .__rvs import rvs_multivariate_normal

import numpy as np
import xarray as xr
import itertools as itt


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

def _synthesis_array( hpars , covs ):##{{{
	
	if np.all( ~np.isfinite(hpars)) or np.all( ~np.isfinite(covs)):
		hpar_S = np.zeros( hpars.shape[1:] ) + np.nan
		cov_S  = np.zeros(  covs.shape[1:] ) + np.nan
		return hpar_S,cov_S
	
	n_mod = hpars.shape[0]
	Si    = np.nansum( covs , axis = 0 ) ## Sum of covariance matrix of the models
	Se    = (n_mod-1) * nancov(hpars)    ## Inter-model covariance matrix
	Su    = ( Se - (1 - 1 / n_mod) * Si ) / (n_mod - 1) ## Climate model uncertainty
	Su    = matrix_positive_part(Su)
	
	hpar_S = np.nanmean( hpars , axis = 0 )
	cov_S  = (1 + 1 / n_mod) * Su + Si / n_mod**2
	cov_S  = matrix_positive_part(cov_S)
	
	return hpar_S,cov_S
##}}}

def _synthesis_zarr( hpars , covs , n_jobs = 1 ):##{{{
	
	##
	if hpars.ndim < 3:
		return _synthesis_array( hpars.zdata[:] , covs.zdata[:] )
	
	##
	hpar_S = np.zeros( hpars.shape[1:] ) + np.nan
	cov_S  = np.zeros( covs.shape[1:] )  + np.nan
	
	##
	jump = max( 0 , int( np.power( n_jobs , 1. / len(hpars.shape[2:]) ) ) ) + 1
	for idx in itt.product(*[range(0,s,jump) for s in hpars.shape[2:]]):
		
		## Define indexes
		s_idx  = tuple([slice(s,s+jump,1) for s in idx])
		iidx1d = (slice(None),slice(None),) + s_idx
		iidx2d = (slice(None),slice(None),slice(None)) + s_idx
		oidx1d = (slice(None),) + s_idx
		oidx2d = (slice(None),slice(None)) + s_idx
		
		## Extract from zarr
		hpar = hpars.get_orthogonal_selection(iidx1d)
		cov  = covs.get_orthogonal_selection(iidx2d)
		
		## Chunk
		hpar = hpar.chunk( { d : 1 for d in hpars.dims[2:] } )
		cov  =  cov.chunk( { d : 1 for d in hpars.dims[2:] } )
		
		## Apply
		h,c = xr.apply_ufunc( _synthesis_array , hpar , cov,
		                    input_core_dims  = [["clim","hpar"],["clim","hpar0","hpar1"]],
		                    output_core_dims = [       ["hpar"],       ["hpar0","hpar1"]],
		                    output_dtypes    = [hpar.dtype,cov.dtype],
		                    vectorize        = True,
		                    dask             = "parallelized"
		                    )
		h = h.transpose(*hpar.dims[1:]).compute()
		c = c.transpose( *cov.dims[1:]).compute()
		hpar_S[oidx1d] = h.values
		cov_S[oidx2d]  = c.values
	
	return hpar_S,cov_S
##}}}

def synthesis( hpars , covs , n_jobs = 1 ):##{{{
	
	if isinstance(hpars,XZarr) or isinstance(covs,XZarr):
		return _synthesis_zarr( hpars , covs , n_jobs )
	else:
		return _synthesis_array( hpars , covs )
##}}}


