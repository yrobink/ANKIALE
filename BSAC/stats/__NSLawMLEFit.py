
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

from .__tools import nslawid_to_class


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


def _nslaw_fit_bootstrap( idxs , xY , X , per , nslaw_class , init_ ):##{{{
	
	idxs = np.array([idxs]).ravel()
	
	nslaw = nslaw_class()
	
	out = []
	for idx in idxs:
		xX = xr.DataArray( np.array( [X.loc[:,per,xY.time].values[idx,:] for _ in range(xY.run.size)] ).T , dims = ["time","run"] , coords = xY.coords )
		
		Yf = xY.values.ravel()
		Xf = xX.values.ravel()
		
		mask = np.isfinite(Yf)
		Yf = Yf[mask]
		Xf = Xf[mask]
		
		if not np.any(mask):
			out.append( np.zeros(len(nslaw.coef_name)) + np.nan )
			continue
		
		## Resample
		idxbs = np.random.choice( Yf.size , Yf.size , replace = True )
		Yf = Yf[idxbs]
		Xf = Xf[idxbs]
		
		## And fit
		nslaw.fit_mle( Yf , Xf , init = init_ )
		out.append(nslaw.coef_)
	
	return np.array(out)
##}}}

## nslaw_fit_bootstrap ##{{{
@log_start_end(logger)
def nslaw_fit_bootstrap( Y , X , hparY , nslawid , n_bootstrap , n_jobs ):
	
	## Find cper
	cper = [ per for per in Y.period.values.tolist() if per not in X.period.values.tolist()][0]
	
	## Variables for a loop on spatial dimension (if exists)
	if Y.ndim == 3:
		spatial = [1]
	else:
		spatial = [Y[d].size for d in Y.dims[3:]]
	
	##
	nslaw_class = nslawid_to_class(nslawid)
	nslaw       = nslaw_class()
	
	## Loop on spatial dimension
	i = 0
	for spatial_idx in itt.product(*[range(s) for s in spatial]):
		
		i += 1
		print( f"{100*i / np.prod(spatial)}%" )
		
		## Indexes to extract data
		iidx = (slice(None),slice(None),slice(None))
		if Y.ndim > 3:
			iidx = iidx + spatial_idx
		
		## Loop on period
		for iper,per in enumerate(X.period.values.tolist()):
			
			## Start with the best estimate
			xY = Y[iidx].sel( period = [cper,per] ).mean( dim = "period" )
			xX = xr.DataArray( np.array( [X.loc["BE",per,xY.time].values for _ in range(xY.run.size)] ).T , dims = ["time","run"] , coords = xY.coords )
			
			Yf = xY.values.ravel()
			Xf = xX.values.ravel()
			
			mask = np.isfinite(Yf)
			
			if not mask.sum() / mask.size > 0.95:
				continue
			Yf = Yf[mask]
			Xf = Xf[mask]
			
			nslaw.fit_mle( Yf , Xf )
			oidx = (slice(None),iper,0)
			if Y.ndim > 3:
				oidx = oidx + spatial_idx
			init_     = nslaw.coef_
			hparY.set_orthogonal_selection( oidx , init_ )
			
			## Prepare dimension for parallelization
			idxs = xr.DataArray( [i+1 for i in range(n_bootstrap)] , dims = ["bootstrap"] , coords = [range(n_bootstrap)] ).chunk( { "bootstrap" : max( n_bootstrap // n_jobs , 1 ) } )
			
			## Parallelization of the bootstrap
			coef_bs = xr.apply_ufunc(
			             _nslaw_fit_bootstrap , idxs ,
			             kwargs             = { "xY" : xY , "X" : X , "per" : per , "nslaw_class" : nslaw_class , "init_" : init_ },
			             input_core_dims    = [[]],
			             output_core_dims   = [["hpar"]],
					     output_dtypes      = X.dtype ,
					     vectorize          = True ,
					     dask               = "parallelized" ,
					     dask_gufunc_kwargs = { "output_sizes" : { "hpar" : init_.size } }
			             ).compute()
			
			oidx = (slice(None),iper,slice(1,None))
			if Y.ndim > 3:
				oidx = oidx + spatial_idx
			hparY.set_orthogonal_selection( oidx , coef_bs.values.T )
	
##}}}

