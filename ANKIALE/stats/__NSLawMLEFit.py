
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

import itertools as itt
import logging
from ..__logs import disable_warnings

import numpy  as np
import xarray as xr


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

## nslaw_fit ##{{{

@disable_warnings
def nslaw_fit( hpar: np.ndarray , hcov: np.ndarray , Y: np.ndarray , samples: np.ndarray , cnslaw: type , proj: xr.DataArray , cname: str ) -> np.ndarray:
    
    ## Find spatial size
    s_spatial = tuple()
    if Y.ndim > 3:
        s_spatial = tuple(Y.shape[:-4])
    
    ## Init output
    hpar_names = proj.hpar.values.tolist()
    s_hparY = hpar.size + cnslaw().nhpar
    dpers   = proj.period.values.tolist()
    ndpers  = Y.shape[-2]-1
    hpars   = np.zeros( s_spatial + (samples.size,s_hparY) ) + np.nan
    nrun    = Y.shape[-1]
    
    ## Draw parameters
    hpars[*([slice(None) for _ in range(hpars.ndim-1)] + [range(hpar.size)] ) ] = np.random.multivariate_normal( mean = hpar , cov = hcov , size = hpars.size // s_hparY ).reshape( hpars.shape[:-1] + (hpar.size,) )
    hpars = xr.DataArray( hpars ,
                          dims = [f"spatial{i}" for i in range(len(s_spatial))] + ["sample","hpar"],
                        coords = [ range(s) for s in s_spatial ] + [range(samples.size),hpar_names+list(cnslaw().h_name)]
                      )
    XF = ( proj.sel( name = cname ) @ hpars )
    hpars = xr.concat( [ hpars for _ in dpers ] , dim = "period" ).assign_coords( period = dpers )
    
    ## Now loop for fit
    init = [None for _ in dpers]
    for idx0 in itt.product( *[ range(s) for s in hpars.shape[1:-1]] ):
        for iper,dper in enumerate(dpers):
            
            ## X / Y and re-sampling
            xX = np.array( [ XF[ (iper,slice(None)) + idx0 ].values for _ in range(nrun) ] ).T.ravel().copy()
            xY = np.nanmean( Y[ idx0[:-1] + (0,slice(None),[0,iper+1],slice(None)) ] , axis = 0 ).ravel().copy()
            
            ## Keep only finite values
            idx = np.isfinite(xY)
            if not idx.any():
                continue
            xX  = xX[idx]
            xY  = xY[idx]
            
            nslaw = cnslaw()
            if init[iper] is None:
                init[iper] = nslaw.fit_mle( xY , xX )
            
            ## Resampling
            p  = np.random.choice( xX.size , xX.size , replace = True )
            
            ## Fit
            ns_hpar = nslaw.fit_mle( xY[p] , xX[p] , init = init[iper] )
            
            ## Save
            hpars[ (iper,) + idx0 + (slice(hpar.size,s_hparY,1),) ] = ns_hpar
    
    odims = [f"spatial{i}" for i in range(len(s_spatial))] + ["sample","hpar","period"]
    hpars = hpars.transpose(*odims)
    
    return hpars.values
##}}}


