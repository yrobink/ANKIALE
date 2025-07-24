
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
from ..__linalg import nancov

import numpy as np


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

def synthesis( hpars , hcovs ):##{{{
    
    if np.all( ~np.isfinite(hpars)) or np.all( ~np.isfinite(hcovs)):
        hpar = np.zeros( hpars.shape[1:] ) + np.nan
        hcov = np.zeros( hcovs.shape[1:] ) + np.nan
        return hpar,hcov
    
    n_mod = hpars.shape[0]
    Si    = np.nansum( hcovs , axis = 0 ) ## Sum of covariance matrix of the models
    Se    = (n_mod-1) * nancov(hpars)    ## Inter-model covariance matrix
    Su    = ( Se - (1 - 1 / n_mod) * Si ) / (n_mod - 1) ## Climate model uncertainty
    Su    = matrix_positive_part(Su)
    
    hpar = np.nanmean( hpars , axis = 0 )
    hcov = (1 + 1 / n_mod) * Su + Si / n_mod**2
    hcov = matrix_positive_part(hcov)
    
    return hpar,hcov
##}}}


