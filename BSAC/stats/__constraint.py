
## Copyright(c) 2024 Yoann Robin
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

def gaussian_conditionning( Xo , A , hpar , cov , cov_o ):##{{{
	K0 = A @ cov
	K1 = ( cov @ A.T ) @ np.linalg.inv( K0 @ A.T + cov_o )
	hpar = hpar + K1 @ ( Xo.squeeze() - A @ hpar )
	cov  = cov  - K1 @ K0
	
	return hpar,cov
##}}}


