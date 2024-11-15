
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

def gaussian_conditionning( *args , A = None ):##{{{
	
	## Extract arguments
	hpar = args[0]
	hcov = args[1]
	lXo  = args[2:]
	gXo  = np.concatenate( args[2:] , axis = 0 )
	
	## Variance of obs
	R      = gXo - A @ hpar
	hcov_o = []
	i      = 0
	for Xo in lXo:
		s = Xo.size
		hcov_o.append( np.ones(s) * float(np.std(R[i:(i+s)]))**2 )
		i += s
	hcov_o = np.diag( np.hstack(hcov_o) )
	
	## Application
	K0 = A @ hcov
	K1 = ( hcov @ A.T ) @ np.linalg.inv( K0 @ A.T + hcov_o )
	hpar = hpar + K1 @ ( Xo.squeeze() - A @ hpar )
	hcov = hcov - K1 @ K0
	
	return hpar,hcov
##}}}


