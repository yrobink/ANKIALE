
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

from .models.__GEVModel    import GEVModel
from .models.__GEVMinModel import GEVMinModel
from .models.__NormalModel import NormalModel


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

def nslawid_to_class( nslawid ):
	if nslawid == "GEV":
		return GEVModel
	if nslawid == "GEVMin":
		return GEVMinModel
	if nslawid == "Normal":
		return NormalModel
	
	raise ValueError( f"NSlaw not known (={nslawid})" )
