
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

#############
## Imports ##
#############

import logging


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###################################
## Matplotlib default parameters ##
###################################

mpl_rcParams = {
               'font.size'         : 7,
               'axes.linewidth'    : 0.5,
               'lines.linewidth'   : 0.5,
               'patch.linewidth'   : 0.5,
               'xtick.major.width' : 0.5,
               'ytick.major.width' : 0.5,
               'font.serif'        : "Liberation Serif",
               'font.sans-serif'   : "Liberation Sans",
               'font.monospace'    : "Liberation Mono"
               }

