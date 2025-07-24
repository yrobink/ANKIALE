
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
from ..__logs import log_start_end
import datetime as dt

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from ..__natural import get_XN
from .__utils    import mpl_rcParams


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## plot_XN ##{{{
@log_start_end(logger)
def plot_XN( ofile = None ):
    
    ## Matplotlib parameters
    mpl.rcdefaults()
    for key in mpl_rcParams:
        mpl.rcParams[key] = mpl_rcParams[key]
    
    ## Data
    time = np.arange( 1850 , 2101 , 1 )
    XN5  = get_XN( time = time , version = "CMIP5")
    XN6  = get_XN( time = time , version = "CMIP6")
    now  = dt.datetime.utcnow().year
    
    ## Figure
    mm  = 1. / 25.4
    fig = plt.figure( figsize = (120*mm,80*mm) , dpi = 120 )
    ax  = fig.add_subplot(1,1,1)
    ax.plot( time , XN5 , color = "red"  , label = "CMIP5" )
    ax.plot( time , XN6 , color = "blue" , label = "CMIP6" )
    ax.axhline( 0 , color = "grey" , linestyle = "-" )
    ax.axvline( now , color = "black" , linestyle = ":" )
    ax.set_xticks([1850,1900,1950,2000,2050,2100])
    ax.set_yticks([-2,-1.5,-1,-0.5,0,0.5,1])
    ax.set_xlim(1840,2110)
    ax.set_xlabel("Year")
    ax.set_ylabel("Natural response (K)")
    ax.legend( loc = "upper right" )
    plt.tight_layout()
    
    if ofile is None:
        plt.show()
    else:
        plt.savefig( ofile , dpi = 300 )
    plt.close(fig)
##}}}


