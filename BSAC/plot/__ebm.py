
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

#############
## Imports ##
#############

import logging
from ..__logs import LINE
from ..__logs import log_start_end
import datetime as dt

import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt

from ..__ebm   import EBM
from .__utils  import mpl_rcParams


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## plot_EBM ##{{{
@log_start_end(logger)
def plot_EBM( ofile = None ):
	
	## Matplotlib parameters
	mpl.rcdefaults()
	for key in mpl_rcParams:
		mpl.rcParams[key] = mpl_rcParams[key]
	
	## Data
	time = np.arange( 1850 , 2101 , 1 )
	XN   = xr.DataArray( EBM().run( t = time ).values.squeeze() , dims = ["time"] , coords = [time] )
	now  = dt.datetime.utcnow().year
	
	## Figure
	mm  = 1. / 25.4
	fig = plt.figure( figsize = (120*mm,80*mm) , dpi = 120 )
	ax  = fig.add_subplot(1,1,1)
	ax.plot( time , XN , color = "red" )
	ax.axhline( 0 , color = "grey" , linestyle = "-" )
	ax.axvline( now , color = "black" , linestyle = ":" )
	ax.set_xticks([1850,1900,1950,2000,2050,2100])
	ax.set_yticks([-1,-0.5,0,0.5,1])
	ax.set_xlim(1840,2110)
	ax.set_ylim(-1,1)
	ax.set_xlabel("Year")
	ax.set_ylabel("EBM response (K)")
	plt.tight_layout()
	
	if ofile is None:
		plt.show()
	else:
		plt.savefig( ofile , dpi = 600 )
	plt.close(fig)
##}}}


