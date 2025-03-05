
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

## Packages
###########

import os

import pandas as pd
import xarray as xr


###############
## Functions ##
###############

def get_XN( time = None , version = "CMIP6" ):##{{{
	"""
	ANKIALE.get_XN
	==============
	
	Arguments
	---------
	time: time axis or None
		Get only a subset of natural forcings
	version: str
		Version of forcings, must be CMIP5 or CMIP6
	
	Returns
	-------
	XN: xarray.DataArray
		A dataarray of natural response
	"""
	cpath = os.path.dirname(os.path.abspath(__file__))
	
	if version not in ["CMIP5","CMIP6"]:
		raise ValueError( "Version of XNGenerator must be CMIP5 or CMIP6")
	
	dX = pd.read_csv( os.path.join( cpath , "data" , f"XN_{version}.csv" ) )
	
	XN = xr.DataArray( dX["XN"].values.astype(float) , dims = ["time"] , coords = [dX["year"].values.astype(int)] )
	
	if time is not None:
		XN = XN.loc[time]
	
	return XN
##}}}

