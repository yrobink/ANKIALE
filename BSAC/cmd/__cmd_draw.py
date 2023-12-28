
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

import itertools as itt
import logging

from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams

import numpy as np
import netCDF4
import cftime


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## run_bsac_cmd_draw_Y ##{{{
@log_start_end(logger)
def run_bsac_cmd_draw_Y():
	
	## Draw data
	logger.info("Draw parameters...")
	draw = bsacParams.clim.rvsY( size = bsacParams.n_samples , add_BE = True , return_hpar = True )
	logger.info("Draw parameters. Done.")
	
	## Extract parameters
	XF   = draw["XF"]
	samples = XF.coords[0]
	time    = XF.coords[1]
	periods = XF.coords[2]
	namesX  = XF.coords[3]
	hpars   = draw["hpar"].coords[1]
	pars    = [key for key in draw if key not in ["XF","XC","hpar"]]
	
	## And save
	logger.info("Save in netCDF...")
	with netCDF4.Dataset( bsacParams.output , "w" ) as ncf:
		
		## Define dimensions
		ncdims = {
		       "sample"   : ncf.createDimension( "sample" , len(samples) ),
		       "time"     : ncf.createDimension(   "time" , len(time)    ),
		       "period"   : ncf.createDimension( "period" , len(periods) ),
		       "name"     : ncf.createDimension(   "name" , len(namesX)  ),
		       "hyper_parameter" : ncf.createDimension( "hyper_parameter" , len(hpars)  ),
		}
		
		if len(XF.coords) > 4:
			d_spatial = tuple(XF.dims[4:])
			c_spatial = XF.coords[4:]
			s_spatial = tuple([len(c) for c in c_spatial])
			for d,s in zip(d_spatial,s_spatial):
				ncdims[d] = ncf.createDimension( d , s )
		else:
			d_spatial = ()
			c_spatial = ()
			s_spatial = ()
		
		## Define variables
		ncvars = {
		       "sample" : ncf.createVariable( "sample" , str       , ("sample",) ),
		       "name"   : ncf.createVariable(   "name" , str       ,   ("name",) ),
		       "period" : ncf.createVariable( "period" , str       , ("period",) ),
		       "time"   : ncf.createVariable(   "time" , "float32" ,   ("time",) ),
		       "hyper_parameter" : ncf.createVariable( "hyper_parameter" , str ,   ("hyper_parameter",) ),
		}
		for d,c in zip(d_spatial,c_spatial):
			ncvars[d] = ncf.createVariable( d , "double" , (d,) )
			ncvars[d][:] = np.array(c).ravel()
		
		ncvars["sample"][:] = np.array( samples , dtype = str )
		ncvars["period"][:] = np.array( periods , dtype = str )
		ncvars[  "name"][:] = np.array(  namesX , dtype = str )
		ncvars["hyper_parameter"][:] = np.array( hpars , dtype = str )
		
		## Fill time axis
		calendar = "standard"
		units    = "days since 1750-01-01 00:00"
		ncvars["time"][:]  = cftime.date2num( [cftime.DatetimeGregorian( int(y) , 1 , 1 ) for y in time] , units = units , calendar = calendar )
		ncvars["time"].setncattr( "standard_name" , "time"      )
		ncvars["time"].setncattr( "long_name"     , "time_axis" )
		ncvars["time"].setncattr( "units"         , units       )
		ncvars["time"].setncattr( "calendar"      , calendar    )
		ncvars["time"].setncattr( "axis"          , "T"         )
		
		## Create final variables
		ncXF   = ncf.createVariable( "XF"   , "float32" , ("sample","time","period","name") + d_spatial , compression = "zlib" , complevel = 5 , chunksizes = (1,1,1,1) + s_spatial )
		ncXC   = ncf.createVariable( "XC"   , "float32" , ("sample","time","period","name") + d_spatial , compression = "zlib" , complevel = 5 , chunksizes = (1,1,1,1) + s_spatial )
		nchpar = ncf.createVariable( "hpar" , "float32" , ("sample","hyper_parameter")      + d_spatial , compression = "zlib" , complevel = 5 , chunksizes =     (1,1) + s_spatial )
		ncPar  = { key : ncf.createVariable( key , "float32" , ("sample","time","period") + d_spatial , compression = "zlib" , complevel = 5 , chunksizes = (1,1,1) + s_spatial ) for key in pars }
		
		t_spatial = tuple([slice(None) for _ in range(len(s_spatial))])
		for s in range(len(samples)):
			
			idxH = (s,slice(None)) + t_spatial
			nchpar[idxH] = draw["hpar"].get_orthogonal_selection(idxH)
			
			for t,p in itt.product(range(len(time)),range(len(periods))):
				idxX = (s,t,p,slice(None)) + t_spatial
				idxP = (s,t,p) + t_spatial
				
				ncXF[idxX] = draw["XF"].get_orthogonal_selection(idxX)
				ncXC[idxX] = draw["XC"].get_orthogonal_selection(idxX)
				for key in pars:
					ncPar[key][idxP] = draw[key].get_orthogonal_selection(idxP)
	logger.info("Save in netCDF. Done.")
	
##}}}

## run_bsac_cmd_draw ##{{{
@log_start_end(logger)
def run_bsac_cmd_draw():
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the show command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["Y"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the show command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "Y":
		run_bsac_cmd_draw_Y()
##}}}


