
## Copyright(c) 2023 / 2024 Yoann Robin
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

import os
import logging

import numpy as np
import xarray as xr
import xesmf

from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams
from ..__climatology import Climatology
from ..__XZarr import XZarr
from ..__XZarr import random_zfile
from ..stats.__tools import nslawid_to_class
from ..stats.__synthesis import synthesis


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## run_bsac_cmd_synthesize ##{{{
@log_start_end(logger)
def run_bsac_cmd_synthesize():
	
	##
	clim = bsacParams.clim
	
	## Read the grid
	try:
		regrid   = True
		gridfile = bsacParams.config.get["grid"]
		gridname = bsacParams.config.get["grid_name"] #, bsacParams.config["names"].split(":")[-1] )
		
		grid = xr.open_dataset(gridfile)
		mask = grid[gridname] > 0
		clim._spatial = { d : grid[d] for d in bsacParams.config["spatial"].split(":") }
		d_spatial = clim.d_spatial
		s_spatial = clim.s_spatial
		c_spatial = tuple([clim._spatial[d]      for d in clim._spatial])
		i_spatial = tuple([slice(None) for _ in d_spatial])
	except:
		regrid    = False
		clim_grid = Climatology.init_from_file( bsacParams.input[0] )
		clim._spatial = clim_grid._spatial
		d_spatial = clim_grid.d_spatial
		s_spatial = clim_grid.s_spatial
		c_spatial = tuple([clim_grid._spatial[d]      for d in clim_grid._spatial])
		i_spatial = tuple([slice(None) for _ in d_spatial])
	
	## Parameters
	ifiles      = bsacParams.input
	d_clim      = "clim"
	s_clim      = len(ifiles)
	c_clim      = range(s_clim)
	clim._nslawid     = bsacParams.config["nslaw"]
	clim._nslaw_class = nslawid_to_class(clim._nslawid)
	clim._names       = bsacParams.config["names"].split(":")
	d_hpar            = "hpar"
	c_hpar            = clim.hpar_names
	s_hpar            = len(c_hpar)
	cvar              = clim._names[-1]
	
	## Temporary files
	zhpar = XZarr.from_value( np.nan,
	                          (s_clim,s_hpar) + s_spatial,
	                          (d_clim,d_hpar) + d_spatial,
	                          (c_clim,c_hpar) + c_spatial,
	                          random_zfile( os.path.join( bsacParams.tmp , "zhpar" ) )
	                        )
	zcov  = XZarr.from_value( np.nan,
	                          (s_clim,s_hpar,s_hpar) + s_spatial,
	                          (d_clim,d_hpar+"0",d_hpar+"1") + d_spatial,
	                          (c_clim,c_hpar,c_hpar) + c_spatial,
	                          random_zfile( os.path.join( bsacParams.tmp , "zcov" ) )
	                        )
	
	##
	clim._bias = { n : 0 for n in clim._names }
	
	## Open all clims, and store in zarr files
	clims = []
	for i,ifile in enumerate(ifiles):
		
		## Read clim
		iclim = Climatology.init_from_file(ifile)
		time  = iclim.time
		bper  = iclim._bper
		bias_ = iclim.bias[cvar]
		mean_ = iclim.xmean_
		cov_  = iclim.xcov_
		
		if regrid:
			## Grid
			igrid     = xr.Dataset( iclim._spatial )
			regridder = xesmf.Regridder( igrid , grid , "nearest_s2d" )
			
			## Regrid
			bias = regridder(bias_).where( mask , np.nan )
			mean = regridder(mean_).where( mask , np.nan )
			cov  = regridder(cov_ ).where( mask , np.nan )
		else:
			bias = bias_
			mean = mean_
			cov  = cov_ 
		
		## Special case, miss scenario(s) in the clim 
		if mean.hpar.size < s_hpar:
			nmean = xr.DataArray( np.nan , dims = (d_hpar,) + d_spatial , coords = (c_hpar,) + c_spatial )
			ncov  = xr.DataArray( np.nan , dims = (d_hpar+"0",d_hpar+"1") + d_spatial , coords = (c_hpar,c_hpar) + c_spatial )
			nmean.loc[(mean.hpar,)+i_spatial] = mean
			ncov.loc[(cov.hpar0,cov.hpar1)+i_spatial] = cov
			mean = nmean
			cov  = ncov
		
		for n in clim.namesX:
			clim._bias[n] += iclim.bias[n]
		clim._bias[cvar] += bias
		
		## Add to zarr
		zhpar.set_orthogonal_selection( (i,slice(None)) + i_spatial , mean )
		zcov.set_orthogonal_selection( (i,slice(None),slice(None)) + i_spatial , cov )
	
	## Final bias
	for n in clim._names:
		clim._bias[n] /= s_clim
	
	## Now the synthesis
	hpar_S,cov_S = synthesis( zhpar , zcov , bsacParams.n_jobs )
	
	clim.mean_ = hpar_S
	clim.cov_  = cov_S
	clim._time = time
	clim._bper = bper
	bsacParams.clim = clim
##}}}


