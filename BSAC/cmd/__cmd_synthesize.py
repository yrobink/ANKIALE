
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

import zxarray as zr

from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams
from ..__climatology import Climatology
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
	logger.info( " * Read the target grid" )
	try:
		regrid   = True
		gridfile = bsacParams.config.get("grid")
		gridname = bsacParams.config.get("grid_name") #, bsacParams.config["names"].split(":")[-1] )
		
		grid = xr.open_dataset(gridfile)
		mask = grid[gridname] > 0
		clim._spatial = { d : grid[d] for d in bsacParams.config["spatial"].split(":") }
		logger.info( "   => Need regrid" )
	except:
		regrid        = False
		clim._spatial = Climatology.init_from_file( bsacParams.input[0] )._spatial
		logger.info( "   => No regrid needed" )
	
	## Parameters
	logger.info( " * Extract parameters" )
	ifiles      = bsacParams.input
	clim._names = bsacParams.config["names"].split(":")
	try:
		clim._nslawid     = bsacParams.config["nslaw"]
		clim._nslaw_class = nslawid_to_class(clim._nslawid)
	except:
		pass
	hpar_names = clim.hpar_names
	
	## Temporary files
	logger.info( " * Create zxarray files" )
	hpars_coords = { **{ "clim" : range(len(ifiles)) , "hpar"  : hpar_names                        } , **clim.c_spatial }
	hcovs_coords = { **{ "clim" : range(len(ifiles)) , "hpar0" : hpar_names , "hpar1" : hpar_names } , **clim.c_spatial }
	hpars = zr.ZXArray( data = np.nan , coords = hpars_coords )
	hcovs = zr.ZXArray( data = np.nan , coords = hcovs_coords )
	
	##
	clim._bias = { n : 0 for n in clim.names }
	
	## Open all clims, and store in zarr files
	logger.info( " * Open all clims, and store in zarr files" )
	for i,ifile in enumerate(ifiles):
		
		logger.info( f"   => {os.path.basename(ifile)}" )
		
		## Read clim
		iclim = Climatology.init_from_file(ifile)
		time  = iclim.time
		bper  = iclim._bper
		bias  = iclim.bias
		hpar  = iclim.hpar.dataarray
		hcov  = iclim.hcov.dataarray
		
		if regrid:
			logger.info( f"    | Regrid" )
			raise Exception
			## Grid
			igrid     = xr.Dataset( iclim._spatial )
#			regridder = xesmf.Regridder( igrid , grid , "bilinear" )
			regridder = xesmf.Regridder( igrid , grid , "nearest_s2d" )
			
			## Regrid
			bias = regridder(bias_).where( mask , np.nan )
			mean = regridder(mean_).where( mask , np.nan )
			cov  = regridder(cov_ ).where( mask , np.nan )
		
		## Store
		hpars.loc[i,hpar["hpar"]] = hpar.values
		hcovs.loc[i,hcov["hpar0"],hcov["hpar1"]] = hcov.values
		
		for n in clim.namesX:
			clim._bias[n] += iclim.bias[n]
		if not clim.onlyX:
			clim._bias[clim.names[-1]] += iclim.bias[iclim.names[-1]]
	
	## Final bias
	for n in clim.names:
		clim._bias[n] /= len(ifiles)
	
	
	## Now the synthesis
	logger.info( " * Run synthesis" )
	if clim.has_spatial:
		raise Exception("Not implemented error")
#		hpar,hcov = zr.apply_ufunc( synthesis , bdims = clim.d_spatial , max_mem = bsacParams.total_memory )
#		hpar,hcov = synthesis( hpars , hcovs , bsacParams.n_jobs )
#		hpar_S,cov_S = synthesis( zhpar , zcov , bsacParams.n_jobs )
	else:
		hpar,hcov = synthesis( hpars.dataarray , hcovs.dataarray )
	
	logger.info( " * Copy to the clim" )
	clim.hpar = hpar
	clim.hcov = hcov
	clim._time = time
	clim._bper = bper
	bsacParams.clim = clim
##}}}


