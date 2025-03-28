
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

import os
import logging
import itertools as itt

import numpy as np
import xarray as xr
import xesmf

import zxarray as zr

from ..__logs import log_start_end

from ..__ANKParams import ankParams
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

def zsynthesis( hpars , hcovs ):##{{{
	
	ssp   = hpars.shape[:-2]
#	nmod  = hpars.shape[-2]
	nhpar = hpars.shape[-1]
	
	hpar = np.zeros( ssp + (nhpar,)      ) + np.nan
	hcov = np.zeros( ssp + (nhpar,nhpar) ) + np.nan
	
	for idx in itt.product(*[range(s) for s in ssp]):
		idx1d = idx + tuple([slice(None) for _ in range(2)])
		idx2d = idx + tuple([slice(None) for _ in range(3)])
		h  = hpars[idx1d]
		c  = hcovs[idx2d]
		
		h,c = synthesis( h , c )
		hpar[idx1d[:-1]] = h
		hcov[idx2d[:-1]] = c
	
	return hpar,hcov
##}}}

## run_ank_cmd_synthesize ##{{{
@log_start_end(logger)
def run_ank_cmd_synthesize():
	
	##
	clim = ankParams.clim
	
	## Read the grid
	logger.info( " * Read the target grid" )
	try:
		regrid   = True
		gridfile = ankParams.config.get("grid")
		gridname = ankParams.config.get("grid_name") #, ankParams.config["names"].split(":")[-1] )
		
		grid = xr.open_dataset(gridfile)
		mask = grid[gridname] > 0
		clim._spatial = { d : grid[d] for d in ankParams.config["spatial"].split(":") }
		logger.info( "   => Need regrid" )
	except Exception:
		regrid        = False
		clim._spatial = Climatology.init_from_file( ankParams.input[0] )._spatial
		logger.info( "   => No regrid needed" )
	
	## Parameters
	logger.info( " * Extract parameters" )
	ifiles      = ankParams.input
	clim._names = ankParams.config["names"].split(":")
	try:
		clim._nslawid     = ankParams.config["nslaw"]
		clim._nslaw_class = nslawid_to_class(clim._nslawid)
	except Exception:
		pass
	hpar_names = clim.hpar_names
	d_spatial = clim.d_spatial
	c_spatial = clim.c_spatial
	
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
		cname = iclim.cname
		time  = iclim.time
		bper  = iclim._bper
		if not clim.onlyX:
			bias  = iclim.bias[iclim.vname]
		hpar  = iclim.hpar.dataarray
		hcov  = iclim.hcov.dataarray
		
		if regrid:
			logger.info( "    | Regrid" )
			
			## Grid
			igrid  = xr.Dataset( iclim._spatial )
			try:
				rgrd2d = xesmf.Regridder( igrid , grid , "bilinear" )
			except Exception:
				rgrd2d = None
			rgrdnn = xesmf.Regridder( igrid , grid , "nearest_s2d" )
			
			## bias is float
			if isinstance( bias , float ):
				logger.info( "    | Convert bias float => xarray" )
				bias = xr.DataArray( [[bias]] , coords = igrid.coords )
			
			## Regrid
			try:
				logger.info( "    | * Bias with bilinear..." )
				raise Exception
				bias = rgrd2d(bias).where( mask , np.nan )
			except Exception:
				logger.info( "    | * Bias with nearest-neighborhood..." )
				bias = rgrdnn(bias).where( mask , np.nan )
			try:
				logger.info( "    | * hpar with bilinear..." )
				raise Exception
				hpar = rgrd2d(hpar).where( mask , np.nan )
			except Exception:
				hpar = rgrdnn(hpar).where( mask , np.nan )
				logger.info( "    | * hpar with nearest-neighborhood..." )
			logger.info( "    | * hcov with nearest-neighborhood" )
			hcov = rgrdnn(hcov).where( mask , np.nan )
		
		## Store
		idx0 = tuple([slice(None) for _ in range(len(d_spatial))])
		hpars.loc[(i,hpar["hpar"])+idx0] = hpar.values
		hcovs.loc[(i,hcov["hpar0"],hcov["hpar1"]) + idx0] = hcov.values
		
		for n in clim.namesX:
			clim._bias[n] += iclim.bias[n]
		if not clim.onlyX:
			clim._bias[clim.vname] += bias
	
	## Final bias
	for n in clim.names:
		clim._bias[n] /= len(ifiles)
	
	## Now the synthesis
	logger.info( " * Run synthesis" )
	if clim.has_spatial:
		hpar_names    = clim.hpar_names
#		nhpar_names   = len(hpar_names)
		output_dims   = [("hpar",) + d_spatial,("hpar0","hpar1") + d_spatial]
		output_coords = [[hpar_names] + [ c_spatial[d] for d in d_spatial ],[hpar_names,hpar_names] + [ c_spatial[d] for d in d_spatial ]]
		output_dtypes = [hpars.dtype,hpars.dtype]
		dask_kwargs   = { "input_core_dims"  : [ ["clim","hpar"] , ["clim","hpar0","hpar1"] ],
		                  "output_core_dims" : [ ["hpar"],["hpar0","hpar1"] ],
		                  "kwargs" : {},
		                  "dask" : "parallelized",
		                  "output_dtypes"  : [hpars.dtype,hpars.dtype]
		                    }
		
		## Block memory function
		nhpar = len(hpar_names)
		block_memory = lambda x : 5 * ( len(ifiles) * (nhpar + nhpar**2) + nhpar + nhpar**2 ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
		
		## Run
		with ankParams.get_cluster() as cluster:
			hpar,hcov = zr.apply_ufunc( zsynthesis , hpars , hcovs, 
			                            block_dims         = d_spatial,
			                            total_memory       = ankParams.total_memory,
			                            block_memory       = block_memory,
			                            output_coords      = output_coords,
			                            output_dims        = output_dims,
			                            output_dtypes      = output_dtypes,
			                            dask_kwargs        = dask_kwargs,
			                            n_workers          = ankParams.n_workers,
			                            threads_per_worker = ankParams.threads_per_worker,
			                            cluster            = cluster,
			                            )
	else:
		hpar,hcov = synthesis( hpars.dataarray , hcovs.dataarray )
	
	logger.info( " * Copy to the clim" )
	clim.hpar = hpar
	clim.hcov = hcov
	clim.cname = cname
	clim._time = time
	clim._bper = bper
	ankParams.clim = clim
##}}}


