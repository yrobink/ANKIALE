
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
from ..stats.__synthesis import synthesis

from ..__exceptions import DevException


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

def zsynthesis( hpars: zr.ZXArray , hcovs: zr.ZXArray ) -> tuple[zr.ZXArray,zr.ZXArray]:##{{{
    
    ssp   = hpars.shape[:-2]
#    nmod  = hpars.shape[-2]
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
def run_ank_cmd_synthesize() -> None:
    
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
    
    ## Set the v parameters
    try:
        vname       = ankParams.config["vname"]
        cname       = ankParams.config["cname"]
        idnslaw     = ankParams.config["nslaw"]
        clim.vconfig._cname = cname
        clim.vconfig._vname = vname
        clim.vconfig.idnslaw = idnslaw
    except:
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
    obias = { n : 0 for n in clim.names }
    
    ## Open all clims, and store in zarr files
    logger.info( " * Open all clims, and store in zarr files" )
    for i,ifile in enumerate(ifiles):
        
        logger.info( f"   => {os.path.basename(ifile)}" )
        
        ## Read clim
        iclim = Climatology.init_from_file(ifile)
        cname = iclim.cname
        time  = iclim.time
        bper  = iclim._bper
        if clim.has_var:
            ibias  = iclim.bias[iclim.vname]
        ihpar  = iclim.hpar.dataarray
        ihcov  = iclim.hcov.dataarray
        
        if regrid:
            logger.info( "    | Regrid" )
            
            ## Grid
            igrid = xr.Dataset( iclim._spatial )
            #rgrd  = xesmf.Regridder( igrid , grid , "bilinear" )
            rgrd  = xesmf.Regridder( igrid , grid , "nearest_s2d" )
            
            ## bias is float
            if isinstance( ibias , float ):
                logger.info( "    | Convert bias float => xarray" )
                ibias = xr.DataArray( [[ibias]] , coords = igrid.coords )
            
            ## Regrid
            logger.info( "    | * Bias with nearest-neighborhood" )
            ibias = rgrd(ibias).where( mask , np.nan )
            
            logger.info( "    | * hpar with nearest-neighborhood" )
            ihpar = rgrd(ihpar).where( mask , np.nan )
            
            logger.info( "    | * hcov with nearest-neighborhood" )
            ihcov = rgrd(ihcov).where( mask , np.nan )
        
        ## Store
        idx0 = tuple([slice(None) for _ in range(len(d_spatial))])
        hpars.loc[(i,ihpar["hpar"])+idx0] = ihpar.values
        hcovs.loc[(i,ihcov["hpar0"],ihcov["hpar1"]) + idx0] = ihcov.values
        
        for n in clim.cnames:
            obias[n] += iclim.bias[n]
        if clim.has_var:
            obias[clim.vname] += ibias
    
    ## Final bias
    logger.info( " * Final bias" )
    for n in clim.names:
        obias[n] /= len(ifiles)
    clim._bias = obias
    
    ## Now the synthesis
    logger.info( " * Run synthesis" )
    if clim.has_spatial:
        hpar_names    = clim.hpar_names
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
    clim._time = time
    clim._bper = bper
    ankParams.clim = clim
##}}}


