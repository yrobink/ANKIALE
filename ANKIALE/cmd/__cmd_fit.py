
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
from ..__logs import log_start_end

from ..__ANKParams import ankParams

from ..stats.__MultiGAM import MPeriodSmoother
from ..stats.__NSLawMLEFit import nslaw_fit
from ..__sys import coords_samples

from ..__exceptions import DevException
from ..__linalg import mean_cov_hpars

import numpy  as np
import xarray as xr
import zxarray as zr


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## run_ank_cmd_fit_X ##{{{
@log_start_end(logger)
def run_ank_cmd_fit_X() -> None:
    
    ##
    clim = ankParams.clim
    
    ## Check the inputs
    logger.info("Check inputs")
    n_X   = len(ankParams.input)
    if n_X == 0:
        raise ValueError("Fit asked, but no input given, abort.")
    inputs = { inp.split(",")[0] : inp.split(",")[1] for inp in ankParams.input }
    for cname in inputs:
        if not os.path.isfile(inputs[cname]):
            raise FileNotFoundError(f"File '{inputs[cname]}' is not found, abort.")
        if cname not in clim.cnames:
            raise ValueError(f"{cname} is not given in the list of variable of the climatology")
    
    ## Parameters
    cnames  = clim.cnames
    periods = clim.dpers
    time    = clim.time
    
    ## Now open the data
    logger.info("Open the data")
    dX = xr.DataArray( dims = ["name","period","time"],
                      coords = [cnames,periods,time]
                      )
    for cname in cnames:
        idata    = xr.open_dataset( inputs[cname] )[cname].mean( dim = "run" )
        for per in clim.cper + clim.dpers:
            if per not in idata.period:
                raise ValueError(f"Period {per} not found in input data {inputs[cname]}, abort")
        for per in periods:
            ## Extract and store
            X = idata.sel( period = ankParams.cper + [per] ).mean( dim = "period" ).sel( time = slice(time[0],time[-1]) )
            dX.loc[cname,per,X.time] = X
            
            ## Re-extract the TOTAL time axis to check if all values are valid
            X = dX.loc[cname,per,:]
            if np.isfinite(X).all():
                continue
            nantime = X.time[~np.isfinite(X)].values
            for t in nantime:
                m = X.sel( time = slice( t - 15 , t + 15 ) ).mean()
                s = X.sel( time = slice( t - 15 , t + 15 ) ).std()
                m = float(m)
                s = float(s)
                if not np.isfinite([m,s]).all():
                    raise ValueError("Impossible to remove nan values, abort.")
                X.loc[t] = float(np.random.normal( loc = m , scale = s , size = 1 ))
            dX.loc[cname,per:] = X
    
    ## Find the bias
    logger.info( "Build bias" )
    bias = dX.sel( time = slice(*clim.bper) ).mean( dim = ["period","time"] )
    dX   = dX - bias
    clim._bias = { cname : float(bias.loc[cname].values) for cname in cnames }
    for cname in cnames:
        logger.info( f" * {cname}: {clim._bias[cname]}" )
    
    ## Find natural forcings version
    clim._vXN = ankParams.XN_version
    
    ## Create smoother
    logger.info( "Create smoother" )
    mps = MPeriodSmoother(
        XN = clim.XN,
        total_dof = clim.cconfig.total_dof,
        n_spl_basis = clim.cconfig.nknot,
        degree = clim.cconfig.degree
    )
    
    ## Fit
    logger.info( "Fit the MultiGAM model" )
    hpar,hcov = mps.fit(dX)
    
    if clim.vconfig.is_init:
        vhpar = xr.DataArray( np.nan, dims = ["hpar"] , coords = [clim.hpar_names] )
        vhcov = xr.DataArray( np.nan, dims = ["hpar0","hpar1"] , coords = [clim.hpar_names,clim.hpar_names] )
        vhpar.loc[hpar.hpar] = hpar
        vhcov.loc[hcov.hpar0,hcov.hpar1] = hcov
        hpar = vhpar
        hcov = vhcov

    ## Store
    logger.info( "Save in clim" )
    clim.hpar = hpar
    clim.hcov = hcov

    ##
    ankParams.clim = clim
    
##}}}

## run_ank_cmd_fit_Y ##{{{
@log_start_end(logger)
def run_ank_cmd_fit_Y() -> None:
    
    ## The current climatology
    clim = ankParams.clim
    logger.info(" * Find parameters")
    
    ## Name of the variable to fit
    vname = clim.vname
    cname = clim.cname
    
    ## Spatial dimensions
    d_spatial = ankParams.spatial
    if d_spatial is not None:
        d_spatial = d_spatial.split(":")
    else:
        d_spatial = []
    d_spatial = tuple(d_spatial)

    ## Open the data
    logger.info(" * Open data")
    ifile = ankParams.input[0]
    idata = xr.open_dataset(ifile).load()
    
    ## Check periods
    for p in clim.dpers:
        if not p in idata.period.values:
            raise ValueError(f"Periods {p} not found in {ifile}")
    idata = idata.sel( period = clim.cper + clim.dpers )
    
    ## Check if variables in idata
    for v in (vname,) + d_spatial:
        if v not in idata:
            raise ValueError( f"Variable '{v}' not in input data" )
    
    ## Spatial coordinates
    c_spatial = { d : idata[d] for d in d_spatial }
    
    ## Find the bias, and remove it
    logger.info(" * Compute bias")
    Y     = idata[vname]
    biasY = Y.sel( time = slice(*clim.bias_period) ).mean( dim = [d for d in Y.dims if d not in d_spatial] )
    Y     = Y - biasY
    try:
        biasY = float(biasY)
    except Exception:
        pass
    
    ## Force to add a spatial dimension
    if len(d_spatial) == 0:
        d_spatial = ("fake",)
        c_spatial = { "fake" : xr.DataArray( [0] , dims = ["fake"] , coords = [[0]] ) }
        Y         = xr.DataArray( Y.values.reshape( Y.shape + (1,) ) , dims = Y.dims + ("fake",) , coords = { **c_spatial , **dict(Y.coords) } )
        biasY     = xr.DataArray( [biasY] , dims = ("fake",) , coords = c_spatial )

    ## Transform in ZXArray
    zY = zr.ZXArray.from_xarray(Y)
    
    ## Find the nslaw
    cnslaw = clim.vconfig.cnslaw
    
    ## Design matrix of the covariate
    chpar_names = clim.chpar_names
    vhpar_names = clim.vhpar_names
    hpar_names  = clim.hpar_names
    projF,_ = clim.projection()
    projF = projF.sel( hpar = chpar_names )
    
    ## Time axis
    zY   = Y.sel( time = slice(clim.time[0],clim.time[-1]) )
    projF = projF.sel( time = zY["time"].values )
    
    ## Samples
    n_samples = ankParams.n_samples
    samples   = coords_samples(n_samples)
    samples   = zr.ZXArray.from_xarray( xr.DataArray( range(n_samples) , dims = ["sample"] , coords = [samples] ).astype(float) )
    
    ## zxarray.apply_ufunc parameters
    output_dims      = [ ("hparY","dperiod","sample") + d_spatial ]
    output_coords    = [ [hpar_names,clim.dpers,samples.dataarray] + [ c_spatial[d] for d in d_spatial ] ]
    output_dtypes    = [ float ]
    dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] , ["time","period","run"] , []],
                         "output_core_dims" : [ ["hparY","dperiod"] ],
                         "kwargs" : { "cnslaw" : cnslaw , "proj" : projF , "cname" : cname },
                         "dask" : "parallelized",
                         "dask_gufunc_kwargs" : { "output_sizes" : { "hparY" : len(hpar_names) , "dperiod" : len(clim.dpers) } },
                         "output_dtypes"  : [clim.hpar.dtype]
                        }
    
    ## Block memory function
    nhpar = len(hpar_names)
    block_memory = lambda x : 2 * ( nhpar + nhpar**2 + clim.time.size * (clim.ndpers+1) * Y["run"].size + nhpar * clim.ndpers ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
    
    hpar  = clim.hpar.dataarray.loc[chpar_names]
    hcov  = clim.hcov.dataarray.loc[chpar_names,chpar_names]
    hpar  = zr.ZXArray.from_xarray(hpar)
    hcov  = zr.ZXArray.from_xarray(hcov)

    ## Fit samples of parameters
    logger.info(" * Fit samples")
    with ankParams.get_cluster() as cluster:
        hpars = zr.apply_ufunc( nslaw_fit , hpar , hcov , zY, samples,
                                block_dims         = ("sample",) + d_spatial,
                                block_memory       = block_memory,
                                total_memory       = ankParams.total_memory,
                                output_dims        = output_dims,
                                output_coords      = output_coords,
                                output_dtypes      = output_dtypes,
                                dask_kwargs        = dask_kwargs,
                                n_workers          = ankParams.n_workers,
                                threads_per_worker = ankParams.threads_per_worker,
                                cluster            = cluster,
                                )
    
    ## And find parameters of the distribution
    output_dims      = [ ("hpar",) + d_spatial , ("hpar0","hpar1") + d_spatial ]
    output_coords    = [ [hpar_names] + [ c_spatial[d] for d in d_spatial ] , [hpar_names,hpar_names] + [ c_spatial[d] for d in d_spatial ] ]
    output_dtypes    = [float,float]
    dask_kwargs      = { "input_core_dims"  : [ ["hparY","dperiod","sample"]],
                         "output_core_dims" : [ ["hpar"] , ["hpar0","hpar1"] ],
                         "kwargs" : {},
                         "dask" : "parallelized",
                         "dask_gufunc_kwargs" : { "output_sizes" : { "hpar" : len(hpar_names) , "hpar0" : len(hpar_names) , "hpar1" : len(hpar_names) } },
                         "output_dtypes"  : [hpars.dtype,hpars.dtype]
                        }
    
    ## Block memory function
    block_memory = lambda x : 2 * ( nhpar * clim.ndpers * n_samples + nhpar + nhpar**2 ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
    
    ## Apply
    logger.info(" * Find hpar and hcov from samples")
    with ankParams.get_cluster() as cluster:
        hpar,hcov = zr.apply_ufunc( mean_cov_hpars , hpars,
                                    block_dims         = d_spatial,
                                    total_memory       = ankParams.total_memory,
                                    block_memory       = block_memory,
                                    output_dims        = output_dims,
                                    output_coords      = output_coords,
                                    output_dtypes      = output_dtypes,
                                    dask_kwargs        = dask_kwargs,
                                    n_workers          = ankParams.n_workers,
                                    threads_per_worker = ankParams.threads_per_worker,
                                    cluster            = cluster,
                                    )
    
    ## Update the climatology
    logger.info(" * Update clim")
    clim.hpar  = hpar
    clim.hcov  = hcov
    clim._bias[vname]  = biasY
    clim._spatial = c_spatial
    
##}}}

## run_ank_cmd_fit ##{{{
@log_start_end(logger)
def run_ank_cmd_fit() -> None:
    
    ## Check the command
    if not len(ankParams.arg) == 1:
        raise ValueError(f"Bad numbers of arguments of the fit command: {', '.join(ankParams.arg)}")
    
    available_commands = ["X","Y"]
    if ankParams.arg[0] not in available_commands:
        raise ValueError(f"Bad argument of the fit command ({ankParams.arg[0]}), must be: {', '.join(available_commands)}")
    
    ## OK, run the good command
    if ankParams.arg[0] == "X":
        run_ank_cmd_fit_X()
    if ankParams.arg[0] == "Y":
        run_ank_cmd_fit_Y()
##}}}


