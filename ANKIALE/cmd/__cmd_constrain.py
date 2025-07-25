
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
import itertools as itt
import gc
import distributed
import netCDF4

import numpy  as np
import xarray as xr
import zxarray as zr

from ..__ANKParams import ankParams

from ..__exceptions import DevException

from ..__logs import log_start_end
from ..__sys     import coords_samples

from ..stats.__MultiGAM   import MPeriodSmoother
from ..stats.__constraint import build_projection_matrix
from ..stats.__constraint import constraint_covar
from ..stats.__constraint import mcmc

from ..__linalg import mean_cov_hpars


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

def zconstraint_covar( *args: np.ndarray , P: np.ndarray | None = None , timeXo: np.ndarray | None = None , method_oerror: str | None = None ) -> tuple[np.ndarray,np.ndarray]:##{{{
    
    ihpar = args[0]
    ihcov = args[1]
    lXo   = args[2:]
    ssp   = ihpar.shape[:-1]
    
    ohpar = np.zeros_like(ihpar) + np.nan
    ohcov = np.zeros_like(ihcov) + np.nan
    
    for idx in itt.product(*[range(s) for s in ssp]):
        idx1d = idx + tuple([slice(None) for _ in range(1)])
        idx2d = idx + tuple([slice(None) for _ in range(2)])
        ih    = ihpar[idx1d]
        ic    = ihcov[idx2d]
        iargs = [ih,ic] + [ Xo[idx1d] for Xo in lXo ]
        
        oh,oc = constraint_covar( *iargs , P = P , timeXo = timeXo , method_oerror = method_oerror )
        ohpar[idx1d] = oh
        ohcov[idx2d] = oc
    
    return ohpar,ohcov
##}}}

## run_ank_cmd_constrain_X ##{{{
@log_start_end(logger)
def run_ank_cmd_constrain_X() -> None:
    
    ## Parameters
    clim = ankParams.clim
    d_spatial = clim.d_spatial
    c_spatial = clim.c_spatial
    tleft  = str(int(clim.time[ 0]))
    tright = str(int(clim.time[-1]))

    ## Load observations
    zXo = {}
    for i,inp in enumerate(ankParams.input):
        
        ## Name and file
        name,ifile = inp.split(",")
        if name not in clim.cnames:
            raise ValueError( f"Unknown variable {name}" )
        
        ## Open data
        idata = xr.open_dataset(ifile).sel( time = slice(tleft,tright) )
        
        ## Time axis
        time = idata.time.dt.year.values
        time = xr.DataArray( time , dims = [f"time{i}"] , coords = [time] )
        
        ## Init zarr file
        dims   = (f"time{i}",) + d_spatial
        coords = [time] + [c_spatial[d] for d in d_spatial]
        Xo     = zr.ZXArray( data = np.nan , dims = dims , coords = coords )
        
        ## Now copy data
        ndimXo = len([s for s in idata[name].shape if s > 1])
        bias   = xr.DataArray( np.nan , dims = d_spatial , coords = c_spatial )
        if bias.ndim == 0:
            bias = xr.DataArray( [np.nan] )
        if ndimXo == 1:
            xXo  = idata[name]
            anom = float(xXo.sel( time = slice(*[str(y) for y in clim.bper]) ).mean( dim = "time" ))
            xXo  = xXo.values.squeeze() - anom
            bias[:] = anom
            for idx in itt.product(*[range(s) for s in clim.s_spatial]):
                Xo[(slice(None),) + idx] = xXo
        else:
            raise NotImplementedError
        clim._bias[name] = bias
        
        ## Store zarr file
        zXo[name] = Xo

    ## Method to find natural variability of obs
    method_oerror = ankParams.config.get("method_oerror","IND")
    if method_oerror not in ["IND","MAR2","KCC"]:
        raise ValueError(f"Observation constraint method must be one of 'IND', 'MAR2' or 'KCC' ({method_oerror} is given)")
    if method_oerror == "KCC" and not len(zXo) == 2:
        logging.warning("'KCC' method can not be used if covariate number '{len(zXo)' != 2. Use 'MAR2' method.")
        method_oerror = 'MAR2'
    logger.info( f"Observation error method used: {method_oerror}" )

    ## And find configuration
    method_constraint = ankParams.config.get("method_constraint")
    if method_constraint is None:
        method_constraint = { cname : "full" for cname in zXo }
    else:
        method_constraint = { cname : goal.lower() for cname,goal in [s.split(":") for s in method_constraint.split("::")] }
        for cname in zXo:
            if cname not in method_constraint:
                method_constraint[cname] = "full"
            if method_constraint[cname] not in ["full"] + clim.dpers:
                raise ValueError(f"Constraint method not coherent: {cname} / {method_constraint['cname']}")
                
    logger.info( f"Constraint method used: {method_constraint}")
    
    ## Init smoother matrix for projection
    time = clim.time
    nper = len(clim.dpers)
    mps = MPeriodSmoother(
        XN = clim.XN,
        cnames = clim.cnames,
        dpers = clim.dpers,
        spl_config = clim.cconfig.spl_config
    )
    P = build_projection_matrix( mps , zXo , method_constraint )
    
    ## Add to projection matrix the design part for variable
    if clim.has_var:
        vP = np.zeros( (P.shape[0],clim.vsize) )
        P  = np.hstack( (P,vP) )

    ## Build apply arguments
    hpar_names       = clim.hpar_names
    c_spatial        = clim.c_spatial
    d_spatial        = clim.d_spatial
    args             = [clim.hpar,clim.hcov] + [ zXo[name] for name in zXo ]
    output_dims      = [ ("hpar",) + d_spatial   , ("hpar0","hpar1") + d_spatial ]
    output_coords    = [ [hpar_names] + [ c_spatial[d] for d in d_spatial ] , [hpar_names,hpar_names] + [ c_spatial[d] for d in d_spatial ] ]
    output_dtypes    = [float,float]
    dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] ] + [ [f"time{i}"] for i in range(len(zXo)) ],
                         "output_core_dims" : [ ["hpar"] , ["hpar0","hpar1"] ],
                     "kwargs" : { "P" : P , "timeXo" : [zXo[name][f"time{iname}"] for iname,name in enumerate(zXo)] , "method_oerror" : method_oerror } ,
                         "dask" : "parallelized",
                         "output_dtypes"  : [clim.hpar.dtype,clim.hcov.dtype]
                        }
    
    ## Block memory function
    nhpar = len(clim.hpar_names)
    block_memory = lambda x : 2 * ( nhpar + nhpar**2 +  len(zXo) * time.size + nhpar + nhpar**2 ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
    
    ##
    with ankParams.get_cluster() as cluster:
        hpar,hcov = zr.apply_ufunc( zconstraint_covar , *args ,
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
    
    ## Save
    clim.hpar = hpar
    clim.hcov = hcov
    ankParams.clim = clim
##}}}


def zmcmc( ihpar , ihcov , Yo , samples , A , size_chain , nslaw_class , use_STAN , tmp ):##{{{
    
    ssp    = ihpar.shape[:-2]
    nhpar  = ihpar.shape[-1]
    hpars  = np.zeros( ssp + (samples.size,nhpar,size_chain) ) + np.nan
    
    for idx in itt.product(*[range(s) for s in ssp]):
        for s in range(samples.size):
            
            ## Extract
            idx1d = idx + (0,) + tuple([slice(None) for _ in range(1)])
            idx2d = idx + (0,) + tuple([slice(None) for _ in range(2)])
            ih    = ihpar[idx1d]
            ic    = ihcov[idx2d]
            iYo   = Yo[idx1d]
            
            if np.any(~np.isfinite(ih)) or np.any(~np.isfinite(ic)):
                if np.all(~np.isfinite(ih)) or np.all(~np.isfinite(ic)):
                    continue
                else:
                    raise ValueError("hpar or hcov partially not finite in mcmc")
            
            if np.all(~np.isfinite(iYo)):
                continue
            
            ## MCMC
            oh = mcmc( ih , ic , iYo , A , size_chain , nslaw_class , use_STAN , tmp )
            if not np.isfinite(oh).all():
                distributed.print("Fail MCMC")
            
            ## Store
            idx2d = idx + (s,) + tuple([slice(None) for _ in range(2)])
            hpars[idx2d] = oh
    
    return hpars
##}}}

## run_ank_cmd_constrain_Y ##{{{
@log_start_end(logger)
def run_ank_cmd_constrain_Y():
    
    ##
    clim = ankParams.clim
    size_chain = int(ankParams.config.get("size-chain", 10))
    
    ## Load observations
    name,ifile = ankParams.input[0].split(",")
    Yo = xr.open_dataset(ifile)[name]
    if clim.spatial_is_fake:
        Yo = xr.DataArray( Yo.values.reshape(-1,1) , dims = ("time",) + clim.d_spatial , coords = { **{ "time" : Yo.time.dt.year.values } , **clim.c_spatial } )
    else:
        Yo = xr.DataArray( Yo.values , dims = Yo.dims , coords = [Yo.time.dt.year.values] + [Yo.coords[d] for d in Yo.dims[1:]] )
    
    ## Bias
    bias = Yo.sel( time = slice(*[str(y) for y in clim.bper]) ).mean( dim = "time" )
    Yo   = Yo - bias
    try:
        bias = float(bias)
    except Exception:
        pass
    if clim.spatial_is_fake:
        bias = xr.DataArray( [bias] , dims = clim.d_spatial , coords = clim.c_spatial )
    clim._bias[clim.vname] = bias
    
    ## Transform in ZXArray
    zYo = zr.ZXArray.from_xarray(Yo)
    
    ## Extract parameters
    use_STAN   = not ankParams.no_STAN
    d_spatial  = clim.d_spatial
    c_spatial  = clim.c_spatial
    hpar_names = clim.hpar_names
    ihpar      = clim.hpar
    ihcov      = clim.hcov
    
    ## Samples
    n_samples = ankParams.n_samples
    samples   = coords_samples(n_samples)
    zsamples  = zr.ZXArray.from_xarray( xr.DataArray( range(n_samples) , dims = ["sample"] , coords = [samples] ).astype(float) )
    
    ##
    chains = range(size_chain)
    
    ## Build projection operator for the covariable
#    time = clim.time
#    spl,lin,_,_ = clim.build_design_XFC()
#    nper = len(clim.dpers)
#    
#    design_ = []
#    for nameX in clim.namesX:
#        if nameX == clim.cname:
#            design_ = design_ + [spl for _ in range(nper)] + [nper * lin]
#        else:
#            design_ = design_ + [np.zeros_like(spl) for _ in range(nper)] + [np.zeros_like(lin)]
#    design_ = design_ + [np.zeros( (time.size,clim.sizeY) )]
#    design_ = np.hstack(design_)
#    
#    T = xr.DataArray( np.identity(design_.shape[0]) , dims = ["timeA","timeB"] , coords = [time,time] ).loc[Yo.time,time].values
#    A = T @ design_ / nper
    time = clim.time
    lin,spl = clim.build_design_basis()
    nper = len(clim.dpers)
    
    ## Build the design_matrix for projection
    design_ = []
    for nameX in clim.namesX:
        if nameX == clim.cname:
            design_ = design_ + [spl[nameX][clim.dpers[iper]] for iper in range(nper)] + [nper * lin]
        else:
            design_ = design_ + [np.zeros_like(spl[nameX][clim.dpers[iper]]) for iper in range(nper)] + [np.zeros_like(lin)]
    design_ = design_ + [np.zeros( (time.size,clim.sizeY) )]
    design_ = np.hstack(design_)
    
    
    T = xr.DataArray( np.identity(design_.shape[0]) , dims = ["timeA","timeB"] , coords = [time,time] ).loc[Yo.time,time].values
    A = T @ design_ / nper
    
    ##
    nslaw_class = clim._nslaw_class
    
    ## Init stan
    if use_STAN:
        logger.info(" * STAN compilation...")
        nslaw_class().init_stan( tmp = ankParams.tmp_stan , force_compile = True )
        logger.info(" * STAN compilation... Done.")
    
    ## Apply parameters
    output_dims      = [ ("hpar","sample","chain") + d_spatial   ]
    output_coords    = [ [hpar_names,samples,chains] + [ c_spatial[d] for d in d_spatial ] ]
    output_dtypes    = [float]
    dask_kwargs      = { "input_core_dims"  : [ ["hpar"] , ["hpar0","hpar1"] , ["time"] , [] ],
                         "output_core_dims" : [ ["hpar","chain"] ],
                         "kwargs" : { "A" : A , "size_chain" : size_chain , "nslaw_class" : nslaw_class , "use_STAN" : use_STAN , "tmp" : ankParams.tmp_stan } ,
                         "dask" : "parallelized",
                         "dask_gufunc_kwargs" : { "output_sizes" : { "chain" : size_chain } },
                         "output_dtypes"  : [clim.hpar.dtype]
                        }
    
    ## Block memory function
    nhpar = len(clim.hpar_names)
    block_memory = lambda x : 5 * ( nhpar + nhpar**2 + time.size + nhpar * time.size + 1 + nhpar * size_chain ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
    
    ## Draw samples
    logger.info(" * Draw samples")
    with ankParams.get_cluster() as cluster:
        ohpars = zr.apply_ufunc( zmcmc , ihpar , ihcov , zYo , zsamples ,
                                 block_dims         = d_spatial + ("sample",),
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
    
    ## Clean memory
    gc.collect()
    
    ## And find parameters of the distribution
    logger.info(" * Compute mean and covariance of parameters")
    output_dims      = [ ("hpar",) + d_spatial , ("hpar0","hpar1") + d_spatial ]
    output_coords    = [ [hpar_names] + [ c_spatial[d] for d in d_spatial ] , [hpar_names,hpar_names] + [ c_spatial[d] for d in d_spatial ] ]
    output_dtypes    = [float,float]
    dask_kwargs      = { "input_core_dims"  : [ ["hpar","sample","chain"]],
                         "output_core_dims" : [ ["hpar"] , ["hpar0","hpar1"] ],
                         "kwargs" : {},
                         "dask" : "parallelized",
                         "dask_gufunc_kwargs" : { "output_sizes" : { "hpar0" : len(hpar_names) , "hpar1" : len(hpar_names) } },
                         "output_dtypes"  : [ohpars.dtype,ohpars.dtype]
                        }
    
    ## Block memory function
    block_memory = lambda x : 10 * ( (nhpar + nhpar**2) * n_samples * size_chain + nhpar + nhpar**2 ) * np.prod(x) * (np.finfo("float32").bits // zr.DMUnit.bits_per_octet) * zr.DMUnit("1o")
    
    ## Apply
    with ankParams.get_cluster() as cluster:
        hpar,hcov = zr.apply_ufunc( mean_cov_hpars , ohpars,
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
                                    chunks             = { d : 1 for d in d_spatial },
                                    )
    
    ## Clean memory
    gc.collect()
    
    ## Store (or not) the samples
    if ankParams.output is not None:
        logger.info(" * Store samples on the disk")
        
        names = lambda n : "hyper_parameter" if n == "hpar" else n
        with netCDF4.Dataset( ankParams.output , "w" ) as oncf:
            
            odims  = []
            oshape = []
            for d,s in zip(ohpars.dims,ohpars.shape):
                if not d == "fake":
                    odims.append(d)
                    oshape.append(s)
            
            ## Create dimensions
            ncdims = { names(d) : oncf.createDimension( names(d) , s ) for d,s in zip(odims,oshape) }
            
            ## Create variables
            ncvars = { names(d) : oncf.createVariable( names(d) , ohpars.coords[d].dtype , (names(d),) ) for d in odims }
            ncvars["hpars"] = oncf.createVariable( "hpars" , ohpars.dtype , [names(d) for d in odims] )
            
            ## And fill
            for d in odims:
                ncvars[names(d)][:] = ohpars.coords[d].values[:]
            idx = [slice(None) for _ in range(len(odims))]
            if clim.spatial_is_fake:
                idx.append(0)
            idx = tuple(idx)
            ncvars["hpars"][:] = ohpars._internal.zdata.get_orthogonal_selection(idx)
    
    ## And save
    clim.hpar = hpar
    clim.hcov = hcov
    ankParams.clim = clim
##}}}


## run_ank_cmd_constrain ##{{{
@log_start_end(logger)
def run_ank_cmd_constrain() -> None:
    ## Check the command
    if not len(ankParams.arg) == 1:
        raise ValueError(f"Bad numbers of arguments of the fit command: {', '.join(ankParams.arg)}")
    
    available_commands = ["X","Y"]
    if ankParams.arg[0] not in available_commands:
        raise ValueError(f"Bad argument of the fit command ({ankParams.arg[0]}), must be: {', '.join(available_commands)}")
    
    ## OK, run the good command
    if ankParams.arg[0] == "X":
        run_ank_cmd_constrain_X()
    if ankParams.arg[0] == "Y":
        run_ank_cmd_constrain_Y()
##}}}

