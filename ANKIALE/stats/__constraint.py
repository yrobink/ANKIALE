
## Copyright(c) 2024, 2025 Yoann Robin
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

##############
## Packages ##
##############


#############
## Imports ##
#############

import logging

import numpy as np
import scipy.stats as sc
import xarray as xr
import zxarray as zr

from typing import Sequence
from typing import Any

from .__MultiGAM import MPeriodSmoother

from .__KCC import KCC
from .__KCC import MAR2

from ..__sys import Error
from ..__exceptions import StanError
from ..__exceptions import StanInitError
from ..__exceptions import MCMCError
from ..__exceptions import DevException
from ..__logs import disable_warnings

from .models.__AbstractModel import AbstractModel


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#############
## Classes ##
#############


###############
## Functions ##
###############

def build_projection_matrix( mps: MPeriodSmoother , zXo: dict[str,zr.ZXArray] , method_constraint: dict | None = None ):##{{{

    time   = mps.time
    cnames = mps.cnames
    dpers  = mps.dpers
    lin    = mps.lin
    SB0    = mps.SB0
    nper = len(dpers)

    if method_constraint is None:
        method_constraint = { cname : "full" for cname in zXo }

    drP = {}
    for icname,cname in enumerate(zXo):
        cst = nper if method_constraint[cname] == "full" else 1
        rP = [lin]

        for per in dpers:
            if method_constraint[cname] == "full":
                rP.append(SB0 / cst)
            else:
                rP.append( (per == method_constraint[cname]) * SB0 )
        idxt = [ t in zXo[cname][f'time{icname}'].values for t in time]
        drP[cname] = np.hstack(rP)[idxt,:]

    P = []
    for cname0 in zXo:
        row = []
        for cname1 in cnames:
            row.append( drP[cname0] * (cname0 == cname1))
        P.append( np.hstack(row) )
    P = np.vstack(P)

    return P
##}}}


def infer_hcov_o_IND( Ros: Sequence[xr.DataArray] , size: int ) -> np.ndarray:##{{{
    hcov_o = np.zeros((size,size))
    b = 0
    for Ro in Ros:
        e = b + Ro.size
        hcov_o[b:e,b:e] = np.diag( np.ones(Ro.size) * float(np.std(Ro)**2))
        b += Ro.size
    
    return hcov_o
##}}}

def infer_hcov_o_MAR2( Ros: Sequence[xr.DataArray] , size: int ) -> np.ndarray:##{{{
    hcov_o = np.zeros((size,size))
    b = 0
    for Ro in Ros:
        e = b + Ro.size
        hcov_o[b:e,b:e] = MAR2.fit( Ro ).cov(Ro.size)
        b += Ro.size
    
    return hcov_o
##}}}

def infer_hcov_o_KCC( Ros: Sequence[xr.DataArray] , size: int ) -> np.ndarray:##{{{
    
    hcov_o_meas0 = 0
    hcov_o_meas1 = 0
    kcc          = KCC().fit( Ros[0] , Ros[1] )
    hcov_o_iv0   = kcc.cov_iv0
    hcov_o_iv1   = kcc.cov_iv1
    hcov_o_iv01  = kcc.cov_iv01
    hcov_o       = np.block( [ [hcov_o_meas0 + hcov_o_iv0 , hcov_o_iv01  ],
                               [hcov_o_iv01.T , hcov_o_meas1 + hcov_o_iv1] ] )
    
    return hcov_o
##}}}

def _infer_hcov_o( hpar: np.ndarray , hcov: np.ndarray , Xos: Sequence[xr.DataArray] , P: np.ndarray , method_oerror: str = 'IND' ) -> np.ndarray:##{{{

    ## Find individual residuals
    X = P @ hpar
    b = 0
    Ros = []
    for Xo in Xos:
        e = b + Xo.size
        Ros.append( Xo.values - X[b:e] )
        b += Xo.size
    hcov_o = np.zeros((b,b))

    match method_oerror.upper():
        case 'IND':
            hcov_o = infer_hcov_o_IND( Ros , b )
        case 'MAR2':
            hcov_o = infer_hcov_o_MAR2( Ros , b )
        case 'KCC':
            hcov_o = infer_hcov_o_KCC( Ros , b )
        case _:
            raise ValueError("Bad observed error method")

    return hcov_o
##}}}

def infer_hcov_o( hpar: np.ndarray , hcov: np.ndarray , Xos: Sequence[xr.DataArray] , P: np.ndarray , method_oerror: str = 'IND' ) -> np.ndarray:##{{{
    try:
        hcov_o = _infer_hcov_o( hpar , hcov , Xos , P , method_oerror )
    except Exception as e:
        match method_oerror:
            case "KCC":
                logger.warning("Fail to use KCC, back to MAR2")
                hcov_o = infer_hcov_o( hpar , hcov , Xos , P , "MAR2" )
            case "MAR2":
                logger.warning("Fail to use MAR2, back to IND")
                hcov_o = infer_hcov_o( hpar , hcov , Xos , P , "IND" )
            case _:
                raise e
    return hcov_o
##}}}


def gaussian_conditionning( hpar: np.ndarray , hcov: np.ndarray , P: np.ndarray , Xo: np.ndarray , hcov_o: np.ndarray ) -> tuple[np.ndarray,np.ndarray]:##{{{
    K0    = P @ hcov
    K1    = ( hcov @ P.T ) @ np.linalg.inv( K0 @ P.T + hcov_o )
    hparC = hpar + K1 @ ( Xo - P @ hpar )
    hcovC = hcov - K1 @ K0

    return hparC,hcovC
##}}}

def constraint_covar( *args: np.ndarray , P: np.ndarray | None = None , timeXo: Sequence[np.ndarray] | None = None , method_oerror: str | None = None ) -> tuple[np.ndarray,np.ndarray]:##{{{
    
    ## Extract data
    hpar = args[0]
    hcov = args[1]
    Xos  = [ xr.DataArray( Xo , dims = ["time"] , coords = [time] )
            for Xo,time in zip(args[2:],timeXo) ]
    
    ## Check data are finite
    if not np.isfinite(hpar).all() or not np.isfinite(hcov).all():
        hparC = np.zeros_like(hpar) + np.nan
        hcovC = np.zeros_like(hcov) + np.nan
        return hparC,hcovC
    
    ## Init
    err = Error()
    hcov_o = infer_hcov_o( hpar , hcov , Xos , P , method_oerror )
    
    ## Loop on constraint until convergence
    gXo = np.hstack( [Xo.values for Xo in Xos] )
    while not err.stop:
        hparC,hcovC = gaussian_conditionning( hpar , hcov , P , gXo , hcov_o )
        hcov_u      = infer_hcov_o( hparC , hcovC , Xos , P , method_oerror )
        err.value   = np.linalg.norm( hcov_o - hcov_u ) / np.linalg.norm(hcov_o)
        hcov_o      = hcov_u
    
    return hparC,hcovC
##}}}



def constraint_var( hpar: np.ndarray , hcov: np.ndarray , Y: np.ndarray , P: np.ndarray , size_chain: int , cnslaw: AbstractModel , use_STAN: bool , tmp_stan: str | None = None , n_try: int = 5 ) -> np.ndarray:##{{{
    
    ## Law
    nslaw   = cnslaw()
    nnshpar = nslaw.nhpar
    
    ## Prior
    prior_hpar = hpar[-nnshpar:]
    prior_hcov = hcov[-nnshpar:,:][:,-nnshpar:]
    prior      = sc.multivariate_normal( mean = prior_hpar , cov = prior_hcov , allow_singular = True )
    
    ## Output
    hpars = np.zeros( hpar.shape + (size_chain,) ) + np.nan
    
    ##
    chain_is_valid = False
    for _ in range(n_try):
        ## Draw covariate parameters
        hpars[:] = np.random.multivariate_normal( mean = hpar , cov = hcov , size = 1 ).reshape(-1,1)
        
        ## Build the covariable
        X = P @ hpars[:,0]
        
        ## Keep finite
        idx = np.isfinite(X) & np.isfinite(Y)
        iX  = X[idx]
        iY  = Y[idx]
        break
        ## Apply constraint
        try:
            draw = nslaw.fit_bayesian( iY , iX , prior , size_chain , use_STAN = use_STAN , tmp = tmp_stan , n_try = n_try )
        except StanError:
            continue
        except StanInitError:
            continue
        
        ##
        chain_is_valid = np.isfinite(draw).all()
        if chain_is_valid:
            hpars[-nnshpar:,:] = draw.T
            break
    
    
    return hpars
##}}}

