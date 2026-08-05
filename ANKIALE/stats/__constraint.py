
## Copyright(c) 2024 / 2026 Yoann Robin
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

from typing import Any
from typing import Sequence

from .__KCC import KCC
from .__KCC import MAR2

from ..__sys import Error
from ..__exceptions import StanError
from ..__exceptions import StanInitError
from ..__exceptions import DevException

from .models.__AbstractModel import AbstractModel

from ..__logs import disable_warnings
import traceback

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
        hcov_o[b:e,b:e] = MAR2.fit( Ro.values ).cov(Ro.size)
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
        Ros.append(
            Xo.copy( data = Xo.values - X[b:e] )
        )
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

def infer_hcov_o( hpar: np.ndarray , hcov: np.ndarray , Xos: Sequence[xr.DataArray] , P: np.ndarray , method_oerror: str = 'IND', errors: str = "raise" ) -> np.ndarray:##{{{
    try:
        hcov_o = _infer_hcov_o( hpar , hcov , Xos , P , method_oerror )
        omethod_oerror = method_oerror
    except Exception as e:
        logger.error( f"Error: {e}" )
        logger.error( f"Traceback:\n{traceback.format_exc()}" )
        if errors == "raise":
            raise e
        match method_oerror:
            case "KCC":
                logger.warning("Fail to use KCC, back to MAR2")
                hcov_o,omethod_oerror = infer_hcov_o( hpar , hcov , Xos , P , "MAR2" )
            case "MAR2":
                logger.warning("Fail to use MAR2, back to IND")
                hcov_o,omethod_oerror = infer_hcov_o( hpar , hcov , Xos , P , "IND" )
            case _:
                raise e
    return hcov_o,omethod_oerror
##}}}


def gaussian_conditionning( hpar: np.ndarray , hcov: np.ndarray , P: np.ndarray , Xo: np.ndarray , hcov_o: np.ndarray ) -> tuple[np.ndarray,np.ndarray]:##{{{
    K0    = P @ hcov
    K1    = ( hcov @ P.T ) @ np.linalg.inv( K0 @ P.T + hcov_o )
    hparC = hpar + K1 @ ( Xo - P @ hpar )
    hcovC = hcov - K1 @ K0

    return hparC,hcovC
##}}}


def _constraint_covar( hpar: np.ndarray, hcov: np.ndarray, Xos: Sequence[xr.DataArray], hcov_o: np.ndarray | None, P: np.ndarray | None = None , hcov_o_meas: np.ndarray | float = 0., method_oerror: str | None = None, errors: str = "raise" ) -> tuple[np.ndarray,np.ndarray,np.ndarray]: ##{{{
    
    ## Init
    err = Error( tol = 1e-3 )
    if hcov_o is None:
        hcov_o_iv,_ = infer_hcov_o( hpar , hcov , Xos , P , "IND" )
        hcov_o = hcov_o_iv + hcov_o_meas
    
    ## Loop on constraint until convergence
    merr = method_oerror
    gXo = np.hstack( [Xo.values for Xo in Xos] )
    logger.debug( f" * Constraint with method {method_oerror}" )
    while not err.stop:
        hparC,hcovC = gaussian_conditionning( hpar , hcov , P , gXo , hcov_o )
        hcov_u_iv,merr = infer_hcov_o( hparC , hcovC , Xos , P , merr, errors = errors )
        hcov_u = hcov_u_iv + hcov_o_meas
#        err.value   = np.linalg.norm( ( np.linalg.inv(hcov_o) @ hcov_u ) - np.identity(hcov_u.shape[0]) )
        err.value   = np.linalg.norm( hcov_o - hcov_u ) / np.linalg.norm(hcov_o)
        hcov_o      = hcov_u
        logger.debug( f"   => Observed matrix convergence error: {err.value}" )
    
    return hparC,hcovC,hcov_o
##}}}

## constraint_covar ##{{{

@disable_warnings
def constraint_covar( hpar: np.ndarray | xr.DataArray,
                      hcov: np.ndarray | xr.DataArray,
                      P: np.ndarray | xr.DataArray,
                      *args: Any,
                      hcov_o_meas: np.ndarray | xr.DataArray | None = None,
                      method_oerror: str = "IND",
                     errors: str = "KCC-MAR2-IND",
                     ) -> tuple[np.ndarray | xr.DataArray,np.ndarray | xr.DataArray]:
    """
    Function for constraining the distribution N(hpar,hcov) using observations.
    The observations Xo1, Xo2,... are provided as additional arguments (*args)
    and must be xarray.DataArray objects with time as the dimension for each
    observed covariate. The projection matrix P must satisfy the following
    relationship:
    [Xo1,Xo2,...] = P @ hpar

    Arguments
    ---------
    hpar: np.ndarray | xr.DataArray
        Mean of parameters to constrain
    hcov: np.ndarray | xr.DataArray
        Covariance matrix of parameters to constrain
    P: np.ndarray | xr.DataArray
        Projection operator to apply gaussian conditionning theorem
    args: Sequence[xr.DataArray]
        Sequence of observations
    method_oerror: str
        Observed error estimation method. Must be one of:
            - 'IND': Observed residuals are independent
            - 'MAR2': Observed residuals follow a sum of two AR(1) process
            - 'KCC': If numbers of covariates is greater than 1, assume MAR2
                     and add a dependency term between covariates.
    errors: str = "raise" or "KCC-MAR2-IND"
        if an error occured, if errors is:
            - "raise": raise an Exception
            - "KCC-MAR2-IND", if method_oerror is KCC, use MAR2, if it is MAR2
              use IND, if it is IND, raise the Exception
        
    """

    ## Convert all in np.ndarray
    _hpar = hpar.values if isinstance(hpar,xr.DataArray) else hpar
    _hcov = hcov.values if isinstance(hcov,xr.DataArray) else hcov
    _P    =    P.values if isinstance(   P,xr.DataArray) else P
    
    _hcov_o_meas = hcov_o_meas
    if isinstance(_hcov_o_meas,xr.DataArray):
        _hcov_o_meas = hcov_o_meas.values
    if _hcov_o_meas is None:
        _hcov_o_meas = 0
    
    ## Observations
    Xos = args

    ## Check data are finite
    if not np.isfinite(_hpar).all() or not np.isfinite(_hcov).all():
        hparC = np.zeros_like(_hpar) + np.nan
        hcovC = np.zeros_like(_hcov) + np.nan
        if isinstance(hpar,xr.DataArray):
            hparC = hpar.copy( data = _hpar )
        if isinstance(hcov,xr.DataArray):
            hcovC = hcov.copy( data = _hcov )
        return hparC,hcovC
    
    ##
    hparC,hcovC,hcov_o = _constraint_covar( _hpar, _hcov, Xos, None, _P, _hcov_o_meas, "IND", errors = "raise" )
    
    if method_oerror in ["MAR2","KCC"]:
        hparC,hcovC,hcov_o = _constraint_covar( _hpar, _hcov, Xos, hcov_o, _P, _hcov_o_meas, method_oerror, errors = errors )
    
    if isinstance(hpar,xr.DataArray):
        hparC = hpar.copy( data = hparC )
    if isinstance(hcov,xr.DataArray):
        hcovC = hcov.copy( data = hcovC )
    
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
        _U,_S,_Vh = np.linalg.svd(hcov)
        scov      = _U @ np.sqrt(np.diag(np.abs(_S))) @ _Vh
        _N        = np.random.normal( size = (hpar.size,1) )
        hpars[:]  = scov @ _N + hpar.reshape(-1,1)
#        hpars[:] = np.random.multivariate_normal( mean = hpar , cov = hcov , size = 1 ).reshape(-1,1)
        
        ## Build the covariable
        X = P @ hpars[:,0]
        
        ## Keep finite
        idx = np.isfinite(X) & np.isfinite(Y)
        iX  = X[idx]
        iY  = Y[idx]
        
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

