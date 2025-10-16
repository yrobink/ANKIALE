
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

##############
## Packages ##
##############


#############
## Imports ##
#############

import logging
import itertools as itt

import numpy  as np
import xarray as xr
import scipy.interpolate as sci

from typing import Sequence
from ..__exceptions import DevException


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#############
## Classes ##
#############


#class SplineSmoother:##{{{
#    
#    _spl: gamapi.BSplines
#    _gam: gamapi.GLMGam | None = None
#    dof: float
#
#    def __init__( self , x: np.ndarray , sbasis: int , dof: float , degree: int , include_intercept: bool = True ) -> None:##{{{
#        self._spl = gamapi.BSplines( x , df = sbasis + int(not include_intercept) , degree = degree , include_intercept = include_intercept )
#        self.dof  = dof
#    ##}}}
#    
#    def fit( self , X: np.ndarray ) -> Self:##{{{
#        
#        alpha = 1e-6
#        edof  = self.sbasis + 1
#        while edof > self.dof:
#            alpha *= 10
#            gam    = gamapi.GLMGam( endog = X , smoother = self._spl , alpha = alpha )
#            res    = gam.fit()
#            edof   = res.edf.sum()
#        
#        alphaL = alpha
#        alphaR = alphaL / 10
#        while np.abs(edof - self.dof) > 1e-2:
#            alpha = ( alphaL + alphaR ) / 2
#            gam    = gamapi.GLMGam( endog = X , smoother = self._spl , alpha = alpha )
#            res    = gam.fit()
#            edof   = res.edf.sum()
#            if edof < self.dof:
#                alphaL = alpha
#            else:
#                alphaR = alpha
#
#        self._gam = gam
#        self._res = res
#
#        return self
#    ##}}}
#    
#    def predict( self , *args: Any , **kwargs: Any ) -> np.ndarray:##{{{
#        if self._res is not None:
#            return self._res.predict( *args , **kwargs )
#    ##}}}
#    
#    ## Properties ##{{{
#    
#    @property
#    def degree(self) -> int:
#        return int(self._spl.degree[0])
#    
#    @property
#    def sbasis(self) -> int:
#        return self._spl.dim_basis
#    
#    @property
#    def include_intercept(self) -> bool:
#        return self._spl.include_intercept
#    
#    @property
#    def basis(self) -> np.ndarray:
#        return self._spl.basis
#    
#    @property
#    def edof(self) -> float:
#        if self._res is not None:
#            return self._res.edf.sum()
#    
#    @property
#    def hpar(self) -> np.ndarray:
#        if self._res is not None:
#            return self._res.params
#    
#    @property
#    def hcov(self) -> np.ndarray:
#        if self._res is not None:
#            return self._res.cov_params()
#    
#
#    ##}}}
#
###}}}


class BSplineBasis:##{{{
    
    def __init__( self , x: np.ndarray , nbasis: int , degree: int , intercept: bool = False ) -> None:##{{{
        self.x         = np.asarray(x)
        self.nbasis    = nbasis
        self.degree    = degree
        self.intercept = intercept
        self._iinter   = int(not intercept)
        
        self.knots     = np.linspace( self.x[0] , self.x[-1] , nbasis - degree + 1 + self._iinter )
    
        self.eknots =  np.concatenate([
            np.repeat(self.knots[0], self.degree),
            self.knots,
            np.repeat(self.knots[-1], self.degree)
        ])
        if not self.nbasis == self.eknots.size - self.degree - 1 - self._iinter:
            raise ValueError("Bad spline basis")
    ##}}}

    def basis( self, der: int = 0 ) -> np.ndarray:##{{{
        nbasis = self.nbasis + self._iinter
        B = np.zeros((self.x.size,nbasis))
        for i in range(nbasis):
            c    = np.zeros(nbasis)
            c[i] = 1.
            bspl = sci.BSpline( self.eknots , c , self.degree )
            B[:,i] = bspl.derivative(der)(self.x)

        if not self.intercept:
            B = B[:,1:]
        return B
    ##}}}
    
##}}}

class MPeriodSmoother:##{{{
    
    ## Attributes ##{{{
    _tdof: xr.DataArray
    _edof: xr.DataArray | None = None
    _bspl: BSplineBasis
    _tol: float
    
    _dtime: str
    _dname: str
    _dperiod: str
    _dhpar: str = "hpar"

    _names: Sequence[str]
    _periods: Sequence[str]
    _chpar: Sequence[str]
    
    _MB0: xr.DataArray
    _MB1: xr.DataArray
    _MB2: xr.DataArray
    _K0: np.ndarray | None = None
    _P: np.ndarray | None = None
    
    ##}}}
    
    def _create_basis( self, XN: np.ndarray ) -> None:##{{{
        B0  = self._bspl.basis(0)
        B1  = self._bspl.basis(1)
        B2  = self._bspl.basis(2)
        MB0 = xr.DataArray( 0.,
                          dims   = [self.dname,self.dperiod,self.dtime,self.dhpar],
                          coords = [self.names,self.periods,self.time ,self.chpar]
                          )
        MB1 = MB0.copy()
        MB2 = MB0.copy()
        for name,per in itt.product(self.names,self.periods):
            hpn = [f"XS{i}_{name}_{per}" for i in range(self.n_spl_basis) ]
            MB0.loc[name,per,:,f"X0_{name}"]  = 1.
            MB0.loc[name,per,:,f"XN_{name}"]  = XN
            MB0.loc[name,per,:,hpn]           = B0
            MB1.loc[name,per,:,f"XN_{name}"]  = 1.
            MB1.loc[name,per,:,hpn]           = B1
            MB2.loc[name,per,:,hpn]           = B2
        
        self._MB0 = MB0
        self._MB1 = MB1
        self._MB2 = MB2
    ##}}}

    def __init__( self , XN: xr.DataArray, total_dof: xr.DataArray , n_spl_basis: int , degree: int = 3 , tol: float = 1e-3 ) -> None:##{{{

        ## Set internal
        self._tdof  = total_dof
        self._dtime = XN.dims[0]
        self._bspl  = BSplineBasis( XN[self.dtime].values, n_spl_basis, degree, False )
        self._tol   = tol

        ## Check
        if not self._tdof.ndim == 2:
            raise ValueError("Target dof must be a table with two dimensions")

        ## Set derived internal
        self._dname   = self._tdof.dims[0]
        self._dperiod = self._tdof.dims[1]
        self._names   = self._tdof[self.dname].values.tolist()
        self._periods = self._tdof[self.dperiod].values.tolist()
        self._chpar   = self.gen_chpar()
        self._create_basis(XN.values)

    ##}}}
    
    def gen_chpar( self, names: Sequence[str] | None = None , periods: Sequence[str] | None = None ) -> Sequence[str]:##{{{
        if names is None: names = self.names
        if periods is None: periods = self.periods
        chpar = []
        for name in names:
            chpar.extend([f"X0_{name}",f"XN_{name}"])
            chpar.extend(
                [f"XS{i}_{name}_{per}"
                    for per,i in itt.product(periods,range(self.n_spl_basis))
                ] )
        
        return chpar
    ##}}}
    
    def _cst_matrix( self , TC: xr.DataArray ) -> np.ndarray:##{{{
        MC = []
        for h in self.chpar:
            if "X0" in h or "XN" in h:
                MC.append(1.)
            else:
                _,name,per = h.split("_")
                MC.append(float(TC.loc[name,per]))
        MC = np.array(MC)
        
        return np.diag(MC)
    ##}}}

    def _build_smoother( self , L: xr.DataArray ) -> None:##{{{:
        ML  = self._cst_matrix(L)
        xB0 = self.MB0.values.reshape(-1,self.nhpar)
        xB2 = self.MB2.values.reshape(-1,self.nhpar)
        
        K0   = np.linalg.inv(xB0.T @ xB0 + ML @ xB2.T @ xB2 )
        P    = xB0 @ K0 @ xB0.T
        edof = xr.DataArray( np.diag(P).reshape(self.nname,self.nperiod,-1).sum(axis=-1) , dims = [self.dname,self.dperiod] , coords = [self.names,self.periods] )

        self._K0   = K0
        self._P    = P
        self._edof = edof
    ##}}}

    def _init_smoother(self) -> None: ##{{{
        L_L  = xr.DataArray( 1e-3 , dims = [self.dname,self.dperiod] , coords = [self.names,self.periods] )
        while True:
            self._build_smoother(L_L)
            if (self._edof > self._tdof).all():
                break
            L_L /= L_L
        
        L_R  = xr.DataArray( 1e3 , dims = [self.dname,self.dperiod] , coords = [self.names,self.periods] )
        while True:
            self._build_smoother(L_R)
            if (self._edof < self._tdof).all():
                break
            L_R *= L_R
        
        while True:
            L = (L_R + L_L) / 2
            self._build_smoother(L)
            if np.abs(self._edof - self._tdof).max() < self._tol:
                break
            L_R = L_R.where( self._edof > self._tdof , L )
            L_L = L_L.where( self._edof < self._tdof , L )
    ##}}}

    def fit( self , X: xr.DataArray ) -> tuple[xr.DataArray,xr.DataArray]:##{{{
        
        ## Create smoother
        if self._edof is None:
            self._init_smoother()

        ## find hpar
        sMB0 = self.MB0.values.reshape(-1,self.nhpar)
        hpar = self._K0 @ sMB0.T @ X.values.reshape(-1)
        hpar = xr.DataArray( hpar , dims = [self.dhpar] , coords = [self.chpar] )
        
        ## find hcov
        S = self.MB0 @ hpar
        R = X - S
        C = self._cst_matrix( R.std( dim = self.dtime )**2 )
        hcov = self._K0 @ C / ( X[self.dtime].size - float(self._edof.sum() / self.nperiod) )
        hcov = xr.DataArray( hcov , dims = [f"{self.dhpar}0",f"{self.dhpar}1"] , coords = [self.chpar,self.chpar] )
        
        return hpar,hcov
    ##}}}
    
    def obs_projection( self , mix_periods: dict[str,str] | None = None , time: dict[str,str] | None = None ) -> np.ndarray:##{{{
        if mix_periods is None and time is None:
            names = self.names
        elif mix_periods is None:
            names = list(time)
        else:
            names = list(mix_periods)
        
        if mix_periods is None:
            mix_periods = { name: "full" for name in names }
        if time is None:
            time = { name: self.time for name in names }
        
        lB0 = []
        for name in names:
            mper = mix_periods[name]
            if mper == "full":
                mper = self.periods
            lB0.append( self.MB0.loc[name,mper,time[name],:].mean( dim = "period" ).values )

        return np.vstack(lB0)

    ##}}}

    ## Properties ##{{{
    
    @property
    def total_dof(self) -> xr.DataArray:
        return self._tdof
    
    @property
    def edof(self) -> xr.DataArray:
        return self._edof
    
    @property
    def dtime(self) -> str:
        return self._dtime
    
    @property
    def dname(self) -> str:
        return self._dname
    
    @property
    def dperiod(self) -> str:
        return self._dperiod
    
    @property
    def dhpar(self) -> str:
        return self._dhpar
    
    @property
    def x(self) -> np.ndarray:
        return self._bspl.x
    
    @property
    def time(self) -> np.ndarray:
        return self.x
    
    @property
    def n_spl_basis(self) -> int:
        return self._bspl.nbasis
    
    @property
    def degree(self) -> int:
        return self._bspl.degree

    @property
    def names(self) -> Sequence[str]:
        return self._names
    
    @property
    def periods(self) -> Sequence[str]:
        return self._periods
    
    @property
    def chpar(self) -> Sequence[str]:
        return self._chpar
    
    @property
    def nname(self) -> int:
        return len(self._names)
    
    @property
    def nperiod(self) -> int:
        return len(self._periods)
    
    @property
    def nhpar(self) -> int:
        return len(self._chpar)
    
    @property
    def MB0(self) -> xr.DataArray:
        return self._MB0

    @property
    def MB1(self) -> xr.DataArray:
        return self._MB1

    @property
    def MB2(self) -> xr.DataArray:
        return self._MB2

    ##}}}
    
##}}}



