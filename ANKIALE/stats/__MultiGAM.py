
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

from ..__linalg import matrix_positive_part
from ..__sys import Error

import numpy  as np
import xarray as xr
import scipy.linalg as scl
import statsmodels.gam.api as smg
import statsmodels.gam.api as gamapi
import distributed

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

class SplineSmoother:##{{{
    
    def __init__( self , x , sbasis , dof , degree , include_intercept = True ):##{{{
        self._spl = smg.BSplines( x , df = sbasis + int(not include_intercept) , degree = degree , include_intercept = include_intercept )
        self._gam = None
        self.dof  = dof
    ##}}}
    
    def fit( self , X ):##{{{
        
        alpha = 1e-6
        edof  = self.sbasis + 1
        while edof > self.dof:
            alpha *= 10
            gam    = smg.GLMGam( endog = X , smoother = self._spl , alpha = alpha )
            res    = gam.fit()
            edof   = res.edf.sum()
        
        alphaL = alpha
        alphaR = alphaL / 10
        while np.abs(edof - self.dof) > 1e-2:
            alpha = ( alphaL + alphaR ) / 2
            gam    = smg.GLMGam( endog = X , smoother = self._spl , alpha = alpha )
            res    = gam.fit()
            edof   = res.edf.sum()
            if edof < self.dof:
                alphaL = alpha
            else:
                alphaR = alpha

        self._gam = gam
        self._res = res

        return self
    ##}}}
    
    def predict( self , *args , **kwargs ):##{{{
        if self._res is not None:
            return self._res.predict( *args , **kwargs )
    ##}}}
    
    ## Properties ##{{{
    
    @property
    def degree(self):
        return int(self._spl.degree[0])
    
    @property
    def sbasis(self):
        return self._spl.dim_basis
    
    @property
    def include_intercept(self):
        return self._spl.include_intercept
    
    @property
    def basis(self):
        return self._spl.basis
    
    @property
    def edof(self):
        if self._res is not None:
            return self._res.edf.sum()
    
    @property
    def hpar(self):
        if self._res is not None:
            return self._res.params
    
    @property
    def hcov(self):
        if self._res is not None:
            return self._res.cov_params()
    

    ##}}}

##}}}

class MPeriodSmoother:##{{{

    XN: xr.DataArray
    cnames: Sequence[str]
    dpers: Sequence[str]
    spl_config: dict[str,int | float]
    _keys = Sequence[str]
    D0: np.ndarray
    D2: np.ndarray
    K0: np.ndarray
    P: np.ndarray
    dof: np.ndarray
    spl: gamapi.BSplines

    def __init__( self , XN: xr.DataArray, ##{{{
                  cnames: Sequence[str],
                  dpers: Sequence[str],
                  spl_config: dict[str,int | float],
                ) -> None:
        self.XN = XN
        self.cnames = cnames
        self.dpers = dpers
        self.spl_config = spl_config
        self._keys = [ f"{cname}_{per}"
                 for cname,per in itt.product(self.cnames,
                                              self.dpers)]

        self._build_design_matrix()
        self._init_smoother()
    ## End __init__ }}}

    def _build_design_matrix(self):##{{{
        """Create design matrix
        """
        hpar_names = ["X0","XN"] + [
            f"XS{i}_{per}"
            for per,i in itt.product( self.dpers,
                                      range(self.nknot)
                                    )
        ]

        self.spl = gamapi.BSplines( self.time,
                                    df = self.nknot + 1,
                                    degree = self.degree,
                                    include_intercept = False )
        
        D0R = xr.DataArray( 0. ,
                            dims = ["time","hpar"],
                            coords = [self.time,hpar_names] )
        D0R.loc[:,"X0"] = 1
        D0R.loc[:,"XN"] = self.XN
        D2R = xr.zeros_like(D0R)

        ## Loop over periods to create the period block
        D0 = []
        D2 = []
        for iper,per in enumerate(self.dpers):

            ## Index
            idx = [h for h in hpar_names if per in h]

            ## Spline basis
            D0R.loc[:,idx] = self.SB0
            D0.append(
                D0R.assign_coords(
                    time = [f"{t}_{per}" for t in self.time]
                ).copy()
            )
            D0R.loc[:,idx] = 0

            ## Spline basis of 2nd der
            D2R.loc[:,idx] = self.SB2
            D2.append(
                D2R.assign_coords(
                    time = [f"{t}_{per}" for t in self.time]
                ).copy()
            )
            D2R.loc[:,idx] = 0


        D0 = xr.concat( D0 , dim = "time" )
        D2 = xr.concat( D2 , dim = "time" )

        ## Loop on cnames to create the final matrix
        self.D0 = []
        self.D2 = []
        for ic0,ic1 in itt.product(range(self.ncnames),
                                   range(self.ncnames)):
            if ic1 == 0:
                self.D0.append([])
                self.D2.append([])
            self.D0[-1].append( (ic0 == ic1) * D0.values )
            self.D2[-1].append( (ic0 == ic1) * D2.values )

        self.D0  = np.block(self.D0)
        self.D2  = np.block(self.D2)
        self.DD0 = self.D0.T @ self.D0
        self.DD2 = self.D2.T @ self.D2
    ## End build_design_matrix }}}

    def _build_projection( self , L: np.ndarray ) -> tuple[##{{{
        np.ndarray,
        np.ndarray,
        np.ndarray]:
        K0   = np.linalg.inv(self.DD0 + L @ self.DD2 )
        P    = self.D0 @ K0 @ self.D0.T
        dof  = np.diag(P).reshape( self.ncnames * self.ndpers, -1 ).sum( axis = 1 )
        return K0,P,dof
    ##}}}

    def _build_L_matrix( self , L: xr.DataArray ) -> np.ndarray:##{{{

        ML = []
        for cname in self.cnames:
            l = np.zeros((self.ndpers,self.nknot))
            idx = [x for x in self._keys if cname in x]
            l[:] = L.loc[idx].values.reshape(-1,1)
            ML.extend([1.,1.])
            ML.extend(l.reshape(-1).tolist())
        ML = np.diag(np.array(ML))

        return ML
    ##}}}

    def _init_smoother(self) -> None:##{{{
        tdof = xr.DataArray( 0. , dims = ["d0"] , coords = [self._keys] )
        cdof = tdof.copy()
        Ldof = tdof.copy()
        Rdof = tdof.copy()
        LR   = tdof.copy() + 1e+3
        LL   = tdof.copy() + 1e-3
        for key in self._keys:
            tdof.loc[key] = self.spl_config[key] + 2

        ## Find init interval
        MLR = self._build_L_matrix( LR )
        MLL = self._build_L_matrix( LL )
        _,_,Ldof[:] = self._build_projection(MLR)
        _,_,Rdof[:] = self._build_projection(MLL)

        while (Ldof > tdof).any():
            LL *= 10
            MLL = self._build_L_matrix( LL )
            _,_,Ldof[:] = self._build_projection(MLL)

        while (Rdof < tdof).any():
            LR /= 10
            MLR = self._build_L_matrix(LR)
            _,_,Rdof[:] = self._build_projection(MLL)

        ## And now loop
        err = Error( tol = 1e-3 )
        while not err.stop:
            L  = (LL + LR) / 2
            ML = self._build_L_matrix(L)
            K0,P,cdof[:] = self._build_projection(ML)
            LR[ (cdof > tdof)] = L[ (cdof > tdof)]
            LL[~(cdof > tdof)] = L[~(cdof > tdof)]
            err.value = float( np.abs(cdof - tdof).max() )

        ## Set
        self.K0 = K0
        self.P = P
        self.dof = cdof
    ##}}} End of _init_smoother

    def fit( self , X ): ##{{{ cname / period / time

        hpar_names = self.hpar_names
        mtime = self.mtime

        xD0 = xr.DataArray( self.D0,
                          dims = ["mtime","hpar"],
                          coords = [mtime,hpar_names]
                          )
        xK0 = xr.DataArray( self.K0,
                          dims = ["hpar0","hpar1"],
                          coords = [hpar_names,hpar_names]
                          )
        xkeys = [f"{cname}_{per}" for cname,per in itt.product(
            X.cname.values,
            X.period.values,
        )]
        xtime = [f"{xkey}_{t}" for xkey,t in itt.product(
            xkeys,
            X.time.values
        )]

        xhpar_names = []
        for h in hpar_names:
            if "X0" in h or "XN" in h:
                xhpar_names.append(h)
                continue
            if "_".join(h.split("_")[1:]) in xkeys:
                xhpar_names.append(h)

        Xu = xr.DataArray( X.values.reshape(-1),
                         dims = ["mtime"],
                         coords = [xtime] )

        D0   = xD0.loc[xtime,xhpar_names].values
        K0   = xK0.loc[xhpar_names,xhpar_names].values
        xdof = self.dof.loc[xkeys].sum().values

        hpar  = K0 @ D0.T @ Xu.values
        R     = (Xu.values - D0 @ hpar).reshape(-1,1)
        hcov  = K0 * (R.T @ R)
        hcov  = hcov  / ( Xu.size - xdof )

        hpar = xr.DataArray( hpar,
                           dims = ["hpar"],
                           coords = [xhpar_names]
                           )
        hcov = xr.DataArray( hcov,
                           dims = ["hpar0","hpar1"],
                           coords = [xhpar_names,xhpar_names]
                           )

        return hpar,hcov
    ##}}}

    ## Properties{{{
    @property
    def hpar_names(self):
        _hpar_names = []
        for _cname in self.cnames:
            _hpar_names.extend( [f"X0_{_cname}",f"XN_{_cname}"] )
            for _per in self.dpers:
                _hpar_names.extend(
                    [f"XS{i}_{_cname}_{_per}" for i in range(self.nknot)]
                )
        return _hpar_names

    @property
    def lin(self):
        return np.vstack((np.ones(self.XN.size),self.XN.values)).T
    
    @property
    def SB0(self):
        return self.spl.basis

    @property
    def SB1(self):
        return self.spl.smoothers[0].der_basis
        
    @property
    def SB2(self):
        return self.spl.smoothers[0].der2_basis
    
    @property
    def xD0(self):
        xD0 = xr.DataArray( self.D0,
                          dims = ["mtime","hpar"],
                          coords = [self.mtime,self.hpar_names]
                          )
        return xD0
        
    
    @property
    def mtime(self):
        return [ f"{key}_{t}" for key,t in itt.product(self._keys, 
                                                        self.time)]

    @property
    def ncnames(self):
        return len(self.cnames)

    @property
    def ndpers(self):
        return len(self.dpers)

    @property
    def nknot(self):
        return self.spl_config["nknot"]

    @property
    def degree(self):
        return self.spl_config["degree"]

    @property
    def time(self):
        return self.XN.time.values
    ##}}}

##}}}



