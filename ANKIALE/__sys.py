
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
import numpy as np

#############
## Classes ##
#############


###############
## Functions ##
###############

def sort_run(key):##{{{
    
    r = key.split("r")[1].split("i")[0]
    i = key.split("i")[1].split("p")[0]
    p = key.split("p")[1].split("f")[0]
    f = key.split("f")[1]
    
    return [int(s) for s in [r,i,p,f]]
##}}}

def as_list( x ):##{{{
    if isinstance( x , np.ndarray ):
        return x.tolist()
    elif not isinstance( x , list ):
        return [x]
    return x
##}}}

def coords_samples( size ):##{{{
    return [ "S{:{fill}{align}{n}}".format( i , fill = "0" , align = ">" , n = int(np.log10(max(1,size-2))) + 1 ) for i in range(size) ]
##}}}

def copy_files( *args ):##{{{
    
    ## Check number of arguments
    if not len(args) > 1:
        raise ValueError( "At least one source and the destination must be set" )
    
    ##
    if len(args) > 2:
        for ifile in args[:-1]:
            copy_files( ifile , os.path.join( args[-1] , ifile ) )
    else:
        ifile = args[0]
        ofile = args[1]
        opath = os.path.dirname(ofile)
        if not os.path.isfile(ifile):
            raise FileNotFoundError( f"The file '{ifile}' does not exist!" )
        if not os.path.isdir(opath):
            raise NotADirectoryError( f"Output path '{opath}' is not a directory" )
        with open( ifile , "r" ) as inf:
            with open( ofile , "w" ) as onf:
                onf.write( "\n".join( inf.readlines() ) )
##}}}

class Error:##{{{
    _tol: float = 1e-3
    _value: float = 1
    _nit: int   = 0
    _maxit: int = 100

    def __init__( self , tol: float = 1e-3 , maxit: int = 100 ) -> None:
        self._tol = tol
        self._maxit = maxit

    def __str__(self) -> str:
        s = "Error(value = {} {} {}, it = {} {} {})".format(
            self._value,
            "<" if self._value < self._tol else ">",
            self._tol,
            self._nit,
            "<" if self._nit < self._maxit else ">",
            self._maxit
        )
        return s

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def nit(self) -> int:
        return self._nit

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value( self , value_: float ) -> None:
        self._value = value_
        self._nit += 1

    @property
    def stop(self) -> bool:
        _stop = (self.value < self._tol) or (self._nit > self._maxit)
        return _stop
##}}}

