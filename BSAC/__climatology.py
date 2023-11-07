
## Copyright(c) 2023 Yoann Robin
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

## Packages
###########

import dataclasses
import logging
import datetime as dt

import numpy as np
import xarray as xr
import netCDF4
import cftime
import statsmodels.gam.api as smg

from .__release import version
from .__sys     import as_list
from .__sys     import coords_samples
from .__ebm     import EBM
from .stats.__tools import nslawid_to_class

##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#############
## Classes ##
#############


class Climatology:##{{{
	
	## Input / output ##{{{
	
	def __init__( self ):##{{{
		self._names = None
		self._cper  = None
		self._bper  = None
		self._dpers = None
		
		self._mean = None
		self._cov  = None
		self._bias = None
		self._time = None
		self._XN   = None 
		
		self._nslawid     = None
		self._nslaw_class = None
		self._spatial     = None
		
		self._Xconfig = { "GAM_dof" : 7 , "GAM_degree" : 3 }
		self._Yconfig = {}
	##}}}
	
	def __str__(self):##{{{
		
		out = "<class BSAC.Climatology>"
		
		try:
			## Build strings
			hpar = ', '.join(self.hpar_names[:3]+['...']+self.hpar_names[-3:])
			bper = '/'.join([str(i) for i in self.bper])
			bias = ", ".join( [f"{name}: {self.bias[name]:.2f}" for name in self.names] )
			time = ', '.join( [ str(y) for y in self.time.tolist()[:3]+['...']+self.time.tolist()[-3:] ] )
			mshape = "" if self._mean is None else str(self._mean.shape)
			cshape = "" if self._cov  is None else str(self._cov.shape)
			spatial = "" if self._spatial is None else ", ".join(self._spatial)
			
			sns = [
			       "onlyX",
			       "names",
			       "nslaw",
			       "hyper_parameter",
			       "bias_period",
			       "common_period",
			       "different_periods",
			       "bias",
			       "time",
			       "mean_shape",
			       "cov_shape",
			       "spatial"
			      ]
			ss  = [
			       str(self.onlyX),
			       ", ".join(self.names),
			       str(self._nslawid),
			       hpar,
			       bper,
			       self.cper[0],
			       ", ".join(self.dpers),
			       bias,
			       time,
			       mshape,
			       cshape,
			       spatial
			      ]
			
			## Output
			for sn,s in zip(sns,ss):
				out = out + "\n" + " * " + "{:{fill}{align}{n}}".format( sn , fill = " " , align = "<" , n = 15 ) + ": " + s
		except:
			pass
		
		return out
	##}}}
	
	def __repr__(self):##{{{
		return self.__str__()
	##}}}
	
	## staticmethod.init_from_file ##{{{
	@staticmethod
	def init_from_file( ifile ):
		
		clim = Climatology()
		
		with netCDF4.Dataset( ifile , "r" ) as incf:
			
			clim.names      = incf.variables["names"][:]
			clim.cper       = incf.variables["common_period"][:]
			clim.dpers      = incf.variables["different_periods"][:]
			
			clim._mean = np.array(incf.variables["mean"][:])
			clim._cov  = np.array(incf.variables["cov"][:])
			clim._bias = {}
			for name in clim.names:
				clim._bias[name] = float(incf.variables[f"bias_{name}"][:])
				clim.bper = str(incf.variables[f"bias_{name}"].getncattr("period")).split("/")
			
			nctime    = incf.variables["time"]
			units     = nctime.getncattr( "units"    )
			calendar  = nctime.getncattr( "calendar" )
			clim.time = [ t.year for t in cftime.num2date( nctime[:] , units , calendar ) ]
			
			
			clim._Xconfig = {}
			for c,t in zip( ["GAM_dof","GAM_degree"] , [int,int] ):
				clim._Xconfig[c] = t(incf.variables["X"].getncattr(c))
			
			try:
				clim._nslawid     = incf.variables["Y"].getncattr("nslawid")
				clim._nslaw_class = nslawid_to_class(clim._nslawid)
			except:
				pass
			
			try:
				spatial = incf.variables["Y"].getncattr("spatial").split(":")
				self._spatial = { s : xr.DataArray( incf.variables[s][:] , dims = [s] , coords = [incf.variables[s][:]] ) for s in spatial }
			except:
				pass
			
		return clim
	##}}}
	
	def save( self , ofile ):##{{{
		
		##
		with netCDF4.Dataset( ofile , "w" ) as ncf:
			
			## Define dimensions
			ncdims = {
			       "hyper_parameter"   : ncf.createDimension(   "hyper_parameter" , self._mean.shape[0] ),
			       "names"             : ncf.createDimension(             "names" , len(self.names) ),
			       "common_period"     : ncf.createDimension(     "common_period" , len(self.cper)  ),
			       "different_periods" : ncf.createDimension( "different_periods" , len(self.dpers) ),
			       "time"              : ncf.createDimension(              "time" , len(self.time)  ),
			}
			spatial = ()
			if self._spatial is not None:
				for d in self._spatial:
					ncdims[d] = ncf.createDimension( d , self._spatial[d].size )
				spatial = tuple([d for d in self._spatial])
			
			## Define variables
			ncvars = {
			       "hyper_parameter"   : ncf.createVariable(   "hyper_parameter" , str       ,   ("hyper_parameter",) ),
			       "names"             : ncf.createVariable(             "names" , str       ,             ("names",) ),
			       "common_period"     : ncf.createVariable(     "common_period" , str       ,     ("common_period",) ),
			       "different_periods" : ncf.createVariable( "different_periods" , str       , ("different_periods",) ),
			       "time"              : ncf.createVariable(              "time" , "float32" ,              ("time",) ),
			         "X"               : ncf.createVariable(                 "X" , "int32"                            ),
			       "mean"              : ncf.createVariable(              "mean" , "float32" ,   ("hyper_parameter",) + spatial ),
			       "cov"               : ncf.createVariable(               "cov" , "float32" ,   ("hyper_parameter","hyper_parameter") + spatial ),
			}
			for name in self.names:
				ncvars[f"bias_{name}"]    = ncf.createVariable( f"bias_{name}" , "float32" )
				ncvars[f"bias_{name}"][:] = float(self._bias[name])
				ncvars[f"bias_{name}"].setncattr( "period" , "{}/{}".format(*self.bper) )
			if self._spatial is not None:
				for d in self._spatial:
					ncvars[d] = ncf.createVariable( d , "double" , (d,) )
					ncvars[d][:] = np.array(self._spatial[d]).ravel()
			
			if not self.onlyX:
				ncvars["Y"] = ncf.createVariable( "Y" , "int32" )
			
			## Fill variables of dimension
			ncvars[  "hyper_parameter"][:] = np.array( self.hpar_names , dtype = str )
			ncvars[            "names"][:] = np.array( self.names      , dtype = str )
			ncvars[    "common_period"][:] = np.array( self.cper       , dtype = str )
			ncvars["different_periods"][:] = np.array( self.dpers      , dtype = str )
			
			## Fill time axis
			calendar = "standard"
			units    = "days since 1750-01-01 00:00"
			ncvars["time"][:]  = cftime.date2num( [cftime.DatetimeGregorian( int(y) , 1 , 1 ) for y in self.time] , units = units , calendar = calendar )
			ncvars["time"].setncattr( "standard_name" , "time"      )
			ncvars["time"].setncattr( "long_name"     , "time_axis" )
			ncvars["time"].setncattr( "units"         , units       )
			ncvars["time"].setncattr( "calendar"      , calendar    )
			ncvars["time"].setncattr( "axis"          , "T"         )
			
			## Fill variables
			ncvars["mean"][:] = self.mean_
			ncvars["cov"][:]  = self.cov_
			
			## Fill informations variables
			ncvars["X"][:] = 1
			ncvars["X"].setncattr( "GAM_dof"    , self.GAM_dof    )
			ncvars["X"].setncattr( "GAM_degree" , self.GAM_degree )
			
			if not self.onlyX:
				ncvars["Y"][:] = 1
				ncvars["Y"].setncattr( "nslawid" , self._nslawid )
				if self._spatial is not None:
					ncvars["Y"].setncattr( "spatial" , ":".join(self._spatial) )
			
			## Global attributes
			ncf.setncattr( "creation_date" , str(dt.datetime.utcnow())[:19] + " (UTC)" )
			ncf.setncattr( "BSAC_version"  , version )
	##}}}
	
	##}}}
	
	def isel( self , per , name ):##{{{
		
		if not per in self.dpers + ["lin","ns"]:
			raise ValueError("Bad index")
		if not name in self.names:
			raise ValueError("Bad index")
		
		size_X     = self.sizeX
		size_Y     = self.sizeY
		size_total = size_X * len(self.namesX) + size_Y
		
		if per == "ns":
			return slice( size_total - size_Y , size_total )
		
		t0 = self.names.index(name) * size_X
		if per in self.dpers:
			t0 += self.dpers.index(per) * (self.GAM_dof - 1)
			t1  = t0 + self.GAM_dof - 1
		else:
			t0 += len(self.dpers) * (self.GAM_dof - 1)
			t1 = t0 + 2
		
		return slice(t0,t1)
	##}}}
	
	## Statistics of X ##{{{ 
	
	def rvsX( self , size , add_BE = False , return_hpar = False ):##{{{
		
		##
		dof  = self.GAM_dof + 1
		time = self.time
		
		## Build the design matrix
		spl         = smg.BSplines( time , df = dof - 1 , degree = self.GAM_degree , include_intercept = False ).basis
		lin         = np.stack( [np.ones(time.size),self.XN.loc[time].values] ).T.copy()
		hpar_coords = [f"s{i}" for i in range(dof-2)] + ["cst","slope"]
		designF_    = xr.DataArray( np.hstack( (spl,lin) )                , dims = ["time","hpar"] , coords = [time,hpar_coords] )
		designC_    = xr.DataArray( np.hstack( (np.zeros_like(spl),lin) ) , dims = ["time","hpar"] , coords = [time,hpar_coords] )
		
		## Extract parameters of the distribution
		idxM = (slice(None),)            + tuple([0 for _ in range(self.mean_.ndim-1)])
		idxC = (slice(None),slice(None)) + tuple([0 for _ in range(self.cov_.ndim-2)])
		coef_ = self.mean_[idxM]
		cov_  = self.cov_[idxC]
		
		## if add BE
		samples = coords_samples(size)
		if add_BE:
			size    = size + 1
			samples = ["BE"] + samples
		
		coefs = xr.DataArray( np.random.multivariate_normal( mean = coef_ , cov = cov_ , size = size ) , dims = ["sample","hpar"] , coords = [samples,self.hpar_names] )
		if add_BE:
			coefs[0,:] = coef_
		
		XF = xr.concat(
		    [
		     xr.concat(
		        [
		         xr.concat( [coefs[:,self.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designF_
		         for per in self.dpers
		        ],
		        dim = "period"
		        )
		     for name in self.namesX
		    ],
		    dim = "name"
		    ).assign_coords( { "period" : self.dpers , "name" : self.namesX } ).transpose("sample","name","period","time")
		XC = xr.concat(
		    [
		     xr.concat(
		        [
		         xr.concat( [coefs[:,self.isel(p,name)] for p in [per,"lin"]] , dim = "hpar" ).assign_coords( hpar = hpar_coords ) @ designC_
		         for per in self.dpers
		        ],
		        dim = "period"
		        )
		     for name in self.namesX
		    ],
		    dim = "name"
		    ).assign_coords( { "period" : self.dpers , "name" : self.namesX } ).transpose("sample","name","period","time")
		XA = XF - XC
		
		out = xr.Dataset( { "XF" : XF , "XC" : XC , "XA" : XA } )
		
		if return_hpar:
			return out,coefs
		else:
			return out
	##}}}
	
	##}}}
	
	## Properties ##{{{
	
	@property
	def time(self):
		return self._time
	
	@time.setter
	def time( self , value ):
		self._time = np.array(value).ravel()
	
	@property
	def bias(self):
		return self._bias
	
	@property
	def XN(self):
		if self._XN is None:
			self._XN = EBM().run( t = self.time )
		return self._XN
	
	@property
	def mean_(self):
		return self._mean
	
	@mean_.setter
	def mean_( self , value ):
		self._mean = value
	
	@property
	def cov_(self):
		return self._cov
	
	@cov_.setter
	def cov_( self , value ):
		self._cov = value
	
	@property
	def onlyX(self):
		return self._nslawid is None
	
	@property
	def names(self):
		return self._names
	
	@property
	def namesX(self):
		return self.names if self.onlyX else self.names[:-1]
	
	@names.setter
	def names( self , value ):
		self._names = as_list(value)
	
	
	@property
	def hpar_names(self):
		hparnames = []
		for name in self.namesX:
			for p in self.dpers:
				hparnames = hparnames + [f"s{s}_{name}_{p}" for s in range(self.GAM_dof-1)]
			hparnames = hparnames + [f"cst_{name}",f"slope_{name}"]
		
		if not self.onlyX:
			hparnames = hparnames + self._nslaw_class().coef_name
		
		return hparnames
	
	
	@property
	def cper(self):
		return self._cper
	
	@property
	def common_period(self):
		return self._cper
	
	@cper.setter
	def cper( self , value ):
		self._cper = as_list(value)
	
	
	@property
	def bper(self):
		return self._bper
	
	@property
	def bias_period(self):
		return self._bper
	
	@bper.setter
	def bper( self , value ):
		self._bper = tuple([int(v) for v in value])
		if not len(self._bper) == 2:
			raise ValueError("Invalid bias period")
	
	
	@property
	def different_periods(self):
		return self._dpers
	
	@property
	def dpers(self):
		return self._dpers
	
	@dpers.setter
	def dpers( self , value ):
		self._dpers = as_list(value)
	
	
	@property
	def GAM_dof(self):
		return self._Xconfig["GAM_dof"]
	
	@property
	def GAM_degree(self):
		return self._Xconfig["GAM_degree"]
	
	
	@property
	def sizeX(self):
		return (self.GAM_dof - 1) * len(self.dpers) + 2
	
	@property
	def sizeY(self):
		if self._nslawid is None:
			return 0
		return len(self._nslaw_class().coef_name)
	
	def size(self):
		return self.sizeX * len(self.namesX) + self.sizeY
	
	##}}}
	
##}}}


