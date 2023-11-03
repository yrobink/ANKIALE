
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

import os

import numpy  as np
import xarray as xr


## Classes
##########

class EBM:
	
	def __init__( self ):##{{{
		
		cpath = os.path.dirname(os.path.abspath(__file__))
		
		ebmp = xr.open_dataset( os.path.join( cpath , "data" , "EBM_param.nc" ) )
		
		self.params  = xr.DataArray( ebmp.ebm_param.values[:-1,:]  , dims = ["model","param"]  , coords = [ebmp.ebm_param.attrs["model_name"].split(" ")[:-1],ebmp.ebm_param.attrs["param_name"].split(" ")] )
		self.forcing = xr.DataArray( ebmp.ebm_forcing.values[:,1:] , dims = ["year","forcing_name"] , coords = [ebmp.ebm_forcing[:,0].values.astype(int),ebmp.ebm_forcing.attrs["forcing_name"].split(" ")[1:]] )
		
		self._XN = self._run()
	##}}}
	
	def run( self , t , model = "AVG" ):##{{{
		
		if not (model in self.params.model or model == "AVG"):
			raise ValueError
		
		out = xr.DataArray( self._XN.loc[model,t].values , dims = ["time"] , coords = [t] )
		
		return out
	##}}}
	
	def _run(self):##{{{
		
		N    = self.forcing.year.size
		nmod = self.params.model.size
		res  = xr.DataArray( np.zeros( (nmod,N+1,3) ) , dims = ["model","time","dim0"] , coords = [self.params.model,range(N+1),range(3)] )
		dt   = 1
		
		c    = self.params.loc[:,"c"]
		c0   = self.params.loc[:,"c0"]
		lamb = self.params.loc[:,"lamb"]
		gamm = self.params.loc[:,"gamm"]
		
		for i in range(2,N+1):
			res[:,i,0] = res[:,i-1,0] + (dt / c) * ( self.forcing[i-1,2] - lamb * res[:,i-1,0] - gamm * ( res[:,i-1,0] - res[:,i-1,1] ) )
			res[:,i,1] = res[:,i-1,1] + (dt / c0 ) * gamm * ( res[:,i-1,0] - res[:,i-1,1] )
		
		res = res[:,1:,0].drop_vars("dim0").assign_coords( time = self.forcing.year.values )
		res = xr.concat( (res,res.mean( dim = "model" ).assign_coords(model = "AVG")) , dim = "model" )
		
		return res
	##}}}
	

