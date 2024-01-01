
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


#############
## Imports ##
#############

import os
import zarr
import numpy  as np
import xarray as xr

#############
## Classes ##
#############

def random_zfile( prefix = "" ):##{{{
	it = 0
	zfile = prefix + f"{it}.zarr"
	while os.path.isfile(zfile):
		it += 1
		zfile = prefix + f"{it}.zarr"
	return zfile
##}}}

class XZarr:##{{{
	
	def __init__( self ):##{{{
		
		self.dims   = None
		self.coords = None
		self.shape  = None
		self.zfile  = None
		self.zdata  = None
	##}}}
	
	def __repr__(self):##{{{
		return self.__str__()
	##}}}
	
	def __str__(self):##{{{
		out = "<BSAC.XZarr>"
		try:
			out = out + " " + str(tuple(self.shape))
			for d in self.dims:
				out = out + "\n" + f" * {d}"
		except:
			pass
		return out
	##}}}
	
	## static.from_xarray ##{{{
	@staticmethod
	def from_xarray( xX , zfile ):
		
		xzarr = XZarr()
		
		xzarr.dims   = xX.dims
		xzarr.coords = xX.coords
		xzarr.shape  = xX.shape
		xzarr.zfile  = zfile
		xzarr.zdata  = zarr.open( xzarr.zfile , mode = "w" , shape = xzarr.shape , dtype = xX.dtype )
		
		xzarr.zdata[:] = xX.values[:]
		
		return xzarr
	##}}}
	
	## static.from_value ##{{{
	@staticmethod
	def from_value( value , shape, dims , coords , zfile , dtype = np.float32 , zarr_kwargs = {} ):
		
		xzarr = XZarr()
		
		xzarr.dims   = dims
		xzarr.coords = coords
		xzarr.shape  = shape
		xzarr.zfile  = zfile
		xzarr.zdata  = zarr.open( xzarr.zfile , mode = "w" , shape = xzarr.shape , dtype = dtype , **zarr_kwargs )
		
		xzarr.zdata[:] = value
		
		return xzarr
	##}}}
	
	## static.like ##{{{
	@staticmethod
	def like( xzarr , zfile , value = 0 , dtype = "float32" ):
		
		copy = XZarr()
		
		copy.dims   = xzarr.dims
		copy.coords = xzarr.coords
		copy.shape  = xzarr.shape
		copy.zfile  = zfile
		copy.zdata  = zarr.open( copy.zfile , mode = "w" , shape = copy.shape , dtype = dtype )
		
		copy.zdata[:] = value
		
		return copy
	##}}}
	
	def get_orthogonal_selection( self , idxs ):##{{{
		
		dims   = []
		coords = []
		for i,idx in enumerate(idxs):
			if isinstance(idx,slice):
				dims.append(self.dims[i])
				coords.append(self.coords[i][idx])
			elif isinstance(idx,int):
				pass
		
		out = xr.DataArray( self.zdata.get_orthogonal_selection(idxs),
		                    dims   = dims,
		                    coords = coords
		                    )
		return out
	##}}}
	
	def set_orthogonal_selection( self , idxs , X ):##{{{
		self.zdata.set_orthogonal_selection( idxs , X )
	##}}}
	
	## Properties ##{{{
	
	@property
	def ndim(self):
		return len(self.shape)
	
	@property
	def xdata(self):
		out = xr.DataArray( self.zdata[:],
		                    dims   = self.dims,
		                    coords = self.coords
		                    )
		return out
	
	##}}}
	
##}}}

