
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

import numpy as np

#############
## Classes ##
#############

class SizeOf:##{{{
	"""
	BSAC.SizeOf
	===========
	
	Class to manage units size for memory
	
	1o == 1octet == 1Byte == 1B == 8 bits == 8b
	
	1kB = 1000^1B
	1MB = 1000^2B
	1GB = 1000^3B
	1TB = 1000^4B
	
	1kiB = 1024^1B
	1MiB = 1024^2B
	1GiB = 1024^3B
	1TiB = 1024^4B
	
	"""
	
	def __init__( self , s = None , n = None , unit = "b" ):##{{{
		
		if s is not None and n is None:
			self._init_from_str(s)
		elif n is not None and s is None:
			self._init_from_value( n , unit )
		elif s is None and n is None:
			raise ValueError( f"'s' or 'n' must be set!" )
		else:
			raise ValueError( f"s = {s} and n = {n} can not be set simultaneously!" )
		
	##}}}
	
	def _init_from_str( self , s ):##{{{
		s = s.replace(" ","")
		
		## Start with unit
		self.unit = s[-1]
		if not self.unit.lower() in ["b","o"]:
			raise Exception(f"Bad unit: {self.unit}")
		
		## Others values
		if s[-2] == "i":
			self.ibase = "i"
			self.base = 1024
			if s[-3].lower() in ["k","m","g","t"]:
				scale = s[-3]
				value = s[:-3]
			else:
				scale = ""
				value = s[:-2]
		else:
			self.ibase = ""
			self.base = 1000
			if s[-2].lower() in ["k","m","g","t"]:
				scale = s[-2]
				value = s[:-2]
			else:
				scale = ""
				value = s[:-1]
		
		## Check value
		try:
			try:
				value = int(value)
			except:
				value = float(value)
		except:
			raise Exception(f"Value error:{value}")
		
		self.scale = scale
		self.value = value
		
		self.bits = self.value * self.base**self.iscale
		if not self.unit == "b":
			self.bits = self.bits * self.bits_per_byte
		if not 10 * int(self.bits) == int(10*self.bits):
			raise Exception(f"Value is a subdivision of a bit, it is not possible! b = {self.bits}" )
		self.bits = int(self.bits)
	##}}}
	
	def _init_from_value( self , n , unit ):##{{{
		
		if not isinstance( n , int ):
			raise ValueError( f"n = {n} must be an integer" )
		self._init_from_str( f"{n}{unit}" )
	##}}}
	
	def __repr__( self ):##{{{
		return self.__str__()
	##}}}
	
	def __str__( self ):##{{{
		
		if int(self.o) == 0:
			return "{:.2f}o".format(self.o)
		elif int(self.ko) == 0:
			return "{:.2f}o".format(self.o)
		elif int(self.Mo) == 0:
			return "{:.2f}ko".format(self.ko)
		elif int(self.Go) == 0:
			return "{:.2f}Mo".format(self.Mo)
		elif int(self.To) == 0:
			return "{:.2f}Go".format(self.Go)
		
		return "{:.2f}To".format(self.To)
		
	##}}}
	
	## Properties ##{{{
	
	@property
	def bits_per_byte(self):
		return 8
	
	@property
	def bits_per_octet(self):
		return 8
	
	##}}}
	
	## property.iscale ##{{{
	@property
	def iscale(self):
		if self.scale.lower() == "k":
			return 1
		if self.scale.lower() == "m":
			return 2
		if self.scale.lower() == "g":
			return 3
		if self.scale.lower() == "t":
			return 4
		return 0
	
	##}}}
	
	## Octet properties ##{{{
	
	@property
	def o( self ):
		return self.bits / self.bits_per_octet / 1000**0
	
	@property
	def ko( self ):
		return self.bits / self.bits_per_octet / 1000**1
	
	@property
	def Mo( self ):
		return self.bits / self.bits_per_octet / 1000**2
	
	@property
	def Go( self ):
		return self.bits / self.bits_per_octet / 1000**3
	
	@property
	def To( self ):
		return self.bits / self.bits_per_octet / 1000**4
	
	##}}}
	
	## iOctet properties ##{{{
	
	@property
	def io( self ):
		return self.bits / self.bits_per_octet / 1024**0
	
	@property
	def kio( self ):
		return self.bits / self.bits_per_octet / 1024**1
	
	@property
	def Mio( self ):
		return self.bits / self.bits_per_octet / 1024**2
	
	@property
	def Gio( self ):
		return self.bits / self.bits_per_octet / 1024**3
	
	@property
	def Tio( self ):
		return self.bits / self.bits_per_octet / 1024**4
	
	##}}}
	
	## Byte properties ##{{{
	
	@property
	def B( self ):
		return self.bits / self.bits_per_byte / 1000**0
	
	@property
	def kB( self ):
		return self.bits / self.bits_per_byte / 1000**1
	
	@property
	def MB( self ):
		return self.bits / self.bits_per_byte / 1000**2
	
	@property
	def GB( self ):
		return self.bits / self.bits_per_byte / 1000**3
	
	@property
	def TB( self ):
		return self.bits / self.bits_per_byte / 1000**4
	
	##}}}
	
	## iByte properties ##{{{
	
	@property
	def iB( self ):
		return self.bits / self.bits_per_byte / 1024**0
	
	@property
	def kiB( self ):
		return self.bits / self.bits_per_byte / 1024**1
	
	@property
	def MiB( self ):
		return self.bits / self.bits_per_byte / 1024**2
	
	@property
	def GiB( self ):
		return self.bits / self.bits_per_byte / 1024**3
	
	@property
	def TiB( self ):
		return self.bits / self.bits_per_byte / 1024**4
	
	##}}}
	
	## bits properties ##{{{
	
	@property
	def b( self ):
		return self.bits / 1000**0
	
	@property
	def kb( self ):
		return self.bits / 1000**1
	
	@property
	def Mb( self ):
		return self.bits / 1000**2
	
	@property
	def Gb( self ):
		return self.bits / 1000**3
	
	@property
	def Tb( self ):
		return self.bits / 1000**4
	
	##}}}
	
	## ibits properties ##{{{
	
	@property
	def ib( self ):
		return self.bits / 1024**0
	
	@property
	def kib( self ):
		return self.bits / 1024**1
	
	@property
	def Mib( self ):
		return self.bits / 1024**2
	
	@property
	def Gib( self ):
		return self.bits / 1024**3
	
	@property
	def Tib( self ):
		return self.bits / 1024**4
	##}}}
	
	## Comparison operators overload ##{{{
	
	def __eq__( self , other ):##{{{
		
		if isinstance(other,str):
			other = SizeOf(other)
		
		return self.bits == other.bits
	##}}}
	
	def __ne__( self , other ):##{{{
		
		if isinstance(other,str):
			other = SizeOf(other)
		
		return self.bits != other.bits
	##}}}
	
	def __lt__( self , other ):##{{{
		
		if isinstance(other,str):
			other = SizeOf(other)
		
		return self.bits < other.bits
	##}}}
	
	def __gt__( self , other ):##{{{
		
		if isinstance(other,str):
			other = SizeOf(other)
		
		return self.bits > other.bits
	##}}}
	
	def __le__( self , other ):##{{{
		
		if isinstance(other,str):
			other = SizeOf(other)
		
		return self.bits <= other.bits
	##}}}
	
	def __ge__( self , other ):##{{{
		
		if isinstance(other,str):
			other = SizeOf(other)
		
		return self.bits >= other.bits
	##}}}
	
	##}}}
	
	## Arithmetic operators overload ##{{{
	
	def __add__( self , other ):##{{{
		if isinstance(other,str):
			other = SizeOf(other)
		
		return SizeOf( n = self.bits + other.bits , unit = "b" )
	##}}}
	
	def __radd__( self , other ):##{{{
		if isinstance(other,str):
			other = SizeOf(other)
		
		return SizeOf( n = self.bits + other.bits , unit = "b" )
	##}}}
	
	def __mul__( self , x ):##{{{
		if not isinstance(x,int):
			raise ValueError( "Only multiplication by an integer is allowed" )
		
		return SizeOf( n = self.bits * x , unit = "b" )
	##}}}
	
	def __rmul__( self , x ):##{{{
		if not isinstance(x,int):
			raise ValueError( "Only multiplication by an integer is allowed" )
		
		return SizeOf( n = self.bits * x , unit = "b" )
	##}}}
	
	def __floordiv__( self , x ):##{{{
		if not isinstance(x,int):
			raise ValueError( "Only division by an integer is allowed" )
		
		return SizeOf( n = self.bits // x , unit = "b" )
	##}}}
	
	def __mod__( self , x ):##{{{
		if not isinstance(x,int):
			raise ValueError( "Only modulo operator by an integer is allowed" )
		
		return SizeOf( n = self.bits % x , unit = "b" )
	##}}}
	
	##}}}
	
##}}}


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
	return [ "S{:{fill}{align}{n}}".format( i , fill = "0" , align = ">" , n = int(np.log10(size-2)) + 1 ) for i in range(size) ]
##}}}

