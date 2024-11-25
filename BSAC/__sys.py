
## Copyright(c) 2023, 2024 Yoann Robin
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
			copy_files( arg , os.path.join( args[-1] , arg ) )
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

