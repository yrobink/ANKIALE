
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

import itertools as itt
import logging
from ..__logs import LINE
from ..__logs import log_start_end
import datetime as dt

import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

from ..__BSACParams import bsacParams
from .__colors import colorsCMIP
from .__colors import colorC
from .__utils  import mpl_rcParams

##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

## plot_covariates ##{{{
@log_start_end(logger)
def plot_covariates( RVS , ofile = None ):
	
	## Matplotlib parameters
	mpl.rcdefaults()
	for key in mpl_rcParams:
		mpl.rcParams[key] = mpl_rcParams[key]
	
	## Parameters
	ci      = bsacParams.config.get("ci")
	nrowcol = bsacParams.config.get("grid")
	
	try:
		ci = float(ci)
	except:
		ci = 0.05
	
	periods = RVS.period.values.tolist()
	try:
		nrow,ncol = [ int(x) for x in nrowcol.split("x") ]
	except:
		total = len(periods)
		if total <= 4:
			nrow,ncol = 2,2
		elif total <= 6:
			nrow,ncol = 2,3
		elif total <= 12:
			nrow,ncol = 3,4
		else:
			ncol = int(np.sqrt(total) * 4/3)
			nrow = total // ncol + 1
	
	##
	XF   = RVS.XF
	XC   = RVS.XC
	XA   = RVS.XA
	time = RVS.time
	now  = dt.datetime.utcnow().year
	mm   = 1. / 25.4
	pt   = 1. / 72
	
	## Build confidence interval
	qXF = XF.quantile( [ci/2 , 0.5 , 1-ci/2] , dim = "sample" , method = "median_unbiased" ).assign_coords( quantile = ["ql","BE","qu"] )
	qXC = XC.quantile( [ci/2 , 0.5 , 1-ci/2] , dim = "sample" , method = "median_unbiased" ).assign_coords( quantile = ["ql","BE","qu"] )
	qXA = XA.quantile( [ci/2 , 0.5 , 1-ci/2] , dim = "sample" , method = "median_unbiased" ).assign_coords( quantile = ["ql","BE","qu"] )
	
	## Find colors
	colorsF = plt.cm.rainbow( np.linspace( 0 , 1 , len(periods) ) ).tolist()
	for iper,per in enumerate(periods):
		if per in colorsCMIP:
			colorsF[iper] = colorsCMIP[per]
	
	## Figure
	if ofile is not None:
		pdf = mpdf.PdfPages(ofile)
	
	for name in RVS.name.values.tolist():
		
		for F in ["FC","A"]:
			
			fig  = plt.figure( figsize = (297*mm,210*mm) , dpi = 120 )
			axes = []
			ylim = [1e9,-1e9]
			
			## Check if original data are available
			inputs = bsacParams.input
			if inputs is None: inputs = []
			idata  = None
			bias   = bsacParams.clim.bias[name]
			for inp in inputs:
				try:
					iname,ifile = inp.split(",")
					if not iname == name:
						continue
					idata = xr.open_dataset(ifile)[name]
					break
				except:
					continue
			
			##
			for iper,per in enumerate(periods):
				
				axes.append( fig.add_subplot( nrow , ncol , iper + 1 ) )
				
				if F == "FC":
					if idata is not None:
						for p,run in itt.product(bsacParams.clim.cper + [per],idata.run):
							axes[-1].plot( idata.time , idata.loc[:,p,run] - bias , color = "grey" , linestyle = "" , marker = "." , markersize = 2 , alpha = 0.5 )
					axes[-1].plot( time , qXF.loc["BE",name,per,:] , color = colorsF[iper] )
					axes[-1].fill_between( time , qXF.loc["ql",name,per,:] , qXF.loc["qu",name,per,:] , color = colorsF[iper] , alpha = 0.5 , label = r"$X_t^{\mathrm{F}}$" )
					axes[-1].plot( time , qXC.loc["BE",name,per,:] , color = colorC )
					axes[-1].fill_between( time , qXC.loc["ql",name,per,:] , qXC.loc["qu",name,per,:] , color = colorC , alpha = 0.5 , label = r"$X_t^{\mathrm{C}}$" )
				else:
					axes[-1].plot( time , qXA.loc["BE",name,per,:] , color = colorsF[iper] )
					axes[-1].fill_between( time , qXA.loc["ql",name,per,:] , qXA.loc["qu",name,per,:] , color = colorsF[iper] , alpha = 0.5 , label = r"$X_t^{\mathrm{A}}$" )
					CC = float(qXA.loc["BE",name,per,now])
					axes[-1].axhline( CC , color = colorsF[iper] , linestyle = ":" , label = "Current Change: {:,g}".format( np.round( CC , 1 ) ) )
				
				axes[-1].axvline( now  , color = "black" , linestyle = ":" , label = "Today" )
				axes[-1].axhline(    0 , color = "black" , linestyle = "-" )
				axes[-1].legend()
				axes[-1].set_title( per.upper() , fontdict = { "family" : "monospace" , "weight" : "bold" , "size" : 9 } )
				
				ylim[0] = min( ylim[0] , axes[-1].get_ylim()[0] )
				ylim[1] = max( ylim[1] , axes[-1].get_ylim()[1] )
			
			for ax in axes:
				ax.set_ylim(ylim)
			
			fig.suptitle( name.upper() + " ({:,g}% C.I.)".format( np.round(100 * (1-ci),2) )  , fontproperties = { "family" : "monospace" , "weight" : "bold" , "size" : 11 } )
			
			plt.tight_layout()
			
			if ofile is None:
				plt.show()
			else:
				pdf.savefig( fig , dpi = 600 )
			plt.close(fig)
	
	if ofile is not None:
		pdf.close()
##}}}

## plot_constrain_CX ##{{{
@log_start_end(logger)
def plot_constrain_CX( RVS , SRVS , ofile = None ):
	
	## Matplotlib parameters
	mpl.rcdefaults()
	for key in mpl_rcParams:
		mpl.rcParams[key] = mpl_rcParams[key]
	
	## Parameters
	ci      = bsacParams.config.get("ci")
	nrowcol = bsacParams.config.get("grid")
	
	try:
		ci = float(ci)
	except:
		ci = 0.05
	
	periods = RVS.period.values.tolist()
	try:
		nrow,ncol = [ int(x) for x in nrowcol.split("x") ]
	except:
		total = len(periods)
		if total <= 4:
			nrow,ncol = 2,2
		elif total <= 6:
			nrow,ncol = 2,3
		elif total <= 12:
			nrow,ncol = 3,4
		else:
			ncol = int(np.sqrt(total) * 4/3)
			nrow = total // ncol + 1
	
	##
	XF   = RVS.XF
	XC   = RVS.XC
	XA   = RVS.XA
	SF   = SRVS.XF
	SC   = SRVS.XC
	SA   = SRVS.XA
	time = RVS.time
	now  = dt.datetime.utcnow().year
	mm   = 1. / 25.4
	pt   = 1. / 72
	
	## Build confidence interval
	qXF = XF.quantile( [ci/2 , 0.5 , 1-ci/2] , dim = "sample" , method = "median_unbiased" ).assign_coords( quantile = ["ql","BE","qu"] )
	qXC = XC.quantile( [ci/2 , 0.5 , 1-ci/2] , dim = "sample" , method = "median_unbiased" ).assign_coords( quantile = ["ql","BE","qu"] )
	qXA = XA.quantile( [ci/2 , 0.5 , 1-ci/2] , dim = "sample" , method = "median_unbiased" ).assign_coords( quantile = ["ql","BE","qu"] )
	qSF = SF.quantile( [ci/2 , 0.5 , 1-ci/2] , dim = "sample" , method = "median_unbiased" ).assign_coords( quantile = ["ql","BE","qu"] )
	qSC = SC.quantile( [ci/2 , 0.5 , 1-ci/2] , dim = "sample" , method = "median_unbiased" ).assign_coords( quantile = ["ql","BE","qu"] )
	qSA = SA.quantile( [ci/2 , 0.5 , 1-ci/2] , dim = "sample" , method = "median_unbiased" ).assign_coords( quantile = ["ql","BE","qu"] )
	
	## Find colors
	colorsF = plt.cm.rainbow( np.linspace( 0 , 1 , len(periods) ) ).tolist()
	for iper,per in enumerate(periods):
		if per in colorsCMIP:
			colorsF[iper] = colorsCMIP[per]
	
	## Figure
	if ofile is not None:
		pdf = mpdf.PdfPages(ofile)
	
	for name in RVS.name.values.tolist():
		
		for F in ["FC","A"]:
			
			fig  = plt.figure( figsize = (297*mm,210*mm) , dpi = 120 )
			axes = []
			ylim = [1e9,-1e9]
			
			##
			for iper,per in enumerate(periods):
				
				axes.append( fig.add_subplot( nrow , ncol , iper + 1 ) )
				
				if F == "FC":
					axes[-1].plot( time , qSF.loc["BE",name,per,:] , color = colorsF[iper] , linestyle = "--" )
					axes[-1].fill_between( time , qSF.loc["ql",name,per,:] , qSF.loc["qu",name,per,:] , color = colorsF[iper] , alpha = 0.2 , label = r"$S_t^{\mathrm{F}}$" )
					axes[-1].plot( time , qSC.loc["BE",name,per,:] , color = colorC , linestyle = "--" )
					axes[-1].fill_between( time , qSC.loc["ql",name,per,:] , qSC.loc["qu",name,per,:] , color = colorC , alpha = 0.2 , label = r"$S_t^{\mathrm{C}}$" )
					axes[-1].plot( time , qXF.loc["BE",name,per,:] , color = colorsF[iper] )
					axes[-1].fill_between( time , qXF.loc["ql",name,per,:] , qXF.loc["qu",name,per,:] , color = colorsF[iper] , alpha = 0.5 , label = r"$X_t^{\mathrm{F}}$" )
					axes[-1].plot( time , qXC.loc["BE",name,per,:] , color = colorC )
					axes[-1].fill_between( time , qXC.loc["ql",name,per,:] , qXC.loc["qu",name,per,:] , color = colorC , alpha = 0.5 , label = r"$X_t^{\mathrm{C}}$" )
				else:
					axes[-1].plot( time , qSA.loc["BE",name,per,:] , color = colorsF[iper] , linestyle = "--" )
					axes[-1].fill_between( time , qSA.loc["ql",name,per,:] , qSA.loc["qu",name,per,:] , color = colorsF[iper] , alpha = 0.2 , label = r"$X_t^{\mathrm{A}}$" )
					axes[-1].plot( time , qXA.loc["BE",name,per,:] , color = colorsF[iper] )
					axes[-1].fill_between( time , qXA.loc["ql",name,per,:] , qXA.loc["qu",name,per,:] , color = colorsF[iper] , alpha = 0.5 , label = r"$X_t^{\mathrm{A}}$" )
					CC = float(qXA.loc["BE",name,per,now])
					axes[-1].axhline( CC , color = colorsF[iper] , linestyle = ":" , label = "Current Change: {:,g}".format( np.round( CC , 1 ) ) )
				
				axes[-1].axvline( now  , color = "black" , linestyle = ":" , label = "Today" )
				axes[-1].axhline(    0 , color = "black" , linestyle = "-" )
				axes[-1].legend()
				axes[-1].set_title( per.upper() , fontdict = { "family" : "monospace" , "weight" : "bold" , "size" : 9 } )
				
				ylim[0] = min( ylim[0] , axes[-1].get_ylim()[0] )
				ylim[1] = max( ylim[1] , axes[-1].get_ylim()[1] )
			
			for ax in axes:
				ax.set_ylim(ylim)
			
			fig.suptitle( name.upper() + " ({:,g}% C.I.)".format( np.round(100 * (1-ci),2) )  , fontproperties = { "family" : "monospace" , "weight" : "bold" , "size" : 11 } )
			
			plt.tight_layout()
			
			if ofile is None:
				plt.show()
			else:
				pdf.savefig( fig , dpi = 600 )
			plt.close(fig)
	
	if ofile is not None:
		pdf.close()
##}}}

