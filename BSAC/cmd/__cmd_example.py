
## Copyright(c) 2023 / 2024 Yoann Robin
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
import logging
import tarfile

from ..__logs import LINE
from ..__logs import log_start_end

from ..__BSACParams import bsacParams



##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

def run_bsac_cmd_example_GMST_script( iopath ):##{{{
	
	## Script
	lsh = ["#!/bin/bash",""]
	
	## Parameters
	lsh.append( "## Parameters" )
	lsh.append( f"N_WORKERS={bsacParams.n_workers}" )
	lsh.append( f"THREADS_PER_WORKER={bsacParams.threads_per_worker}" )
	lsh.append( f"TOTAL_MEMORY={bsacParams.total_memory}" )
	lsh.append( f"WDIR={iopath}" )
	lsh.append( f"N_SAMPLES={bsacParams.n_samples}" )
	lsh.append( f"BIAS_PERIOD='1900/1950'" )
	lsh.append( f"GAM_DOF=7" )
	lsh.append( f"GAM_DEGREE=3" )
	
	## Build path
	lsh.append("")
	lsh.append( r"## Clean and build path" )
	lsh.append( r"for p in LOG FIT SYNTHESIS CONSTRAIN FIGURES" )
	lsh.append( r"do" )
	lsh.append( r"    if [[ -d $WDIR/$p ]]; then rm -rf $WDIR/$p; fi" )
	lsh.append( r"    mkdir $WDIR/$p" )
	lsh.append( r"done" )
	
	## Find the model list
	lsh.append("")
	lsh.append( r"## Find the model list" )
	lsh.append( r"MODELS=$(ls $WDIR/INPUT/GMST/X)" )
	
	## Loop on models
	lsh.append("")
	lsh.append( r"## Loop on models to fit the statistical model" )
	lsh.append( r"for MODNC in $MODELS" )
	lsh.append( r"do" )
	lsh.append( r"    MOD=$(basename $MODNC .nc)" )
	lsh.append( r"    bsac fit X --n-samples $N_SAMPLES --log info $WDIR/LOG/'FITX_'$MOD.log" + "\\" )
	lsh.append( r"               --bias-period $BIAS_PERIOD" + "\\" )
	lsh.append( r"               --input GMST,$WDIR/INPUT/GMST/X/$MODNC" + "\\" )
	lsh.append( r"               --save-clim $WDIR/FIT/$MOD'_fitX.nc'" + "\\" )
	lsh.append( r"               --common-period historical --different-periods ssp126,ssp245,ssp370,ssp585" + "\\" )
	lsh.append( r"               --config GAM_dof=$GAM_DOF,GAM_degree=$GAM_DEGREE" + "\\" )
	lsh.append( r"               --total-memory $TOTAL_MEMORY" + "\\" )
	lsh.append( r"               --n-workers $N_WORKERS" + "\\" )
	lsh.append( r"               --threads-per-worker $THREADS_PER_WORKER" )
	lsh.append( "" )
	lsh.append( r"    bsac show X --n-samples $N_SAMPLES --log info $WDIR/LOG/'SHOWX_'$MOD.log" + "\\" )
	lsh.append( r"                --bias-period $BIAS_PERIOD" + "\\" )
	lsh.append( r"                --load-clim $WDIR/FIT/$MOD'_fitX.nc'" + "\\" )
	lsh.append( r"                --input GMST,$WDIR/INPUT/GMST/X/$MODNC" + "\\" )
	lsh.append( r"                --output $WDIR/FIGURES/BSAC_SHOW_X_$MOD.pdf" + "\\" )
	lsh.append( r"                --total-memory $TOTAL_MEMORY" + "\\" )
	lsh.append( r"                --n-workers $N_WORKERS" + "\\" )
	lsh.append( r"                --threads-per-worker $THREADS_PER_WORKER" )
	lsh.append( "" )
	lsh.append( r"done" )
	
	## Synthesis
	lsh.append("")
	lsh.append( r"## Run synthesis" )
	lsh.append( r"bsac synthesize --log info $WDIR/LOG/SYNTHESIS.log" + "\\" )
	lsh.append( r"                --bias-period $BIAS_PERIOD" + "\\" )
	lsh.append( r"                --input $WDIR/FIT/*.nc" + "\\" )
	lsh.append( r"                --common-period historical --different-periods ssp126,ssp245,ssp370,ssp585" + "\\" )
	lsh.append( r"                --config GAM_dof=$GAM_DOF,GAM_degree=$GAM_DEGREE,names=GMST" + "\\" )
	lsh.append( r"                --save-clim $WDIR/SYNTHESIS/SYNTHESIS.nc" + "\\" )
	lsh.append( r"                --total-memory $TOTAL_MEMORY" + "\\" )
	lsh.append( r"                --n-workers $N_WORKERS" + "\\" )
	lsh.append( r"                --threads-per-worker $THREADS_PER_WORKER" )
	lsh.append( "" )
	lsh.append( r"bsac show X --n-samples $N_SAMPLES --log info $WDIR/LOG/SHOWX_SYNTHESIS.log" + "\\" )
	lsh.append( r"            --bias-period $BIAS_PERIOD" + "\\" )
	lsh.append( r"            --load-clim $WDIR/SYNTHESIS/SYNTHESIS.nc" + "\\" )
	lsh.append( r"            --output $WDIR/FIGURES/BSAC_SHOW_X_SYNTHESIS.pdf" + "\\" )
	lsh.append( r"            --total-memory $TOTAL_MEMORY" + "\\" )
	lsh.append( r"            --n-workers $N_WORKERS" + "\\" )
	lsh.append( r"            --threads-per-worker $THREADS_PER_WORKER" )
	
	## Constraint
	lsh.append("")
	lsh.append( r"## Run constraint" )
	lsh.append( r"bsac constrain X --log info $WDIR/LOG/CONSTRAINX.log" + "\\" )
	lsh.append( r"                 --bias-period $BIAS_PERIOD" + "\\" )
	lsh.append( r"                 --load-clim $WDIR/SYNTHESIS/SYNTHESIS.nc" + "\\" )
	lsh.append( r"                 --save-clim $WDIR/CONSTRAIN/CONSTRAINX.nc" + "\\" )
	lsh.append( r"                 --input GMST,$WDIR/INPUT/GMST/Xo/GISTEMP_tas_year_1880-2023.nc" + "\\" )
	lsh.append( r"                 --total-memory $TOTAL_MEMORY" + "\\" )
	lsh.append( r"                 --n-workers $N_WORKERS" + "\\" )
	lsh.append( r"                 --threads-per-worker $THREADS_PER_WORKER" )
	lsh.append( "" )
	lsh.append( r"bsac show X --n-samples $N_SAMPLES --log info $WDIR/LOG/SHOWX_CONSTRAINX.log" + "\\" )
	lsh.append( r"            --bias-period $BIAS_PERIOD" + "\\" )
	lsh.append( r"            --load-clim $WDIR/CONSTRAIN/CONSTRAINX.nc" + "\\" )
	lsh.append( r"            --input $WDIR/INPUT/GMST/Xo/GISTEMP_tas_year_1880-2023.nc" + "\\" )
	lsh.append( r"            --output $WDIR/FIGURES/BSAC_SHOW_X_CONSTRAINX.pdf" + "\\" )
	lsh.append( r"            --total-memory $TOTAL_MEMORY" + "\\" )
	lsh.append( r"            --n-workers $N_WORKERS" + "\\" )
	lsh.append( r"            --threads-per-worker $THREADS_PER_WORKER" )
	lsh.append( "" )
	lsh.append( r"bsac show CX --n-samples $N_SAMPLES --log info $WDIR/LOG/SHOWCX_CONSTRAINX-VS-SYNTHESIS.log" + "\\" )
	lsh.append( r"             --bias-period $BIAS_PERIOD" + "\\" )
	lsh.append( r"             --load-clim $WDIR/CONSTRAIN/CONSTRAINX.nc" + "\\" )
	lsh.append( r"             --input $WDIR/SYNTHESIS/SYNTHESIS.nc" + "\\" )
	lsh.append( r"             --output $WDIR/FIGURES/BSAC_SHOW_CX_CONSTRAINX-VS-SYNTHESIS.pdf" + "\\" )
	lsh.append( r"             --total-memory $TOTAL_MEMORY" + "\\" )
	lsh.append( r"             --n-workers $N_WORKERS" + "\\" )
	lsh.append( r"             --threads-per-worker $THREADS_PER_WORKER" )
	
	## End
	lsh.append("")
	lsh.append("## END OF SCRIPT")
	
	return "\n".join(lsh)
##}}}

## run_bsac_cmd_example_GMST ##{{{
@log_start_end(logger)
def run_bsac_cmd_example_GMST():
	
	## Find output folder
	iopath = os.path.abspath(bsacParams.output)
	if not os.path.isdir(iopath):
		raise NotADirectoryError( f"{iopath} is not a path" )
	logger.info( f" * Output path found: {iopath}" )
	
	## Copy data
	logger.info( f" * Copy data" )
	cpath = os.path.dirname(os.path.abspath(__file__))
	idata = os.path.join( cpath , ".." , "data" , "GMST.tar.gz" )
	with tarfile.open( idata , mode = "r" ) as ifile:
		ifile.extractall( os.path.join( iopath , "INPUT" ) )
	
	## Copy script
	logger.info( f" * Copy script" )
	sh = run_bsac_cmd_example_GMST_script( iopath )
	with open( os.path.join( iopath , "RUN_BSAC_EXAMPLE_GMST.sh" ) , "w" ) as ofile:
		ofile.write(sh)
	print(sh)
##}}}

## run_bsac_cmd_example ##{{{
@log_start_end(logger)
def run_bsac_cmd_example():
	
	## Check the command
	if not len(bsacParams.arg) == 1:
		raise ValueError(f"Bad numbers of arguments of the example command: {', '.join(bsacParams.arg)}")
	
	available_commands = ["GMST"]
	if not bsacParams.arg[0] in available_commands:
		raise ValueError(f"Bad argument of the fit command ({bsacParams.arg[0]}), must be: {', '.join(available_commands)}")
	
	## OK, run the good command
	if bsacParams.arg[0] == "GMST":
		run_bsac_cmd_example_GMST()
##}}}



