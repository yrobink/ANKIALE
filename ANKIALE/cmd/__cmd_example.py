
## Copyright(c) 2024, 2025 Yoann Robin
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
import datetime
import toml
import shutil
import tarfile
import logging
import itertools as itt

from ..__exceptions import DevException
from ..__logs import log_start_end

from ..__ANKParams import ankParams



##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

def find_path( output: str ) -> tuple[str,str]:##{{{
    
    ## Find output folder
    if "," in output:
        hiopath,diopath = ankParams.output.split(",")
        hiopath = os.path.abspath(hiopath)
        diopath = os.path.abspath(diopath)
        for p in [hiopath,diopath]:
            if not os.path.isdir(p):
                raise NotADirectoryError( f"{p} is not a path" )
    else:
        iopath = os.path.abspath(output)
        if not os.path.isdir(iopath):
            raise NotADirectoryError( f"{iopath} is not a path" )
        hiopath = os.path.join( iopath , "home" )
        diopath = os.path.join( iopath , "data" )
        for p in [hiopath,diopath]:
            if not os.path.isdir(p):
                os.makedirs(p)
    logger.info( f" * Home path found: {hiopath}" )
    logger.info( f" * Data path found: {diopath}" )
    
    return hiopath,diopath
##}}}

## run_ank_cmd_example ##{{{
@log_start_end(logger)
def run_ank_cmd_example( short: bool ) -> None:
    
    cpath = os.path.dirname(os.path.abspath(__file__))
    
    ## Check the command
    if not len(ankParams.arg) == 1:
        raise ValueError(f"Bad numbers of arguments of the example command: {', '.join(ankParams.arg)}")

    ## Path
    hpath,dpath = find_path( ankParams.output )
    opath = os.path.join( dpath , "INPUT" )
    epath = os.path.join( cpath , ".." , "data" , "EXAMPLE" )
    
    ## Find available commands
    full_config  = toml.load( os.path.join( epath , "CONFIGURATION.toml" ) )
    available_commands = list(full_config)
    cmd = ankParams.arg[0].upper()
    if cmd not in available_commands:
        raise ValueError(f"Bad argument of the fit command ({cmd}), must be: {', '.join(available_commands)}")
    
    ## Open configuration
    config  = full_config[cmd]
    cnames  = list(config["X"])
    bper    = config["bper"]
    dpers   = config["dpers"].split(",") if ankParams.dpers is None else ankParams.dpers
    if not isinstance(dpers,list):
        dpers = [dpers]
    cconfig = " ".join([f"{cname}:{dper}:8" for cname,dper in itt.product(cnames,dpers)])
    mconfig = ""
    for v in ["time","names","cname","vname","nslaw","spatial"]:
        if v in config:
            mconfig = mconfig + f" --{v} {config[v]}"

    ## Step 1, initial configuration of the script ##{{{
    
    step1_start = "#!/bin/bash"
    step1_licence = """
## Copyright(c) 2024 / {} Yoann Robin
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
## along with ANKIALE.  If not, see <https://www.gnu.org/licenses/>.""".format( datetime.datetime.now(datetime.UTC).year )
    
    step1_par = """
## Parameters
N_WORKERS={}
THREADS_PER_WORKER={}
MEMORY_PER_WORKER={}
HPATH={}
DPATH={}
N_SAMPLES={}""".format( ankParams.n_workers,
                ankParams.threads_per_worker,
                ankParams.memory_per_worker,
                hpath,
                dpath,
                ankParams.n_samples,
                )
    
    step1_fpar = """
## Fixed Parameters
BIAS_PERIOD='{}'
DPERS='{}'
MISC_CONFIG='{}'
CCONFIG='{}'""".format( bper,
                        ",".join(dpers),
                        mconfig,
                        cconfig,
                    )
    
    step1_path = """
## Clean and build path
for p in $HPATH/LOG $DPATH/FIT $DPATH/SYNTHESIS $DPATH/ANALYSIS $DPATH/CONSTRAIN $DPATH/FIGURES
do
    if [[ ! -d $p ]]
    then
        mkdir $p
    fi
done
    """
    script = "\n".join([ step1_start,
                        step1_licence,
                        step1_par,
                        step1_fpar,
                        step1_path,
                        ])

    ##}}}
    
    ## Step 2, Fit climate models ##{{{
    cX = config.get("X")
    cY = config.get("Y")
    
    ## Step 2.1, copy required files ##{{{
    
    for key in cX:
        with tarfile.open( os.path.join( epath , f"{cX[key]}.tar.gz" ) , mode = "r:gz" ) as igz:
            igz.extractall( opath )
        p = os.path.join( opath , f"{cX[key]}" )
        for f in os.listdir(p):
            if f.startswith("."):
                os.remove( os.path.join( p , f ) )
    
    if cY is not None:
        with tarfile.open( os.path.join( epath , f"{cY}.tar.gz" ) , mode = "r:gz" ) as igz:
            igz.extractall( opath )
        p = os.path.join( opath , cY )
        for f in os.listdir(p):
            if f.startswith("."):
                os.remove( os.path.join( p , f ) )
    
    ##}}}
    
    ## Step 2.2, update the script ##{{{
    gen_script_model_begX = r"""
## Build output dir
mkdir $DPATH/FIT/X

## Find the model list
MODELS=$(ls $DPATH/INPUT/{})

    """.format
    
    gen_script_model_begY = lambda s0,s1: r"""
## Build output dir
mkdir $DPATH/FIT/X
mkdir $DPATH/FIT/Y

## Find the model list
MODELS0=$(ls $DPATH/INPUT/{})
MODELS1=$(ls $DPATH/INPUT/{})
MODELS=""
for MOD in $MODELS1
do
    if [[ ${}MODELS0[@]{} =~ $MOD ]]
    then
        MODELS=$MODELS" $MOD"
    fi
done
    """.format( s0 , s1 , "{","}")

    gen_script_model_start_loop = r"""
## Loop on models to fit the statistical model
echo "Loop on models"
for MODNC in $MODELS
do
    MOD=$(basename $MODNC .nc)
    echo " * $MOD"
    """.format
    
    gen_script_model_start_loop_short = r"""
## Loop on models to fit the statistical model
echo "Loop on models"
ILOOP=0
for MODNC in $MODELS
do
    ILOOP=$((ILOOP+1))
    if (( $ILOOP > 5 )); then break; fi
    
    MOD=$(basename $MODNC .nc)
    echo " * $MOD"
    """.format
    
    gen_script_model_X = r"""
    echo "   => fit X"
    ank fit X --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/'FITX_'$MOD.log\
               --input {}\
               --save-clim $DPATH/FIT/X/FITX_$MOD'.nc'\
               --common-period historical --different-periods $DPERS\
               --bias-period $BIAS_PERIOD\
               --covar-config $CCONFIG\
               $MISC_CONFIG\
               --memory-per-worker $MEMORY_PER_WORKER\
               --n-workers $N_WORKERS\
               --threads-per-worker $THREADS_PER_WORKER
    """.format

    gen_script_model_showX = r"""
    echo "   => show X"
    ank show X --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/'SHOWX_'$MOD.log\
                --input {}\
                --output $DPATH/FIGURES/SHOWX_$MOD.pdf\
                --load-clim $DPATH/FIT/X/FITX_$MOD'.nc'\
                --common-period historical --different-periods $DPERS\
                --bias-period $BIAS_PERIOD\
                --covar-config $CCONFIG\
                $MISC_CONFIG\
                --memory-per-worker $MEMORY_PER_WORKER\
                --n-workers $N_WORKERS\
                --threads-per-worker $THREADS_PER_WORKER
    """.format
    
    gen_script_model_Y = r"""
    echo "   => fit Y"
    ank fit Y --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/'FITY_'$MOD.log\
               --input {}\
               --load-clim $DPATH/FIT/X/FITX_$MOD'.nc'\
               --save-clim $DPATH/FIT/Y/FITY_$MOD'.nc'\
               --common-period historical --different-periods $DPERS\
               --bias-period $BIAS_PERIOD\
               --covar-config $CCONFIG\
               $MISC_CONFIG\
               --memory-per-worker $MEMORY_PER_WORKER\
               --n-workers $N_WORKERS\
               --threads-per-worker $THREADS_PER_WORKER
    """.format
    if cY is not None:
        script = script + gen_script_model_begY( cX[list(cX)[0]] , cY )
    else:
        script = script + gen_script_model_begX(cX[list(cX)[0]])
    if short:
        script = script + gen_script_model_start_loop_short()
    else:
        script = script + gen_script_model_start_loop()
    script = script + gen_script_model_X( " ".join(["{},$DPATH/INPUT/{}/$MODNC".format( key , cX[key] ) for key in cX]) )\
                    + gen_script_model_showX( " ".join(["{},$DPATH/INPUT/{}/$MODNC".format( key , cX[key] ) for key in cX]) )
    if cY is not None:
        script = script + gen_script_model_Y( r"$DPATH/INPUT/{}/$MODNC".format(cY) )
    script = script + "\n" + "done" + "\n"
    
    ##}}}
    
    ##}}}
    
    ## Step 3, synthesis ##{{{
    
    ## Step 3.1 Copy required file ##{{{
    
    if "GRID" in config:
        os.makedirs( os.path.join( opath , "INFOS" ) )
        shutil.copy( os.path.join( epath , "INFOS" , config["GRID"]["FILE"] ),
                     os.path.join( opath , "INFOS" , config["GRID"]["FILE"] )
                   )

    ##}}}

    ## Step 3.2 Generator ##{{{
    gen_script_synthesis = r"""
## Run synthesis
echo " * synthesis"
ank synthesize -v --log-file $HPATH/LOG/SYNTHESIS.log\
                --input $DPATH/FIT/{}/*.nc\
                --save-clim $DPATH/SYNTHESIS/SYNTHESIS.nc\
                --common-period historical --different-periods $DPERS\
                --bias-period $BIAS_PERIOD\
                --covar-config $CCONFIG\
                {} $MISC_CONFIG\
                --memory-per-worker $MEMORY_PER_WORKER\
                --n-workers $N_WORKERS\
                --threads-per-worker $THREADS_PER_WORKER
    """.format
    
    gen_script_show_synthesis = r"""
echo " * show X of synthesis"
ank show X --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/SHOWX_SYNTHESIS.log\
            --load-clim $DPATH/SYNTHESIS/SYNTHESIS.nc\
            --output $DPATH/FIGURES/SHOWX_SYNTHESIS.pdf\
            --common-period historical --different-periods $DPERS\
            --bias-period $BIAS_PERIOD\
            --covar-config $CCONFIG\
            $MISC_CONFIG\
            --memory-per-worker $MEMORY_PER_WORKER\
            --n-workers $N_WORKERS\
            --threads-per-worker $THREADS_PER_WORKER
    """.format

    _config = ""
    if cY is not None:
        if "GRID" in config:
            _config = "--config grid=$DPATH/INPUT/INFOS/{},grid_name='{}'".format( config["GRID"]["FILE"] , config["GRID"]["NAME"] )
    p = "X" if "Y" not in config else "Y"
    script = script + gen_script_synthesis( p , _config )
    script = script + gen_script_show_synthesis()
    
    ##}}}
     
    ##}}}
    
    ## Step 4, constraint of co-variables ##{{{

    ## Step 4.1, copy files ##{{{
    if not config["Xo"] == "None":
        cnamesXo = [ cname for cname in cnames if cname in config["Xo"] ]
        os.makedirs( os.path.join( opath , "Xo" ) )
        for cname in cnamesXo:
            shutil.copy( os.path.join( epath , "Xo" , config["Xo"][cname] ),
                         os.path.join( opath , "Xo" , config["Xo"][cname] ),
                        )
    
    ##}}}

    ## Step 4.2, scripts ##{{{
    gen_constraint_X = r"""
## Run constraint
echo " * constraint X of observations"
ank constrain X -v --log-file $HPATH/LOG/CONSTRAINX.log\
                 --input {}\
                 --load-clim $DPATH/SYNTHESIS/SYNTHESIS.nc\
                 --save-clim $DPATH/CONSTRAIN/CONSTRAINX.nc\
                 --common-period historical --different-periods $DPERS\
                 --bias-period $BIAS_PERIOD\
                 --covar-config $CCONFIG\
                 $MISC_CONFIG\
                 --config method_oerror=IND,method_constraint='{}'\
                 --memory-per-worker $MEMORY_PER_WORKER\
                 --n-workers $N_WORKERS\
                 --threads-per-worker $THREADS_PER_WORKER
    """.format

    gen_constraint_showX = r"""
echo " * show X of constraint"
ank show X --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/SHOWX_CONSTRAINX.log\
            --load-clim $DPATH/CONSTRAIN/CONSTRAINX.nc\
            --output $DPATH/FIGURES/SHOWCX_CONSTRAINX.pdf\
            --common-period historical --different-periods $DPERS\
            --bias-period $BIAS_PERIOD\
            --covar-config $CCONFIG\
            $MISC_CONFIG\
            --memory-per-worker $MEMORY_PER_WORKER\
            --n-workers $N_WORKERS\
            --threads-per-worker $THREADS_PER_WORKER
    """.format
    
    gen_constraint_showXS = r"""
echo " * show X of constraint VS synthesis"
ank show CX --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/SHOWCX_CONSTRAINX-VS-SYNTHESIS.log\
             --load-clim $DPATH/CONSTRAIN/CONSTRAINX.nc\
             --input $DPATH/SYNTHESIS/SYNTHESIS.nc\
             --output $DPATH/FIGURES/SHOWCX-VS-S_CONSTRAINX.pdf\
             --common-period historical --different-periods $DPERS\
             --bias-period $BIAS_PERIOD\
             --covar-config $CCONFIG\
             $MISC_CONFIG\
             --memory-per-worker $MEMORY_PER_WORKER\
             --n-workers $N_WORKERS\
             --threads-per-worker $THREADS_PER_WORKER
    """.format
    
    if not config["Xo"] == "None":
        script = script + gen_constraint_X( " ".join( [f"{cname},$DPATH/INPUT/Xo/{config['Xo'][cname]}" for cname in cnamesXo] ), "::".join( [f"{cname}:full" for cname in cnamesXo] )  )\
                        + gen_constraint_showX()\
                        + gen_constraint_showXS()
    ##}}}

    ##}}}
    
    ## Step 5, constraint of variable ##{{{
    
    ## Step 5.1, copy data ##{{{
    if 'Yo' in config:
        vname = list(config["Yo"])[0]
        os.makedirs( os.path.join( opath , "Yo" ) )
        shutil.copy( os.path.join( epath , "Yo" , config["Yo"][vname] ),
                     os.path.join( opath , "Yo" , config["Yo"][vname] ),
                    )
    
    ##}}}

    ## Step 5.2, build script ##{{{
    gen_script_constraintY = r"""
echo " * constraint Y of observations"
ank constrain Y --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/CONSTRAINY.log\
                 --load-clim $DPATH/CONSTRAIN/CONSTRAINX.nc\
                 --save-clim $DPATH/CONSTRAIN/CONSTRAINY.nc\
                 --input {},$DPATH/INPUT/Yo/{}\
                 --output $DPATH/CONSTRAIN/SAMPLES_CONSTRAINY.nc\
                 --common-period historical --different-periods $DPERS\
                 --bias-period $BIAS_PERIOD\
                 --covar-config $CCONFIG\
                 $MISC_CONFIG\
                 --config size-chain=50,method_constraint=full\
                 --memory-per-worker $MEMORY_PER_WORKER\
                 --n-workers $N_WORKERS\
                 --threads-per-worker $THREADS_PER_WORKER
    """.format
    
    if "Yo" in config:
        script = script + gen_script_constraintY( vname, config["Yo"][vname] )
    
    ##}}}
    
    ##}}}
    
    ## Step 6, attribution ##{{{
    
    ## Step 6.1, copy data ##{{{
    if "ATTRIBUTION" in config:
        keys = list(config["ATTRIBUTION"])
        for key in keys:
            try:
                os.makedirs( os.path.join( opath , "Yo" ) )
            except:
                pass
            if "FILE" in config["ATTRIBUTION"][key]:
                shutil.copy( os.path.join( epath , "Yo" , config["ATTRIBUTION"][key]["FILE"] ),
                             os.path.join( opath , "Yo" , config["ATTRIBUTION"][key]["FILE"] ),
                )

    ##}}}
    
    ## Step 6.2, build the script ##{{{
    gen_script_attribution0 = r"""
## Attribution
echo " * Attribution: freturnt"
ank attribute freturnt --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/ANALYSIS_ATTR-FRETURNT.log\
                        --load-clim $DPATH/CONSTRAIN/CONSTRAINY.nc\
                        --input 2,10,30,50,100,1000\
                        --output $DPATH/ANALYSIS/FRETURNT.nc\
                        --common-period historical --different-periods $DPERS\
                        --bias-period $BIAS_PERIOD\
                        --covar-config $CCONFIG\
                        $MISC_CONFIG\
                        --config mode=quantile,side={}\
                        --memory-per-worker $MEMORY_PER_WORKER\
                        --n-workers $N_WORKERS\
                        --threads-per-worker $THREADS_PER_WORKER
    """.format

    gen_script_attribution1 = r"""
echo " * Attribution: fintensity"
ank attribute fintensity --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/ANALYSIS_ATTR-FINTENSITY.log\
                          --load-clim $DPATH/CONSTRAIN/CONSTRAINY.nc\
                          --input {},$DPATH/INPUT/Yo/{}\
                          --output $DPATH/ANALYSIS/FINTENSITY-MAP.nc\
                          --common-period historical --different-periods $DPERS\
                          --bias-period $BIAS_PERIOD\
                          --covar-config $CCONFIG\
                          $MISC_CONFIG\
                          --config mode=quantile,side={}\
                          --memory-per-worker $MEMORY_PER_WORKER\
                          --n-workers $N_WORKERS\
                          --threads-per-worker $THREADS_PER_WORKER
    """.format

    gen_script_attribution2 = r"""
echo " * Attribution: fintensity"
ank attribute fintensity --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/ANALYSIS_ATTR-FINTENSITY.log\
                          --load-clim $DPATH/CONSTRAIN/CONSTRAINY.nc\
                          --input {}\
                          --output $DPATH/ANALYSIS/FINTENSITY-VALUE.nc\
                          --common-period historical --different-periods $DPERS\
                          --bias-period $BIAS_PERIOD\
                          --covar-config $CCONFIG\
                          $MISC_CONFIG\
                          --config mode=quantile,side={}\
                          --memory-per-worker $MEMORY_PER_WORKER\
                          --n-workers $N_WORKERS\
                          --threads-per-worker $THREADS_PER_WORKER
    """.format

    gen_script_attribution3 = r"""
echo " * Attribution: event"
ank attribute event --n-samples $N_SAMPLES -v --log-file $HPATH/LOG/ANALYSIS_ATTR-EVENT{}.log\
                     --load-clim $DPATH/CONSTRAIN/CONSTRAINY.nc\
                     --input {},$DPATH/INPUT/Yo/{}\
                     --output $DPATH/ANALYSIS/EVENT{}.nc\
                     --common-period historical --different-periods $DPERS\
                     --bias-period $BIAS_PERIOD\
                     --covar-config $CCONFIG\
                     $MISC_CONFIG\
                     --config time={},mode=quantile,side={}\
                     --memory-per-worker $MEMORY_PER_WORKER\
                     --n-workers $N_WORKERS\
                     --threads-per-worker $THREADS_PER_WORKER
    """.format
    
    if "ATTRIBUTION" in config:
        if "FRETURNT" in config["ATTRIBUTION"]:
            d = config["ATTRIBUTION"]["FRETURNT"]
            script = script + gen_script_attribution0( d["SIDE"] )
        
        if "FINTENSITY-MAP" in config["ATTRIBUTION"]:
            d = config["ATTRIBUTION"]["FINTENSITY-MAP"]
            script = script + gen_script_attribution1( d["NAME"] , d["FILE"] , d["SIDE"] )
        
        if "FINTENSITY-VALUE" in config["ATTRIBUTION"]:
            d = config["ATTRIBUTION"]["FINTENSITY-VALUE"]
            script = script + gen_script_attribution2(  d["VALUE"] , d["SIDE"] )
        
        if "EVENT" in config["ATTRIBUTION"]:
            d = config["ATTRIBUTION"]["EVENT"]
            script = script + gen_script_attribution3(  d["YEAR"], d["NAME"] , d["FILE"] , d["YEAR"] , d["YEAR"] , d["SIDE"] )

    ##}}}

    ##}}}

    ## End of script ##{{{
    script = script + "\n\n## End of script"
    
    ##}}}
    
    ## And save the script ##{{{
    with open( os.path.join( hpath , f"RUN_ANKIALE_EXAMPLE_{cmd}.sh" ) , "w" ) as ofile:
        ofile.write(script)

    ##}}}
    
##}}}


