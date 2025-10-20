
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

from .__release import version
from .__release import long_description
from .__release import license
from .__release import license_txt
from .__release import src_url
from .__release import authors_doc

###############
## Variables ##
###############

doc = """\

ANKIALE ({})
{}

{}

================================================================================

If you are using ANKIALE for the first time, start by running the following
command in a directory <output_dir>:

ank example NORMAL-FRA03 --output <output_dir>

And take a look at the <output_dir>/home/RUN_ANKIALE_EXAMPLE_NORMAL_FRA03.sh
script, which runs the attribution of the August 2003 heatwave that took
place in France.

================================================================================

Common arguments
----------------
--help
    Ask to see the documentation
-V
    Show the version
--log-level
    Set the log level, default is 'warning'.
--log-file 
    Set the output log file, the default output is the console.
-v --verbose
    Set the log-level to INFO
-d --debug
    Set the log-level to DEBUG
--cluster
    Cluster used by dask, can be 'PROCESS' or 'THREADING'
--n-workers
    Numbers of CPU used.
--threads-per-worker
    Numbers of threads per worker
--memory-per-worker
    Memory per worker. If not given, the value is total_memory / n_workers
--total-memory
    Total memory. If not given, 80% of the system memory is used.
--tmp
    Root of the temporary folders. If not given, default value of the tempdir
    package is used.

--load-clim ifile
    Load a climatology from the disk.
--save-clim ofile
    Save a climatology to the disk.

--set-seed int
    Set the seed for reproductibility

--n-samples
    Numbers of samples used, when resample is required
--common-period
    Name of time period common, classicaly this is 'historical'
--different-periods
    A comma separated list of the different future periods. For CMIP6, it can
    be a list of the form 'ssp126,ssp245,ssp370,ssp585'
--bias-period
    Anomaly time period, in the forme 'year_start/year_end', for example
    '1961/1990'
--covar-config
    Degree of freedom of the spline for each covariate and period. For example,
    for two covariate 'GMST' and 'EU', and two periods 'ssp245' and 'ssp585',
    the expected value takes the form
    'GMST:ssp245:8 GMST:ssp585:8 EU:ssp245:8 EU:ssp585:8'
--time
    Time axis, of the form
    'common_year_start/common_year_end/different_year_end'.
    For CMIP6, it is classicaly '1850/2014/2100'
--names
    List of name of covariable and variable.
--cname
    Set the covariable used for the regression.
--vname
    Set the name of the variable in the list '--names'
--nslaw
    Set the non-stationary distribution used. Currently, it can be 'Normal',
    'GEV' or 'GEVMin'
--spatial
    Spatial coordinates, separated by ':'.
--XN-version
    Set the version of natural response used. Can be 'CMIP5' or 'CMIP6'.
--no-STAN
    Do not use STAN for the MCMC, but the old algorithm from SDFC, slower.

The example command
-------------------

ank example EXAMPLE_NAME --output OUTPUT_DIR

This command copies the EXAMPLE_NAME example into the OUTPUT_DIR directory. Two
sub-directories are created: OUTPUT_DIR/data, which contains the data, and
OUTPUT_DIR/home, which contains the script used to run the example. If
OUTPUT_DIR is of the form home_path,data_path (two paths separated by a ‘,’)
then home_path is used as home and data_path as data. Global parameters passed
to ANKIALE (as the number of samples, CPU, memory, periods, etc) are propagated
to the scripts.

Possible examples for scenarios SSP1-2.6, SSP2-4.5, SSP3-7.0 and SSP5-8.5:
- GMST: Global warming estimates.
- GMST-EU: Global and Regional (Europe) warming estimates.
- NORMAL-FRA03: Normal distribution analysis of the average August temperature
  in France. Used to analyse the August 2003 heatwave.
- GEV-PARIS: GEV law analysis of TX3x in Paris (France). Enable analysis of
  the July 2019 heatwave.
- GEV-IDF: GEV analysis of TX3x over the Ile de France (France).
- GEVMIN-FRA12: Analysis with the GEVMin law of the TN10n over France. Enable
  analysis of the February 2012 cold snap.


The 'show' command
------------------
* show XN
    Plot a figure of the response of the natural forcings used for covariates.
    --output ofile
        A file to save the figure. If not given the function
        matplotlib.pyplot.show() is called.
* show X (dont forgive the --load-clim parameter)
    Plot the covariates fitted.
    --input cvar,ifile0 cvar,ifile1,...
        Data used to fit the climatology. Optional, added to the plot only if
        given. Must be given in the format cvar,ifile where cvar is the name of
        the variable and ifile the name of the file
    --output ofile.pdf
        A file to save the figure. The file must be a pdf file because multiple
        figures are produced. If not given the function
        matplotlib.pyplot.show() is called.
    --config param0=value0,param1=value1,...
        Parameters to custumize the plot. Currently:
        grid: a value of the form 'nrowxncol', to have a plot with nrow subplots
              and ncol subplots.
        ci: Level of the confidence interval. Default is 0.05 (95% C.I.)
* show CX (dont forgive the --load-clim parameter)
    Plot the covariates constrained and the initial climatology.
    --input synthesis
        Climatology of the synthesis.
    --output ofile.pdf
        A file to save the figure. The file must be a pdf file because multiple
        figures are produced. If not given the function
        matplotlib.pyplot.show() is called.
    --config param0=value0,param1=value1,...
        Parameters to custumize the plot. Currently:
        grid: a value of the form 'nrowxncol', to have a plot with nrow subplots
              and ncol subplots.
        ci: Level of the confidence interval. Default is 0.05 (95% C.I.)


The 'fit' command
-----------------
* fit X
    Fit the covariable(s) (dont forgive the --save-clim parameter). Two
    important points:
    - The time axis used is given by the parameter '--time'. If some values are
      missing (as example the year 1900), a random values is used to fill the
      time series. The method is to draw a value from a normal distribution,
      with the mean and the standard deviation computed on 30 years centered on
      the missing point.
    - The different periods of the parameter --different-periods must be in the
      input file, else an exception is raised.
    
    --input name0,ifile0 name1,ifile1,...
        Data used to fit the climatology, in the form 'name,file'
* fit Y
    Fit the variable with the covariable fitted from --load-clim (again, dont
    forgive the --save-clim parameter). Two important points:
    - Here, unlike “fit X”, missing values are not a problem and regression is
      performed according to the available time steps.
    - The different periods of the parameter --different-periods must be in the
      input file, else an exception is raised.
    
    --input ifile
        Data used to fit the climatology.


The 'synthesize' command
------------------------
* synthesize
    This command constructs the multi-model synthesis (the prior) from a
    climatology list constructed with 'fit X' or 'fit Y'.
    --input clim0.nc, clim1.nc,...
        Climatology fitted to synthesize
    --config param0=value0,param1=value1,...
        grid: a file describing the grid of the synthesize (clim will be remap
              on this grid)
        grid_name: name of the variable for the grid
        method_synthesis: Synthesis method used. This one of 'Ribes2017' or
              'classic' (covariance matrix do not use the covariance matrix
              of each individual model).


The 'constrain' command
-----------------------
* constrain X
    --input name0,ifile0 name1,ifile1,...
        Observations used for the constraint, in the form 'name,file'. The file
        is a single time series, applied to all grid point.
    --config param0=value0,param1=value1,...
        method_oerror: 3 methods are available:
            * 'IND', observed error assume independence for
               autocorrelation, default;
            * 'MAR2': Mixture of two AR1 process, but multiple covariates are
              independent;
            * 'KCC': As MAR2, with dependence between 2 covariates.
        method_constraint: set which covariate is constrain, and with what
        period. For example, if covariates are 'GMST' and 'EU', and 'GMST' is
        constrained by all scenarios, but 'EU' only by the ssp245, the input
        is: 'GMST:full::EU:ssp245'
* constrain Y
    --input name,ifile
        Observations used for the constraint, in the form 'name,ifile'.
    --config param0=value0,param1=value1,...
        size_chain: Length of each MCMC chain
        method_constraint: if all periods (use 'full' as value in this case) or
        only one (use the name of the period) is used to constrain.
    --output ofile
        Optionnal, if given samples are stored in the file ofile. Be careful,
        the size of this dataset can be very very large


The 'draw' command
------------------
* Command used to draw samples
    --output ofile
        Output file to save data drawn


The 'attribute' command
-----------------------
This command enables an attribution to be made by calculating the
probabilities, return times and intensities in the factual and counter-factual
world. Changes in intensity and probability are also calculated. All sub
command here use the following '--config' parameters:

--output ofile
    Output file to save the result of the attribution
--config param0=value0,param1=value1,...
    mode: 'sample' (store samples used) or 'quantile' (store only the
    median and the confidence interval)
    side: Tails used: 'right' or 'left'
    ci: Level of the confidence interval (default is 0.05 for 95%)
    time: Year of the event (for 'attribute event')

* attribute event
    Perform the attribution of the event with intensity IF 'ifile' occuring at
    year 'time'.
    --input name,ifile
        Observations of the event, in the form 'name,file'.
* attribute freturnt
    Performs the attribution by setting the value of the return time in the
    factual world. Several values can be given.
    --input float,float,float,...
        Factual return time of the event.
* attribute creturnt
    Performs the attribution by setting the value of the return time in the
    counter-factual world. Several values can be given.
    --input float,float,float,...
        Counter-factual return time of the event.
* attribute fintensity
    Performs the attribution by setting the value of the intensity in the
    factual world.
    --input float or ifile
        Factual intensity of the event. Can be a float (for all spatial
        dimensions) or a map.


The 'misc' command
------------------
* misc wpe
    wpe: Worst Possible Event. Find the smallest extreme event occuring at least
    one time between 'year0' and 'year1' with probabilities 'prob'.
    --input float,float,float...
        A list of probabilities separated by ':', or 'IPCC', which means prob =
        [0.01,0.1,0.33,0.5,0.66,0.9,0.99]
    --output ofile
        Output file to save the event.
    --n-samples int
        Numbers of samples drawn
    --config param0=value0,param1=value1,...
        period: year0/year1
        mode: 'sample' (store samples used) or 'quantile' (store only the
        median and the confidence interval)
        side: Tails used: 'right' or 'left'
        ci: Level of the confidence interval (default is 0.05 for 95%)


License {}
{}
{}


Sources and author(s)
---------------------
Sources   : {}
Author(s) : {}
""".format( version , "=" * (12+len(version)) ,
            long_description,
            license , "-" * ( 8 + len(license) ) , license_txt ,
            src_url , authors_doc )


