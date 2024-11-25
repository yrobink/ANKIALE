
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

BSAC ({})
{}

{}


Common arguments
----------------
--help
    Ask to see the documentation
--log [loglevel,logfile]
    Set the log level, default is 'warning'. If '--log' is passed without
    arguments, 'debug' is used. The default output is the console, and the
    second argument is a file to redirect the logs.
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


The 'show' command
------------------
* show XN
    Plot a figure of the response of the natural forcings used for covariates.
    --output ofile
        A file to save the figure. If not given the function
        matplotlib.pyplot.show() is called.
* show X (dont forgive the --load-clim parameter)
    Plot the covariates fitted.
    --input ifile0, ifile1,...
        Data used to fit the climatology. Optional, added to the plot only
        if given
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
* show Y
    Not implemented


The 'fit' command
-----------------
* fit X
    Fit the covariable(s) (dont forgive the --save-clim parameter)
    --n-samples int
        Numbers of resamples used for the bootstrap
    --input ifile0, ifile1,...
        Data used to fit the climatology, in the form 'name,file'
    --common-period per
        Name of the common period between scenarios (typically the historical)
    --different-periods per0,per1,...
        Names of the differents periods (typically the scenarios)
    --config param0=value0,param1=value1,...
        GAM_dof: Number of degree of freedom of the GAM model, default = 7
        GAM_degree: Degree of the splines in the GAM model, default = 3
        XN_version: set the version of natural forcings, must be CMIP5 or CMIP6
* fit Y
    Fit the variable with the covariable fitted from --load-clim (again, dont
    forgive the --save-clim parameter)
    --n-samples int
        Numbers of resamples used for the bootstrap
    --input ifile
        Data used to fit the climatology.
    --config param0=value0,param1=value1,...
        name: Name of the variable
        spatial: Names of the spatial coordinates (if exists), separated by a ':'
        nslaw: Identifier of the non-stationary distribution


The 'draw' command
------------------
* Y
    --output ofile
        Output file to save data drawn


The 'synthesize' command
------------------------
* synthesize
    --input clim0.nc, clim1.nc,...
        Climatology fitted to synthesize
    --common-period per
        Name of the common period between scenarios (typically the historical)
    --different-periods per0,per1,...
        Names of the differents periods (typically the scenarios)
    --config param0=value0,param1=value1,...
        GAM_dof: Number of degree of freedom of the GAM model, default = 7
        GAM_degree: Degree of the splines in the GAM model, default = 3
        grid: a file describing the grid of the synthesize (clim will be remap
              on this grid)
        grid_name: name of the variable for the grid
        nslaw: Identifier of the non-stationary distribution
        spatial: Names of the spatial coordinates (if exists), separated by a ':'
        names: Names of the covariate, followed by the variable, with ':' as separator


The 'constrain' command
-----------------------
* constrain X
    --input ifile0, ifile1,...
        Observations used for the constraint, in the form 'name,file' The file
        can be a single time series, applied to all grid point, or a spatial
        observations for differentes covariates.
* constrain Y
    --input ifile
        Observations used for the constraint, in the form 'name,file'.
    --n-samples int
        Numbers of samples drawn to find the posterior
    --config param0=value0,param1=value1,...
        n-mcmc-min: Minimal number of MCMC drawn.
        n-mcmc-max: Maximal number of MCMC drawn.


The 'attribute' command
-----------------------
* attribute event
    Perform the attribution of the event with intensity 'ifile' occuring at year
    'time'.
    --input ifile
        Observations of the event, in the form 'name,file'.
    --output ofile
        Output file to save the result of the attribution
    --n-samples int
        Numbers of samples drawn
    --config param0=value0,param1=value1,...
        time: Year of the event
* attribute freturnt
    Perform the attribution of the event with factual return time '--input'
    occuring at year 'time'.
    --input float
        Factual return time of the event.
    --output ofile
        Output file to save the result of the attribution
    --n-samples int
        Numbers of samples drawn
    --config param0=value0,param1=value1,...
        time: Year of the event
* attribute creturnt
    Perform the attribution of the event with counter factual return time
    '--input' occuring at year 'time'.
    --input float
        Counter-factual return time of the event.
    --output ofile
        Output file to save the result of the attribution
    --n-samples int
        Numbers of samples drawn
    --config param0=value0,param1=value1,...
        time: Year of the event


The 'misc' command
------------------
* misc bpc
    bpc: Best Possible Case. Find the smallest extreme event occuring at least
    one time between 'year0' and 'year1' with probabilities 'prob'.
    --output ofile
        Output file to save the event.
    --n-samples int
        Numbers of samples drawn
    --config param0=value0,param1=value1,...
        period: year0/year1
        prob: a list of probabilities separated by ':', or 'IPCC', which means
              prob = [0.01,0.1,0.33,0.5,0.66,0.9,0.99]


Examples
--------
TODO


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


