
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


The 'show' command
------------------
* show EBM
    Plot a figure of the response of the EBM used for covariates.
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
* show Y
    Not implemented


The 'fit' command
-----------------
* fit X
    Fit the covariable(s) (dont forgive the --save-clim parameter)
    --n-samples int
        Numbers of resamples used for the bootstrap
    --input ifile0, ifile1,...
        Data used to fit the climatology.
    --common-period per
        Name of the common period between scenarios (typically the historical)
    --different-periods per0,per1,...
        Names of the differents periods (typically the scenarios)
    --config param0=value0,param1=value1,...
        GAM_dof: Number of degree of freedom of the GAM model, default = 7
        GAM_degree: Degree of the splines in the GAM model, default = 3
* fit Y
    Not implemented


The 'draw' command
------------------
Not implemented


The 'synthesize' command
------------------------
Not implemented


The 'constrain' command
-----------------------
Not implemented


The 'attribution' command
-------------------------
Not implemented



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


