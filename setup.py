
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


## Start by import release details
import os
from distutils.core import setup

## Release elements
class Release:##{{{
    def __init__( self , pkg ):
        self._release = {}
        cpath = os.path.dirname(os.path.abspath(__file__)) ## current-path
        with open( os.path.join( cpath , pkg , "__release.py" ) , "r" ) as f:
            lines = f.readlines()
        exec( "".join(lines) , {} , self._release )
    
    @property
    def name(self):
        return self._release['name']
    
    @property
    def version(self):
        return self._release['version']
    
    @property
    def description(self):
        return self._release['description']
    
    @property
    def long_description(self):
        return self._release['long_description']
    
    @property
    def author(self):
        return self._release['author']
    
    @property
    def author_email(self):
        return self._release['author_email']
    
    @property
    def src_url(self):
        return self._release['src_url']
    
    @property
    def license(self):
        return self._release['license']
##}}}

release = Release("ANKIALE")

## Required elements
package_dir = { "ANKIALE" : "ANKIALE" }
requires    = [
               "toml",
               "numpy",
               "scipy",
               "xarray",
               "dask",
               "netCDF4",
               "cftime",
               "matplotlib",
               "SDFC (>=0.9.0)",
               "statsmodels (>= 0.14)",
               "xesmf",
               "zxarray (>=0.12.0)",
               "cmdstanpy",
              ]
scripts     = ["scripts/ank"]
keywords    = []
platforms   = ["linux","macosx"]
packages    = [
    "ANKIALE",
    "ANKIALE.cmd",
    "ANKIALE.plot",
    "ANKIALE.stats",
    "ANKIALE.stats.models",
    ]

classifiers      = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Natural Language :: English",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
    "Intended Audience :: Science/Research",
    ]


## Now the setup

setup(  name             = release.name,
        version          = release.version,
        description      = release.description,
        long_description = release.long_description,
        author           = release.author,
        author_email     = release.author_email,
        url              = release.src_url,
        packages         = packages,
        package_dir      = package_dir,
        requires         = requires,
        scripts          = scripts,
        license          = release.license,
        keywords         = keywords,
        platforms        = platforms,
        classifiers      = classifiers,
        include_package_data = True
    )


