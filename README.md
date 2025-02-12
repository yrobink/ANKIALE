# BSAC
Bayesian Statistical Analysis of the Climatology

## Description

BSAC -- Bayesian Statistical Analysis of the Climatology -- is a program for
the inference, analysis and attribution of extreme events. The methodology is
as follows:

- Inference of the law (which can be specified) for several climate models,
- Estimation of a multi-model synthesis,
- The synthesis is considered as a prior of reality, and the observations are
  used to derive the posterior using Bayesian methods.

These different steps can be called up from the command line. Attributions and
analyses of the inferred laws can then be performed.


## Installation

Don't forget to install the [SDFC](https://github.com/yrobink/SDFC-python) and
[zxarray](https://github.com/yrobink/zxarray) packages either from GitHub or
with pip:

~~~bash
pip3 install zxarray SDFC
~~~

BSAC can then be installed with pip.


## Where to start ?

BSAC includes several examples described in the documentation. You can build,
for example, the analysis of the Ile de France using the GEV law with the
following command:

~~~bash
bsac example GEV_IDF --output <directory>
~~~

The documentation is available with:

~~~bash
bsac --help
~~~


## License

Copyright(c) 2023 / 2024 Yoann Robin

This file is part of BSAC.

BSAC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BSAC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BSAC.  If not, see <https://www.gnu.org/licenses/>.

