
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

version_major = 1
version_minor = 0
version_patch = 4
version_extra = ""
version      = f"{version_major}.{version_minor}.{version_patch}{version_extra}"

name         = "ANKIALE"
description  = "ANalysis of Klimate with bayesian Inference: AppLication to extreme Events"
author       = "Yoann Robin"
author_email = "yoann.robin.k@gmail.com"
license      = "GNU-GPL3"
src_url      = "https://github.com/yrobink/ANKIALE"

long_description = """\
ANKIALE -- ANalysis of Klimate with bayesian Inference: AppLication to extreme
Events -- is a program for the inference, analysis and attribution of extreme
events. The methodology is as follows:

- Inference of the law (which can be specified) for several climate models,
- Estimation of a multi-model synthesis,
- The synthesis is considered as a prior of reality, and the observations are
  used to derive the posterior using Bayesian methods.

These different steps can be called up from the command line. Attributions and
analyses of the inferred laws can then be performed.
"""

authors_doc = author
license_txt = """\
Copyright(c) 2023 / 2025 Yoann Robin

This file is part of ANKIALE.

ANKIALE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ANKIALE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ANKIALE.  If not, see <https://www.gnu.org/licenses/>.
"""

