
// Copyright(c) 2024 / 2025 Yoann Robin
// 
// This file is part of ANKIALE.
// 
// ANKIALE is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// ANKIALE is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with ANKIALE.  If not, see <https://www.gnu.org/licenses/>.

functions
{
	real norm_lpdf( vector Y , vector mu , vector sigma )
	{
    	real ll ;
		ll = - sum(log(sigma^2)) / 2 - sum( (Y - mu)^2 ./ sigma^2 ) / 2 ;
		return ll ;
	}
}

data
{
	int nhpar;
	vector[nhpar] prior_hpar ;
	cov_matrix[nhpar] prior_hcov ;
	matrix[nhpar,nhpar] prior_hstd ;
	int nXY;
	vector[nXY] X;
	vector[nXY] Y;
}

parameters
{
	vector[nhpar] hpar;
}

transformed parameters
{
	vector[nXY] mu  ;
	vector[nXY] sig ;
	mu  = hpar[1] + hpar[2] * X ;
	sig = exp( hpar[3] + hpar[4] * X ) ;
}

model
{
	hpar ~ multi_normal( prior_hpar , prior_hcov );
	Y ~  norm( mu , sig ) ;
}

