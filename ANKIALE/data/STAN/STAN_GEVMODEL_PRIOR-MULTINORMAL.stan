
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
	real gev_lpdf( vector Y , vector loc , vector scale , vector shape )
	{
		real ll ;
    	int  nXY;
    	nXY = num_elements(Y);
		vector[nXY] tmp ;
		real shp ;
		real ishp ;
		shp  = shape[1] ;
		ishp = 1 / shp ;
		tmp = log( 1 + shp * (Y - loc) ./ scale ) ;
		ll  = - sum(log(scale)) - (1 + ishp) * sum(tmp) - sum( exp( ( -ishp ) * tmp ) ) ;
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
	vector[nXY] loc  ;
	vector[nXY] scale ;
	vector[nXY] shape ;
	loc  = hpar[1] + hpar[2] * X ;
	scale = exp( hpar[3] + hpar[4] * X ) ;
	shape  = hpar[5] + 0 * X ;
}

model
{
	hpar ~ multi_normal( prior_hpar , prior_hcov );
	Y ~  gev( loc , scale , shape ) ;
}

