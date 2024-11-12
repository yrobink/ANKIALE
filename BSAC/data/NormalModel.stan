
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

