
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
	vector[nhpar] npar;
}

transformed parameters
{
	vector[nXY] mu  ;
	vector[nXY] sig ;
	vector[nhpar] hpar = prior_hstd * npar + prior_hpar ;
	mu  = hpar[1] + hpar[2] * X ;
	sig = exp( hpar[3] + hpar[4] * X ) ;
}

model
{
	hpar ~ normal( 0 , 1 );
	Y ~  norm( mu , sig ) ;
}

