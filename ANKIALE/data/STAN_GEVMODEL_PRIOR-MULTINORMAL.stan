
functions
{
	real gev_lpdf( vector Y , vector mu , vector sigma , real xi )
	{
		real ll ;
		real ixi ;
    	int  nXY;
    	nXY = num_elements(Y);
		vector[nXY] tmp ;
		tmp = log( 1 + xi * (Y - mu) ./ sigma ) ;
		ixi = 1 / xi ;
		ll  = - sum(log(sigma)) - (1 + ixi) * sum(tmp) - sum( exp( ( -ixi ) * tmp ) ) ;
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
	real xi ;
	mu  = hpar[1] + hpar[2] * X ;
	sig = exp( hpar[3] + hpar[4] * X ) ;
	xi  = hpar[5] ;
}

model
{
	hpar ~ multi_normal( prior_hpar , prior_hcov );
	Y ~  gev( mu , sig , xi ) ;
}

