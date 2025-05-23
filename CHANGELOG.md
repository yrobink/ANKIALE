
# Changelog

## [Unreleased]

### Added
- The spline basis of the covariate is highly customizable: the size, the dof
  and the degree can be set for each covariate, and each differents periods.
  These new parameters can be passed to ANKIALE with the new option
  `--Xconfig`, which takes arguments of the form:
       `--Xconfig name_covariate:period:size_basis:dof:degree`
  If not given the default configuration is `6:6:3`. Note that the dof is only
  for the spline basis, without the intercept. So the total numbers of degree
  of freedom is `dof+2` for each different periods.
- Add the method `ANKIALE.Climatology.build_design_basis method`, which return
  the linear part of the covariate, and a dict for the spline part.
- Add a class `ANKIALE.stats.SplineSmoother` to smooth on a spline basis
  controlling the dof and degree.
- New `DevException` class, used only for development.
- Add a `pyproject.toml` file

### Changed
- Documentation has been changed accordingly to the new spline basis
  parameters.
- Examples have been changed accordingly to the new spline basis parameters.

### Removed
- In the `--config` option, `GAM_dof` and `GAM_degree` have been removed.
- In the `ANKIALE.Climatology` class, the method `build_design_XFC` has been
  removed because design matrix is more complex with the new spline basis. See
  the `projection` method, or the new `build_design_basis` method.

### Fixed
- Remove the use of KCC for GMST example.

## [1.0.3] - 2025-04-15
- Fix: `rowvar=False` to compute covariance matrix
- Fix: bug when data are masked before the fit command
- Fix: bug in wpe

## [1.0.0] - 2025-03-17
- New: First release!

