
# Changelog

## [Unreleased]

### Added
- The spline basis of the covariate is highly customizable: the dof and the
  degree can be set for each covariate, and each differents periods. These new
  parameters can be passed to ANKIALE with the new option `--covar-config`, which
  takes arguments of the form:
     `--covar-config name_covariate:period:dof`
  If not given the default configuration is `6`. Note that the dof is only
  for the spline basis, without the intercept. So the total numbers of degree
  of freedom is `dof+2` for each different periods.
- Add the method `ANKIALE.Climatology.build_design_basis` method, which return
  the linear part of the covariate, and a dict for the spline part.
- Add a class `ANKIALE.stats.SplineSmoother` to smooth on a spline basis
  controlling the dof and degree.
- New `DevException` class, used only for development.
- Add a `pyproject.toml` file
- For the constraint of the covariate, the KCC and MAR2 methods now works.
  Default behavior is the independent case, but the KCC or MAR2 can be enabled
  by setting `--config method=KCC` or `--config method=MAR2` when the command
  `constraint X` is called.
- Add a new command `sexample` for `short example`. This command has the same
  effect as the `example` command, but the model numbers is limited to 5.

### Changed
- Documentation has been changed accordingly to the new spline basis
  parameters.
- Examples have been changed accordingly to the new spline basis parameters.
- name for variable (and not covariate) is renamed vname

### Removed
- In the `--config` option, `GAM_dof` and `GAM_degree` have been removed.
- In the `ANKIALE.Climatology` class, the method `build_design_XFC` has been
  removed because design matrix is more complex with the new spline basis. See
  the `projection` method, or the new `build_design_basis` method.
- The old options `--use-KCC` and `--use-MAR2` have been removed.

### Fixed
- Remove the use of KCC for GMST example.

## [1.0.3] - 2025-04-15
- Fix: `rowvar=False` to compute covariance matrix
- Fix: bug when data are masked before the fit command
- Fix: bug in wpe

## [1.0.0] - 2025-03-17
- New: First release!

