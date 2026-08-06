# Changelog

## [2.0.3] - 2026-08-07
### Fixed
- Final polish: unified API (lists) across all tests and source code.
- Renamed ambiguous variable 'I' to 'I_sec' to pass Ruff E741.
- Fixed singular matrix bug in `test_golden_simple_truss` (roller without X-support).
- Fixed physical bug: double-counting of thermal forces in postprocessing.
- Fixed strain energy calculation to use mechanical deformation only.
- Unified exception hierarchy (SingularMatrixError, EnergyValidationError).
- Added strict schema validation for JSON input files (loads).
- Fixed Imperial unit conversion for temperature differences (5/9).
- Removed unused scipy dependency and dead use_sparse parameter.
- Added ruff to CI pipeline and bumped Python requirement to >=3.9.
- Removed logger.py side-effect creating files in CWD at import time.

## [2.0.1] - 2026-08-06
### Fixed
- Enforce fail-fast on zero-length elements in assembly.py.
