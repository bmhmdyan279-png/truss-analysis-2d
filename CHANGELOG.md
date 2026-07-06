# Changelog

تمام تغییرات مهم این پروژه در این فایل مستند می‌شوند.

## [1.5.0] - 2026-07-06

### Added
- All 65 tests now passing (0 skipped)

### Fixed
- Unskipped 9 tests by updating to new input_data API
- Fixed test_single_element.py: removed duplicate @pytest.mark.skip decorators
- Fixed test_solver.py: rewrote input_data from dict to list format
- Fixed calculate_total_energy() calls with F_global parameter

### Changed
- Improved test coverage from 56/65 to 65/65 (100% pass rate)

## [1.4.1] - 2026-07-06

### Fixed
- **Critical**: 11 skipped tests now passing
  - `test_edge_cases.py`: Rewritten to use list-based input_data API
  - `test_single_element.py`: Migrated from non-existent add_node/add_element API
  - `test_solver.py`: Fixed dict→list structure mismatch in nodes/elements

### Added
- `tests/test_api_contract.py`: Regression tests for API structure validation
- Validation in `TrussModel._create_nodes()` to reject dict input with clear error

### Documentation
- Added API migration guide in README.md
- Documented input_data schema in docs/schema.md

## [1.4.0] - 2026-07-06

### Major Changes (Professional Polish)
- **Standard `src/` structure**: Moved all modules to `src/truss_analysis/` for professional packaging
- **CLI Entry Point**: Added `truss-analyze` command via `console_scripts`
- **Removed extra files**: Deleted old `setup.py` and root `__init__.py`
- **Unified imports**: Used relative imports in package core and absolute imports in tests
- **Backward compatibility shim**: Kept `main.py` at root for easy execution

### Infrastructure Improvements
- Configured `pythonpath` in `pytest` for running tests without installation
- Updated `pyproject.toml` with complete metadata and version 1.4.0
- Fixed apparent contradictions in README (emphasized Python 3.8+)

## [1.3.1] - 2026-07-06

### Fixed
- Convert NumPy bool to Python bool in buckling warning to ensure consistent type checking
- Fix test assertion compatibility with NumPy boolean values

### Changed
- Improved type consistency in solver results

## [1.1.0] - 2026-06-30

### Major Changes
- **Repository organization**: Moved test files to `tests/` and example files to `examples/` for consistency with documentation
- **Code readability**: Rewrote `solve_displacements` function in `solver.py` by extracting helper functions (`_solve_elimination`, `_solve_penalty`)
- **Increased stability**: Removed automatic conversion of sparse matrices to dense on error (errors are now reported transparently)

### Minor Improvements
- Removed extra files (`model.py.bak`)
- Fixed installation instructions in README
- Added `pytest.ini` file for automatic test path configuration

### Infrastructure
- Set up GitHub Actions for automatic test execution on Python 3.8 to 3.11

## [1.0.0] - 2026-01-01

### Added
- Complete 2D truss analysis
- Thermal effects and fabrication errors
- Euler buckling analysis
- Support for multiple unit systems
- Graphical outputs and CSV/JSON/Markdown reports
- 65 comprehensive tests with 100% coverage
