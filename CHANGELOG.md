# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [2.4.0] — 2026-08-28

### 🏗️ Infrastructure Overhaul (Phase 0 Complete)

#### Added
- Unified CLI entry point at `src/truss_analysis/main.py`
- Comprehensive test suite (39 tests, 90.5% coverage)
- GitHub Actions CI/CD pipeline (pytest, ruff, mypy)
- Backward compatibility for legacy JSON schemas with space-padded keys
- Pre-commit hooks for code quality enforcement
- `PROJECT_DOCUMENTATION/` folder structure (git-ignored)
- Visual output example (plot image) in README

#### Changed
- **BREAKING:** Removed root-level `main.py` shim
  - Migration: use `python -m truss_analysis.main` or `truss-analysis` CLI
- Migrated to `src/` layout for proper package structure
- Pinned all dependencies in `pyproject.toml`
- Updated README with output/citation sections and fixed Python version

#### Fixed
- JSON schema tolerance for legacy files (e.g., `example1.json`)
- Coverage tracking for CLI execution paths
- Type hints and mypy compliance across all modules
- Test failures from invalid example files (skip invalid ones)
- Removed accidental `sample_output.json` from examples

---

## [2.3.0] — 2026-08-24

### 📚 Professional Documentation Overhaul

#### Added
- Complete README rewrite with badges, examples, and full API reference
- CONTRIBUTING.md with detailed guidelines for contributors
- CHANGELOG.md with complete version history from v1.0.0
- Dynamic versioning with `setuptools_scm`

#### Changed
- Coverage threshold adjusted to 85%
- Removed unused `src/utils/` directory

#### Fixed
- Version/coverage conflicts
- Cleanup of accidental files

---

## [2.2.1] — 2026-08-24

#### Fixed
- mypy-clean root shim with `hasattr` guard for `TextIO.reconfigure`

---

## [2.2.0] — 2026-08-24

#### Changed
- Dynamic versioning with `setuptools_scm` (applied properly)

---

## [2.1.9] — 2026-08-24

#### Fixed
- mypy-clean root shim with `hasattr` guard (re-applied)

---

## [2.1.8] — 2026-08-24

#### Added
- Dynamic versioning with `setuptools_scm` (initial setup)

---

## [2.1.7] — 2026-08-24

#### Removed
- Unused legacy files: `i18n`, `logger`, `plotter`, `report`

---

## [2.1.6] — 2026-08-24

#### Fixed
- Correct equilibrium key names in tests

---

## [2.1.5] — 2026-08-24

#### Changed
- Removed temporary fix script and formatted code

---

## [2.1.4] — 2026-08-24

#### Fixed
- CI: remove unused variable and add PyPI trusted publishing permissions
- CI: remove `scipy` dependency and ensure ruff format compliance

---

## [2.1.0] — 2026-08-24

#### Added
- Phase 1 scientific correctness: reactions, equilibrium check, analytical tests
- Phase 0 crash fixes

#### Fixed
- Move UTF-8 wrapper inside `__main__` guard to prevent `sys.stderr` loss
- Duplicate TOML keys in `pyproject.toml`

#### Removed
- Temporary debug scripts and `.venv` from tracking

---

## [2.0.9] — 2026-08-06

#### Fixed
- Conditional import `tomllib` for Python < 3.11 compatibility

---

## [2.0.8] — 2026-08-06

#### Documentation
- Finalize v2.0.8 Production-Ready documentation
- Synchronized README with Critic-Proof Architecture
- Added Thermodynamic Consistency section (generalized Clapeyron's theorem)
- Documented API changes: `assembly` returns 4 values, `postprocess` returns 3
- Documented `prestress_work` calculation for thermal loads
- Updated badges to v2.0.8

---

## [2.0.7] — 2026-08-06

#### Fixed
- Resolve kinematic mechanisms in golden tests
- `test_golden_simple_truss`: Node 3 must be a pin support (`support_dx=True`)
- `test_golden_thermal_loading`: Node 2 must be constrained in Y (`support_dy=True`)

---

## [2.0.6] — 2026-08-06

#### Fixed
- Correct API unpack and physics in test suite
- `test_dof_mapping`: unpack 4 values from `assembly`
- `test_golden_simple_truss`: node 3 as roller (`support_dx=False`)
- `test_golden_thermal_loading`: node 2 free for zero strain energy
- Add `test_golden_thermal_constrained` for locked rod

---

## [2.0.5] — 2026-08-06

#### Fixed
- Resolve thermodynamic inconsistency in energy validation
- Split force vectors: `F_ext` (total) vs `F_mechanical` (external only)
- Updated `check_energy` with generalized Clapeyron's theorem:
  `W_mech = U_strain + W_prestress`
- Added `prestress_work` calculation in `postprocess.py`
- Fixed API: `assembly` now returns 4 values (`K`, `F_ext`, `F_mech`, `fixed_dofs`)
- Added `test_golden_thermal_loading` for pure thermal case

---

## [2.0.4] — 2026-08-06

#### Documentation
- Align README with v2.0.3 Critic-Proof Architecture
- Update badges to Python >=3.9 and v2.0.4
- Remove unused `scipy` dependency from docs
- Fix JSON 'loads' schema example to match flat list structure
- Update Energy Validation exception to `EnergyValidationError`
- Add `exceptions.py` to architecture diagram

---

## [2.0.3] — 2026-08-06

#### Added
- Final Polish: API Alignment, Physics & Formatting
- Strict schema validation for JSON input files (loads)
- Imperial unit conversion for temperature differences (5/9)

#### Fixed
- Empty path crash in file writer (`os.makedirs`)
- Ambiguous variable name `'I'` → `I_sec` (E741)
- Aligned `test_assembly` and `test_postprocess` APIs with list-based core
- Singular matrix bug in `test_golden_simple_truss`
- Scale factor limit bug (clamp to 1000.0) in `test_postprocess`
- Double-counting of thermal forces in postprocessing
- Removed `logger.py` side-effect creating files in CWD at import time

---

## [2.0.2] — 2026-08-06

#### Fixed
- Roller support, golden tests, and CI fully green

---

## [2.0.1] — 2026-08-06

#### Added
- Pure DTOs, fail-fast validation, centralized units
- Roller support, golden tests, comprehensive README
- Scipy dependency (later removed)

---

## [2.0.0] — 2026-08-06

#### Added
- **BREAKING:** Critic-Proof Architecture (Phase 0)
- Complete redesign with DTOs and fail-fast validation

---

## [1.7.0] — 2026-08-05

#### Fixed
- Resolve `B007` with ruff unsafe fixes
- Finalize exception chaining, formatting, and cleanup
- One-liner logic for thermal energy balance (zero-energy guard)
- Correct energy balance error metric (avoid division by near-zero)
- Cross-platform font rendering (Linux, macOS, Windows)

#### Changed
- Single source version, remove side-effects, sync constants

#### Documentation
- Update README for production readiness (py3.9, real coverage, fail-fast)
- Comprehensive CHANGELOG and version bump

---

## [1.6.2] — 2026-07-31

#### Fixed
- Restore missing `[project]` and `[build-system]` sections in `pyproject.toml`
- Correct ruff `target-version` to `py311`

---

## [1.6.1] — 2026-07-31

#### Added
- Phase 4: import-linter, CI matrix, AI guardrails, release workflow

---

## [1.6.0] — 2026-07-31

#### Added
- Phase 3 Masterplan v6.0: safe Persian font, data contract, Jinja2 RTL template
- Performance guard, version sync, and right-sized coverage guard (70% core)
- Error codes documentation and UTF-8 safety

#### Changed
- Enforce LF line endings globally via `.gitattributes`
- Remove temporary fix scripts and repo artifacts
- Cleanup cache, update version sync test, init i18n phase 1
- Deprecate root `main.py`

#### Fixed
- Correctly use `error_code` in exception logging and remove duplicate lines

---

## [1.5.0] — 2026-07-06

#### Fixed
- Unskip 9 tests by updating to new `input_data` API
- Restore `model.py` from v1.4.0 and fix sparse matrix handling
- Convert all absolute imports to relative for CLI

#### Changed
- Update to v1.5.0 – all 65 tests passing
- Cleanup temporary fix scripts and update pre-commit config

---

## [1.4.0] — 2026-07-06

#### Added
- `src` layout for proper package structure
- CLI entry point and root shim for backward compatibility
- Professional Polish: code quality refactor

#### Changed
- Move tests and source into organized structure

#### Fixed
- Convert numpy bool to Python bool in buckling warning
- Add missing `TOLERANCES` keys and fix test assertions

---

## [Initial Development] — v1.0.0 to v1.3.1 (2026-06-16 – 2026-07-05)

### Added
- Initial 2D truss analysis solver (FEM)
- Basic thermal loading support
- Equilibrium checking and buckling analysis
- CLI interface and JSON input/output
- Custom exceptions and robust file I/O
- Magic numbers extracted to `constants.py`
- Logging automation and linting fixes

### Changed
- Applied ruff formatting and auto-fixed linting issues
- Refactored code quality baseline

### Removed
- Temporary debug scripts
- Unused test files (reorganized)

---

*Note: Versions prior to 1.4.0 represent the early development phase.*
*Starting from 1.4.0, the project follows structured releases with full test coverage and CI/CD.*
