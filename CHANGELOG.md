# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [2.4.0] — 2026-08-29

### 🏗️ Infrastructure Overhaul (Phase 0 Complete)

#### Added
- Unified CLI entry point at `src/truss_analysis/main.py`
- Comprehensive test suite (39 tests, 90.5% coverage)
- GitHub Actions CI/CD pipeline (pytest, ruff, mypy)
- Backward compatibility for legacy JSON schemas with space-padded keys
- Pre-commit hooks for code quality enforcement
- `PROJECT_DOCUMENTATION/` folder structure (git-ignored)

#### Changed
- **BREAKING:** Removed root-level `main.py` shim
  - Migration: use `python -m truss_analysis.main` or `truss-analysis` CLI
- Migrated to `src/` layout for proper package structure
- Pinned all dependencies in `pyproject.toml`

#### Fixed
- JSON schema tolerance for legacy files (e.g., `example1.json`)
- Coverage tracking for CLI execution paths
- Type hints and mypy compliance across all modules

---

## [2.3.0] — 2026-08-24

### 📚 Professional Documentation Overhaul
- Complete README rewrite with badges, examples, and full API reference
- CONTRIBUTING.md with detailed guidelines for contributors
- CHANGELOG.md with complete version history from v1.0.0
- Dynamic versioning with `setuptools_scm`
- Removed unused `src/utils/` directory
- Coverage threshold adjusted to 85%

---

## [2.2.1] — 2026-08-24
- Automated release of version 2.2.1

## [2.2.0] — 2026-08-24
- fix: mypy-clean root shim with hasattr guard for TextIO.reconfigure

## [2.1.9] — 2026-08-24
- fix: mypy-clean root shim with hasattr guard for TextIO.reconfigure

## [2.1.8] — 2026-08-24
- chore: add dynamic versioning with setuptools_scm

## [2.1.7] — 2026-08-24
- chore: add dynamic versioning with setuptools_scm

## [2.1.6] — 2026-08-24
- chore: remove unused legacy files (i18n, logger, plotter, report)

## [2.1.5] — 2026-08-24
- fix: correct equilibrium key names in test

## [2.1.4] — 2026-08-24
- chore: remove temporary fix script and format code

---

## [1.x.x] — Initial Development
- Initial implementation of 2D truss analysis solver
- Basic FEM solver with thermal loading support
- Equilibrium checking and buckling analysis
- CLI interface and JSON input/output

---

*Note: Versions prior to 2.4.0 represent the pre-research development phase.*
*Version 2.4.0 marks the beginning of the research project phases.*
