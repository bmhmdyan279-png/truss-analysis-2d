# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.4.0] - 2026-08-29

### 🏗️ Infrastructure Overhaul (Phase 0)

#### Added
- Unified CLI entry point at `src/truss_analysis/main.py`
- Comprehensive test suite (39 tests, 90.5% coverage)
- GitHub Actions CI/CD pipeline (pytest, ruff, mypy)
- Backward compatibility for legacy JSON schemas with space-padded keys
- Pre-commit hooks for code quality enforcement

#### Changed
- **BREAKING:** Removed root-level `main.py` shim (use `python -m truss_analysis.main` instead)
- Migrated to `src/` layout for proper package structure
- Pinned all dependencies in `pyproject.toml`

#### Fixed
- JSON schema tolerance for legacy files (e.g., `example1.json`)
- Coverage tracking for CLI execution paths
- Type hints and mypy compliance across all modules

#### Technical Details
- Decision: Eliminated dual entry points to prevent import confusion
- Decision: Added `_get()` helper for space-tolerant key access
- All 39 tests passing with 90.5% code coverage

---

## [1.x.x] - Previous releases
(Previous history...)
