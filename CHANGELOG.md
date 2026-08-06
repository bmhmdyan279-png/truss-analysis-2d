# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2.0.8] - 2026-08-07

### 📚 Documentation
- Complete sync of README with Critic-Proof Architecture
- Added "Thermodynamic Consistency" section explaining generalized Clapeyron's theorem
- Documented API change: `assemble_global_matrices` returns 4 values
- Documented new `prestress_work` return value from `calculate_element_forces`
- Added comprehensive changelog entries for v2.0.5, v2.0.6, v2.0.7

## [2.0.7] - 2026-08-07

### 🔧 Fixed
- **test_golden_simple_truss**: Changed node 3 from roller to pin support.
  A vertical member (1-3) provides zero X-stiffness, causing a kinematic
  mechanism if the support is a roller.
- **test_golden_thermal_loading**: Constrained node 2 in Y direction.
  A single horizontal member allows infinite rotation (rigid body mode)
  if Y is unconstrained.
- All 29 tests now pass with mathematically stable stiffness matrices.

## [2.0.6] - 2026-08-07

### 🔧 Fixed
- **test_dof_mapping**: Updated to unpack 4 values from `assemble_global_matrices`
  (K, F_ext, F_mechanical, fixed_dofs) to match new API introduced in v2.0.5.

## [2.0.5] - 2026-08-07

### 🚀 Major - Thermodynamic Consistency
- **Fixed critical physics bug**: Resolved thermodynamic inconsistency in energy
  validation when thermal loads are present.
- **Split force vectors**: `assemble_global_matrices` now returns two separate
  force vectors:
  - `F_ext`: Total forces (mechanical + thermal) for solving K·U = F
  - `F_mechanical`: External mechanical loads only for energy validation
- **Generalized Clapeyron's theorem**: Updated `check_energy` to use the
  correct formula: `W_mech = U_strain + W_prestress`
- **Added `prestress_work`**: `calculate_element_forces` now computes and
  returns the prestress work: `W_prestress = Σ(k·ΔL_thermal·ΔL_mech)`
- **New tests**:
  - `test_golden_thermal_loading`: Verifies zero strain energy under free
    thermal expansion
  - `test_golden_thermal_constrained`: Verifies correct energy storage in a
    fully constrained rod
  - `test_check_energy_pass_with_thermal`: Unit test for the generalized
    energy balance formula
- **API change**: `assemble_global_matrices` signature changed from
  `(nodes, elements) -> (K, F_ext, fixed_dofs)` to
  `(nodes, elements) -> (K, F_ext, F_mechanical, fixed_dofs)`
- **API change**: `calculate_element_forces` signature changed from
  `-> (results, strain_energy)` to
  `-> (results, strain_energy, prestress_work)`
- **API change**: `check_energy` signature changed from
  `(U, F, strain_energy, tol)` to
  `(U, F_mechanical, strain_energy, prestress_work, tol)`

## [2.0.4] - 2026-08-07

### 📚 Documentation
- Synced README with Critic-Proof Architecture
- Updated badges to Python ≥3.9 and v2.0.3
- Fixed JSON `loads` schema example to match flat list structure
- Updated Energy Validation exception name to `EnergyValidationError`
- Added `exceptions.py` to architecture diagram

## [2.0.3] - 2026-08-07

### 🚀 Critic-Proof Architecture
- **Unified API**: Aligned all modules and tests to use `list[Node]` and
  `list[Element]` consistently
- **Centralized exceptions hierarchy**: Moved all custom exceptions to
  `exceptions.py` with `TrussError` as base class
- **Strict JSON schema validation**: Added validation for `loads` field in
  `fileio.py` to prevent silent failures on malformed input
- **Fixed Imperial `delta_T` conversion**: Applied correct factor of `5/9`
  (Δ°C = Δ°F × 5/9), previously was incorrectly set to `1.0`
- **Removed dead dependencies**: Removed unused `scipy` from `pyproject.toml`
  and dead `use_sparse` parameter from `solver.py`
- **Code quality**: Bumped to Python `>=3.9`, added `ruff` to CI pipeline,
  renamed ambiguous variable `I` to `I_sec` to pass E741
- **Removed logger side-effect**: `logger.py` no longer creates files in CWD
  at import time

## [2.0.1] - 2026-08-06

### ✨ Added
- **Pure DTO Architecture**: Complete separation of data (`model.py`),
  assembly (`assembly.py`), and solving (`solver.py`)
- **Fail-Fast Rank Check**: Explicit detection of singular stiffness matrices
  via `SingularMatrixError`
- **Roller Support**: Independent boundary conditions in X and Y directions
- **Centralized Unit System**: `units.py` as single source of truth for
  SI ↔ Imperial conversions
- **Golden Tests**: Integration tests with analytical accuracy of 0.01%
- **DOF Mapping Tests**: Verification of correct degree-of-freedom numbering

### 🐛 Fixed
- Enforce fail-fast on zero-length elements in `assembly.py`
  (raises `AssemblyError` instead of silent failure)
- Fixed `ImportError` in `__init__.py`: `solve_truss` alias properly exported
- Aligned `main.py` with `example1.json` structure (both now use lists)
