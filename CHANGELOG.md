# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

This release is the outcome of a nine-part external technical audit of the
computational core. Every defect listed below was reproduced against the code
before being fixed; nothing here is a speculative change. Several items
**alter reported numbers** — they are marked ⚠ and collected at the end under
*Behaviour changes*.

### Fixed

- ⚠ **`check_energy` rejected the textbook thermal-stress problem.** The
  generalized Clapeyron identity is `W_mech = U_strain + ½·W_prestress`; setting
  `W_mech = 0` gives `U_strain = −½·W_prestress`, which is generally non-zero.
  The self-equilibrated branch nevertheless demanded `U_strain ≈ 0`, so a fully
  restrained bar heated by `delta_T` raised `EnergyValidationError` out of
  `run()`. The balance is now tested in its unified form with no special case.
- ⚠ **`calculate_buckling` ignored `effective_length_factor`.** It used the
  pin-ended `π²EI/L²` while `docs/theory.md` documented `π²EI/(KL)²` and
  `limitstates` *did* apply `K` — a user supplying `K = 0.7` got `K = 1` from
  `analyze` and a contradictory number from `dcr_field`. All three paths now
  delegate to `sections.euler_buckling_load`.
- ⚠ **A compressed member with no `I_sec` was reported `safe = true`.** Missing
  data was interpreted as safety. It now returns `status = "unknown"`,
  `safe = false`, and raises `BucklingCheckWarning`.
- **`check_equilibrium` scaled the moment residual by the force reference
  twice**, giving a bound with units of force squared — far too tight for a
  large bridge, far too loose for a small one. Force and moment now have
  separate scales; the moment bound is force × bounding-box diagonal.
- **`sensitivity.py` reported total rather than mechanical strain energy.**
  `½·uₑᵀkₑuₑ = ½·k·ΔL_total²` includes imposed thermal/fabrication strain. A
  freely expanding member was scored as storing energy and a fully restrained
  heated member as storing none. Both routes are now computed and retained
  (`strain_energy`, `strain_energy_total`).
- **`_parse_model` dropped `effective_length_factor` and `density` entirely**
  and never read the nested `properties` block. `example1.json` shipped
  `I_sec = 2.63e-08` *and* `properties.I_sec = 8.33e-06` for one member — a
  factor of 317 — and the parser silently picked one. Precedence is now
  explicit (top level wins) and a disagreement raises `InputIgnoredWarning`.
- **Self-weight read `rho` from the raw payload, bypassing `to_si()`**, so any
  Imperial model got self-weight in the wrong units while `Element.density` sat
  at its default. Both now flow through the parsed element.
- **A restraint declared without `is_support` was silently ignored**, producing
  a different structure from the one described. Both inconsistent support
  declarations are now rejected by `validate_inputs`.
- ⚠ **Fire compression capacity used bare Euler.** See *Behaviour changes*.
- **Axial-force sign classification used an absolute `1e-9 N` cut-off**, so a
  1e-6 N force read as `"Tension"` in a kilonewton model and 1e3 N as `"Zero"`
  in a giganewton one. It now compares the strain `N/(EA)` against a relative
  band.

### Added

- **`truss_analysis.numerics`** — one policy object for every relative threshold
  in the library. The rank cutoff was `1e-13` in `solver.py` but `1e-9` in
  `graph_validation.py`, so a model could be reported as a mechanism by the
  validator and as solvable by the solver. Both now read one
  `NumericalTolerances`, in which the two values are deliberately different and
  documented: a tight hard gate for the solver, a loose diagnostic for
  validation. `NumericalStatus` separates `stable` / `ill_conditioned` /
  `singular`, because ill-conditioned is not the same as invalid.
- **Sparse assembly and solve.** `assemble_global_matrices(..., sparse=True)`
  returns CSR, built from rank-one dyads with no Python loop over the 4×4 block
  and summed by `coo_matrix` in C. `solve` accepts dense or sparse and uses
  SuperLU for the latter. A 20 000-node model needed ~13 GB of dense `K` and now
  needs tens of megabytes.
- **Cholesky-first factorisation.** `K_ff` of a stable truss is symmetric
  positive definite, so Cholesky is the correct choice: about half the cost of
  the LU that `np.linalg.solve` performed, and a sharper singularity test that
  fails exactly when positive definiteness — hence stability — is lost.
- **`solve_with_diagnostics`** returning rank, condition number, numerical
  status and the factorisation actually used, so a result can carry its
  numerical evidence rather than only its value.
- **Penalty boundary conditions** (`apply_penalty_bc`, `solve_penalty`,
  `solve_penalty_with_energy`), which the README and `docs/theory.md` had
  advertised but the code did not implement. Accepts an absolute `penalty_value`
  [N/m] or derives a dimensionless multiple of `max|diag K|`, and warns in
  **both** directions when the ratio falls outside `[1e4, 1e12]`.
- **Multi-criteria criticality indices** (`criticality.criteria`): displacement,
  peak member force, the perturbed member's own force, total strain energy and
  support reaction — five indices from the same exact rank-one field, with a
  `governing` label and a conservative `composite`.
- **EN 1993-1-2 §4.2.3.1 buckling reduction** `chi(lambda_bar_theta)` with the
  EN 1993-1-1 curve table and the 0.65 fire imperfection factor
  (`sections.buckling_reduction_factor`, `non_dimensional_slenderness`).
- **Imperial `weight_density`** (lbf/ft³, pcf) alongside `density`
  (slug/ft³), plus `UnitAmbiguityWarning` when a density converts outside the
  plausible range for matter.
- **`InputIgnoredWarning`** for recognised-but-unapplied and unrecognised input
  keys, in both the `options` block and element payloads.
- **`postprocess.imposed_strain_energy`** — the characteristic energy
  `Σ ½·k·ΔL_prestress²` used to place a unit-agnostic round-off floor under the
  energy balance.
- **`docs/theory.md`** expanded from 32 lines into a full reference: model scope
  and what is *not* included, the rank-one identity everything rests on, thermal
  limits, numerical scaling policy, the penalty trade-off, three-level buckling
  separation, the CI definition and its physical interpretation, and a
  verification matrix.

### Changed — performance

- Memoised Table 3.1 lookups (`_column_arrays`, `_interp_scalar`). Results are
  bit-identical; the array path is untouched. **The full test suite went from
  42.5 s to 27.4 s (−36 %)**, since temperature sweeps dominate the criticality
  and limit-state paths.
- Member geometry (lengths, direction cosines, stiffness, DOF maps) is computed
  once by `assembly.member_geometry` and shared by both storage layouts,
  replacing independent recomputation in five modules.
- Dense assembly scatter uses `np.ix_` instead of a nested 16-iteration Python
  loop per member.
- The node → constrained-DOF map is defined once in `model.fixed_dof_indices`
  and consumed by both the assembler and the criticality engine, which
  previously carried private copies.

### Changed — data

- ⚠ The `I_sec` values in all five `examples/*.json` were regenerated from
  `sections.idealised_square_hss(A, b/t = 25)`, the section model the library
  documents. The shipped values were inconsistent by construction: the ratio
  `I_sec / (A²/12)` ranged from `5e-5` to `5e-4` across the examples, and two
  members of identical area in different files disagreed by a factor of ten.
  Since buckling capacity scales with `I`, every buckling figure those examples
  produced was a data artefact. Conflicting nested `properties.I_sec` blocks were
  removed. `tests/test_examples_section_provenance.py` now pins them, following
  the same single-source-of-truth pattern as the Eurocode material fixture.

### Tests

357 → **488 passing**, coverage 94.6 %.

- `test_self_equilibrated_energy.py` — both thermal limits against closed forms,
  the scale-invariance of the balance, and an explicit pin on the known
  limitation of the absolute fallback floor.
- `test_cross_module_consistency.py` — `K = Bᵀ diag(k) B`, `K = Σ kₑ bₑ bₑᵀ`,
  symmetry/PSD with exactly three rigid-body modes, sparse ≡ dense assembly and
  solve, validator ≡ post-processor strain energy under imposed strain.
- `test_property_invariance.py` — linearity and superposition, `1/E` scaling,
  translation/rotation/renumbering/orientation invariance, plus **negative
  controls** so the invariance tests cannot pass vacuously.
- `test_multi_criteria_ci.py` — all five indices against a brute-force
  perturbation that shares no factorisation with the engine, on two topologies.
- `test_penalty_bc_and_options.py` — convergence to elimination, the measured
  error↔conditioning trade-off, both warning directions, and a regression test
  for the sparse+penalty path discarding applied loads.
- `test_units.py` — every conversion factor against its documented value, the
  slug/pcf invariant (`g = 32.174 ft/s²`), and `delta_T` as a difference.

### Behaviour changes ⚠

These alter numbers a user may already have published. Each is a correction
towards the governing standard, and each is either reversible or accompanied by
a warning.

1. **Fire compression capacity is now `chi · N_Rd`, not `min(N_cr, N_Rd)`.**
   The old form took the smaller of two asymptotes and so overestimated capacity
   at intermediate slenderness, understating DCR and overstating `theta_sys`.
   On the validation case (600 °C, `lambda_bar = 0.655`) capacity drops 18 % and
   DCR rises 22 %. Reproduce the old numbers with
   `BucklingModel.EULER_ONLY`; a test asserts `chi` never yields a larger
   capacity, so the reduction cannot be wired backwards silently.
2. **Buckling now applies `K`.** Any model with `effective_length_factor ≠ 1.0`
   that previously got `K = 1` from `analyze` now gets the value it asked for.
3. **A compressed member with no `I_sec` reports `safe = false`.** Code that
   treated `safe = true` as "nothing to do" will now see these members.
4. **Shipped examples' `I_sec` changed**, so their buckling output changes.
5. **Contradictory input now warns or errors** rather than being silently
   dropped: conflicting `I_sec`, unrecognised element or `options` keys,
   inconsistent support declarations, badly scaled penalties.

### Removed

- The unreachable `ZERO_LENGTH` branch behaviour is retained but no longer the
  only guard: assembly raises `AssemblyError` first, so the branch is documented
  as defensive rather than load-bearing.

## [2.5.0] — previous

### Added

- Temperature-dependent material model per EN 1993-1-2 (Tables 3.1, constitutive law)
- Parametric topology generation for Warren, Pratt and Howe families
- Member criticality index with exact rank-1 perturbation engine
- Uncertainty quantification via LHS and Monte Carlo with rank correlation
- Retrofit prioritisation utilities
- Graph and topology validation with rank and condition reporting
- Energy validation via the generalized Clapeyron theorem
- Command-line interface with `analyze`, `validate`, `generate` and `version` subcommands
- Multi-stage Dockerfile and repository documentation in English and Persian

### Changed

- Expanded test suite and type annotations
- Broadened CI matrix

