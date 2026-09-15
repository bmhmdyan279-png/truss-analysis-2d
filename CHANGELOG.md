# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Round-6 external audit. 22 tracked items (7 P0 scientific/safety, 5 P1
physical coverage, 10 P2 performance/architecture) plus the three failures
that were live on `main` at the start of the round. As in rounds 1-5, each
fix carries a pinning test, and the two items where the *ticket itself* was
technically wrong are implemented correctly with the discrepancy recorded
rather than followed.

### Fixed

- **`.gitignore` was committed wrapped in markdown code fences**, so git
  never honoured a single line of it. That is how 40 `__pycache__/*.pyc`
  files, `.coverage` and the generated `_version.py` came to be tracked, and
  why `test_no_bytecode_tracked_in_git` and
  `test_setuptools_scm_fallback_configured` were both failing on `main`.
  Fences removed, the `setuptools_scm` `write_to` target and the
  coverage/type caches added, all 42 offenders dropped from the index.
- **`tests/test_stability_tangent.py` duplicated the finite-difference
  tangent oracle already in `test_stability.py` (A5)** and called
  `member_forces()` with the full displacement vector instead of the free-DOF
  slice, so it failed with a shape mismatch on every run. Deleted; its one
  non-redundant assertion survives as
  `test_geometric_stiffness_is_homogeneous_in_the_base_state`, which pins
  `K_G(alpha N) = alpha K_G(N)` at matrix level through the production demand
  path.
- **`tangent_verification.py` was dead scaffolding that always passed.**
  `verify_tangent_stiffness()` looped over `pass` and returned
  `(True, 0.0, ...)` unconditionally; `TangentVerifier.compute_internal_force()`
  raised `NotImplementedError`; the module was imported by nothing and had
  0 % coverage. A verifier that always passes is worse than no verifier, so
  it was rewritten around the geometrically-exact internal force of the
  pin-jointed assembly, its exact two-dyad tangent (material + geometric, at
  the *deformed* cosines), a central-FD oracle, and
  `verify_linearization_convergence()`. The first draft of the rewrite had a
  real bug the oracle caught immediately: the transverse extractor's second
  node block was not sign-flipped, giving an 8.8e-3 relative error against
  1e-10 once fixed.
- **153 ruff findings on `main`** (83 whitespace, 16 over-length lines, 5
  unused locals, 4 unused imports, naming and docstring rules) reduced to
  zero; `mypy --strict` green on all 48 source files. The matplotlib
  `Normalize` reference now imports from `matplotlib.colors`.
- **`stability.geometric_stiffness` documented its own sign convention
  backwards**: it claimed `K_G` is positive semi-definite on the *compressed*
  members. With tension-positive `N_e`, tension stiffens and compression
  softens. Corrected.
- **`ShallowSystemWarning` was listed under `Raises`** though it is issued,
  not raised; moved to a `Notes` section.
- **`n_modes` was accepted and then ignored.** The dense path always computed
  and returned every positive eigenvalue, so `BucklingResult.modes` held
  `n_free` entries regardless of what was asked for. Now honoured through
  `eigh(subset_by_index=...)` / `eigsh(k=...)`.
- **`BucklingResult.modes` was documented as "the first element equals
  `mode`" while being sorted ascending by `nu`**, i.e. the first element was
  the *least* critical mode. Now sorted descending by `nu` (ascending
  `lambda_cr`) so `modes[0]` is the critical one and the docstring is true.
- **`_rise_span_ratio` flagged every collinear model as shallow.** The
  orientation-robust rewrite initially returned `0.0` for a degenerate
  bounding box, which made every member-level buckling check in `limitstates`
  emit a shallow-system warning for a single column. Such a model now returns
  `inf` (the heuristic is undefined, not "shallow"), and `dcr_field()` passes
  `warn_shallow=False` since the advisory belongs to a deliberate stability
  study, not to every DCR call.
- **`IllConditionedPerturbationWarning` existed since round 5 and was never
  raised**, and its docstring pointed at `truss_analysis.reliability.perturb_multi`
  -- not where that function lives. Wired up (C2) and the docstring corrected.
- **`perturb_multi` returned noise from a merely ill-conditioned core.**
  `_check_lu` only rejects cores singular to working precision; a core at
  `cond ~ 1e13` sailed through and its digits fed straight into retrofit
  ranking. The solve is now *measured* -- exact `cond` of the `r x r` core
  plus the backward error `||core x - f|| / denom` -- and warns above
  `PERTURB_COND_WARN` (1e10) or `PERTURB_RESID_WARN` (1e-8).
- **`apply_decision` hard-coded `b/t = 25` for every member it enlarged**
  (C7), with the same literal duplicated in `sections.idealised_square_hss`
  and `TrussConfig`. For a slender section the error is unsafe: the enlarged
  `I` comes out too large, so the retrofit looks better than it is. The ratio
  is now recovered per member from its own `(A, I_sec)` pair.
- **`docs/api` had no page for `thermal.fire_curve`** -- one of the largest
  modules in the library -- nor for `thermal.protection` or
  `tangent_verification`. Added, both toctrees updated.

### Added

- **C1: a sparse Lanczos eigen-path for large models.** `eigen_solver=
  {"auto","dense","sparse"}` on `linearized_buckling_load_factor`, with CSR
  assembly and one SuperLU factorisation shared between the
  positive-definiteness probe and the eigensolve (handed to ARPACK as `Minv`).
  Two details are easy to get wrong and are documented on the functions:
  the whitened problem needs the largest *algebraic* `nu` (`which="LA"`),
  not the smallest magnitude, since `nu = 1/lambda_cr`; and the definiteness
  probe shifts to `sigma = 0`, not to `-||A||_F` -- the far shift brackets
  the spectrum and looks more rigorous but clusters every transformed
  eigenvalue at `1/||A||_F`, and *measured* it failed to converge in 12811
  iterations at 1121 free DOFs where `sigma = 0` converges in under ten.
  `SPARSE_EIGEN_THRESHOLD` is set from measurement (see *Behaviour changes*),
  and new `solver_path` / `n_free_dof` fields report which path actually ran,
  because `eigen_solver="sparse"` cannot be honoured when ARPACK has no room
  (`k < n`) and that fallback should be visible.
- **C3: a redundancy screen before perturbation.** Removing `k` members from
  an assembly with redundancy `n_s` leaves `n_s - k` (a bar goes away without
  the DOF count changing), so a negative result is a *certain* mechanism.
  `perturb_multi` now says so combinatorially, before any factorisation, in
  engineering terms. New `setup_indeterminacy()` derives it as `m - n_free`,
  and `graph_validation.static_indeterminacy` / `model_indeterminacy` are now
  the single definition of `m + r - 2j`.
- **C16: parallel Monte Carlo.** `ReliabilityEngine(n_jobs=..., parallel_backend=...)`
  plus a per-call `run_convergence(n_jobs=...)` override. Bit-identical to
  the serial run at every worker count and on both backends -- the draws come
  from one up-front stream and margins accumulate strictly in index order,
  with each index taken from its own block rather than a running counter that
  could drift. Threads are the default (the callback is never pickled and the
  work is numpy/scipy, which releases the GIL); evaluation is chunked so
  resident responses are bounded by a chunk rather than `max_n`.
- **B4: `imperfection_sensitivity()`.** Imposes `eps * L_ref * phi` on the
  node coordinates and re-runs the bifurcation analysis, returning the sweep,
  the first-order gradient `d(lambda/lambda_0)/d(eps)` and a sensitivity
  verdict. **Both imperfection signs are probed and the adverse one
  reported**: an eigenvector's sign is arbitrary, and on a shallow toggle the
  two signs are wildly asymmetric -- one deepens the arch and `lambda_cr`
  roughly quadruples with the rise, the other flattens it and the reserve
  collapses. Reporting only the favourable sign would be the most dangerous
  output this function could produce. When the perfect geometry has no
  bifurcation at all (`lambda_cr = inf`, mode identically zero) it refuses to
  invent a direction and asks for an explicit `mode=`.
- **B2: the EN 1991-1-2 Annex A parametric temperature-time curve.**
  `ParametricFire` (callable, so it drops into `steel_temperature(fire_curve=...)`)
  and `parametric_fire_temperature`, derived from the compartment's opening
  factor, fire load density, boundary thermal inertia and Table A.2 limiting
  duration -- with a real peak and a cooling phase, which a nominal
  fire-rating curve cannot express. Implemented against the Access Steel
  worked example SX042a-EN-EU rather than from recollection, because two
  details are easy to get wrong and invisible in the curve's shape: `Gamma`
  is formed from the *opening factor*, `(O/b)^2 / (0.04/1160)^2`, not from
  `q_td`; and `t_max = max(0.2e-3 q_td / O, t_lim)`, so a lightly loaded or
  well ventilated compartment is bounded below by the fire-growth duration
  and becomes fuel-controlled. The example's `Gamma = 5.791`,
  `t_max = 0.355 h`, `t*_max = 2.056 h`, `theta_max = 1052 degC`, cooling
  rate 250 K/h and the line `theta = theta_max - 250 (t* - t*_max)` are all
  reproduced and pinned; the heating branch matches a hand evaluation of
  A.1(1) to 1e-12, and all three A.2 cooling-rate branches have a parameter
  set that lands in them.
- **B1: EN 1993-1-2 clause 4.2.5.2 heating of an insulated member** -- new
  module `truss_analysis.thermal.protection`. The library could heat an
  unprotected member but had no path for a protected one, so any insulated
  design had to be hand-computed outside the library and fed in as a
  prescribed field. Three deliberate choices: the integrator is **explicit
  Euler, not RK4**, because 4.2.5.2 states a *recursion* with a
  non-negativity clip and a 30 s ceiling rather than an ODE, and a test
  measures the order as ~1 so "improving" it to RK4 fails the suite (a
  fire-resistance duration quoted against a different integrator than the
  code's is not a code-compliant duration); `c_a = c_a(theta_a)` is
  re-evaluated every step from the library's own EN single source of truth, so
  `mu` moves with the specific-heat spike near 730 degC; and `k_sh` defaults
  to the conservative 1.0, with `box_protection_shadow_factor()` computing the
  `0.9 (A_p/V)_box / (A_p/V)` reduction and rejecting reversed arguments --
  which would otherwise yield `k_sh > 1`, a *less* conservative temperature
  from a call that looks valid. Ships a ten-product catalogue in the
  project's provenance-fixture style, which states plainly that it is a
  SECONDARY source and not a substitute for certified product data.
- **B7: `LumpedCapacityWarning` when `A_m/V < 50 1/m`.** The
  uniform-cross-section assumption of clause 4.2.2.2 was documented in the
  module docstring and then applied silently to every section factor,
  including stocky ones where it does not hold. The message carries the
  Biot number rather than just the threshold: `lumped_capacity_biot()` derives
  `Bi = h_eff (V/A_m) / lambda_a` from the module's own `h_net` linearised
  about a representative exposure, and `biot_critical_section_factor()`
  inverts it. For the default exposure the classical `Bi < 0.1` criterion is
  reached at `A_m/V ~ 34 1/m`, so the code's 50 fires first -- a test pins
  that ordering, because quoting a Biot number that passed its own stated
  criterion would be self-contradicting.
- **C5: `iman_conover_with_report()` / `RankCorrelationReport`.** The bare
  function claimed the realised Spearman correlation matched the target
  "within sampling noise" and returned only the matrix, leaving the caller to
  re-derive the noise. Rank correlation is not exactly attainable at finite
  `n` (the reordering permutes a fixed value set, so the achievable
  correlations are discrete), which made the claim unfalsifiable. The report
  carries target, achieved (recomputed with `spearmanr`, not self-reported),
  maximum off-diagonal deviation and `n`.
- **C6: `tolerance` on `member_critical_temperature[_detailed]`**, default
  `BISECT_XTOL` (1e-3 degC, now public with the private name kept as an
  alias). Because the returned `theta` is the bracket's *upper* end,
  `DCR(theta) >= 1` holds at any tolerance, so loosening it widens a
  conservative band rather than introducing error.
- **C15: `assembly_mode` and `reduction_order` in `solver_metadata`** and in
  the documented JSON schema. `bc_method` said which strategy was
  *requested*; these say what the assembler and linear algebra then did. This
  earned its place immediately: requesting `use_sparse=true` with
  `bc_method=penalty` silently falls back to dense assembly, which was
  already warned about at run time but is now machine-readable in the report
  that survives.
- **`tangent_verification` as a real module**: `internal_force` (the
  geometrically-exact operator), `exact_tangent_stiffness`,
  `linearized_tangent_stiffness`, `verify_tangent_stiffness` and
  `verify_linearization_convergence`. The last one converts the word
  "linearised" from a docstring sentence into a measured rate: the relative
  Frobenius gap between `K_E + K_G` and the exact tangent closes at fitted
  order **1.000** over a 16x load range on the toggle.
- **`EigenConvergenceError`** in the exception hierarchy: a buckling load
  factor is safety-critical, so an unconverged Ritz value is never returned as
  if it were an answer.

### Changed

- **C12: the README statistics are now gated.** `make stats` regenerated the
  test/coverage/module numbers quoted in four places in each of two READMEs,
  but nothing enforced that it had been run: round 5 found them advertising
  356 tests / 95.44 %, round 6 found 611 / 95.3 % against a suite that had
  moved past both. `scripts/update_readme_stats.py --check` renders the same
  patches in memory and exits 1 on any difference (a pattern that no longer
  matches is also a failure -- the READMEs changed shape and the numbers are
  no longer maintained at all). Wired as `make stats-check`, into
  `make check-all`, as a CI `stats` job, and as a **pre-push** hook: it
  re-runs the suite plus mypy (~100 s), and gating every commit on that is how
  gates get bypassed with `--no-verify`. `make pre-commit-setup` now installs
  the pre-push hook type too. The CI job lets the script do its own measuring
  rather than parsing the coverage job's log in shell, so `--check` and
  `make stats` share one parser and a green gate genuinely implies
  `make stats` would be a no-op.
- **`BucklingResult` gained `load_factors`, `solver_path` and `n_free_dof`.**
  A list of mode shapes without the loads they belong to cannot be checked,
  plotted or ranked by the caller, so each returned mode now carries its
  factor.
- **C8: `beta_hat` renamed to `beta_mom`** on `MarginStatistics` and
  `HeterogeneityResult`, with the old names kept as `DeprecationWarning`
  alias properties. `mean/std` of a sampled margin is not a Hasofer-Lind
  reliability index -- a FORM index is the distance from the origin to a
  design point minimised over the actual limit-state surface in standard
  normal space, and no such search happens here. The two coincide only for a
  Gaussian margin *and* a linear limit state. `_beta_hat` -> `_beta_mom`.
- `geometric_stiffness(sparse=True)` and the new sparse dyad assembler map
  global DOFs to free positions by fancy indexing rather than a Python
  comprehension over `geom.dofs`, which was a measurable share of the cost at
  the sizes the sparse path exists for. Local dyad vectors are now built
  straight from the shared cosines instead of being gathered out of the full
  `(m, n_dof)` matrix, removing a restricted-vs-unrestricted indexing trap
  that the first draft of this change fell into and a test caught.
- `DEFAULT_THICKNESS_RATIO` / `MIN_THICKNESS_RATIO` are now the single source
  for the section-ratio literals.

### Behaviour changes ⚠

Collected for quick scanning; each is detailed above.

1. **C19 (recorded late): `TopologyResult.tau_vs_base`** was added in 2.8.0
   and never appeared in that release's *Behaviour changes*, though it puts a
   new ranking quantity -- tau-b of the perturbed topology against the base
   -- into `compute_ci_for_topology` output. Recorded here for completeness;
   it is additive and does not alter any pre-existing number.
2. **`BucklingResult.modes` is now ordered by ascending `lambda_cr`** (was
   ascending `nu`, i.e. the *opposite*), is truncated to `n_modes` (was all
   positive eigenvalues), and `modes[0]` now genuinely equals `mode`. Code
   that indexed `modes[-1]` for the critical mode must change to `modes[0]`.
3. **`lambda_cr` from `eigen_solver="auto"`** switches to sparse Lanczos at
   `SPARSE_EIGEN_THRESHOLD = 400` free DOFs. Agreement with the dense path is
   `< 2e-13` relative on `lambda_cr` and on mode subspace angles across
   241-2001 DOF, with and without a thermal prestress field, so no published
   number moves at reporting precision; the threshold was set from measurement
   (sparse/dense time ratio 1.7x, 3.6x, 0.42x, 0.88x, 0.52x, 0.40x, 0.36x,
   0.29x, 0.29x at `n_free` = 33, 65, 121, 181, 241, 481, 801, 1281, 2001),
   choosing 400 over the ~120 crossover because the dense path also buys an
   unconditional Cholesky verdict and the full spectrum.
4. **`beta_hat` -> `beta_mom`** on two public dataclasses. The old attribute
   still works and emits `DeprecationWarning`; keyword construction with
   `beta_hat=` does not.
5. **`apply_decision(thickness_ratio=...)` now defaults to `None`** (derive
   per member) instead of `25.0`. Enlarged `I_sec` values therefore change for
   any member whose real `b/t` is not 25 -- which is the point, and the
   direction is *less* optimistic for slender sections. Pass an explicit
   ratio to restore the old behaviour.
6. **New warnings, no silent number changes:** `LumpedCapacityWarning` for
   `A_m/V < 50 1/m`, `IllConditionedPerturbationWarning` from `perturb_multi`,
   a `UserWarning` when a parametric fire's `q_td` is outside [50, 1000], and
   `MechanismError` (instead of a cryptic factorisation failure) when
   simultaneous removals exceed the system's redundancy.
7. **`solver_metadata` gained `assembly_mode` and `reduction_order`**
   (additive; strict JSON preserved).
8. Models solved through the unchanged dense path, uncorrelated sampling,
   unprotected members below the `A_m/V` threshold, and all legacy scalar APIs
   remain **bit-for-bit identical**.

### Deferred with rationale (recorded, not dropped)

- **C11 -- separate `MaterialState` from `MemberResponse`.** The two are
  genuinely conflated (`MemberResponse` carries both the response quantities
  `axial_force`/`E` and the section/material state `A`/`I_sec`/`yield_stress`/
  `temperature`). Splitting them is the right call but touches every
  constructor site in the reliability chain, and doing it at the end of a
  round with no budget for a full re-verification is how a clean refactor
  becomes a silent behaviour change in a safety-critical path. Deferred to a
  round of its own.
- **C10 -- complex-step verification of the DDM.** The sensitivity module's
  analytic derivatives are already cross-checked against central differences;
  complex-step would remove the round-off floor and let the agreement be
  asserted to machine precision. Worth doing, but it needs `complex`-safe
  paths through the EN material interpolation (piecewise-linear tables with
  `np.interp` do not accept complex input), which is a change to the material
  layer rather than to the sensitivity layer.
- **C13/C14 -- a machine-readable `physics_boundary.yaml` and its propagation
  into `AnalysisResult`/JSON.** `docs/theory.md` already states the boundary
  matrix in prose; making it a shipped artefact that the result carries is a
  good idea and a schema decision (what belongs in a *result* versus in
  documentation) that deserves its own discussion rather than a late-round
  guess.
- **C9 -- mandatory triple ranking output.** Ranking currently returns one
  ordering chosen by the caller; making all three unconditional would change
  the shape of a public result type.
- **C4 -- calibrating `dpocon` on more than two fixtures.** Needs a reference
  corpus with known condition numbers; the benchmark suite is the right home
  and does not have them yet.

## [2.8.0] — 2026-09-15

This release closes the fifth external audit round (nine independent
critiques). As in rounds 1-4, every concrete finding was **reproduced
against `d79f8c8` before being fixed**, each fix carries a pinning test,
and findings that did not reproduce are recorded under *Not reproduced*
with the measurement. Seven of the twelve rows in the most detailed
code-level critique described the pre-2.7 codebase; they are answered
with file/line evidence rather than changes. Items that **alter reported
numbers** are marked ⚠ and collected under *Behaviour changes*.

### Fixed

- ⚠ **The reliability engine's buckling margin used bare Euler while the
  DCR chain used the code-correct `chi` model** (critic 8, finding 1 -
  reproduced). `_buckling_margin` computed `P_cr - |N|` with
  `P_cr = pi^2 EI/(kL)^2`, the model `BucklingModel` itself documents as
  optimistic (~15 % at `lambda_bar ~ 2`, far more near `lambda_bar ~ 1`).
  On the reference member at `lambda_bar ~ 0.5` the Euler margin was
  **6x** the `chi` margin (2.73 MN vs 0.45 MN), so every reported
  buckling `beta` was systematically optimistic and inconsistent with the
  library's own capacity model. The margin is now
  `chi * A * f_y / gamma_M - |N|`, built from the *same*
  `sections.non_dimensional_slenderness` +
  `sections.buckling_reduction_factor` pair the limit-state layer uses
  (fire imperfection factor above ambient, EN 1993-1-1 curve at 20 degC),
  cross-pinned against `limitstates._member_limit_state` to 1e-12
  (`test_buckling_margin_matches_limitstates_capacity_at_fire_temperature`).
  `MemberResponse` gained a `temperature` field (default 20.0) that only
  selects the code regime; `E` and `yield_stress` remain as-given.
  `ReliabilityEngine(buckling_model=..., buckling_curve=...)` exposes the
  choice; `EULER_ONLY` reproduces the legacy numbers bit-for-bit
  (`test_buckling_margin_euler_only_is_bit_for_bit_legacy`). A missing
  `yield_stress` now yields `NaN` under `EUROCODE_CHI` (no silent Euler
  fallback - that would re-create the two-capacity-models split).
- **`perturb_multi` emitted garbage instead of failing when simultaneous
  damage created a mechanism** (critic 8 - reproduced). Deleting two
  members of a determinate triangle leaves the Woodbury core
  *mathematically* singular but *numerically* invertible (~1e-17 pivot):
  `np.linalg.solve` returned a finite displacement vector instead of
  raising, and the exactly-singular single-member case leaked a raw
  `LinAlgError` instead of the library's `MechanismError` contract. The
  core is now screened with the same relative criterion as the base state
  (`_check_lu`, now parameterised by context), so both cases raise
  `MechanismError` naming the perturbed members
  (`test_perturb_multi_raises_mechanism_error_not_linalg`).
- **`scf_alpha_min` silently meant "last probed alpha", not "smallest
  alpha"** (critic 8 - reproduced). `DamageOperator` took `scfs[-1]`,
  which only equals the documented "response ratio at the smallest
  non-singular alpha" when the caller passes `alphas` in descending order.
  The point is now selected by value (`np.argmin`), making the profile
  order-invariant (`test_scf_alpha_min_is_order_invariant`).
- **`ThermalDegradation.apply` raised a bare `KeyError` for members
  missing from the temperature map** (critic 8 - reproduced). Now a
  `ValueError` naming every missing member id and the fix
  (`test_thermal_degradation_missing_temperature_names_members`).
- **The Gaussian copula accepted asymmetric and non-unit-diagonal targets
  silently** (critic 8, finding 8 - reproduced; the Cholesky reads only
  the lower triangle). `_kruskal_inverse` now validates squareness,
  symmetry (1e-12), unit diagonal and off-diagonal range before mapping,
  in both correlated-sampling paths
  (`test_correlation_validation_rejects_asymmetric_and_bad_diagonal`).
- **`DamageOperator._check_mechanism` ran a full `O(n^3)` SVD per member
  with a hard-coded `1e-5` rank cutoff** (critic 8, perf finding 3 -
  reproduced). Replaced with the Cholesky + LAPACK `dpocon` pattern the
  criticality engine already uses (factorisation failure = hard mechanism;
  `rcond < _KEY_ELEMENT_RCOND` = near-mechanism at the `alpha = 1e-6`
  probe scale). Classification verified **identical on every member of
  both regression fixtures** (essential: `rcond ~ 1e-7`; redundant:
  `rcond ~ 6e-2`; threshold sits mid-gap) and the threshold is now a named
  constant documented as probe-calibrated, deliberately distinct from the
  `NumericalTolerances` policy cutoffs
  (`test_key_element_detection_matches_mechanism_semantics`).
- **`DamageOperator._solve` re-implemented member-force recovery by
  hand** - the exact "two implementations of one physics" pattern that
  produced the 2.6.0 demand-chain bugs (critic 8 - reproduced). Forces now
  come from `postprocess.calculate_element_forces`, the canonical path
  (`test_solve_forces_match_postprocess`, with thermal + fabrication
  strain active so the prestress convention is exercised).
- **`sensitivity.compute_all` re-assembled the global matrix it had just
  solved** (critic 8, perf - reproduced). One shared
  `_assemble_and_solve` feeds both `compute_baseline` and the DDM pass.
- **`load_vector` rebuilt the node-index dict inside the load loop**
  (critic 8, perf - reproduced); hoisted.
- **`UniformForceScan.forces_at` re-imported `MechanismError` locally**
  although the module imports it top-level (critic 8 - reproduced);
  removed.
- ⚠ **Correlated Latin-hypercube draws lost their stratification**
  (critic 8, finding 3 - reproduced). `sample_spec_matrix` coupled the
  design through `gaussian_copula_correlate`, whose linear mix
  `z @ L.T` recombines all columns: marginals stayed uniform but the
  "exactly one sample per stratum per dimension" property - the entire
  variance reduction LHS provides - was destroyed. The pipeline now uses
  **Iman-Conover rank reordering** (`iman_conover_correlate`): each output
  column is an exact permutation of the input column (stratification
  provably preserved, `test_iman_conover_preserves_lhs_stratification_exactly`),
  the realised Spearman correlation reproduces the target within sampling
  noise (0.6 -> 0.5964 at n=2e4, same accuracy as the copula path), and
  the transform is deterministic given `(u, correlation)` - no extra seed,
  consistent with the library's reproducibility rule.
  `gaussian_copula_correlate` remains for non-LHS inputs with an explicit
  docstring warning about stratification.
- **86 `__pycache__/*.pyc` files were tracked in git** (critic 8, hygiene -
  reproduced: `git ls-files | grep __pycache__ | wc -l` = 86, cpython-312
  bytecode in history). Purged from the index; `check_public_hygiene.py`
  now rejects `.pyc/.pyo` names and `__pycache__` paths; and
  `test_no_bytecode_tracked_in_git` pins the index itself so
  re-introduction fails the suite.
- **README/badge/version drift** (critic 8, hygiene - reproduced: README
  advertised "356 tests / 95.44 %" while HEAD ran 539 tests at 95.06 %,
  and `__init__.py`'s version fallback was stuck at "2.5.0"). New
  `scripts/update_readme_stats.py` (Makefile target `stats`) measures the
  real values (one coverage run + one mypy run) and patches all nine
  quoted locations across both READMEs (English digits and Persian);
  the fallback chain is `_version.py` -> `importlib.metadata` -> literal,
  so a stale hardcode can no longer outlive two release cycles.

### Added

- **`stability.py` - geometric stiffness and linearised bifurcation**
  (critics 2-A, 5-1, 6-2: the top scientific gap; round-5 scope decision:
  linearised system bifurcation now, nonlinear continuation deferred).
  `geometric_stiffness()` assembles `K_G = sum (N_e/L_e) g_e g_e^T` from
  the shared `MemberGeometry` (dense or CSR-sparse, full or free-DOF);
  `linearized_buckling_load_factor()` solves
  `[K_E + K_G(N_imposed) + lambda K_G(N_mech)] u = 0` for the smallest
  positive `lambda` via symmetric whitening (`C = L^-1 B L^-T`,
  `lambda_cr = 1/nu_max`), returning `BucklingResult(lambda_cr, mode,
  n_compressed, base_forces)`. Temperature fields feed the *same*
  `build_engine` setup the fire chain uses, so bifurcation and DCR cannot
  disagree about the base state; an imposed-force state that is itself
  unstable raises `MechanismError` instead of returning a factor.
  Verification (docs/theory.md §9, matrix rows): shallow-toggle closed
  form `P_cr = 2EAh^3/(b^2 L_0)` matched to 1e-10 relative at four rise
  ratios; dense LAPACK QZ eigensolve cross-check on eight campaign
  topologies; `b ⟂ g` orthogonality and the `u^T K_G u = sum N L phi^2`
  energy identity; single-member golden 4x4; rigid-motion and load-scaling
  invariance; tension-only `inf`; zero-force irrelevance; mode residual
  < 1e-10; and the headline physics - a redundant shallow fan under
  growing fabrication misfit loses its load factor monotonically
  (297 -> 214 -> 131 -> 48 -> 6.4) and then destabilises, exactly the
  restrained-prestress effect a first-order DCR chain cannot see.
- **`thermal/fire_curve.py` - ISO 834 exposure and EN 1993-1-2 §4.2.2.2
  lumped-capacitance member heating** (critics 1-9, 3-9, 6-3 - the
  exposure-to-temperature layer whose absence made "fire analysis" claims
  incomplete; critic 4's layering demand). `iso_834_temperature()` matches
  the six published table points (576/679/739/842/945/1049 degC) to
  < 1 degC; `steel_temperature()` integrates
  `rho_a c_a(theta_a) V dtheta/dt = k_sh A_m h_net` with RK4 at <= 5 s
  steps and the standard's temperature-dependent `c_a` (including the
  ~735 degC endothermic spike) from the existing Table 3.1 fixture.
  Oracles: constant-property convection limit matches the closed-form
  exponential to 1e-10 relative; independent `solve_ivp` RK45 at tight
  tolerance matches to 2e-4 degC; surface heat input closes the enthalpy
  balance `rho int c_a dtheta` to 5e-5 relative; monotonicity, gas bound,
  `A_m/V` and `k_sh` orderings, step-halving convergence (2e-4 degC
  spread), custom-curve acceptance, and an end-to-end bridge test where a
  fire-heated restrained bar reproduces the analytical
  `N = -k_E(theta) E A alpha dT` to 1e-12. Scope documented: uniform
  section temperature, unprotected steel, gas temperature given - no
  through-thickness conduction, no zone model.
- **`thermal_strain()` / `effective_alpha()` in the material layer**
  (critic 8, finding 4 - reproduced by measurement: the ambient constant
  `1.2e-5` understates the standard's elongation by **+20.7 %** at
  600 degC, secant slope `1.448e-5`). `effective_alpha(theta, theta_0)`
  is the chord slope of the clause-3.4.1.1 elongation curve; setting
  `elem.alpha = effective_alpha(T)` makes the framework's existing
  `alpha * delta_T * L` prestress term reproduce the standard's free
  thermal elongation *exactly* (dense-grid identity test). Element
  defaults are unchanged, so models without imposed strain stay
  bit-for-bit; the correction is an explicit, documented user choice.
- **`failure_mode` on critical temperatures** (critic 8, finding 5 -
  reproduced: an essentially unloaded member's scan ends at the `k_E = 0`
  table endpoint and pre-2.8 reported ~1200 degC as its "critical
  temperature" with no way to tell system collapse from a material limit
  state). New `FailureMode` enum (`material` / `stiffness_collapse` /
  `none`), `CriticalTemperatureResult(theta, failure_mode)`, and
  `member_critical_temperature_detailed` /
  `system_critical_temperature_detailed`; the legacy scalar functions
  delegate and are bit-for-bit unchanged. Under the real Eurocode law
  `k_y` vanishes before `k_E`, so loaded members fail in `material` mode
  first - pinned by tests on both branches.
- **`pf_empirical` + exact Clopper-Pearson interval on every margin
  statistic** (critics 7, 8-2 - reproduced: `pf_approx = Phi(-beta_hat)`
  silently assumes a normal margin while the input models are lognormal /
  Gumbel). `MarginStatistics` now carries the assumption-free observed
  rate and its 95 % exact-binomial interval; at zero observed failures the
  upper bound (~3/n) states the study's resolution instead of implying
  safety. `pf_approx` is relabelled in the docstring as a first-order
  normal approximation.
- **`solver_metadata` in the analysis result and JSON export** (critic
  1-12): library/python/numpy/scipy versions, BC method and penalty value,
  sparse flag, model counts, the factorisation actually used, rank and
  estimated condition number of `K_ff`, the numerical-status verdict, and
  the full `NumericalTolerances` policy - every number needed to
  reproduce and interpret a report. `run()` now takes its solve through
  `solve_with_diagnostics` (to which `solve` already delegates, so the
  computation is bit-for-bit identical). Non-finite values are exported as
  JSON `null`; the export stays strict JSON (no `Infinity`/`NaN` tokens).
- **`LargeDisplacementWarning` + `check_displacement_magnitude`**
  (critics 2, 5-1, 9-8.3): the linear-kinematics validity boundary is now
  observable. `run()` warns when `max|u|` exceeds 10 % of the
  characteristic length (bounding-box diagonal, the same measure the
  buckling report uses). The three golden-analytical tests whose unit-like
  numbers legitimately trip the guard suppress it locally, documented.
- **`BucklingCheckWarning` in the fire chain for compressed members with
  `I_sec <= 0`** (critic 7 conceptual-3, direction corrected - see *Not
  reproduced*): `dcr_field` and friends were silent while reporting
  `DCR = +inf`; the static-report path already refused to pass such
  members quietly. Both paths are now equally loud, with the member ids in
  the message.
- **Dimensional-similarity (scaling) invariance tests** (critic 4, #12):
  geometry x s, areas x s^2, loads x s^2 => displacements x s, forces
  x s^2, stresses exactly invariant - including a restrained-thermal
  variant. The dimensional-analysis oracle that catches unit/exponent bugs
  coverage cannot see.

### Changed

- `docs/theory.md` gains §9 (geometric stiffness and bifurcation, with the
  toggle derivation), §10 (fire layering, ISO 834, lumped capacitance,
  secant strain), §11 (reliability conventions: one capacity model,
  empirical pf, Iman-Conover, failure modes) and §12 (**Physics Boundary**
  - the formal supported/not-supported matrix critic 4 asked for, with the
  deferred-feature rationale versioned). The §8 verification matrix grows
  eleven rows for the new subsystems; the honest-gaps paragraph is updated
  to say exactly what is now covered (linearised bifurcation, exposure ->
  member temperature) and what remains open (nonlinear continuation,
  section gradients, rare-event reliability, experimental benchmarks).
- `DamageOperator` probe threshold documented as probe-calibrated
  (`_KEY_ELEMENT_RCOND`, tied to the `alpha = 1e-6` probe magnitude) and
  deliberately separate from the `NumericalTolerances` policy; mechanism
  probes cost one Cholesky + `O(n^2)` `dpocon` instead of a full SVD.
- `retrofit/actions.py` now labels the `theta_offset` protection levels as
  the **first-order decision proxy** they are (critic 7-12): constant
  offsets independent of section factor, thickness and exposure time,
  valid for triage ranking, not insulation design; above ~700 degC the
  misestimate can exceed 150 degC, and `thermal/fire_curve.py` is named as
  the physics-based path.
- The criticality engine module docstring states its **dense memory
  ceiling** explicitly (critic 8, perf 1): `O(n_free x n_members)` for
  `z` and `u_pert`, ~1.6 GB for a 10k-member truss, with the sparse
  assembly/solve layers scaling beyond it - the scope limit is now
  documented where the earlier silence implied otherwise.
- `sensitivity.SensitivityResult.ddm_sensitivity` documents two silent
  conventions (critic 8, finding 7): the derivative is a **subgradient**
  of `max|u|` at the base-state argmax node (ties/node switches are
  kinks), and the `d_max < 1e-15 -> 1.0` floor switches the reported
  quantity to the un-normalised numerator with no physical unit.
- `pyproject.toml` `fallback_version` and the README status blocks are
  regenerated from measurement, not handwriting.

### Not reproduced (recorded with the measurement)

- **Critic 7's table rows 1, 2, 3, 4, 5, 6, 8, 9 describe the pre-2.7
  codebase.** At HEAD (`d79f8c8`): `solver.py` factorises Cholesky-first
  with LU fallback (row 1); `limitstates` uses `chi * n_rd` under the
  default `BucklingModel.EUROCODE_CHI` with the legacy `min(P_cr, N_Rd)`
  retained only behind `EULER_ONLY` (row 2); `numerics.NumericalTolerances`
  unifies the rank thresholds and *enforces* their ordering in
  `__post_init__` (row 3 - the 1e-13/1e-9 pair is deliberate: hard gate vs
  diagnostic, documented); `check_energy` tests the unified Clapeyron form
  with a relative `energy_scale` floor and no absolute-joule branch (row 4
  - its docstring even derives the self-equilibrated identity the critique
  re-derives); `sensitivity.py` reports mechanical and total strain energy
  separately (row 5, fixed in 2.7.0); assembly is vectorised COO->CSR with
  C-level duplicate summation (row 6); `tests/test_uniform_invariance.py`
  pins the `tau = 1` uniform-temperature invariance (row 8);
  `postprocess.calculate_buckling` delegates to
  `sections.euler_buckling_load(..., effective_length_factor)` and reports
  `k_factor` (row 9). "theory.md is 26 lines": it is 729 lines at HEAD.
  "48 refactorisations per theta_sys scan": `UniformForceScan` serves the
  whole grid from ONE ambient factorisation (counted in its tests).
- **Row 10 (copula PSD guard)** was fixed in 2.7.0 (Kruskal inverse +
  `ValueError` naming the mapping); the *residual* asymmetry/diagonal gap
  was real and is fixed this round (see Fixed).
- **Row 11 (asymmetric scenario partition)** does not reproduce: both
  boundaries snap toward `mid` within `relative_eps`, and the half-open
  intervals `[0, L/3) | [L/3, 2L/3] | (2L/3, L]` are mirror-symmetric - a
  member at either exact boundary lands in `mid`, and the assignment is
  invariant under `x -> max_x + min_x - x`. Pinned by
  `tests/test_scenario_partition.py`.
- **Row 7 (`perturb_multi` not wired into retrofit)** is factually correct
  but the proposed wiring does not remove any solves: `evaluate()` needs
  the full `dcr_field` and `theta_sys` per decision, not just perturbed
  displacements, so a Woodbury rank-r update cannot replace the re-solve
  there. Recorded as analysed-and-declined rather than silently ignored.
- **Critic 9 §4.2's direction is reversed for `I_sec = 0`:** such members
  were *not* evaluated as "yield only" - `lambda_bar -> inf`, `chi -> 0`,
  capacity 0, `DCR = +inf`: they fail hard. The real gap was silence, now
  closed by the fire-chain `BucklingCheckWarning` (see Added).
- **Critic 6's "unsymmetric K under thermal loads"** does not reproduce:
  thermal effects enter the right-hand side (equivalent nodal forces),
  never the matrix; `K` stays symmetric positive definite and
  Cholesky-first with LU fallback and penalty-scale warnings (visible in
  `test_penalty_bc_and_options.py`) is exactly the requested behaviour.
  Penalty `beta`-sweep sensitivity is measured across four decades in
  `test_penalty_error_and_conditioning_trade_off`.
- **Critic 1's rows 5-7** (solver strategy parameter, COO assembly,
  bisection on `DCR(T)-1`) were landed in 2.7.0; `solver_metadata` now
  also *reports* which factorisation ran. Critic 1-8's tunable guard
  exists as `EngineSetup.guard_tol` + `ci_sweep(guard_tol=...)` with
  condition-scaled default, and `flagged` already reports brute-force
  routing.
- **Critic 3's "iterative thermo-mechanical equilibrium"** is unnecessary
  for this model class: with linear kinematics and eigenstrain loads the
  coupled thermal-mechanical solve is *exact in one step*; iteration would
  only be needed for the (out-of-scope) geometric/material nonlinearity.

### Deferred with rationale (recorded, not dropped)

- Nonlinear continuation (Newton-Raphson / arc-length, post-buckling,
  snap-through): supersedes rather than extends the §9 linearised check;
  needs its own verification suite. The validity boundary is now guarded
  (`LargeDisplacementWarning`) and diagnosable (`lambda_cr`).
- Creep and transient thermal strain: requires time-dependent material
  laws plus validation data beyond the Table 3.1 fixture.
- Torsional-flexural buckling: needs `I_z, I_t, I_w` in the section model.
- PCE / Sobol / subset simulation / FORM-SORM: a UQ subsystem with its own
  validation burden; `pf_empirical` + exact CI covers the honest-reporting
  gap in the meantime.
- Gumbel/Clayton/vine copulas: the Gaussian path now validates its input
  and preserves designs; tail-dependence families change the reliability
  semantics and belong with the UQ rework.
- Per-DOF CI vectors, `ci_sweep` blocking, sparse rank-1 updates,
  `scaled_engine` for uniform fields: the dense ceiling is documented;
  chunking changes the `CiSweep` contract and needs an API design pass.
- Aluminium/concrete materials, 3D, frames, semi-rigid joints, distributed
  loads, section thermal gradients, experimental benchmarks (Cardington),
  OpenSeesPy in public CI: outside the declared scope or blocked on
  external assets; the §12 boundary matrix states each.

### Behaviour changes ⚠

Collected for quick scanning; each is detailed under *Fixed*/*Added* above.

1. **Reliability buckling margins** under the new default
   `BucklingModel.EUROCODE_CHI` are *smaller* (conservative) wherever
   `lambda_bar > 0`; reported `beta_buckling` and `pf` move accordingly.
   `ReliabilityEngine(buckling_model=BucklingModel.EULER_ONLY)` restores
   pre-2.8 numbers bit-for-bit.
2. **Correlated `sample_spec_matrix` draws** use Iman-Conover reordering:
   same target rank correlation, exact LHS marginals, different sample
   values than the pre-2.8 copula mix (still a pure function of the seed).
   Uncorrelated draws are untouched.
3. **`run()` results and JSON exports** carry a `solver_metadata` block
   (additive; strict JSON preserved).
4. **New warnings, no silent number changes:** `LargeDisplacementWarning`
   for gross displacements, `BucklingCheckWarning` from the fire chain for
   compressed members with `I_sec <= 0`, `ValueError` (instead of
   `KeyError`) for missing degradation temperatures, `MechanismError`
   (instead of garbage or `LinAlgError`) from `perturb_multi`.
5. Models **without** imposed strain, uncorrelated sampling, and all
   legacy scalar APIs remain **bit-for-bit identical**.

## [2.7.0] — 2026-09-15

This release closes the fourth external audit round (six independent
audits). Every concrete finding was **reproduced against `06de4a8` before
being fixed**, and each fix carries a pinning test; findings that did not
reproduce are recorded under *Changed* with the measurement. Items that **alter reported numbers**
are marked ⚠ and collected at the end under *Behaviour changes*.

### Fixed

- ⚠ **The DDM sensitivity in `sensitivity.py` differentiated the wrong
  elongation** (critic 5, finding 1 — reproduced). The member area `A_i`
  enters the solved system through *two* channels: the stiffness
  `k_i = E_i A_i / L_i` **and** the imposed-force term
  `F_pre,i = k_i dL_pre,i b_i` that the assembler puts on the right-hand
  side. Differentiating only `K` left the adjoint numerator as the *total*
  elongation `b_i^T U_f` instead of the *mechanical* elongation
  `b_i^T U_f - dL_pre,i = N_i / k_i` — the same convention the criticality
  engine's rank-1 numerator has used since 2.6.0. Reproduced on a three-bar
  redundant model with one heated member: the reported sensitivity was
  `-2.485` where the central difference gives `+1.4467e-2` — wrong sign,
  ~170x wrong magnitude. After the fix the DDM matches central differences
  to seven digits in both the heated and the cold case
  (`test_ddm_matches_central_finite_difference`). Models without imposed
  strain are bit-for-bit unchanged.
- ⚠ **The Gaussian copula fed the Spearman target straight into the
  Cholesky** (critic 1, finding 2 — reproduced). By Kruskal's theorem,
  normals with Pearson correlation `rho_P` carry rank correlation
  `rho_S = (6/pi) arcsin(rho_P / 2)`; injecting the target directly made the
  *realised* rank correlation the Kruskal image of the target — a systematic
  contraction (target 0.70 realised 0.682; 0.90 realised 0.891) biasing every
  correlated reliability draw. `gaussian_copula_correlate` now inverts the
  relation (`rho_P = 2 sin((pi/6) rho_S)`, diagonal pinned to 1) before the
  factorisation; the realised `rho_S` reproduces the target within sampling
  noise (0.70 -> 0.6992 at n = 2e5). An indefinite target raises `ValueError`
  naming the mapping instead of a bare `LinAlgError`.
- ⚠ **`ci_two_component` referenced the DCR component to the COLD structure**
  (critic 3, P0 — reproduced). `docs/theory.md` 5.2 defines the baseline as
  "the undamaged structure at the same temperature field", and the index is a
  damage counterfactual; the code instead divided by the 20 degC DCR "so that
  temperature degradation does NOT cancel out". Exposing test: at `alpha = 1`
  — no perturbation at all — a 600 degC redundant truss reported
  `dcr_component ~ +2.1` with `governing = buckling`, i.e. fire degradation
  masquerading as damage criticality, contaminating `governing` and every
  ranking built on `ci_values`. The two effects are now separated and
  reported explicitly: `dcr_component = DCR_pert(T)/DCR_base(T) - 1`
  (exactly 0 at `alpha = 1`), a new `fire_component =
  DCR_base(T)/DCR_base(20) - 1` (independent of `alpha`), and a new
  `dcr_combined = (1 + dcr)(1 + fire) - 1` which is bit-for-bit the legacy
  cold-referenced ratio for triage contexts that want both effects in one
  explicit number. The composite `ci = max(u_component, dcr_component)` is
  pure criticality. `theory.md` 5.2 carries the derivation; the MC ladders
  and the probabilistic ranking now build their fire-responsive statistic
  from `max(u_component, dcr_combined)` explicitly, with their estimands
  re-documented.
- **The Sherman-Morrison guard used a fixed `1e-8` threshold** (critic 1,
  finding 4 — mechanism confirmed analytically). The round-off level of
  `denom = 1 + Delta_i d_i` is `~ cond(K_ff) * eps`; at `cond = 1e8` that is
  `~2e-8`, above the fixed threshold, so a healthy member of an
  ill-conditioned structure divided by a ~20%-contaminated denominator and
  emitted a silently inaccurate rank-1 CI instead of routing to the exact
  brute-force solve. `build_engine` now estimates `cond` from the LU factors
  it already computed (LAPACK `dgecon`, `O(n^2)`, no extra factorisation) and
  `EngineSetup` carries
  `guard_tol = min(max(1e-8, 100 cond eps), 1e-4)`; `ci_sweep` uses it unless
  the caller overrides. The cap trades brute-force time, never accuracy
  (guarded members are re-solved exactly), and `denom >= alpha`
  mathematically bounds healthy members away from any `tol <= alpha`.
  Reproduced before the fix (`denom ~ 1e-7` at `cond ~ 1e8` passed unflagged);
  now flagged, routed, and equal to the exact solve bit-for-bit
  (`tests/test_guard_tolerance.py`, 5 tests).
- **`member_critical_temperature` linearly interpolated a curved `DCR(T)`**
  across a 25 degC cell while its docstring claimed "root-found" (critic 3,
  P1). It now brackets on the grid and bisects on exact
  `UniformForceScan.forces_at` evaluations — `O(m)` each, so the
  one-factorisation-per-scan guarantee is untouched (the decomposition-count
  test stays green) — down to `1e-3` degC, and returns the upper bracket end,
  making `DCR(theta) >= 1` true by construction: conservative by at most the
  tolerance, with no guarantee-free interpolation error of up to a grid step.
  Pinned two-sided through the independent `dcr_field` path.
- **The zero-stiffness endpoint crashed the temperature scans** (critic 3,
  P1 — path hardened). A grid point with `k_E(T) <= 0` (the Eurocode
  zero-stiffness endpoint at 1200 degC) raised `MechanismError` out of the
  middle of a scan. It is now modelled as what it physically is — failure by
  collapse at that point: `system_critical_temperature` returns the last safe
  temperature, `member_critical_temperature` converges to the endpoint. With
  the real Eurocode law the capacity collapse at 1100 degC fires first, so
  this only affects custom grids and direct `forces_at` users; both are
  covered by tests.

### Changed

- **`system_critical_temperature`'s definition is now explicit** (critic 3,
  P1 — documentation defect, code correct). The docstring said "highest
  scanned temperature with no member at DCR >= 1", which read over the WHOLE
  grid and contradicted the first-failure algorithm whenever `DCR(T)` is
  non-monotone. The algorithm is right for a heating fire: once a member
  crosses `DCR = 1` at `T*` it has failed at `T*`, whatever redistribution
  does above it. θ_sys is now documented as the first loss of acceptability
  along the monotone heating path — the largest `T` such that every scanned
  point up to `T` is safe. **No numerical change.**
- **`UniformForceScan` is documented as an exact solver for the special case
  of uniform scalar stiffness degradation only** (critics 3 and 5): one
  shared factor `s(T)` must scale every member's modulus (one material law,
  one temperature for all members). Per-member materials, protection or
  temperature histories break `K(T) = s(T) K_0` and must fall back to the
  per-point engine rebuild. It is not a generic thermal-scan engine, and the
  class docstring now says so.
- **The `O(N)` `list.index()` anti-pattern is gone from `sensitivity.py`**
  (critic 1, finding 1): the 4M-iteration compatibility-assembly loop and the
  per-member critical-DOF resolution now use an `O(1)` dict mapping, and the
  `next(n for n in nodes ...)` scans in `_limit_states_from_forces` /
  `_length` were replaced the same way.
- **The sparse free-DOF extraction in `solver.py` stays `np.ix_`, with the
  measurement on record** (critic 1, finding 3 — claim did not reproduce).
  SciPy dispatches array-x-array indexing on CSC/CSR to its C++ IndexMixin,
  not Python loops; measured on a 40k-node band truss (80k DOF, 480k nnz) the
  slice costs ~12 ms against the seconds-scale SuperLU factorisation of the
  same matrix — well under 1% of the solve. The proposed second assembly
  path (building `K_ff` directly at element level) is not worth its
  synchronisation risk against `assembly.py`; the comment in `solver.py`
  records the numbers.
- **`lambda_bar_theta` was verified to be built from `E(theta)`** (critic 5,
  finding 4 — did not reproduce; the code was already correct) and is now
  pinned by regression: `lambda_bar(theta)/lambda_bar(20) = sqrt(k_y/k_E)`
  against the closed form, the absolute value against a hand-built `N_cr`
  from `k_E(theta) E`, plus an explicit not-the-ambient-modulus guard.

### Added

- **Cross-path demand invariants** (critic 5, finding 2):
  `tests/test_cross_path_demand.py` pins that the full public pipeline
  (JSON -> `run()` -> dense block assembler) and the limit-state chain
  (rank-1 engine) report the same member forces on a redundant heated model
  with restrained expansion *and* fabrication strain — two assemblers that
  share no code — and that the `ReactionInfluence` reactions close globally
  against the mechanical load in both directions (critic 5, finding 6).
- 22 new tests total (521 vs 499), covering every fix above: DDM vs central
  differences (heated/cold), copula target fidelity (negative, 3x3,
  indefinite, marginal preservation, spec-level path), the `alpha = 1`
  exposure case, fire/damage component isolation on determinate and
  redundant frames, bisection bracketing, collapse endpoint, and guard
  routing.

### Behaviour changes ⚠

Numbers published before 2.7.0 change as follows (each traced to a
reproduced defect — none is a refactoring side effect):

1. **`ci_two_component` on heated models**: `dcr_component`, the composite
   `ci` and `governing` now measure the damage counterfactual at the fire
   state; the legacy cold-referenced ratio is exactly `dcr_combined`.
   Ambient-temperature models are unchanged (at `T = 20` the two baselines
   coincide). Consumers that need the legacy composite build
   `max(u_component, dcr_combined)` — the MC tests do.
2. **Correlated sampling**: for the same seed, `gaussian_copula_correlate`
   and `sample_spec_matrix(..., correlation=...)` draws change so that the
   realised rank correlation equals the target instead of its Kruskal image.
3. **`IndependentValidator.compute_all`**: `ddm_sensitivity` values change
   for models with imposed (thermal/fabrication) strain — they were wrong
   (wrong sign possible); cold models are bit-for-bit unchanged.
4. **`member_critical_temperature`**: returns the bisected crossing
   (conservative upper end, `<= 1e-3` degC above the true first crossing of
   the bracketed cell) instead of a linear interpolation that could land on
   either side by up to a grid step. `system_critical_temperature` is
   numerically unchanged.
5. **`EngineSetup`** gains a `guard_tol` field (default `GUARD_TOL`, so
   direct constructions keep working); `ci_sweep`'s `guard_tol` parameter
   now defaults to `None` = "use the setup's adaptive value" — an explicit
   float still overrides, and results only change where the old fixed
   threshold was below the round-off band of an ill-conditioned `denom`,
   i.e. exactly where the old numbers were noise.

## [2.6.0] — 2026-09-15

This release is the outcome of a nine-part external technical audit of the
computational core, followed by a second judging round that reproduced the
first round's fixes against the exact commit and surfaced six further
findings; all six are closed here. Every defect listed below was reproduced
against the code before being fixed; nothing here is a speculative change.
Several items **alter reported numbers** — they are marked ⚠ and collected at
the end under *Behaviour changes*.

### Fixed

- ⚠ **The fire demand chain was blind to restrained thermal expansion**
  (second-round finding 1, the audit's top priority). `DCR`, `theta_sys`, the
  CI engine and the retrofit triage degraded `E(T)` but solved against the
  *mechanical* right-hand side only, while `run()` applied the same
  temperature through the equivalent nodal forces of restrained expansion. On
  a redundant structure a heated member developed real compression that the
  whole demand chain never saw — non-conservative exactly in the local-fire
  scenarios the library targets. The engine now carries the imposed
  elongation `dL_pre = alpha (T - 20) L + delta_L_free` end to end:
  `total_load_vector` solves against `F_mech + B^T diag(k(T)) dL_pre`,
  `member_forces` reports `N = k(T)(b.u - dL_pre)`, the rank-1 numerator
  becomes the member's *mechanical* elongation (softening a member scales its
  thermal force with its stiffness — derivation in the engine docstring and
  `docs/theory.md` §6), and reaction indices subtract the direct fixed-DOF
  load (`ReactionInfluence.f_ext_fixed`). The brute-force reference and the
  OpenSeesPy bridge carry the identical physics. Elements without
  `alpha`/`delta_L_free` reproduce every previous number bit-for-bit; heated
  models are pinned against closed forms and against the independent
  assembler path in `tests/test_thermal_demand.py` (18 tests).
- **A fully restrained model (zero free DOFs) crashed the criticality
  engine** on empty-array reductions. It now reports the pure imposed-strain
  state `N = -k dL_pre` that such a model physically has.
- **`_solve_sparse` leaked a raw SciPy `RuntimeError`** when SuperLU failed
  after the SVD screen had passed, breaking the library's documented error
  contract (second-round finding 4). Every factorisation failure now raises
  `SingularMatrixError ... from exc`, with the screened case saying so.
- **`buckling_reduction_factor` glued the 0.65 fire imperfection reduction
  onto any buckling curve** (second-round finding 5). EN 1993-1-2 4.2.3.1(3)
  defines `alpha = 0.65 alpha_c` for curve `c` only; `fire=True` with another
  curve now raises `BucklingCheckWarning` naming the combination an
  extrapolation beyond the code.
- **`graph_validation` kept a dead private copy of the fixed-DOF rule**,
  contradicting the "exactly one place" guarantee in `model.py`
  (second-round finding 6). Removed; `model.fixed_dof_indices` is now the
  single definition in fact as well as in documentation.

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
- **`UniformForceScan`** (second-round finding 2) — one ambient
  factorisation serving an entire uniform-temperature force grid, exact by
  algebra (`u(T) = z_m/k_E(T) + (T-T0) z_alpha + z_free`): under a uniform
  field `K(T) = k_E(T) K_0` and the thermal right-hand side is affine in `T`,
  so `member_critical_temperature` / `system_critical_temperature` went from
  one `O(n^3)` engine build per grid point (48 points by default, 13 per
  retrofit decision) to one build per scan plus `O(m)` arithmetic per point.
  A counted-factorisation test pins "exactly one `lu_factor` per scan"; an
  exactness test pins the scan against per-point `member_axial_forces`
  including fabrication strain.
- **Demand-state primitives in `criticality.engine`** — `prestress_lengths`,
  `imposed_load_vector`, `total_load_vector`, `member_forces`: one public
  definition of the fire chain's right-hand side and member force, consumed
  by the engine, `limitstates`, the retrofit strategies and the OpenSeesPy
  bridge, so the paths cannot drift apart again.
- **`run(..., check_condition=False)`** — exposes the repeated-solve
  recommendation the `solve` docstring already made (temperature sweeps,
  Monte Carlo, retrofit triage); the SVD screen is the dominant cost there.
- **`ReactionInfluence.f_ext_fixed`** and `reaction_influence(..., loads=,
  temps=)` — reactions are the residual `(K u)[fixed] - F_ext[fixed]`, which
  is only equal to `(K u)[fixed]` when nothing loads the supports; a heated
  restrained structure always does.

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
- `reaction_influence` builds `K[fixed, free]` directly as
  `(b[fixed] * k)^T @ b[free]` — `O(m n_fixed n_free)` — instead of
  materialising the dense `(2n, 2n)` global matrix and slicing it
  (second-round finding 3): the reaction index no longer rebuilds the exact
  memory wall the sparse assembly exists to remove.
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

357 → 488 → **517 passing** (499 under the CI ignore list for the three
OpenSees-dependent validation files), coverage ≥ 94 %.

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
- `test_thermal_demand.py` (second round) — restrained bar against the closed
  form `-k_E(T) E A alpha (T-20)`, engine ≡ assembler path on a heated
  redundant frame (base *and* rank-1-perturbed columns), all five
  multi-criteria indices against direct measurement with a load on a support
  node, determinate free expansion ⇒ zero force ⇒ CI exactly 0, `theta_sys`
  and member critical temperature *falling* when expansion is restrained, the
  uniform-scan exactness and single-factorisation guards, and both sides of
  the refined §5.4 invariance claim.

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
6. **DCR, `theta_sys`, CI and retrofit metrics now include restrained thermal
   expansion demand** for models whose elements carry `alpha > 0` or
   `delta_L_free ≠ 0`. Direction: heated restrained members carry *more*
   compression than the old chain reported, so DCR rises and `theta_sys`
   falls — the old numbers were non-conservative. Models built without
   imposed strain (including every campaign fixture and the uniform-T
   invariance theorem's precondition) reproduce all previous values exactly.
7. **`CI_E` is the change in mechanical strain energy**
   `½ Σ k (b.u − dL_pre)²` — the naming the audit asked for; it equals the
   previous quantity whenever no imposed strain is present.
8. **`fire=True` off curve `c` now warns** (`BucklingCheckWarning`): the 0.65
   imperfection reduction is a curve-`c` clause of EN 1993-1-2, and combining
   it with another curve is reported as the extrapolation it is.

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
