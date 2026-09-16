# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **`rank_correlation`** (`src/truss_analysis/validation/metrics.py`)
  now detects constant input explicitly and returns `nan` without
  emitting `scipy.stats.ConstantInputWarning`.  The warning was an
  implementation detail of `spearmanr` leaking through a function whose
  documented contract is "return `nan` for degenerate input" — under
  `filterwarnings = error` it turned a documented return value into a
  test failure.  Discovered by the mandatory `reference-solver` bridge
  job added in v2.10.0, which was the first CI gate to run
  `tests/validation/test_level4_rho_branches.py` on Linux.

### CI

- `reference-solver` job: install `libblas-dev`/`liblapack-dev` and
  symlink versioned shared objects into the linker's default path, so
  OpenSeesPy's Linux wheel can load.  Then use `pyversion` (with
  `getattr` fallback to `getVersion`) — the API name had changed between
  OpenSeesPy releases and the runner's wheel does not expose `getVersion`.

## [2.10.0] — 2026-09-16

Round-8 external audit, on `main@db67ee4`. Eight independent critiques were
read. **Four of the eight had reviewed an older tree** — two said so explicitly
("the remote `main` is still at Sep 14 and has 0 tags"), one reproduced
findings against code that had since been fixed, and one was a restatement of
this project's own changelog with illustrative code attached. Every finding was
therefore reproduced or refuted against `db67ee4` before anything was changed,
and the refutations are recorded below with the file and line that refutes
them. Accepting a stale finding would have meant re-fixing a fixed bug and
calling it progress.

Suite: **1066 → 1130 tests, 0 → 0 warnings, 94.43 % → 94.55 % coverage.**
Mutation score on `limitstates.py`, measured by the new probe: **93.3 %**
(42/45), with the three survivors shown to be mathematically equivalent.

The round closed three findings that three separate critics reached
independently, and the two structural ones were both cases of the same defect
class the round-7 audit named: **a quantity whose name promises something the
value does not deliver.** `dcr` that was not `|N| / capacity`. A warning telling
the caller to pass a parameter no reachable function accepted.

### Changed

- ⚠ **`dcr_field` no longer mutates its own result, and no longer checks system
  stability by default.** With `lambda_cr < 1` the compression DCRs were
  multiplied by `1 / lambda_cr` through `object.__setattr__` on a frozen
  dataclass — no warning, no field recording that it had happened. Reproduced on
  a two-bar toggle at `lambda_cr = 0.543088`:

      dcr(check_system_stability=False) = 275.861571
      dcr(check_system_stability=True)  = 507.950504   (x1.8413 = 1/lambda_cr)
      |axial_force| / capacity          = 275.861571   != dcr
      warnings emitted                  = 0

  So an amplified ratio was indistinguishable from a member-level one, in a
  library whose rule is never to emit a silent number, on the one quantity in
  the fire chain that decides whether a member is acceptable. It was also on by
  default, so every call — including once per candidate inside the retrofit
  search — paid a dense motor build, a factorisation and a Lanczos sweep.

  And when the base state was already a mechanism, only the compression members
  were set to `inf`. On the fixture now pinned in
  `tests/test_dcr_system_stability.py` the post was reported at `dcr = 11.7`
  while the rafters were `inf` — a structure that could not stand up, reported
  as having a member in reserve.

  Now: `dcr` is the member check and `dcr == |axial_force| / capacity` holds in
  **every** branch including collapse; the system verdict lives in three new
  fields (`system_dcr`, `system_stability_factor`, `lambda_cr`) carried per
  member so it survives warning filters and reaches the payload; amplification
  is compression-only because a bifurcation is compression-driven, while a
  mechanism is system-wide because the load cannot be carried at all; the
  comparison is `<= 1` and not `< 1`, so a system with exactly zero reserve is
  reported rather than called unremarkable; `lambda_cr = 0.0` rather than `NaN`
  records the collapse, because `NaN` does not survive serialisation; and the
  adjustment is never silent — `SystemInstabilityWarning` is mandatory and
  carries both numbers plus the statement that `lambda_cr` is a linearised
  tangent verdict and not an ultimate load.

  **Twenty-two tests called `dcr_field` and not one passed
  `check_system_stability`.** The entire adjustment ran only through its default
  and was never asserted, which is how it came to mutate a frozen dataclass
  unnoticed. `tests/test_dcr_system_stability.py` adds eleven.

- ⚠ **`use_effective_alpha` now reaches the whole demand chain.** The flag
  existed in exactly two functions in `src/` — `prestress_lengths` and
  `build_engine` — while `member_axial_forces`, `dcr_field`,
  `ci_two_component`, `compute_ci_for_topology`, `UniformForceScan` and both
  brute-force reference paths built their base state on a constant `alpha`. The
  consequence was that `ConstantAlphaWarning` told the caller to "pass
  `use_effective_alpha=True`" and **no function they could reach accepted it**:
  a warning naming a parameter the public API does not take is a promise the
  library cannot keep.

  Worse, `_solve_perturbed_full` (`engine.py:641`) and `brute_force_ci`
  (`engine.py:874`) — the paths a caller goes to when they have stopped trusting
  the fast one — silently disagreed with the engine they were verifying. The
  comparison then measured the gap between two different prestress fields
  instead of the correctness of the Woodbury update.

  The flag now reaches every place the chain forms a base state, including all
  four in `compute_ci_for_topology` (hot engine, hot fallback column, cold
  reference engine, cold reference column), because a ranking is only comparable
  to its own reference if both rest on the same field. The default stays
  `False` everywhere: changing it would silently move every result ever
  published with this library, which is not a change to make quietly.

### Added

- **`UniformForceScan` keeps its closed form under the secant coefficient.**
  Under a uniform field `alpha_eff(T)` is a single scalar, so the thermal
  right-hand side is still a scalar multiple of one temperature-independent
  vector — it only needs a *masked* basis,
  `z_unit = K_0^-1 B^T (k0 · m · L)` with `m_e = 1` where the member expands and
  `0` where it was modelled fixed in length. That mask is the same rule
  `prestress_lengths` applies, so the scan and the per-point engine cannot
  disagree about which members are heated. Cost: one extra back-substitution
  against a factorisation that already exists. Measured against a full per-point
  rebuild over six temperatures in both modes: **worst relative deviation
  5.5e-16**. Both bases are always built, and `use_effective_alpha` is recorded
  on the instance so a scan cannot be mistaken for the other mode afterwards.

- **A criticality-engine benchmark with a genuinely independent oracle.** The
  suite verified the solver, the thermal chain and the tangent stiffness, but not
  the four modules where the most complicated logic lives.
  `criticality_rank1_vs_direct_resolve` pairs the rank-1 sweep against a direct
  stiffness method assembled from geometry *inside the benchmark* and solved with
  `numpy.linalg.solve` — no shared assembly path, factorisation or linear-algebra
  entry point. On a seven-member redundant girder at `alpha = 0.7` the engine
  reproduces it to a **relative error of 1.45e-16** against a 1e-12 tolerance:
  margin 6895x, machine epsilon rather than a fitted bound. The model was chosen
  by searching candidate member sets for the best-conditioned stable one
  (assembled eigenvalue ratio 5.0e-2); a six-member layout tried first was
  **singular**, which is worth recording because a mechanism has no
  redistribution to measure and both sides would have agreed on garbage.

- **`benchmarks/performance.py`: the engine's speed claim, measured.** 165.8x at
  29 members, 562.9x at 61, 1840.5x at 93, against a floor of `m / 4`. It is
  deliberately **not** a `ReferenceProblem`: a wall-clock ratio has no oracle,
  because nothing outside this machine can say how long this machine takes, and
  filing it under the verification matrix would be the category error
  `theory.md` §8.1 exists to prevent. Speed is never reported without agreement —
  every measurement also compares the two paths' criticality indices
  (max `|dCI|` = 2.2e-14 to 1.5e-13) and voids the speedup if they diverge,
  because a path that returns garbage is very fast and a guard that only timed
  would reward exactly that.

- **A mandatory `reference-solver` CI job.** The three files the canonical stats
  configuration ignores need a reference solver the other runners lack. Excluding
  them from the published *count* is correct — the number must be the same on
  every machine, which is what round 7 fixed. Excluding them from *every job* was
  a different decision wearing the same clothes: the cross-validation column of
  the verification matrix was enforced nowhere, while `physics_boundary` shipped a
  digest of that evidence inside `solver_metadata`. A claim was travelling in the
  payload that no gate checked. The job asserts `openseespy` imported before
  running them, because an `importorskip` that skips is green.
  `test_the_ignored_files_are_run_by_a_mandatory_job` parses the workflow and
  fails if the job is deleted, renamed, made conditional, or made to re-introduce
  the ignores it exists to undo.

- **`theory.md` §8.1 and §8.3: verification is not validation.** All five
  columns of the matrix answer the same question — are the equations solved
  correctly — and none answers the one a reader needs: do these equations
  represent a real structure. §8.1 now names four kinds of evidence and marks
  which three this project has, places the OpenSeesPy bridge explicitly in the
  *independent numerical* row (two implementations of the same model is a strong
  check on the algebra and no check on the physics), and calls out the entries
  whose "independent" label is weaker than it looks. §8.3 states the validation
  status in the vocabulary `physics_boundary.yaml` uses, so the prose and the
  machine-readable file cannot drift: **verification strong, validation absent**,
  and the absence is a property of the evidence rather than a defect to assume
  away.

- **`MemberLimitState.lambda_cr` / `system_stability_factor` / `system_dcr`.**
  Carried per member rather than returned once, on the reasoning that
  `SteelHeatingResult.out_of_range` already set: a validity limit that lives only
  in a warning does not survive a caller who filters warnings, and the payload is
  what gets read afterwards.

- **`SystemInstabilityWarning`** (`exceptions.py`), documented with the reason it
  exists rather than only its trigger condition.

### Fixed

- **`ConstantAlphaWarning` counted members that do not expand, and quoted a
  number 3.4x too large.** A member with `alpha == 0` is an explicit statement
  that it does not expand — `prestress_lengths` already refused to override it
  under `use_effective_alpha=True`, with a paragraph explaining why — but the
  warning's statistics did not get the same treatment. Because the quoted
  percentage is built from `np.mean(alphas)`, one zero halved the mean:

      two members, both alpha = 1.2e-5   -> "2 member(s) ... 17.1%"   correct
      one at 1.2e-5, one fixed-length    -> "2 member(s) ... 58.6%"   wrong

  Both the count and the number are now pinned, so a future filter cannot quietly
  widen. A model of only non-expanding members now warns not at all: there is no
  constant-`alpha` approximation in play, and the percentage it could quote would
  be a division against a zero mean.

- **`effective_alpha`'s documentation conflated two ratios of the same pair.**
  At 600 degC a constant `1.2e-5` against the secant `1.448e-5` gives **17.1 %**
  understatement of the *strain* and **20.7 %** overstatement of the *alpha
  needed to fix it*. Both are correct; neither is interchangeable; and
  `1/(1-x) > 1+x` means the second is always the larger. The 2.8.0 entry labelled
  20.7 % as "understates the standard's elongation", which is the wrong label on
  a right measurement, and a test repeated it. Both ratios are now asserted to
  four significant figures with the ordering invariant pinned, so a swap fails.
  The 2.8.0 entry is **annotated rather than rewritten**: a changelog that edits
  its own history cannot be audited against it.

- **`imperfection_sensitivity` bought a full base eigen-solve before validating
  its input.** Both rejected conditions are pure shape and norm tests on the
  `mode=` argument. They now run first, so a malformed mode is an immediate
  `ValueError` rather than a discarded factorisation plus Lanczos. The zero-norm
  test on the solver's *own* mode stays after the solve, because it depends on it.

- **`stability.py`'s module docstring contradicted its own contents.** Line 52
  listed "imperfection sensitivity" among the things that "remain out of scope",
  while `imperfection_sensitivity` (line 1495) and `ImperfectionStudy` (line
  1327) are implemented in that same file. This is the mirror image of the defect
  class round 7 was about — a scope note disclaiming capability the module has —
  and it is the first thing a reviewer checks against the code. The note now says
  what is actually out of scope (post-buckling paths, arc-length continuation,
  Newton-Raphson load stepping) and adds the boundary the linearised criterion
  needs: `lambda_cr` is the criticality of the linearised tangent state and must
  not be read as the ultimate load of the real structure.

- **The changelog had two `[Unreleased]` sections and the released 2.9.0 had no
  heading.** Both the round-7 and round-6 blocks were sitting under
  `## [Unreleased]` although v2.9.0 was tagged on 2026-09-16 and shipped both.
  `test_stats_gate.py` aligns the version across README, `CITATION.cff` and the
  tag, and nothing checked the changelog's own structure, so the drift was
  invisible. Both blocks are now labelled with the release they shipped in, and
  `test_the_changelog_has_one_unreleased_and_a_heading_per_tag` makes the
  property a gate.

- **`scripts/mutation_probe.py`: a targeted mutation probe, and what it found.**
  Round 7 added 246 tests and moved coverage by +0.23 pp, which is the signature
  of parameterised variants over branches that were already covered. Line
  coverage asks whether a line *ran*; it cannot ask whether any test would notice
  if the line computed something else. This round had already proved the
  difference by accident — `dcr_field`'s system-stability branch had twenty-two
  callers in the suite and **zero** assertions, so a path that silently rewrote a
  safety-relevant number was fully covered and fully unverified.

  The probe applies curated semantic edits (comparison, boolean and arithmetic
  operators) to one module and runs that module's own tests against each, with
  `-x` so a killed mutant costs seconds. `limitstates.py` scored **35/45 (77.8 %)**
  on the first run and **42/45 (93.3 %)** after the findings were closed; the
  three survivors left are mathematically equivalent mutants, recorded with the
  reason in `tests/test_limitstates_boundaries.py`, so every mutant that any test
  could kill now dies.

  Two of the ten survivors were in code written **earlier in this same round**.
  Flipping `lambda_cr <= 1.0` to `< 1.0` left the whole suite green — including
  the test written specifically to pin that comparison, because it reached the
  boundary by scaling a load by the measured factor and round-off lands that at
  `1 ± 1e-16`, so no load-scaling test can sit *on* the boundary. The decision is
  now a pure function, `system_stability_amplification(lambda_cr, collapsed)`,
  that can be asked about exactly 1.0, with the whole domain parameterised across
  it. The general lesson is in its docstring: **a boundary that cannot be reached
  from the public API is a boundary that cannot be tested.**

  Chasing a second survivor produced a better result than a killing test would
  have: `ls.compression and ls.p_cr is not None` cannot be separated, because
  `_member_limit_state` sets `p_cr` only inside its compression branch, so the two
  halves are the same statement for every reachable input. That invariant is now
  pinned directly, since redundancy whose redundancy is unstated gets
  "simplified" by someone who does not know it was load-bearing.

  Six more were real boundaries no fixture ever landed on: a member length
  computed as `x_j - x_i` on models whose `node_i` was always at the origin (where
  `+` and `-` agree, and `hypot` is sign-insensitive besides); `axial_force < 0`
  with no member at exactly zero force; `lambda_bar <= 0.2` with no member at
  exactly the limit; `p_cr <= n_rd` in the `EULER_ONLY` governing rule with no
  member at exact equality; `model is EULER_ONLY and p_cr is not None` flipped to
  `or`, which would let a *Eurocode* member fall into the legacy governing rule
  on the default code path; and `len(unassessable) > 5` with no fixture at exactly
  five and six.

  Two fixtures written to close these were themselves **vacuous in ways that
  looked correct and passed**: the translation-invariance test built its elements
  with the default `alpha = 0.0`, so `lengths` only ever fed two zero vectors;
  and fixing that still did not kill the mutant, because the fixture was a
  determinate two-bar toggle in which restrained thermal expansion produces no
  force at all. Setting `alpha`, translating the model and comparing both scan
  bases were each necessary and none sufficient — it took one degree of redundancy
  before a length error became a force error.

  The probe is **not** a CI gate and says so in its own docstring: `cosmic-ray`
  runs the whole suite per mutant, which on 1100+ tests at ~65 s is weeks of
  compute for three modules, and a nightly job that cannot finish is not a gate
  either. What it is instead is a measurement with a named scope, where the
  survivor list is the deliverable rather than the score.

  It also caught a defect in itself, which is the strongest available evidence
  that its baseline guard is load-bearing. Its outer `finally` ran
  `git checkout -- <module>` as a "leave no dirty tree" safety net and destroyed
  uncommitted work in the module being probed, because `git checkout` restores
  HEAD rather than the state the file was in when the probe started. The next
  run's baseline check refused to proceed — correctly, since a probe against a red
  baseline measures nothing and every mutant would look killed — and that refusal
  is what surfaced the damage.

### Not reproduced — recorded with the measurement that refutes them

Four of the eight critiques targeted a tree that no longer existed. Listed
because a finding that is not recorded gets made again, and because "we checked
and it was already fixed" needs the line that proves it.

- **"The solver still uses `np.linalg.solve`, no Cholesky."** `solver.py:39`
  imports `cho_factor, cho_solve`; `:265-266` uses them and reports
  `"cholesky"` as the factorisation kind.
- **"`check_energy` rejects self-equilibrated states via
  `if abs(W_mech) < 1e-12`."** No such branch. `solver.py:742` is the unified
  identity `expected = strain_energy + 0.5 * prestress_work + penalty_energy`,
  which is exactly the Clapeyron form the finding asked for.
- **"`limitstates.py` still uses `min(p_cr, n_rd)`."** `BucklingModel.EUROCODE_CHI`
  is the default and `limitstates.py:497` computes `capacity = chi * n_rd`. The
  `min` model survives only as `EULER_ONLY`, retained so results computed with it
  stay reproducible bit for bit, and it emits `LegacyBucklingModelWarning`.
- **"`postprocess.py` computes Euler without the effective-length factor."**
  `postprocess.py:421` calls `euler_buckling_load(e.I_sec, L, e.E,
  e.effective_length_factor)`.
- **"`BucklingResult` returns only one mode."** It has `modes: list[ndarray]`,
  `load_factors: tuple[float, ...]` and `multiplicity: int`, with `mode` kept as
  `modes[0]` for backward compatibility. `imperfection_sensitivity` takes
  `n_modes`.
- **"The validation extra is not installed in any mandatory job."** `ci.yml:77`
  installs `.[validation]` in the mandatory `test` job, and `:102`/`:128` in
  `coverage` and `stats`. The *half* of the finding that was true — the three
  bridge files were `--ignore`d in every job, so nothing enforced them — is fixed
  above by the `reference-solver` job.
- **"The `Table31ReductionFactors` docstring overclaims about a transcription
  slip."** There is no class of that name in `steel_eurocode.py` (the module
  exposes functions: `k_y`, `k_E`, `eps_y`, ...) and no such phrase in it.
- **"You measured 15.1 % in v2.9.0 against 20.7 % in v2.8.0."** The string
  `15.1` does not occur anywhere in the repository. The underlying concern was
  nonetheless correct — two numbers *were* in circulation for one quantity — and
  is fixed above; the real pair is 17.1 % and 20.7 %.

### Deferred, with the reason

- **Newton-Raphson and arc-length continuation.** Both critiques that asked for
  them are right that `exact_tangent_stiffness` without a nonlinear solver is an
  academic exercise. They are also outside a library whose stated boundary is
  linear statics with a linearised stability companion, and adding them in a
  final round would put an unverified nonlinear path next to a verified linear
  one. The scope note now says so precisely instead of disclaiming capability the
  module already has.
- **Distributed and member loads.** The largest functional gap, and the one most
  likely to matter in practice. It needs equivalent nodal forces in the
  load-vector assembly *and* a decision about whether a member load splits the
  member, which is an API question rather than an implementation one — the same
  reason it was deferred in round 7, unchanged.
- **Mutation testing.** The coverage plateau is real: 246 new tests in round 7
  bought +0.23 pp, which is the signature of parameterised variants over branches
  that were already covered. `cosmic-ray` runs the whole suite per mutant, and at
  ~105 s a run on 1103 tests that is weeks of compute for the three modules worth
  mutating — not a gate, and a nightly job that cannot finish is not a gate
  either. What this round did instead is targeted at the same failure mode: the
  defects it found were all in code that *was* covered and was still wrong
  (`dcr_field`'s mutation path had 22 callers and zero assertions), so the
  lever was tests that can fail on the branches that matter, not a higher
  percentage.

---

## [2.9.0] — 2026-09-16

Round-7 external audit, on `main@ee0e9bf`. Nine independent critiques were
read; the two that targeted this exact commit were verified line by line against
the tree before anything was changed. Every finding was reproduced or refuted
with a measurement first — none was accepted on description alone.

Suite: **820 → 1066 tests, 68 → 0 warnings, 94.19 % → 94.4 % coverage.**
The warning count is not a cosmetic change: `filterwarnings = ["error"]` is now
in `pyproject.toml`, so a diagnostic that is not asserted somewhere is a build
failure. That gate found two real defects on its first run (below).

### Fixed

- **The sparse positive-definiteness gate could return a confident, wrong
  buckling load.** `_sparse_smallest_eigenvalue` shift-inverts at `sigma = 0`,
  so it finds the eigenvalue of smallest *magnitude* — the right question at the
  loss-of-definiteness crossing and the wrong one past it. On a base state already
  deeply indefinite the probe returns a positive number, `lam_min <= pd_tol`
  passes, and `eigsh` is handed an indefinite `M` for a problem documented as
  symmetric positive definite. ARPACK does not check that assumption; it returns
  a number. Reproduced on a real model, not a synthetic matrix: a three-fan chain
  with the middle fan over-prestressed has a 6-DOF base spectrum of about
  `{-8.5e5, -2.9e5, +1.0e5, ...}` and answered

      dense : MechanismError                                (correct)
      sparse: lambda_cr = 236.85, path = "sparse-lanczos"   (bogus)

  The docstring admitted the gap and told the caller to pass
  `eigen_solver="dense"` when such a state was suspected — but `"auto"` reaches
  exactly that corner at `n_free >= 400`, so a safety-critical verdict was left
  to the user guessing. The gate is now unconditional: Sylvester inertia read off
  the `U` diagonal of the same `splu` factorisation the path needs anyway, which
  costs nothing and depends on no iteration, no tolerance and no spectral
  assumption. Verified against dense eigenvalue sign counts on 400 random
  matrices with a prescribed number of negative eigenvalues: zero mismatches.
  `diag_pivot_thresh=0.0` is load-bearing for this and is now documented as such.
- **Four `warnings.warn` calls embedded their own category name in the
  message**, and Python's warning machinery prefixes it again, so every CI log
  for several releases read
  `BucklingCheckWarning: BucklingCheckWarning: member 1: ...`. The doubled text
  was visible in every log and read as normal — which is what 68 unasserted
  warnings per run do. Fixed in `retrofit/actions.py`, `criticality/engine.py`,
  `thermal/fire_curve.py` and `stability.py`.
- **One diagnostic had two categories depending on which module emitted it.**
  `limitstates` raised the `EULER_ONLY` capacity warning as a bare `UserWarning`
  while `reliability` raised the identical message as `BucklingCheckWarning`, so
  a caller filtering on one silently missed the other — and a bare `UserWarning`
  cannot be filtered precisely at all, since every third-party library uses it.
  New `LegacyBucklingModelWarning`, deriving from `BucklingCheckWarning` so
  existing filters and `pytest.warns` entries keep working.
- **`steel_temperature` crashed on a scalar-only fire curve.** Its RK4 stages
  called the curve with scalars but its output history called it with an array
  unconditionally, so a `lambda t_min: ...` — the declared `FloatOrArray`
  signature and the natural way to write a bespoke or measured exposure — raised
  `TypeError` at the end of an otherwise successful integration. Reproduced on the
  baseline commit; it was not introduced by this round. Both solvers now go
  through one `_gas_temperature_grid` that tries the vector call and falls back
  point by point, so a performance optimisation cannot break the API again.
- **`max_heating_rate()` understated the peak on a downsampled history.** A
  finite difference across widened output intervals measures less than the true
  rate, and the peak occurs in the first minute where the curve is steepest.
  Measured on a 30-minute ISO 834 exposure at `A_m/V = 200`: 1.0821 degC/s from
  12 output points against a true 1.0966. Now measured on the integration grid
  and carried as `max_heating_rate_full`, which the method prefers.
- **`theta_a0` was the one argument `protected_steel_temperature` never
  validated.** A NaN propagated through every step and an `inf` produced a
  history of NaNs, both inside a normally-returned result object. Now rejected in
  both solvers, identically — their results are documented as interchangeable, so
  their strictness has to be too.
- **The material model clamped silently above 1200 degC.** Every accessor clips
  to Table 3.1's tabulated range, which is right for an interpolator and
  unacceptable for a fire calculation: a thin member in a severe exposure passes
  the ceiling, gets the 1200 degC row, and returns a history that looks ordinary.
  New `SteelTemperatureRangeWarning` from both solvers, plus an `out_of_range`
  field on `SteelHeatingResult` so the fact survives into a payload produced with
  warnings suppressed. Note that ISO 834 cannot trigger it — its gas temperature
  asymptotes near 1193 degC — so the test uses a severe ~1400 degC curve.
- **`normalized_gradient` was a global secant, not the first-order sensitivity
  its docstring claimed.** It fitted a straight line across every probed
  amplitude, a span of twenty-fold by default. Measured on the two-bar toggle,
  whose bifurcation load is cubic in the rise: `-42.3` against a true derivative
  of `-119.9`, a 65 % error in the *unconservative* direction on the one number
  that tells a designer how much to distrust `lambda_cr`. Now the derivative at
  `eps = 0` of a degree-<=3 fit to the amplitudes inside the EN 1993-1-1 Table
  5.1 band, anchored at the exact point `(0, 1)` — measured error 0.0002 %. The
  old slope survives as `secant_gradient`, honestly named.
- **The imperfection verdict was decided by the least realistic amplitude
  probed.** `DEFAULT_IMPERFECTION_AMPLITUDES` topped out at 2 % of the bounding
  box: on a 24 m truss an initial out-of-straightness of 0.48 m, several times
  any code tolerance. Because `imperfection_sensitive` was
  `max(relative_drop) > 5 %`, that single point set the flag. Defaults now span
  the Table 5.1 band `(0.001, L/350, L/200, 0.02)` — still four amplitudes, still
  eight solves — and the verdict is anchored at `verdict_amplitude`, the largest
  probed amplitude inside the band.
- **`_perturbed_nodes` moved the supports.** A solver-returned mode has
  exactly-zero entries on restrained DOFs, so the default path was safe, but a
  caller-supplied `mode=` is arbitrary and any support component was consumed as
  a coordinate change — a support-settlement analysis reported as an
  imperfection study, with a different demand path. Restrained components are
  now masked and the masked magnitude reported once, not once per amplitude per
  sign.
- **The "shallow" screen conflated geometric curvature with structural
  slenderness, and the two modules did not even share a measure.**
  `postprocess.check_shallow_system` computed `ptp(y)/ptp(x)` while
  `stability._rise_span_ratio` computed the orientation-robust extent ratio, so
  the same model could be shallow on one path and not on the other, and a
  *vertical* truss was reported with a ratio above one and never flagged at all.
  Worse, `depth/span < 0.1` was reported as a snap-through risk: a 30 m x 2 m
  Pratt girder has `depth/span = 0.067`, straight chords, and no snap-through
  mode at all — its instability is member buckling, which `limitstates` already
  assesses. Warning about a phenomenon the geometry cannot exhibit, on nearly
  every real model, is how a reader learns to ignore warnings. The screen now
  measures curvature directly (at each node, the pair of incident directions most
  nearly opposite is the candidate chord continuation; if it is within 135 deg of
  straight, its kink `pi - angle` is measured) and claims snap-through only when
  a chord is genuinely bent *and* the depth/span is small. Measured: Pratt and
  Warren girders return exactly 0.0, a toggle returns `2 atan(h/b)` verified
  against the closed form to 1e-9, invariant under a 37 deg rotation. Both
  modules now route through one `shallow_system_screen`.
- **`BucklingResult` violated its own contract when `lambda_cr` was infinite.**
  It returned `modes=[zeros]` with `load_factors=(inf,)` and `multiplicity=1`, so
  `zip(res.modes, res.load_factors)` paired a zero vector with an infinite load
  factor: "here is the buckling mode, it buckles at infinity". The docstring's
  promise that every mode is unit 2-norm was false in exactly that branch, and
  `load_factors` claimed "`inf` entries mean that mode does not buckle" when the
  structure only retains `nu > 1e-14`, making an `inf` entry unreachable in the
  finite case and mandatory in the infinite one — the same token meaning two
  things. Absence is now reported as absence.
- **`content_hash` covered the entries but not the revision.** Re-certifying the
  same envelope against a new edition of EN 1993-1-2 — bumping `updated` and
  `audit_round` while the wording stays identical because the standard's numbers
  did not move — left the hash unchanged, so a consumer pinned to it could not
  tell the two certifications apart. `audit_round` and each entry's verification
  class are now hashed; `updated` and `version` deliberately are not, because a
  date is metadata about the artefact and a version changes every release.
- **The README statistics gate depended on the machine it ran on.** `measure()`
  probed whether `openseespy` imported and passed the three reference-solver
  `--ignore` flags only when it did not, so HEAD advertised 820 tests / 94.2 %
  measured without the extra while earlier commits on the same tree advertised
  838 / 94.19 % measured with it. `make stats-check` was red on one machine and
  green on another for the same commit, and the two `fix(stats)` commits
  immediately preceding this round were attempts to reconcile numbers that were
  never measuring the same suite. The canonical configuration is now
  unconditional — always the same three ignores CI runs — and the with-validation
  count is printed as `informational:` output that nothing compares against a
  file. A number that is not canonical is not a gate.
- **The version was never patched, so one tree gave three answers.** The CLI
  transcript in both READMEs said 2.5.0, the BibTeX records said 2.5.0, and
  `CITATION.cff` said 2.8.0, while the tree built `2.8.1.dev21+gee0e9bfd5`. The
  script synchronised counts and coverage in four places per README and left the
  version alone — the drift it exists to prevent, in the field it did not cover,
  and the field a reader is most likely to copy. `_released_version()` now takes
  the latest release tag (not the working-tree setuptools-scm string, because a
  citation is about the release rather than the commit) and all five occurrences
  are patched and gated together.
- **The hygiene scanner was performing quality control rather than doing it.**
  It declared its forbidden terms as concatenated string fragments
  (`"review" + "er"`, `"Pha" + "se"`) so its own source would not match its own
  rules, and then exempted itself a second time through `SELF_PATHS`. Both
  mechanisms existed for one purpose: to keep the checker from catching itself.
  The vocabulary now lives in `scripts/hygiene_terms.yaml`, so the scanner's
  source contains no forbidden term and **passes its own scan on the merits, with
  no self-exemption** — a claim that is now tested rather than asserted.
- **`.baseline_perf.json` slipped through the artefact filter.** `ARTIFACT_NAME`
  anchors its alternative group at `$`, so every alternative has to reach the end
  of the path; `\.baseline_perf` matched the bare name only. `\.coverage` in the
  same regex already carried the `(\..*)?` suffix — this one had been missed. A
  filter that rejects the name but not the file it names guards nothing.
- **`scan_file` bound its vocabulary as a default argument**, evaluated once when
  the `def` ran, so the module attribute and the parameter were two different
  objects and replacing one left the other scanning against stale rules. The same
  late-binding trap the benchmark driver had, in a hook whose entire job is to
  notice things.
- **`CHANGELOG` listed the C11 deferral twice**, verbatim, in the round-6
  "Deferred with rationale" section. Removed, and the survivor carries the
  round-7 note about `MemberResponse.temperature` having to migrate with the
  split.
- **`docs/theory.md` §1.2 still said "No geometric nonlinearity"** while §9
  documented the geometric stiffness and the linearised bifurcation load.
  Replaced with the distinction that actually matters — *where* the second-order
  term enters — as a table: `solver.solve` is first-order, `lambda_cr` and
  `exact_tangent_stiffness` both use `K_G`, and what remains absent is path
  following.

### Added

- **`benchmarks/reference_problems.py` was rebuilt from nothing into something
  that runs.** The previous version declared ten canonical reference problems and
  was dead code: `run_all_benchmarks(engine, ...)` took an `AnalysisEngine` that
  existed nowhere in `src/` — the name appeared once, in its own docstring — and
  no test imported the module, so a contract with no consumer could not be broken
  by any change to the library. Three reference values were
  `538.0  # typical`, `710.0  # approximate` and `420.0` with no source at all,
  against tolerances of 10-15 %; the real unprotected-steel temperature at 30 min
  and `A_m/V = 200` is 828.17 degC, so the 710 degC figure was wrong by 118 degC
  and the tolerance was wide enough that it would have passed anyway. A fourth
  declared `expected = 1e-4` with `tolerance = 1.0` — ten thousand times the
  quantity it named, which no finite-difference error can exceed. Two more built a
  Koiter cylindrical *shell* and a Williams shallow *arch*, in a pin-jointed truss
  library that can represent neither. This is the defect round 6 identified in
  `tangent_verification.py` ("a verifier that always passes is worse than none")
  reproduced one release later at larger scale.

  Ten problems now run, each stating which library function it exercises, which
  independent oracle produces the reference, and what tolerance connects them in
  the units of the quantity:

  | problem | measured | bound |
  |---|---|---|
  | `euler_column_finite_difference` | 3.4e-11 rel | 1e-9 rel |
  | `toggle_bifurcation_closed_form` | 4.4e-16 rel | 1e-10 rel |
  | `toggle_imperfection_closed_form` | 0.0 abs | 1e-9 abs |
  | `restrained_bar_thermal_force` | 1.8e-16 rel | 1e-12 rel |
  | `unprotected_steel_heating_iso834` | 7.5e-6 degC | 0.05 degC |
  | `protected_steel_heating_code_step` | 0.691 degC | 1.5 degC |
  | `protected_steel_heating_converged` | 0.023 degC | 0.10 degC |
  | `rk4_convergence_order` | 1.2e-8 | 0.02 |
  | `table_3_1_reduction_factors` | 0.0 | 1e-12 |
  | `tangent_stiffness_finite_difference` | 8.0e-9 | 1e-7 |

  The oracles are independent by construction: the exact spectrum of the
  central-difference column operator Richardson-extrapolated (and cross-checked
  against a sparse Lanczos solve), hand-derived closed forms for the toggle and
  its imperfection sweep, `solve_ivp` DOP853 at `rtol=1e-10` on the 4.2.2.2 ODE,
  the 4.2.5.2 recursion reimplemented from the clause text, the shipped Table 3.1
  JSON read and interpolated directly row by row, and two independent central
  differences of `internal_force`.
- **`tests/test_benchmarks.py` (17 tests)** — the file whose absence was the root
  cause. Three mechanisms matter more than the count: a **negative control**
  (a problem perturbed by 0.1 % must be reported failed, must keep its row, and
  must flip the CLI exit status — a suite whose failure path has never been
  exercised is a suite nobody knows can fail); an **anti-vacuity rule** (every
  relative tolerance <= 1e-6, every absolute tolerance <= 1 % of its reference,
  calibrated so the old 10-15 % and 1e4x bounds fail it while the physically
  justified 1.5 degC truncation bound passes at 0.28 %), paired with a required
  observed margin above 1.2x so a tolerance cannot be fitted to its own
  measurement; and **oracle cross-checks**, because an oracle has to be checked
  too or a typo in it silently becomes the reference for the thing it validates.
- **`use_effective_alpha` on `prestress_lengths` and `build_engine`** — the last
  surviving scientific item from round 5. `effective_alpha(T)` was exposed but
  nothing in the default chain consumed it, so a user with `alpha = 1.2e-5`
  running a 600 degC analysis got a restrained force about 17 % low against an
  unchanged capacity: an un-conservative DCR with no notice. Understatement
  measured across the range: 3.9 % at 100 degC, 12.3 % at 400, 17.1 % at 600,
  19.4 % at 700. With the flag set, the imposed strain reproduces `eps_th(T) * L`
  to a relative 1e-14 — not a correction factor tuned to one temperature but the
  standard's own curve through the framework's existing term. `k_axial` is
  bit-identical either way, since stiffness degradation is a separate question.
- **`ConstantAlphaWarning`**, above `ALPHA_CONSTANCY_LIMIT = 150 degC` (the
  temperature at which the strain error first exceeds 5 %, measured rather than
  rounded). The percentage it quotes is computed from the `alpha` values the
  caller actually used, so a model that already carries a secant coefficient is
  told the truth about its own input instead of being warned about somebody
  else's. A member whose `alpha` is exactly zero is never overwritten under
  either setting: zero is an explicit statement that the member does not expand,
  not an unfilled default.
- **`SteelHeatingResult.step_error_estimate`** via a new `estimate_error=` flag on
  both solvers, from one step-halving Richardson pair. This settles an argument
  that was previously open: 4.2.2.2 states an ODE and earns fourth-order RK4,
  while 4.2.5.2 states a *recursion* with a clip and a step ceiling, so the
  protected path reproduces it with explicit Euler because a fire-resistance
  duration quoted against a different integrator than the code's is not
  code-compliant. Both choices are right; what was missing is that they are not
  comparable in accuracy. Measured at each solver's default step:

      unprotected, RK4 at 5 s        4.3e-6 degC
      protected,   Euler at 30 s     0.7123 degC

  and the 0.7123 agrees with the 0.691 the reference suite measures against an
  independently written recursion, so the estimate bounds the real error rather
  than decorating the result.
- **`verification` and `verification_supplement` on every physics-boundary entry,
  plus `verification_summary()`.** A result could previously carry a valid
  `content_hash` for a boundary whose verification had never run in that
  environment: the OpenSeesPy bridge is an optional extra, its tests skip, and the
  digest reported the same bytes either way, so "verified" was indistinguishable
  from "verification not attempted here". The summary reports counts per evidence
  class, names the entries whose evidence could not run, and gives a single
  `complete` flag a consumer can check without knowing which classes depend on an
  extra. Embedded in `digest()`, so it reaches a result payload. Two invariants
  are enforced at load: an unknown evidence class is rejected, and a row claiming
  a modelled status may not declare `declared-only` evidence.
- **`tests/test_hygiene_scanner.py` (65 tests)** — the scanner had none. Writing
  them found the two artefact/late-binding bugs above.
- **`tests/test_stats_gate.py` (14 tests)**, including a drift guard that parses
  `.github/workflows/ci.yml` and asserts its `--ignore` paths are the same three
  the script uses. If CI changes and the script does not follow, the READMEs would
  quote a suite no runner executes — a subtler version of the same defect, and one
  a comment could never catch.
- **A self-checking hygiene vocabulary.** Every rule in `hygiene_terms.yaml`
  carries `matches` and `not_matches`, and `load_vocabulary` refuses a rule whose
  examples contradict its pattern in either direction. That makes the file a
  *checked specification* rather than a comment: a regex that has drifted from
  what its author thought it matched fails at load with the offending string
  quoted. The `not_matches` half is what makes an over-broad rule load-bearing
  rather than cosmetic. The examples live in the data file and never in `tests/`,
  because a test file is scanned by `--all` and inlining a real positive case
  would make the repository fail its own hook — and the fix for that must not be
  to reassemble terms from fragments again.
- **`CONTRIBUTING.md` / `CONTRIBUTING.fa.md`: commit provenance and tool
  assistance.** `Co-authored-by` attribution for agent-assisted commits, commit
  signing, the single-identity expectation, and the practice of recording audit
  findings that did not reproduce. The repository's subject is provenance —
  results carry `solver_metadata`, the physics boundary is hash-pinned, every
  reference value names its oracle — so its own history should be held to the
  same standard.
- **Sparse/dense equivalence pinned at two levels** (matrix and verdict). The
  matrix-level check covers four levels of DOF restriction — none, one, both, and
  alternating, so the global-to-free map is exercised as a non-contiguous
  permutation rather than only as a prefix — plus symmetry and an
  `nnz <= 16 * n_members` bound, since linear-in-members memory is the entire
  reason the sparse path exists. The verdict-level check sweeps
  `lambda_min / (n eps ||A||_F)` across `[1e-3, 1e3]` using a prescribed spectrum,
  because a physical model never reaches that band: on the fan used elsewhere the
  ratio jumps from 2.9e10 straight to negative. The asserted property is the
  *safety* direction — sparse must never accept what dense rejects — and the test
  also asserts the sweep straddles the boundary, so it cannot pass vacuously.

### Changed

- **`test_convergence_order_is_four` now tests the convergence order.** It read
  `_, errors = compute_convergence_order(...)`, discarding the estimate its own
  name advertised, and asserted only that the errors did not increase. Finding out
  *why* it could not simply assert the discarded value is the useful part: the
  reference was too weak, not the method. `_reference_solution` uses `solve_ivp`
  at `rtol=1e-12`, but the EN specific-heat table has a kink near 735 degC, so the
  adaptive integrator saturates around 4e-5 degC while the RK4 sequence resolves
  to 5e-6 degC — measured against that reference the apparent order is 2.4 and
  then collapses. Against a Richardson limit of the library's own sequence, whose
  successive pairs agree to ~7e-5 degC, the order is **3.999999988**. The library's
  RK4 was always fourth order; the test that claimed to check this never did, and
  the reference it relied on was weaker than the thing it measured. The monotonic
  decrease property is kept as its own separately named test.
- **`filterwarnings = ["error"]` in `pyproject.toml`.** This library's diagnostics
  are the product, not incidental logging: a missing `I_sec`, a penalty too small
  to approximate a constraint, a section factor below the lumped-capacitance
  limit, a steel temperature past Table 3.1's ceiling. Each has a dedicated test
  asserting it fires, so a warning leaking unasserted into the summary is either a
  diagnostic nobody pinned or one firing on a model that should not have triggered
  it — and both are defects. Twelve modules that trigger a diagnostic deliberately
  carry a module-level `pytestmark` naming the exact category, with a comment
  saying which test asserts it and why the rest of the module may not. Nothing is
  filtered by bare `UserWarning`, nothing by `ignore` alone, and no filter is wider
  than the categories that module actually produces.
- **`ShallowScreen` is returned on `BucklingResult.shallow_screen`**, so the
  verdict is auditable rather than inferred from the absence of a warning. The
  depth/span fact is still reported for a slender girder; what changed is that it
  no longer licenses a claim about snap-through.
- **`ImperfectionStudy` gained five fields** (`secant_gradient`,
  `gradient_amplitudes`, `gradient_in_code_band`, `verdict_amplitude`,
  `relative_drop_at_verdict`) so the derivative carries its provenance instead of
  being an unverifiable scalar.
- **`ParametricFire`'s out-of-range `q_td` notice has its own category.** It was a
  bare `UserWarning` whose text described itself as
  "LumpedCapacityWarning-adjacent" — naming a class about the lumped-capacitance
  *member* model, for a finding about the *gas curve*, so a caller filtering on
  category could not catch it and a caller reading it was pointed at the wrong
  physics. Now `ParametricFireRangeWarning`.
- **The redundant per-step work in the protected-heating recursion is hoisted.**
  `capacity_ratio_mu` re-derived `specific_heat(theta_a)` (already computed one
  line earlier) and `unit_mass()` (already hoisted above the loop) on every step —
  three table lookups where one sufficed, 480 redundant lookups for a two-hour
  exposure at 30 s. Inlined, with the public function kept and pinned equal to the
  inline expression across 20-1200 degC, because two expressions of one clause is
  exactly how they drift. The gas-temperature grid is evaluated once rather than
  twice per step. `theta_final` is unchanged to 1e-12.
- **The `referee_term` hygiene rule was narrowed.** It matched the bare plural and
  singular of a common English noun, applied to `CONTRIBUTING.md` — a document
  that has to talk about people reviewing pull requests. The repository's earlier
  response to the rule firing was commit `f3265a4`, which rewrote prose around the
  word instead of asking whether the rule was right. Erasing a term is not the same
  as removing what the term stood in for. It now matches the *phrasing* of a
  private assessment workflow — a numbered assessor, the compound and report
  forms, and the reply-to-assessors heading — rather than the ordinary noun, which
  is what the rule was always for. Both halves are pinned: five phrasings that must
  fire and four ordinary sentences that must not. The examples live in
  `scripts/hygiene_terms.yaml` and are checked by `load_vocabulary`, because this
  file is scanned too and quoting the phrasings verbatim here would trip the very
  rule being described.

### Behaviour changes ⚠

- `BucklingResult.modes[0]` raises `IndexError` when `lambda_cr` is infinite,
  where it previously returned a zero vector. Deliberate: the old behaviour
  converted "no bifurcation" into a plausible-looking mode shape, and a loud
  `IndexError` is the right failure mode for a caller that assumed a mode exists.
  `mode` remains a fixed-shape zero vector and `multiplicity` is `0`.
- `ImperfectionStudy.normalized_gradient` changes value for any model whose
  response is curved over the probed range. That is the fix, not a regression.
  `DEFAULT_IMPERFECTION_AMPLITUDES` changes; `imperfection_sensitive` can change
  verdict where it was previously set by an un-code-like amplitude.
- `ShallowSystemWarning` no longer fires on parallel-chord girders, and its
  message now names the chord kink and the node carrying it. `check_shallow_system`
  returns the orientation-robust ratio, so a vertical truss is measured the same
  way as a horizontal one; it also accepts an optional `elements=` argument,
  without which the screen stays conservative and assumes arch-like.
- `run_all_benchmarks(engine, verbose)` is now
  `run_all_benchmarks(verbose, problems)`; `ReferenceProblem.setup()`,
  `get_reference_value()`, `extract_computed_value(result)` and the `result: dict`
  plumbing are replaced by `computed_value()` / `reference_value()`. Nothing in the
  repository consumed the old API — that was the defect.
- `physics_boundary().content_hash()` changes, having absorbed `audit_round` and
  the verification classes. Any pinned expectation must be updated; the hash is a
  compatibility assertion, so it *should* move.
- `boundary_digest()` gains a `verification` key. Consumers comparing digests for
  exact equality will see the change.
- `SteelHeatingResult` gains three optional fields, all defaulting to `None`, so
  hand-constructed instances keep working.
- The `EULER_ONLY` capacity warning is now `LegacyBucklingModelWarning` rather
  than `UserWarning` / `BucklingCheckWarning`. It subclasses
  `BucklingCheckWarning`, so filters on that class keep working; filters written
  against bare `UserWarning` for this specific message will not match by name.

### Not reproduced (recorded with the measurement)

- **"`thickness_ratio_from_section` returns a raw scipy `ValueError` at exactly
  `kappa = 1/12`."** It does not. `brentq` tolerates `f(a) == 0` and returns `a`,
  so the solid limit resolves to `r = 2.000000001`:

      kappa = 1/12 exactly    ->  r = 2.000000001000
      kappa = kappa_min       ->  r = 2.000000001000
      kappa_min - 1 ulp       ->  ValueError (the library's own message)
      kappa_min + 1 ulp       ->  r = 2.000000012167

  Recorded rather than silently "fixed", because the underlying concern is still
  worth acting on: the `<` guard is exact in the comparison it makes, but
  `kappa_min` is itself a floating-point evaluation `_R_EPS` inside the boundary,
  so a section supplied at exactly the solid limit lands on either side of it by a
  rounding — and which side depends on the platform. The answer was resting on
  brentq's *undocumented* tolerance of a zero endpoint. The bracket is now checked
  explicitly and the solid limit returned, pinned by a 200-point sweep across four
  decades of `I/A^2` asserting no scipy bracket message can escape.
- **"`expected = 1e-4` with `tolerance = 1.0` cannot fail."** Correct, and the
  problem is deleted — but replacing it needed more than a tighter number. A
  *forward* difference of `internal_force` at `eps = 1e-6` measures 2.8e-5 against
  the analytic tangent, four orders worse than the central difference's 8.0e-9, so
  a naive rewrite would have been testing the difference scheme's error rather than
  the operator's. The shipped problem uses two independent *central* differences
  and bounds the worse of them at 1e-7.
- **"The RK4 scheme does not achieve fourth order on the fire problem."** It does;
  the reference was the weak link. See `Changed` above for the measurement.

### Deferred with rationale (recorded, not dropped)

- **An experimental anchor (Cardington or equivalent).** Both round-7 critiques
  name this as the largest remaining validation gap and they are right: every
  oracle in the suite is analytical or numerical, so the whole chain is verified
  against independent *computation* rather than against a measured fire. It is
  deferred because doing it properly requires test data with citable provenance —
  a report number, a specimen description, a measurement uncertainty — and
  transcribing numbers from memory is precisely the failure this round removed
  from `benchmarks/`. `538.0  # typical` is what a fabricated experimental anchor
  looks like. The `empirical_validation` boundary entry stays `not-supported` and
  now declares `verification: declared-only`, so the absence is machine-readable
  rather than a sentence in a document.
- **`physics_assumptions` on every result type.** Round 7 asked that
  `AnalysisResult`, `BucklingResult` and `SteelHeatingResult` each carry an
  explicit list of the assumptions they were computed under. Half of it is done:
  `BucklingResult.shallow_screen` and `SteelHeatingResult.out_of_range` /
  `step_error_estimate` are exactly such fields, and
  `boundary_digest()["verification"]` now states whether the evidence ran. What
  remains is a uniform `physics_assumptions` shape across all three result types,
  which is a public-API change best made once rather than incrementally.
- **Rendering `docs/theory.md` §12 from the YAML.** The current pin compares the
  prose table against the artefact through a normalisation step, which works but
  is fragile: any edit to §12 that changes its structure breaks the pin without
  the content having moved. Inverting the dependency — generate the section from
  the YAML, compare the generation against the file — leaves one source of truth
  and makes the prose an artefact rather than an input. Deferred as a standalone
  change because it touches the documentation build, not the library.
- **Newton-Raphson load stepping and arc-length continuation.** `K_G`,
  `exact_tangent_stiffness` and `verify_linearization_convergence` are the
  prerequisites and all three now exist and are measured, so the infrastructure is
  in place — but path following is a multi-week commitment with its own validation
  burden, and shipping a half-tested nonlinear branch into a library whose value
  is that its claims are checked would cost more than the linearised scope costs
  now. The boundary declares `post_buckling` and `geometric_nonlinearity` as
  `not-supported` and says so in every result payload.
- **Chunking / sparse rank-1 for `ci_sweep`, and parallel Monte Carlo.** Correctly
  identified as the remaining bottlenecks, and correctly ranked below everything
  above: they are performance, not correctness, and the round-7 measurements show
  the current paths are accurate to 1e-11 or better where it matters.
- **Member (distributed and point) loads.** Round 1's first recommendation and
  still open. It requires equivalent nodal forces in the load-vector assembly and
  a decision about whether a member load splits the member, which is an API
  question rather than an implementation one.

---

## [2.9.0] — 2026-09-16 · the round-6 audit, shipped in the same release

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
- **C10: complex-step verification of the adjoint DDM.** `sensitivity.py`
  cross-checked its analytic derivative only against central differences, which
  cap out near `sqrt(machine epsilon)` because the cancellation in
  `(f(x+h) - f(x-h))/2h` grows as `h` shrinks. Complex-step has no
  cancellation -- `f'(x) = Im[f(x + ih)]/h`, exact as `h -> 0` -- so it can
  drive `h` to `1e-30`. Two preconditions were checked rather than assumed:
  the DDM's `d(|u|_max)/dA_i` is not analytic as written (`|.|` conjugates,
  `max` is non-smooth), but at a *fixed* base-state argmax node the quantity is
  `sqrt(u_x^2 + u_y^2)`, whose analytic continuation uses the principal complex
  root and no conjugation -- exactly the subgradient convention
  `SensitivityResult` already documents. And the chain must be complex-safe:
  `k = EA/L`, `K = sum k b b^T`, the solve and `B^T (k dL_pre)` are all
  polynomial or linear in `A`, while the EN piecewise-linear material tables
  are *not* (`np.interp` rejects complex input), so this covers the ambient
  chain only and extending it to the fire chain needs a complex-safe
  interpolation in the material layer first. The oracle re-derives assembly,
  reduction and solve from scratch, because a check that shares its code path
  with the thing it checks can only catch typos. **Result: the DDM agrees to
  2e-16 .. 8e-15 relative.** Two further tests pin what makes this oracle
  stronger rather than merely different -- insensitivity to `h` over
  `1e-10..1e-40`, and the best of seven central-difference step sizes being at
  least 100x worse.
- **C4: `_KEY_ELEMENT_RCOND` calibrated on a corpus, not on two fixtures.**
  `DamageOperator._check_mechanism` decides whether a member is kinematically
  essential by comparing a `dpocon` reciprocal condition number against `1e-5`
  -- a threshold in a safety-relevant classification, previously pinned by two
  fixtures and a hard-coded expected-members dictionary. That catches a
  regression; it does not tell you whether `1e-5` is the right number or merely
  one that happens to work on those two models. Now calibrated on a **12-model
  / 193-member corpus** against an *independent* ground truth (delete the
  member outright, `rank(K_ff)` by SVD -- no `dpocon`, no Cholesky, no
  threshold), spanning determinate and redundant frames, Pratt and Warren
  trusses at several panel counts, X-braced variants, a shallow fan and a
  cantilever, with and without fabrication prestrain. Measured: essential
  members reach `rcond <= 7.3e-07`, merely-important ones stay at
  `rcond >= 1.9e-04` -- a **2.41-decade gap** with the constant near the middle
  (1.1 decades below the essential maximum, 1.3 above the redundant minimum) --
  and `dpocon` reproduces the SVD verdict on all 193. The threshold is required
  to sit inside the gap with a factor of 10 clearance on *both* sides, and the
  measured distributions are recorded in the constant's own docstring so the
  number is traceable to evidence. Two fixture bugs surfaced while building the
  corpus, either of which would have quietly gutted the calibration: the plain
  Pratt generator produces a *determinate* truss (`m = 4n+1 = n_free`), so
  every member is essential and it contributes no data for the "merely
  important" class at all (X-braced variants took that class from 3 members to
  96); and the "cantilever" fixture had 5 members against 6 free DOFs, i.e. it
  was a mechanism, whose every member is trivially essential -- data that looks
  meaningful and is not.
- **C13/C14: a machine-readable physics boundary, propagated into results.**
  `docs/theory.md` §12 stated the validity envelope in prose, which answers the
  question for a reader but not for a program. New `data/physics_boundary.yaml`
  (22 entries, closed 5-value status vocabulary, stable ids, explicit `limits`
  on every `supported-with-limits` row) plus `truss_analysis.physics_boundary`
  to load and *validate* it -- a boundary file that is internally inconsistent
  is worse than none, since a consumer calling `covers()` would be relying on
  it. Every analysis result now carries `solver_metadata["physics_boundary"]`:
  a compact hash-pinned digest (schema, version, content hash, counts) rather
  than the whole table, because embedding prose in a report invites consumers
  to parse prose. The content hash changes when and only when the envelope
  changes, which is what makes it a compatibility assertion instead of
  decoration. `docs/api` page added; package-data extended so the YAML ships.
  Tests pin the artefact against §12 in both directions, so the documentation
  cannot drift from what ships -- and the first run of that test found exactly
  such a drift: five entries existed only in the YAML, three from punctuation
  differences ("Thermal and fabrication" vs "Thermal / fabrication") and two
  that were genuinely absent from the prose table.

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

- **C9 -- mandatory triple ranking output.** `ProbabilisticRanking` already
  carries two orderings (mean-CI probabilistic and deterministic-at-mean-inputs)
  plus the sample standard deviations a third would be built from; making a
  third *ordering* unconditional changes the shape of a public result type, and
  which third ordering is the right one (upper confidence bound on CI?
  probability of being most critical?) is a modelling decision that should not
  be guessed at the end of a round.
- **C11 -- separate `MaterialState` from `MemberResponse`.** The two are
  genuinely conflated (`MemberResponse` carries both the response quantities
  `axial_force`/`E` and the section/material state `A`/`I_sec`/`yield_stress`/
  `temperature`). Splitting them is the right call but touches every
  constructor site in the reliability chain, and doing it at the end of a round
  with no budget for a full re-verification is how a clean refactor becomes a
  silent behaviour change in a safety-critical path. Deferred to a round of its
  own. Note added in round 7: `MemberResponse.temperature`, introduced by the
  round-5 chi fix, now carries fire/ambient regime selection and has to migrate
  with this split -- which is one more reason it needs a round of its own rather
  than a slot at the end of one.

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
  600 degC, secant slope `1.448e-5`).
  > **Label corrected in 2.10.0.** The measurement above is right and the
  > label is wrong: `20.7 %` is `1.448e-5 / 1.2e-5 - 1`, the overstatement of
  > the *coefficient needed to fix the strain*. The understatement of the
  > *elongation itself* -- the quantity this sentence names, and the one a
  > restrained member's force and therefore its DCR is wrong by -- is
  > `1 - 1.2e-5 / 1.448e-5` = **17.1 %**. Both ratios are now pinned to four
  > significant figures in `tests/test_material_golden.py` and explained in
  > the `effective_alpha` docstring, so the pair cannot be conflated again.
  > The entry is left as written rather than silently edited: a changelog
  > that rewrites its own history cannot be audited against it.
  `effective_alpha(theta, theta_0)`
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
