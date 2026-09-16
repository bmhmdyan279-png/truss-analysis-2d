"""The reference-problem suite must be able to fail.

The round-7 audit found ``benchmarks/reference_problems.py`` declaring ten
reference problems that nothing ran: the driver took an ``AnalysisEngine``
argument for a class that did not exist anywhere in ``src/``, three reference
values were ``538.0  # typical``, ``710.0  # approximate`` and ``420.0`` with no
source at all against tolerances of 10-15%, and a fourth declared
``expected = 1e-4`` with ``tolerance = 1.0`` -- a bound ten thousand times wider
than the quantity it named, which no measurement could exceed.  There was no
``tests/test_benchmarks.py``, so none of it could be caught.

These tests exist to make that class of defect unrepeatable.  The important one
is :func:`test_a_broken_problem_actually_fails`: a benchmark suite whose failure
path has never been exercised is a suite nobody knows can fail.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from benchmarks.reference_problems import (
    ALL_BENCHMARKS,
    BenchmarkResult,
    EulerColumnFiniteDifference,
    ReferenceProblem,
    ToleranceKind,
    main,
    results_to_dict,
    run_all_benchmarks,
    summary_report,
)

#: Expected problem names.  A registry that silently loses a row is
#: indistinguishable from a row nobody wrote, so the membership is pinned.
EXPECTED_NAMES = frozenset(
    {
        "euler_column_finite_difference",
        "toggle_bifurcation_closed_form",
        "toggle_imperfection_closed_form",
        "restrained_bar_thermal_force",
        "unprotected_steel_heating_iso834",
        "protected_steel_heating_code_step",
        "protected_steel_heating_converged",
        "rk4_convergence_order",
        "table_3_1_reduction_factors",
        "tangent_stiffness_finite_difference",
        "criticality_rank1_vs_direct_resolve",
    }
)


# --------------------------------------------------------------------------
# registry shape
# --------------------------------------------------------------------------


def test_registry_is_exactly_the_expected_problem_set() -> None:
    assert {p.name for p in ALL_BENCHMARKS} == EXPECTED_NAMES
    assert len(ALL_BENCHMARKS) == len(EXPECTED_NAMES), "duplicate names"


def test_every_problem_declares_a_real_oracle_and_tolerance() -> None:
    """No reference value without provenance, and no tolerance without a kind."""
    for problem in ALL_BENCHMARKS:
        assert isinstance(problem, ReferenceProblem)
        assert problem.name
        assert len(problem.description) > 20, problem.name
        # provenance, not a comment: a citation names a source or a method
        oracle = problem.oracle.lower()
        assert len(problem.oracle) > 40, problem.name
        assert not any(
            hedge in oracle for hedge in ("typical", "approximate", "roughly")
        ), f"{problem.name}: oracle text hedges instead of citing"
        assert problem.tolerance > 0.0, problem.name
        assert math.isfinite(problem.tolerance), problem.name
        assert problem.tolerance_kind in ("absolute", "relative"), problem.name
        assert problem.units, problem.name


def test_no_tolerance_is_wider_than_the_quantity_it_bounds(
    results: list[BenchmarkResult],
) -> None:
    """The structural guard against ``expected=1e-4, tolerance=1.0``.

    A tolerance is only meaningful relative to the scale of what it bounds.
    The reference magnitudes come from the measured results rather than from a
    second call into the oracles, which would double this file's runtime for no
    extra evidence.
    """
    by_name = {r.problem_name: r for r in results}
    for problem in ALL_BENCHMARKS:
        reference = abs(by_name[problem.name].reference_value)
        tol = problem.tolerance
        if problem.tolerance_kind == "relative":
            # a relative bound must be a genuine fraction, not "anything"
            assert tol <= 1e-6, (
                f"{problem.name}: relative tolerance {tol:g} is not a check"
            )
        else:
            # An absolute bound is only meaningful against the scale of the
            # quantity, except where zero *is* the reference -- a deviation
            # check, where the bound is already relative by construction.
            #
            # 1% is the rule, and it is calibrated against the defects it
            # replaces rather than chosen by taste: the previous suite bounded
            # temperatures at 10-15% of the reference and bounded an
            # ``expected = 1e-4`` finite-difference error with ``tolerance =
            # 1.0``, i.e. 1e4 times the quantity it named.  Both fail a 1%
            # rule.  A genuinely physical bound survives it: the protected
            # heating check allows 1.5 degC against a 538 degC reference
            # (0.28%), because that 1.5 degC is the first-order truncation
            # error EN 1993-1-2's own 30 s step ceiling produces, not a
            # numerical accuracy claim.
            #
            # This rule alone is not sufficient -- a tolerance can satisfy it
            # and still be fitted to the measurement -- so
            # test_every_reference_problem_passes additionally requires every
            # observed margin to exceed 1.2x.
            if reference > 0.0:
                assert tol <= 1e-2 * reference, (
                    f"{problem.name}: absolute tolerance {tol:g} against a "
                    f"reference of {reference:g} is {tol / reference:.1%} of "
                    "the quantity -- wide enough that it cannot fail"
                )


def test_no_problem_depends_on_a_missing_engine() -> None:
    """The contract that made the previous suite dead code."""
    import inspect

    signature = inspect.signature(run_all_benchmarks)
    assert "engine" not in signature.parameters
    source = inspect.getsource(ReferenceProblem)
    assert "engine" not in source.lower()
    for problem in ALL_BENCHMARKS:
        # computed_value must call the library, and must not take arguments it
        # cannot be given
        assert inspect.signature(problem.computed_value).parameters == {}


# --------------------------------------------------------------------------
# the suite runs, and every problem passes
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results() -> list[BenchmarkResult]:
    """Run the whole suite once and share the measurements.

    The protected-heating oracles integrate the EN 1993-1-2 4.2.5.2 recursion
    for 60,000 sequential steps, so the suite costs about 15 s of wall clock.
    Four separate tests need the same measurements; running the oracles four
    times to get them would make this file the slowest in the repository for no
    additional evidence.
    """
    return run_all_benchmarks(verbose=False)


def test_every_reference_problem_passes(results: list[BenchmarkResult]) -> None:
    """The whole point: run them all, and require a real margin on each.

    A margin of 1.0 would mean the tolerance was tuned to the answer that was
    measured; requiring some headroom keeps the bound honest without making it
    so loose that it stops being a check.
    """
    assert len(results) == len(ALL_BENCHMARKS)

    failures = []
    for r in results:
        if r.passed:
            continue
        error = r.absolute_error if r.tolerance_kind == "absolute" else r.relative_error
        note = f" ({r.notes})" if r.notes else ""
        failures.append(
            f"{r.problem_name}: {r.tolerance_kind} error {error:.6g} "
            f"vs bound {r.tolerance:g}{note}"
        )
    assert not failures, "\n".join(failures)

    for result in results:
        assert math.isfinite(result.computed_value), result.problem_name
        assert math.isfinite(result.reference_value), result.problem_name
        # every check must have at least 20% headroom, or the tolerance was
        # reverse-engineered from the measurement
        assert result.margin > 1.2, (
            f"{result.problem_name}: margin {result.margin:.4g}x means the "
            "tolerance is fitted to the answer"
        )


def test_results_are_serialisable_and_complete(results: list[BenchmarkResult]) -> None:
    payload = results_to_dict(results)
    assert len(payload) == len(ALL_BENCHMARKS)
    # must survive a JSON round trip: this is what a CI artefact would carry
    restored = json.loads(json.dumps(payload))
    assert {row["problem"] for row in restored} == EXPECTED_NAMES
    for row in restored:
        assert isinstance(row["passed"], bool)
        assert row["tolerance_kind"] in ("absolute", "relative")
        assert row["oracle"]


def test_summary_report_counts_match_the_results(
    results: list[BenchmarkResult],
) -> None:
    report = summary_report(results)
    passed = sum(1 for r in results if r.passed)
    assert f"total: {len(results)}   passed: {passed}" in report
    assert "TIGHTEST PASSING CHECK" in report


def test_cli_list_and_exit_status(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list"]) == 0
    listed = capsys.readouterr().out.split()
    assert set(listed) == EXPECTED_NAMES


# --------------------------------------------------------------------------
# negative control: the harness must be able to fail
# --------------------------------------------------------------------------


class _BrokenProblem(ReferenceProblem):
    """A problem whose library answer is deliberately wrong by a factor."""

    factor = 1.001

    @property
    def name(self) -> str:
        return "deliberately_broken_control"

    @property
    def description(self) -> str:
        return "Negative control: the harness must report a failure as a failure"

    @property
    def oracle(self) -> str:
        return "The Euler column's own reference value, perturbed on purpose"

    @property
    def units(self) -> str:
        return "N"

    @property
    def tolerance(self) -> float:
        return 1e-9

    @property
    def tolerance_kind(self) -> ToleranceKind:
        return "relative"

    def reference_value(self) -> float:
        return EulerColumnFiniteDifference().reference_value()

    def computed_value(self) -> float:
        return self.reference_value() * self.factor


def test_a_broken_problem_actually_fails() -> None:
    """If this passes, the benchmark suite is decoration.

    A 0.1% perturbation of a quantity bounded at 1e-9 relative must be reported
    as a failure, must keep its row in the results, and must flip the CLI exit
    status.  Nothing in the previous version of this module could fail, which
    is why nothing noticed that it was not running.
    """
    broken = _BrokenProblem()
    result = broken.validate()
    assert isinstance(result, BenchmarkResult)
    assert not result.passed
    assert result.relative_error == pytest.approx(1e-3, rel=1e-3)
    assert result.margin < 1.0
    assert "FAILED" in summary_report([result])


def test_a_problem_that_raises_is_reported_not_dropped() -> None:
    """An exception must produce a failed row, never a missing one."""

    class _Raising(_BrokenProblem):
        @property
        def name(self) -> str:
            return "deliberately_raising_control"

        def computed_value(self) -> float:
            msg = "deliberate failure to prove the harness reports it"
            raise RuntimeError(msg)

    result = _Raising().validate()
    assert not result.passed
    assert result.absolute_error == float("inf")
    assert "RuntimeError" in result.notes
    assert math.isnan(result.computed_value)


def test_cli_exit_status_is_nonzero_when_a_problem_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.reference_problems as mod

    monkeypatch.setattr(mod, "ALL_BENCHMARKS", (_BrokenProblem(),))
    assert mod.main(["--quiet"]) == 1
    monkeypatch.setattr(mod, "ALL_BENCHMARKS", (EulerColumnFiniteDifference(),))
    assert mod.main(["--quiet"]) == 0


# --------------------------------------------------------------------------
# oracle cross-checks: the oracles themselves are checked, not trusted
# --------------------------------------------------------------------------


def test_euler_closed_form_spectrum_matches_a_sparse_lanczos_solve() -> None:
    """The oracle uses the discrete operator's analytic spectrum; verify it.

    ``lambda_k = (4 EI / h^2) sin^2(k pi h / 2L)`` is asserted against an
    actual sparse eigensolve of ``(EI/h^2) tridiag(-1, 2, -1)``.  Without this,
    a typo in the closed form would silently become the reference for the
    library's own buckling function -- an oracle has to be checked too.
    """
    problem = EulerColumnFiniteDifference()
    length, i_sec = problem.length, problem.i_sec
    e_mod = 210e9
    for n in (50, 200, 800):
        h = length / (n + 1)
        scale = e_mod * i_sec / h**2
        operator = sp.diags(
            [
                -np.ones(n - 1) * scale,
                np.full(n, 2.0 * scale),
                -np.ones(n - 1) * scale,
            ],
            [-1, 0, 1],
            format="csr",
        )
        numerical = float(
            eigsh(operator, k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0]
        )
        closed = problem._fd_eigenvalue(n)
        assert numerical == pytest.approx(closed, rel=1e-11), f"n={n}"


def test_euler_richardson_extrapolation_converges_to_the_textbook_value() -> None:
    """The extrapolated oracle must land on pi^2 E I / L^2 to 1e-9."""
    problem = EulerColumnFiniteDifference()
    reference = problem.reference_value()
    textbook = math.pi**2 * 210e9 * problem.i_sec / problem.length**2
    assert reference == pytest.approx(textbook, rel=1e-9)


def test_toggle_oracle_is_derived_not_borrowed() -> None:
    """The closed form is checked against a plain eigenvalue solve.

    ``_toggle_lambda_cr`` is hand-derived in the benchmark module.  Confirming
    it against a dense generalised eigensolve of the assembled dyads -- a
    route that shares nothing with the derivation -- is what makes it an oracle
    rather than a restatement.
    """
    from scipy.linalg import eigh

    from benchmarks.reference_problems import _toggle, _toggle_lambda_cr
    from truss_analysis.assembly import assemble_global_matrices
    from truss_analysis.stability import geometric_stiffness

    h, b, area = 0.08, 1.0, 2e-3
    nodes, elements = _toggle(h, b, area)
    p_cr = _toggle_lambda_cr(h, b, 210e9, area)

    from truss_analysis.model import fixed_dof_indices

    k_e, _, _, _ = assemble_global_matrices(nodes, elements)
    k_e = np.asarray(k_e)
    restrained = set(fixed_dof_indices(nodes))
    free = [d for d in range(2 * len(nodes)) if d not in restrained]
    # unit load at the apex, downwards: the pattern lambda amplifies
    load = np.zeros(2 * len(nodes))
    load[5] = -1.0
    n_unit = _toggle_member_forces(nodes, elements, load, free, k_e)
    k_g = np.asarray(geometric_stiffness(nodes, elements, n_unit))

    k_ff = k_e[np.ix_(free, free)]
    g_ff = -k_g[np.ix_(free, free)]
    # ``k_ff`` is symmetric positive definite and ``g_ff`` symmetric, so this
    # is a symmetric-definite pencil and ``eigh`` returns real eigenvalues.
    # (``eig`` would return them as complex conjugate pairs with zero imaginary
    # part and a ComplexWarning on the cast -- a reminder that the *structure*
    # of the problem is part of what makes the oracle trustworthy.)
    nu = eigh(g_ff, k_ff, eigvals_only=True)
    positive = np.asarray(nu, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    assert positive.size, "the toggle must have a positive bifurcation eigenvalue"
    lambda_cr = 1.0 / float(np.max(positive))
    assert lambda_cr == pytest.approx(p_cr, rel=1e-10)


def _toggle_member_forces(nodes, elements, load, free, k_e):
    """Solve for the unit-load member forces, without the library's engine."""
    u = np.zeros(len(load))
    u[free] = np.linalg.solve(k_e[np.ix_(free, free)], load[free])
    forces = {}
    for e in elements:
        i = next(k for k, n in enumerate(nodes) if n.id == e.node_i)
        j = next(k for k, n in enumerate(nodes) if n.id == e.node_j)
        d = np.array([nodes[j].x - nodes[i].x, nodes[j].y - nodes[i].y], dtype=float)
        length = float(np.linalg.norm(d))
        c, s = d / length
        b_vec = np.zeros(2 * len(nodes))
        b_vec[[2 * i, 2 * i + 1, 2 * j, 2 * j + 1]] = (-c, -s, c, s)
        forces[e.id] = float(e.E * e.A / length * (b_vec @ u))
    return forces


def test_table_oracle_reads_the_shipped_artefact() -> None:
    """The JSON the accessor memoises must be the JSON the oracle reads."""
    from benchmarks.reference_problems import _TABLE_PATH

    assert _TABLE_PATH.exists(), "the table artefact must ship in the tree"
    raw = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
    table = raw["table"]
    assert table["id"] == "3.1"
    assert raw["clause"] == "3.2.1(3)"
    # provenance is recorded in the artefact itself; a benchmark that cites it
    # must be citing something that is actually pinned
    assert len(raw["standard"]["primary_source_sha256"]) == 64
    temps = np.asarray(table["temperatures_c"], dtype=float)
    k_y = np.asarray(table["k_y"], dtype=float)
    assert float(k_y[np.where(temps == 600.0)[0][0]]) == pytest.approx(0.47)


def test_rk4_order_is_measured_against_a_self_consistent_limit() -> None:
    """The reference quality problem that hid the order-4 behaviour.

    ``solve_ivp`` at ``rtol=1e-12`` saturates around 4e-5 degC on this ODE
    because the EN specific-heat table has a kink near 735 degC, so it is a
    *weaker* reference than the library's own step-refinement sequence -- which
    is why an order-4 claim could not be demonstrated against it.  Two
    independent Richardson limits from successive step pairs must therefore
    agree far more tightly than the errors being measured.
    """
    from benchmarks.reference_problems import Rk4ConvergenceOrder

    problem = Rk4ConvergenceOrder()
    f_10 = problem._final(10.0)
    f_5 = problem._final(5.0)
    f_2_5 = problem._final(2.5)
    f_1_25 = problem._final(1.25)

    limit_a = f_5 + (f_5 - f_10) / 15.0
    limit_b = f_2_5 + (f_2_5 - f_5) / 15.0
    limit_c = f_1_25 + (f_1_25 - f_2_5) / 15.0

    # successive limits agree to ~7e-5 degC, while the errors being measured
    # span 2.4e-4 down to 5e-6 degC
    assert abs(limit_a - limit_b) < 1e-3
    assert abs(limit_b - limit_c) < 1e-3
    assert abs(f_5 - limit_c) > 10.0 * abs(f_1_25 - limit_c), (
        "the error sequence must actually shrink, or an 'order' is meaningless"
    )
    assert problem.computed_value() == pytest.approx(4.0, abs=0.05)


def test_oracle_constants_match_the_standard_not_the_library() -> None:
    """The hardcoded oracle constants are the standard's, and the library agrees.

    ``_unprotected_reference`` and ``_protected_reference`` use a literal
    7850 kg/m^3 rather than calling ``unit_mass()``, because an oracle that asks
    the library what steel is cannot detect the library being wrong about it.
    That only works if the literal is *checked* against the standard's value and
    against the library, so a future change to either shows up here rather than
    as a mysterious benchmark disagreement.
    """
    from benchmarks.reference_problems import RHO_STEEL
    from truss_analysis.material.steel_eurocode import unit_mass

    assert pytest.approx(7850.0) == RHO_STEEL  # EN 1993-1-2:2005 clause 3.2.2
    assert float(unit_mass()) == pytest.approx(RHO_STEEL)


def test_the_criticality_oracle_is_not_a_constant() -> None:
    """The new oracle must be sensitive to the perturbation it measures.

    A comparison between two numbers that both ignore ``alpha`` would pass
    whatever the engine did, so the oracle is anchored at the one point where
    its answer is known without computing it: at ``alpha = 1`` nothing is
    perturbed, every member's CI is exactly zero by definition, and the norm is
    therefore exactly zero.  That the same oracle returns 0.191 at
    ``alpha = 0.7`` is what makes the agreement at 0.7 evidence rather than a
    coincidence of two constants.
    """
    from benchmarks.reference_problems import (
        CriticalityRank1AgainstDirectResolve,
        _oracle_direct_resolve_ci,
    )

    problem = CriticalityRank1AgainstDirectResolve()
    at_unity = _oracle_direct_resolve_ci(1.0)
    assert at_unity == pytest.approx(0.0, abs=1e-12)
    assert problem.reference_value() > 0.1, "the oracle must move with alpha"
    # and the library reproduces the moved value, not the trivial one
    assert problem.computed_value() == pytest.approx(
        problem.reference_value(), rel=1e-12
    )


# --------------------------------------------------------------------------
# performance regression guard -- explicitly NOT a verification oracle
# --------------------------------------------------------------------------


def test_criticality_engine_speedup_is_measured_and_agrees() -> None:
    """The engine's reason to exist, measured rather than asserted.

    ``docs/theory.md`` section 8.1 separates verification from everything else,
    and a wall-clock ratio belongs to the everything else: there is no
    independent source of truth for how long this machine takes to do this work.
    So this is a *regression guard*, kept out of ``ALL_BENCHMARKS`` on purpose,
    guarding the claim that one factorisation serving every member beats one
    factorisation per member.

    The anti-vacuity discipline is the agreement check, and it is the half that
    matters.  A path that returns garbage is very fast, so a guard that only
    timed would reward the wrong thing: ``measure_criticality_engine`` refuses
    to call the speedup meaningful unless the rank-1 sweep and the direct
    resolve produced the same criticality indices, and ``passed`` requires both.

    The floor is ``m / 4``, deliberately far below the measured ratio (5.7m at
    29 members, 19.8m at 93).  Wall-clock ratios move with machine load, BLAS
    build and thread count; a floor at the measured value would flake, and a
    flaky gate gets disabled, which is worse than no gate.  ``m / 4`` is the
    point below which the architectural claim would be false rather than
    diluted -- if the engine ever degenerated to per-member factorisation the
    ratio would fall towards 1 and this would fail.
    """
    from benchmarks.performance import measure_criticality_engine

    measurement = measure_criticality_engine(n_panels=12, repeats=3)

    # the agreement half first: without it the timing half means nothing
    assert measurement.agrees, (
        f"the two paths disagree by {measurement.max_ci_deviation:.3e}, so the "
        "speedup is void -- a fast wrong answer is not a result"
    )
    assert measurement.max_ci_deviation < 1e-9
    assert measurement.speedup >= measurement.floor, measurement.report()
    assert measurement.passed
    # and the model is big enough for the claim to be about something
    assert measurement.n_members >= 40
    assert measurement.engine_s > 0.0
    assert measurement.direct_s > measurement.engine_s
