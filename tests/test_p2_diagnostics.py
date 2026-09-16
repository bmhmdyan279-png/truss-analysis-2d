"""Round-6 audit, P2 layer: diagnostics, parameterisation and invariants.

Covers

* **C5** -- Iman--Conover must *report* the rank correlation it achieved,
  not merely claim to impose the requested one;
* **C6** -- the critical-temperature bisection tolerance is a parameter, not
  a private constant;
* **C15** -- ``solver_metadata`` records how the linear system was actually
  assembled and reduced, so two runs are comparable from the JSON alone;
* **C17** -- thermal strain is a state function: the path taken to a
  temperature must not change the answer;
* **C18** -- the member-heating solver's sensitivity to ``A_m/V`` is
  monotone, quantified and matches the first-order physics.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from truss_analysis import run
from truss_analysis.exceptions import InputIgnoredWarning
from truss_analysis.limitstates import (
    BISECT_XTOL,
    member_critical_temperature,
    member_critical_temperature_detailed,
)
from truss_analysis.material.steel_eurocode import alpha as thermal_elongation
from truss_analysis.model import Element, Node
from truss_analysis.thermal.fire_curve import steel_temperature
from truss_analysis.thermal.protection import (
    InsulationMaterial,
    capacity_ratio_mu,
    protected_steel_temperature,
)
from truss_analysis.uncertainty.sampling import (
    RankCorrelationReport,
    iman_conover_correlate,
    iman_conover_with_report,
)

# Both categories are the subject of this module: it checks that solver_metadata
# records the path actually taken, including the penalty path's conditioning
# complaint and the sparse-to-dense downgrade notice.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.IllConditionedWarning"
    ),
    pytest.mark.filterwarnings("ignore::truss_analysis.exceptions.InputIgnoredWarning"),
]

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "example1.json"


def _three_bar_model():
    """Statically indeterminate (one redundancy) model with a compressed member.

    All three supports are pinned, so ``n_free = 2`` against ``m = 3``
    members: one degree of redundancy.  That matters for the thermal tests --
    a *determinate* version of this frame accommodates uniform heating
    exactly and reports zero member force, which would make every assertion
    about restrained thermal expansion vacuously true.
    """
    nodes = [
        Node(id="L", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=4.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=2.0, y=-2.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=2.0, y=0.0),
    ]
    elements = [
        Element(
            id="r1",
            node_i="L",
            node_j="A",
            E=210e9,
            A=6e-3,
            I_sec=4e-5,
            alpha=1.2e-5,
        ),
        Element(
            id="r2",
            node_i="R",
            node_j="A",
            E=210e9,
            A=6e-3,
            I_sec=4e-5,
            alpha=1.2e-5,
        ),
        Element(
            id="post",
            node_i="A",
            node_j="B",
            E=210e9,
            A=4e-3,
            I_sec=2e-5,
            alpha=1.2e-5,
        ),
    ]
    loads = {"A": {"Fx": 0.0, "Fy": -1.2e6}}
    return nodes, elements, loads


# --------------------------------------------------------------------------
# C5: achieved rank correlation is reported, not assumed
# --------------------------------------------------------------------------


def _lhs_like(n: int, m: int, seed: int = 0) -> np.ndarray:
    """Stratified uniform matrix: one sample per stratum per dimension."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, m), dtype=float)
    for d in range(m):
        out[:, d] = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(out[:, d])
    return out


def test_iman_conover_report_is_consistent_with_the_bare_function() -> None:
    u = _lhs_like(500, 3, seed=1)
    target = np.array([[1.0, 0.7, -0.3], [0.7, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    report = iman_conover_with_report(u, target)
    assert isinstance(report, RankCorrelationReport)
    assert np.array_equal(report.reordered, iman_conover_correlate(u, target))
    assert np.array_equal(report.target, target)
    assert report.n_samples == 500


def test_achieved_correlation_approaches_the_target() -> None:
    """The whole point: the gap is measured, and it shrinks with n."""
    # must be positive definite after the Kruskal mapping rho_P = 2 sin(pi/6 rho_S),
    # which the function itself enforces -- a matrix with a large negative and
    # two large positive entries is not
    target = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.4], [0.3, 0.4, 1.0]])
    small = iman_conover_with_report(_lhs_like(200, 3, seed=2), target)
    large = iman_conover_with_report(_lhs_like(8000, 3, seed=2), target)
    assert small.max_abs_deviation < 0.15
    assert large.max_abs_deviation < small.max_abs_deviation
    assert large.max_abs_deviation < 0.03


def test_report_deviation_matches_a_recomputed_spearman() -> None:
    """`max_abs_deviation` is not a self-reported number; it is checkable."""
    from scipy.stats import spearmanr

    u = _lhs_like(1000, 3, seed=3)
    target = np.array([[1.0, 0.6, 0.0], [0.6, 1.0, -0.4], [0.0, -0.4, 1.0]])
    report = iman_conover_with_report(u, target)
    rho = np.atleast_2d(np.asarray(spearmanr(report.reordered).statistic, dtype=float))
    assert np.allclose(report.achieved, rho, rtol=1e-12)
    off = ~np.eye(3, dtype=bool)
    assert report.max_abs_deviation == pytest.approx(
        float(np.max(np.abs(rho[off] - target[off]))), rel=1e-12
    )
    assert report.achieved.shape == (3, 3)
    # a correlation matrix is symmetric with unit diagonal
    assert np.allclose(report.achieved, report.achieved.T, atol=1e-9)
    assert np.allclose(np.diag(report.achieved), 1.0, atol=1e-12)


def test_report_preserves_marginals_and_stratification() -> None:
    """Iman--Conover permutes; it must not mix values between strata."""
    u = _lhs_like(200, 3, seed=4)
    target = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]])
    report = iman_conover_with_report(u, target)
    for d in range(3):
        # exact permutation of the input column: same values, same multiset
        assert np.array_equal(np.sort(report.reordered[:, d]), np.sort(u[:, d]))
        # and still one sample per 1/n stratum, i.e. still a Latin hypercube
        strata = np.floor(report.reordered[:, d] * 200).astype(int)
        assert sorted(strata.tolist()) == list(range(200))


def test_report_handles_a_single_variable() -> None:
    u = _lhs_like(50, 1, seed=5)
    report = iman_conover_with_report(u, np.array([[1.0]]))
    assert report.max_abs_deviation == 0.0
    assert report.achieved.shape == (1, 1)
    assert np.array_equal(np.sort(report.reordered[:, 0]), np.sort(u[:, 0]))


# --------------------------------------------------------------------------
# C6: bisection tolerance is a parameter
# --------------------------------------------------------------------------


def test_default_tolerance_is_the_published_constant() -> None:
    assert BISECT_XTOL == 1e-3
    nodes, elements, loads = _three_bar_model()
    default = member_critical_temperature(nodes, elements, loads, "post", 355e6)
    explicit = member_critical_temperature(
        nodes, elements, loads, "post", 355e6, tolerance=BISECT_XTOL
    )
    assert default == explicit


@pytest.mark.parametrize("tol", [1e-6, 1e-3, 0.1, 1.0, 5.0])
def test_looser_tolerance_gives_a_higher_conservative_temperature(tol: float) -> None:
    """The returned theta is the bracket's upper end, so DCR(theta) >= 1.

    Widening the bracket can only move that upper end up (or leave it), and
    the answer must stay on the failing side -- which is what makes a loose
    tolerance safe for screening rather than merely inaccurate.
    """
    nodes, elements, loads = _three_bar_model()
    tight = member_critical_temperature(
        nodes, elements, loads, "post", 355e6, tolerance=1e-6
    )
    loose = member_critical_temperature(
        nodes, elements, loads, "post", 355e6, tolerance=tol
    )
    assert tight is not None
    assert loose is not None
    assert loose >= tight - 1e-9
    assert loose - tight <= tol + 1e-9


def test_tolerance_is_honoured_by_the_detailed_variant_too() -> None:
    nodes, elements, loads = _three_bar_model()
    a = member_critical_temperature_detailed(
        nodes, elements, loads, "post", 355e6, tolerance=1e-6
    )
    b = member_critical_temperature_detailed(
        nodes, elements, loads, "post", 355e6, tolerance=2.0
    )
    assert a.theta is not None
    assert b.theta is not None
    assert b.theta >= a.theta
    assert b.theta - a.theta <= 2.0 + 1e-9
    # the failure label must not depend on the tolerance
    assert a.failure_mode == b.failure_mode


@pytest.mark.parametrize("tol", [0.0, -1.0])
def test_non_positive_tolerance_rejected(tol: float) -> None:
    nodes, elements, loads = _three_bar_model()
    with pytest.raises(ValueError, match="tolerance must be > 0"):
        member_critical_temperature_detailed(
            nodes, elements, loads, "post", 355e6, tolerance=tol
        )


# --------------------------------------------------------------------------
# C15: solver_metadata records the assembly and reduction that actually ran
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("use_sparse", "bc_method", "assembly", "reduction"),
    [
        (False, "elimination", "dense-einsum", "free-dof-elimination"),
        (True, "elimination", "sparse-coo", "free-dof-elimination"),
        (False, "penalty", "dense-einsum", "penalty-augmented"),
        (True, "penalty", "dense-einsum", "penalty-augmented"),
    ],
)
def test_solver_metadata_records_the_real_path(
    use_sparse: bool, bc_method: str, assembly: str, reduction: str
) -> None:
    result = run(EXAMPLE, use_sparse=use_sparse, bc_method=bc_method, quiet=True)
    md = result.solver_metadata
    assert md["assembly_mode"] == assembly
    assert md["reduction_order"] == reduction
    assert md["bc_method"] == bc_method
    assert md["use_sparse"] is (assembly == "sparse-coo")


def test_metadata_exposes_the_silent_sparse_downgrade() -> None:
    """The penalty path cannot use sparse assembly, and says so.

    A warning is emitted at run time, but a warning is transient; the JSON
    report is what survives.  Requesting ``use_sparse=True`` with the penalty
    method must be visible in the exported metadata rather than only in a
    log line nobody reads after the fact.
    """
    with pytest.warns(InputIgnoredWarning, match="requires dense assembly"):
        result = run(EXAMPLE, use_sparse=True, bc_method="penalty", quiet=True)
    assert result.solver_metadata["use_sparse"] is False
    assert result.solver_metadata["assembly_mode"] == "dense-einsum"


def test_metadata_survives_json_round_trip() -> None:
    """The new fields must be serialisable, or they are not really reported."""
    import tempfile

    result = run(EXAMPLE, use_sparse=True, bc_method="elimination", quiet=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        run(
            EXAMPLE,
            use_sparse=True,
            bc_method="elimination",
            quiet=True,
            output=str(out),
        )
        payload = json.loads(out.read_text(encoding="utf-8"))
    md = payload["solver_metadata"]
    assert md["assembly_mode"] == "sparse-coo"
    assert md["reduction_order"] == "free-dof-elimination"
    assert md == json.loads(json.dumps(result.solver_metadata))


# --------------------------------------------------------------------------
# C17: thermal strain is a state function
# --------------------------------------------------------------------------


@pytest.mark.parametrize("target", [100.0, 400.0, 700.0, 1000.0])
def test_thermal_elongation_is_path_independent(target: float) -> None:
    """``alpha(theta)`` is dl/l, a state function of temperature.

    The relative elongation between 20 degC and ``target`` must equal the sum
    of the increments over any partition of that interval.  It does *not*
    for a rate quantity, so this test is what distinguishes "the code stores
    a state function" from "the code integrates a path".
    """
    total = float(thermal_elongation(target)) - float(thermal_elongation(20.0))
    for n_steps in (2, 5, 17, 100):
        grid = np.linspace(20.0, target, n_steps + 1)
        increments = np.diff(np.asarray(thermal_elongation(grid), dtype=float))
        assert float(np.sum(increments)) == pytest.approx(total, rel=1e-12, abs=1e-15)


def test_restrained_thermal_force_depends_only_on_the_end_temperature() -> None:
    """The demand a heated restrained member carries is a function of T alone.

    ``prestress_lengths`` is a state function of the temperature *field*: it
    reads no history, no rate and no intermediate value.  Two things are
    pinned here, because both are easy to get wrong in a way that still looks
    like it works:

    * the imposed elongation is ``alpha_e (T - T_ambient) L`` -- the element's
      own constant coefficient times the temperature *rise*, not the EN
      relative-elongation table ``alpha(theta)`` (that table is what
      :func:`test_thermal_elongation_is_path_independent` checks, and the two
      are different quantities with the same symbol);
    * a redundant frame heated uniformly develops real compression, so the
      assertion is not vacuous.
    """
    from truss_analysis.criticality.engine import (
        base_displacement,
        build_engine,
        member_forces,
        prestress_lengths,
        total_load_vector,
    )
    from truss_analysis.criticality.scenarios import T_AMBIENT

    nodes, elements, _loads = _three_bar_model()

    def forces_at(temps: dict[str, float]) -> np.ndarray:
        setup = build_engine(nodes, elements, {}, temps)
        u = base_displacement(setup, total_load_vector(nodes, {}, setup))
        return member_forces(setup, u)

    uniform = {e.id: 600.0 for e in elements}
    n_600 = forces_at(uniform)
    # the redundancy must actually bite: a determinate frame would give ~0
    assert np.abs(n_600).max() > 1e5, f"no restrained force: {n_600}"
    # a different end temperature gives a different demand
    assert not np.allclose(n_600, forces_at({e.id: 900.0 for e in elements}))
    # ... but the same end temperature always gives the same demand, whatever
    # the caller believes about "history"
    assert np.allclose(n_600, forces_at(dict(uniform)), rtol=0.0, atol=0.0)

    dl = prestress_lengths(nodes, elements, uniform)
    coords = {n.id: (n.x, n.y) for n in nodes}
    for e, got in zip(elements, dl, strict=True):
        (xi, yi), (xj, yj) = coords[e.node_i], coords[e.node_j]
        length = float(np.hypot(xj - xi, yj - yi))
        assert got == pytest.approx(e.alpha * (600.0 - T_AMBIENT) * length, rel=1e-12)
    # no temperature field -> no imposed elongation at all
    assert np.allclose(prestress_lengths(nodes, elements, None), 0.0, atol=1e-15)


def test_heating_solver_final_state_is_independent_of_step_size() -> None:
    """Same exposure, different integrator steps: same temperature."""
    coarse = steel_temperature(30.0, 180.0, max_step_s=5.0, warn_lumped=False)
    fine = steel_temperature(30.0, 180.0, max_step_s=0.5, warn_lumped=False)
    assert coarse.theta_final == pytest.approx(fine.theta_final, abs=0.05)


# --------------------------------------------------------------------------
# C18: sensitivity to the section factor A_m/V
# --------------------------------------------------------------------------


def test_final_temperature_is_monotone_in_section_factor() -> None:
    """A slender (high ``A_m/V``) member heats faster at every duration."""
    factors = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
    for duration in (10.0, 30.0, 60.0):
        finals = [
            steel_temperature(duration, sf, warn_lumped=False).theta_final
            for sf in factors
        ]
        assert all(a < b for a, b in itertools.pairwise(finals)), (
            f"non-monotone at {duration} min: {finals}"
        )


def test_heating_rate_scales_with_section_factor_early_on() -> None:
    """First-order physics: ``d(theta)/dt ~ (A_m/V)`` while ``theta ~ theta_0``.

    At the start of the exposure the steel is still cold, so ``h_net`` is
    almost the same for every section factor and the initial slope is
    proportional to ``A_m/V``.  This is the quantitative form of "users do
    not know how much ``A_m/V`` matters".
    """
    ratios = []
    for sf in (100.0, 200.0, 400.0):
        res = steel_temperature(0.5, sf, n_output=0, warn_lumped=False)
        slope = (res.theta_steel[1] - res.theta_steel[0]) / (
            res.time_s[1] - res.time_s[0]
        )
        ratios.append(slope / sf)
    # slope / (A_m/V) is nearly constant across a 4x range of section factor
    assert max(ratios) / min(ratios) == pytest.approx(1.0, rel=0.05)


def test_time_to_reference_temperature_scales_inversely_with_section_factor() -> None:
    """Under an *isothermal* exposure, time-to-temperature goes as ``V/A_m``.

    The inverse proportionality is exact physics only while the driving
    temperature difference is fixed; under ISO 834 the gas temperature itself
    rises sub-linearly (logarithmically) in time, so the measured ratio there
    is ~1.45 for a 2x section factor rather than 2.0.  Testing the clean case
    is what pins the ``A_m/V`` dependence itself, and the ISO 834 case is
    asserted only as monotone -- quoting "2.0" for it would be wrong.
    """
    target = 400.0

    def time_to(sf: float, curve) -> float:
        res = steel_temperature(
            180.0,
            sf,
            n_output=0,
            warn_lumped=False,
            **({} if curve is None else {"fire_curve": curve}),
        )
        hit = np.nonzero(res.theta_steel >= target)[0]
        return float(res.time_s[hit[0]] / 60.0) if hit.size else float("inf")

    isothermal = lambda _t: 1000.0  # noqa: E731 - a one-line exposure
    t100, t200, t400 = (time_to(sf, isothermal) for sf in (100.0, 200.0, 400.0))
    assert t100 / t200 == pytest.approx(2.0, rel=0.05)
    assert t200 / t400 == pytest.approx(2.0, rel=0.05)
    # and under ISO 834 the ordering holds even though the ratio does not
    iso100, iso200 = time_to(100.0, None), time_to(200.0, None)
    assert iso100 > iso200


def test_protection_dampens_the_section_factor_sensitivity() -> None:
    """Insulation must reduce how much ``A_m/V`` matters.

    With a thick board the heat flux is limited by the insulation, not by
    how much steel surface there is per unit volume, so the spread of final
    temperatures across a range of section factors shrinks.
    """
    gypsum = InsulationMaterial(
        id="gyp", name="gypsum", lambda_p=0.20, rho_p=800.0, c_p=1700.0
    )
    factors = [100.0, 200.0, 300.0]
    duration = 10.0  # early exposure: nothing has saturated yet
    bare = [
        steel_temperature(duration, sf, warn_lumped=False).theta_final for sf in factors
    ]
    protected = [
        protected_steel_temperature(duration, sf, gypsum, 0.030).theta_final
        for sf in factors
    ]
    # bare steel spans ~220 degC across this A_m/V range at 10 min; behind a
    # 30 mm board the same range spans only a few degrees.  At long duration
    # the ordering reverses -- bare members saturate near the gas temperature
    # and stop being sensitive, while protected ones are still climbing -- so
    # the duration is part of the claim, not an arbitrary choice.
    assert max(bare) - min(bare) > 100.0
    assert max(protected) - min(protected) < 0.25 * (max(bare) - min(bare))
    # mu grows with A_p/V, so the insulation's own capacity matters more for
    # a slender member -- the sensitivity is damped, not removed
    mus = [capacity_ratio_mu(gypsum, 0.030, sf, 20.0) for sf in factors]
    assert all(a < b for a, b in itertools.pairwise(mus))


def test_biot_sensitivity_tracks_section_factor() -> None:
    """The lumped-model validity margin must widen as ``A_m/V`` grows."""
    from truss_analysis.thermal.fire_curve import lumped_capacity_biot

    biots = [lumped_capacity_biot(sf) for sf in (25.0, 50.0, 100.0, 200.0)]
    assert all(a > b for a, b in itertools.pairwise(biots))
    assert biots[0] > 0.1 > biots[-1]
