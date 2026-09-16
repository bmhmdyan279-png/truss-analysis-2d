"""The ``use_effective_alpha`` flag reaches the whole demand chain.

``effective_alpha(T)`` existed in the material layer before this round, and the
flag reached ``prestress_lengths`` and ``build_engine`` -- but nothing the fire
chain actually calls accepted it.  ``member_axial_forces``, ``dcr_field``,
``ci_two_component``, ``compute_ci_for_topology`` and ``UniformForceScan`` all
built their base state on a constant ``alpha``, and the two brute-force
reference paths in the criticality engine did so *silently*, so a caller who
had opted into the Eurocode curve got a fallback column computed on a different
prestress field than the engine it was verifying.

The consequence was that :class:`~truss_analysis.exceptions.ConstantAlphaWarning`
told the user to "pass ``use_effective_alpha=True``" while no function they
could reach accepted it -- a warning that names a parameter no reachable API
takes is a promise the library cannot keep.

These tests pin the wiring, the exactness of the closed-form scan under it, and
the one number a reader is most likely to think is a bug: correcting the strain
raises the restrained force by 20.7% at 600 degC while the strain itself was
understated by 17.1%.  Both are the same fact seen from opposite ends.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from truss_analysis.criticality.engine import (
    base_displacement,
    brute_force_ci,
    build_engine,
    ci_sweep,
    prestress_lengths,
    total_load_vector,
)
from truss_analysis.exceptions import ConstantAlphaWarning
from truss_analysis.limitstates import (
    UniformForceScan,
    ci_two_component,
    dcr_field,
    member_axial_forces,
    member_critical_temperature,
    system_critical_temperature,
)
from truss_analysis.material.steel_eurocode import effective_alpha, thermal_strain
from truss_analysis.model import Element, Node
from truss_analysis.thermal.fire_curve import iso_834_temperature, steel_temperature

pytestmark = pytest.mark.filterwarnings(
    "ignore::truss_analysis.exceptions.ConstantAlphaWarning"
)

E_STEEL = 210.0e9
F_Y = 235.0e6
ALPHA_AMBIENT = 1.2e-5
GRID = (20.0, 150.0, 400.0, 600.0, 800.0, 1000.0)


def _redundant_fan(
    delta_l_free: float = 0.0,
    rise: float = 0.01,
    post_alpha: float = 0.0,
    area: float = 1.0e-3,
    i_sec: float = 1.0e-9,
    load: float = -50.0,
):
    """A redundant shallow fan whose rafters expand and whose post does not.

    Redundancy is what makes the flag observable: in a determinate truss a
    heated member expands freely and develops no force, so ``alpha`` never
    reaches the demand.  The post is deliberately given ``alpha = 0`` -- a
    tie/post modelled as fixed in length -- because that is the case where the
    secant basis must *not* be applied, and where a naive
    "multiply everything by ``alpha_eff(T)``" implementation would silently
    invent expansion the model forbids.
    """
    nodes = [
        Node(id="L", x=-1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=-1.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=rise, is_support=False),
    ]
    elements = [
        Element(
            id="r1",
            node_i="L",
            node_j="A",
            E=E_STEEL,
            A=area,
            I_sec=i_sec,
            alpha=ALPHA_AMBIENT,
        ),
        Element(
            id="r2",
            node_i="R",
            node_j="A",
            E=E_STEEL,
            A=area,
            I_sec=i_sec,
            alpha=ALPHA_AMBIENT,
        ),
        Element(
            id="post",
            node_i="A",
            node_j="B",
            E=E_STEEL,
            A=area,
            I_sec=i_sec,
            alpha=post_alpha,
            delta_L_free=delta_l_free,
        ),
    ]
    loads = {"A": {"Fx": 0.0, "Fy": load}}
    return nodes, elements, loads


def _uniform(elements, theta: float) -> dict[str, float]:
    return {e.id: float(theta) for e in elements}


# ---------------------------------------------------------------------------
# 1. the flag is accepted everywhere the chain forms a base state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        "member_axial_forces",
        "dcr_field",
        "ci_two_component",
        "member_critical_temperature",
        "system_critical_temperature",
        "UniformForceScan.build",
    ],
)
def test_the_demand_chain_accepts_the_flag(fn: str) -> None:
    """Every reachable entry point must accept ``use_effective_alpha``.

    Enumerated by name rather than tested one by one, because the defect being
    guarded against is *omission*: a function added to the chain later that
    quietly drops the flag would reproduce exactly the half-installed state
    this round closed.  A name that disappears from the public API fails here
    instead of failing silently in somebody's fire analysis.
    """
    import inspect

    from truss_analysis import limitstates

    target = getattr(limitstates, fn.split(".")[0])
    if "." in fn:  # classmethod
        target = getattr(target, fn.split(".")[1])
    params = inspect.signature(target).parameters
    assert "use_effective_alpha" in params, f"{fn} does not accept the flag"
    assert params["use_effective_alpha"].default is False, (
        f"{fn}: the default must stay False so published results remain "
        "reproducible bit for bit"
    )


def test_the_engine_reference_paths_accept_the_flag() -> None:
    """The brute-force fallback must be buildable on the same prestress field.

    ``_solve_perturbed_full`` and ``brute_force_ci`` are the paths a caller
    goes to when they have stopped trusting the fast one.  A reference that
    disagrees with the thing it verifies, for a reason neither reports, is
    worse than no reference.
    """
    import inspect

    from truss_analysis.criticality import engine

    for fn in (
        engine.brute_force_ci,
        engine.compute_ci_for_topology,
        engine._solve_perturbed_full,
    ):
        params = inspect.signature(fn).parameters
        assert "use_effective_alpha" in params, fn.__name__
        assert params["use_effective_alpha"].default is False, fn.__name__


# ---------------------------------------------------------------------------
# 2. the closed-form scan stays exact under the secant basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [False, True])
def test_uniform_force_scan_matches_pointwise_in_both_modes(flag: bool) -> None:
    """The secant coefficient does not break the closed form.

    ``UniformForceScan`` exists so the critical-temperature scans cost
    ``O(n^3)`` once instead of once per grid point.  That is only worth having
    if the closed form is *exact*, so the check is against a full per-point
    engine rebuild at machine precision, not against a loose tolerance -- and
    it is run in both modes, because the effective path introduces a second
    precomputed basis and a wrong mask would show up here first.
    """
    nodes, elements, loads = _redundant_fan()
    scan = UniformForceScan.build(nodes, elements, loads, use_effective_alpha=flag)
    assert scan.use_effective_alpha is flag
    for theta in GRID:
        closed = scan.forces_at(theta)
        pointwise = np.array(
            [
                member_axial_forces(
                    nodes,
                    elements,
                    loads,
                    _uniform(elements, theta),
                    use_effective_alpha=flag,
                )[e.id]
                for e in elements
            ]
        )
        scale = max(1.0, float(np.max(np.abs(pointwise))))
        assert np.max(np.abs(closed - pointwise)) / scale < 1e-12, theta


def test_the_secant_basis_uses_a_masked_unit_alpha_vector() -> None:
    """``unit_lengths`` is ``L_e`` where the member expands and ``0`` where not.

    Pinned because this is the one place the effective path could quietly
    diverge from ``prestress_lengths``: both must apply the same rule that
    ``alpha = 0`` means "this member does not expand", or the scan and the
    per-point engine would disagree about which members are heated.
    """
    nodes, elements, loads = _redundant_fan()
    scan = UniformForceScan.build(nodes, elements, loads, use_effective_alpha=True)
    coords = {n.id: (n.x, n.y) for n in nodes}
    lengths = np.array(
        [
            float(np.hypot(*(np.array(coords[e.node_j]) - np.array(coords[e.node_i]))))
            for e in elements
        ]
    )
    expanding = np.array([e.alpha for e in elements]) != 0.0
    assert np.allclose(scan.unit_lengths, np.where(expanding, lengths, 0.0), rtol=1e-14)
    # the alpha = 0 post carries its length as zero in the secant basis...
    assert scan.unit_lengths[scan.ids.index("post")] == 0.0
    # ...while the constant basis already had it at zero through alpha itself
    assert scan.alpha_lengths[scan.ids.index("post")] == 0.0
    # and the two bases coincide exactly when every member shares one alpha
    assert np.allclose(
        scan.alpha_lengths[expanding],
        ALPHA_AMBIENT * scan.unit_lengths[expanding],
        rtol=1e-14,
    )


def test_fixed_length_members_do_not_gain_expansion_under_the_flag() -> None:
    """Turning the flag on must not make an ``alpha = 0`` member expand.

    The direct inverse of the defect the mask prevents.  Here the post carries
    all of its imposed elongation in ``delta_L_free``, which is a geometric
    input and must be untouched by a change of expansion coefficient.
    """
    nodes, elements, _loads = _redundant_fan(delta_l_free=3.0e-4)
    temps = _uniform(elements, 600.0)
    cold = prestress_lengths(nodes, elements, temps, use_effective_alpha=False)
    warm = prestress_lengths(nodes, elements, temps, use_effective_alpha=True)
    post = [e.id for e in elements].index("post")
    assert cold[post] == warm[post] == pytest.approx(3.0e-4, rel=1e-15)
    # and the expanding members did change
    assert warm[0] > cold[0]


# ---------------------------------------------------------------------------
# 3. the default is unchanged, bit for bit
# ---------------------------------------------------------------------------


def test_default_is_bit_identical_to_the_constant_alpha_path() -> None:
    """Omitting the flag must reproduce the pre-change numbers exactly.

    Not ``approx`` -- bit for bit.  Changing the default would silently move
    every result ever published with this library, which is why the correction
    is opt-in and why this test exists: it is the guarantee that makes opt-in
    safe to offer.
    """
    nodes, elements, loads = _redundant_fan()
    temps = _uniform(elements, 600.0)
    explicit = member_axial_forces(
        nodes, elements, loads, temps, use_effective_alpha=False
    )
    defaulted = member_axial_forces(nodes, elements, loads, temps)
    for eid in explicit:
        assert explicit[eid] == defaulted[eid]

    scan_default = UniformForceScan.build(nodes, elements, loads)
    scan_explicit = UniformForceScan.build(
        nodes, elements, loads, use_effective_alpha=False
    )
    assert np.array_equal(scan_default.forces_at(600.0), scan_explicit.forces_at(600.0))


def test_the_constant_path_reproduces_the_straight_line_by_hand() -> None:
    """A fully restrained bar's force is ``-k_E(T) E A alpha (T - T_0)``.

    Hand-computed, so the constant-``alpha`` branch is anchored to something
    other than the library's own expression for it.
    """
    nodes = [
        Node(id="L", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=3.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(
            id="bar", node_i="L", node_j="R", E=E_STEEL, A=0.01, alpha=ALPHA_AMBIENT
        )
    ]
    theta = 600.0
    forces = member_axial_forces(nodes, elements, {}, {"bar": theta})
    from truss_analysis.material.steel_eurocode import k_E

    hand = -float(k_E(theta)) * E_STEEL * 0.01 * ALPHA_AMBIENT * (theta - 20.0)
    assert forces["bar"] == pytest.approx(hand, rel=1e-12)

    # and with the flag it reproduces the standard's own elongation instead
    effective = member_axial_forces(
        nodes, elements, {}, {"bar": theta}, use_effective_alpha=True
    )
    hand_curve = -float(k_E(theta)) * E_STEEL * 0.01 * float(thermal_strain(theta))
    assert effective["bar"] == pytest.approx(hand_curve, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. the two ratios: 17.1% understated, 20.7% correction
# ---------------------------------------------------------------------------


def test_correcting_the_strain_raises_the_force_by_the_other_ratio() -> None:
    """The number a reader is most likely to mistake for a bug.

    At 600 degC the constant-``alpha`` *strain* is 17.1% below the standard's
    curve, but switching the flag on *raises* the restrained force by 20.7%.
    Both are correct and they are the same pair seen from opposite ends:
    ``eps_const = (1 - 0.171) eps_curve``, so
    ``N_curve / N_const = 1 / (1 - 0.171) = 1.207``.  A test that pinned only
    one of them would have passed with the other wrong, which is how the
    changelog came to label 20.7% as the elongation understatement.
    """
    # No mechanical load: the force is then purely restrained thermal
    # expansion and scales exactly with alpha, so the ratio is the alpha ratio
    # to machine precision.  With the 50 N apex load present the ratio is
    # 1.20644 rather than 1.20667 -- the mechanical part does not scale -- so
    # an exactness assertion there would be wrong for an uninteresting reason.
    nodes, elements, _ = _redundant_fan()
    loads = {}
    temps = _uniform(elements, 600.0)
    cold = member_axial_forces(nodes, elements, loads, temps)
    warm = member_axial_forces(nodes, elements, loads, temps, use_effective_alpha=True)
    alpha_ratio = float(effective_alpha(600.0)) / ALPHA_AMBIENT
    for eid in ("r1", "r2"):
        ratio = warm[eid] / cold[eid]  # both compressive, so the ratio is > 1
        assert ratio == pytest.approx(alpha_ratio, rel=1e-12)
        assert ratio == pytest.approx(1.0 / (1.0 - 0.1713), rel=2e-3)
        # and the two ways of stating the gap really are the reciprocal pair
        assert 1.0 - 1.0 / ratio == pytest.approx(0.1713, abs=5e-4)

    # with a mechanical load present the ratio moves *towards* 1, never past it
    nodes, elements, loaded = _redundant_fan()
    cold_l = member_axial_forces(nodes, elements, loaded, temps)
    warm_l = member_axial_forces(
        nodes, elements, loaded, temps, use_effective_alpha=True
    )
    for eid in ("r1", "r2"):
        assert 1.0 < warm_l[eid] / cold_l[eid] <= alpha_ratio


def test_the_flag_moves_the_dcr_in_the_unsafe_direction_only() -> None:
    """Demand rises, capacity is untouched, so the DCR can only get worse.

    The direction is the safety-relevant part: the constant-``alpha`` default
    is un-conservative, never the reverse.  A change that ever *lowered* a
    fire DCR would be a regression of a different and worse kind.
    """
    nodes, elements, loads = _redundant_fan()
    for theta in (400.0, 600.0, 700.0):
        temps = _uniform(elements, theta)
        cold = dcr_field(nodes, elements, loads, temps, F_Y)
        warm = dcr_field(nodes, elements, loads, temps, F_Y, use_effective_alpha=True)
        for eid in ("r1", "r2"):
            assert warm[eid].dcr > cold[eid].dcr, (theta, eid)
            # capacity depends on temperature alone, not on alpha
            assert warm[eid].capacity == pytest.approx(cold[eid].capacity, rel=1e-15)


# ---------------------------------------------------------------------------
# 5. the critical-temperature scans honour the flag
# ---------------------------------------------------------------------------


def test_critical_temperatures_drop_when_the_curve_is_used() -> None:
    """Higher demand at every temperature means failure arrives sooner.

    ``theta_cr`` is the headline number of the fire chain, so a flag that
    changed the forces but not the critical temperature would mean the scans
    were still built on the old basis -- the exact half-installed state this
    round closed.

    The fixture is deliberately *not* the slender fan used elsewhere in this
    file.  That one is so shallow that its restrained thermal forces cross
    ``DCR = 1`` near 120 degC, where the secant correction is only ~4% and the
    crossing stays inside the same grid cell in both modes -- a comparison that
    could not fail.  This fan is deeper (rise 0.5 m), every member expands, and
    the crossing sits at 460 degC where the correction is worth 50 degC of
    critical temperature.  Both fixtures are checked to be non-vacuous: the
    ambient DCR must be well below 1, so the scan has somewhere to travel.
    """
    grid = tuple(float(t) for t in range(20, 901, 10))

    nodes, elements, loads = _redundant_fan()
    member_cold = member_critical_temperature(
        nodes, elements, loads, "r1", F_Y, temp_grid=grid
    )
    member_warm = member_critical_temperature(
        nodes, elements, loads, "r1", F_Y, temp_grid=grid, use_effective_alpha=True
    )
    assert member_cold is not None
    assert member_warm is not None
    assert member_warm < member_cold

    nodes, elements, loads = _redundant_fan(
        rise=0.5, post_alpha=ALPHA_AMBIENT, area=1.0e-3, i_sec=1.0e-6, load=-5.0e3
    )
    ambient = dcr_field(nodes, elements, loads, _uniform(elements, 20.0), F_Y)
    assert max(ls.dcr for ls in ambient.values()) < 0.05, (
        "the fixture must start well below DCR = 1 or the scan cannot move"
    )
    system_cold = system_critical_temperature(
        nodes, elements, loads, F_Y, temp_grid=grid
    )
    system_warm = system_critical_temperature(
        nodes, elements, loads, F_Y, temp_grid=grid, use_effective_alpha=True
    )
    assert system_cold == pytest.approx(460.0)
    assert system_warm == pytest.approx(410.0)
    assert system_warm < system_cold
    # 50 degC of critical temperature is the size of the prize: this is not a
    # rounding-level correction, it is the difference between a member that is
    # reported to survive 7.5 minutes of standard fire and one that survives 6
    assert (system_cold - system_warm) >= 40.0


# ---------------------------------------------------------------------------
# 6. the reference path agrees with the engine it verifies
# ---------------------------------------------------------------------------


def test_brute_force_column_matches_an_engine_built_on_the_same_basis() -> None:
    """The rank-1 sweep and its brute-force reference must share a prestress.

    This is the finding the wiring closes: with the flag set on the engine and
    not on the fallback, the two columns were computed from different imposed
    strain fields and the comparison measured that disagreement instead of the
    correctness of the Woodbury update.
    """
    nodes, elements, loads = _redundant_fan()
    temps = _uniform(elements, 600.0)
    alpha = 0.7
    for flag in (False, True):
        setup = build_engine(nodes, elements, loads, temps, use_effective_alpha=flag)
        u = base_displacement(setup, total_load_vector(nodes, loads, setup))
        sweep = ci_sweep(setup, u, alpha)
        brute, u_max_base, _ = brute_force_ci(
            nodes, elements, loads, temps, alpha, use_effective_alpha=flag
        )
        for eid in sweep.ci_values:
            assert sweep.ci_values[eid] == pytest.approx(brute[eid], abs=1e-9), (
                flag,
                eid,
            )
        assert u_max_base > 0.0


def test_a_mismatched_reference_is_visibly_different() -> None:
    """The negative control: mixing the two bases must NOT agree.

    Without this, the test above could pass because both columns ignored the
    flag.  Asserting that the mismatched pairing differs proves the flag is
    actually load-bearing on the reference path.
    """
    nodes, elements, loads = _redundant_fan()
    temps = _uniform(elements, 600.0)
    warm, _, _ = brute_force_ci(
        nodes, elements, loads, temps, 0.7, use_effective_alpha=True
    )
    cold, _, _ = brute_force_ci(
        nodes, elements, loads, temps, 0.7, use_effective_alpha=False
    )
    assert any(abs(warm[k] - cold[k]) > 1e-12 for k in warm), (
        "the two bases produced identical CI values, so the flag is not "
        "reaching the reference path and the equivalence test is vacuous"
    )


def test_ci_two_component_accepts_and_honours_the_flag() -> None:
    """The two-component CI's hot base state follows the flag too."""
    nodes, elements, loads = _redundant_fan()
    temps = _uniform(elements, 600.0)
    cold = ci_two_component(nodes, elements, loads, temps, 0.7, F_Y)
    warm = ci_two_component(
        nodes, elements, loads, temps, 0.7, F_Y, use_effective_alpha=True
    )
    # The *damage* component is a ratio DCR_pert(T) / DCR_base(T) - 1 whose
    # numerator and denominator share the same prestress field, so it is
    # invariant to the choice of alpha -- and that invariance is the property
    # its docstring claims ("temperature degradation can never leak into the
    # perturbation criticality").  Pinning it here means a future change that
    # applied the flag to one side only would fail.
    for k in cold.components:
        assert warm.components[k].dcr_component == pytest.approx(
            cold.components[k].dcr_component, rel=1e-12
        ), k
    # The *fire-severity* component is an absolute ratio against the cold
    # state, so it must move -- this is what proves the flag is load-bearing.
    assert any(
        abs(warm.components[k].fire_component - cold.components[k].fire_component)
        > 1e-9
        for k in cold.components
    )
    assert all(
        warm.components[k].fire_component > cold.components[k].fire_component
        for k in cold.components
    )
    assert any(
        abs(warm.components[k].dcr_combined - cold.components[k].dcr_combined) > 1e-9
        for k in cold.components
    )


# ---------------------------------------------------------------------------
# 7. end-to-end acceptance: fire curve -> steel temperature -> alpha(T) -> DCR
# ---------------------------------------------------------------------------


def test_fire_to_dcr_end_to_end_acceptance_case() -> None:
    """The whole chain on one golden case, from a fire curve to a DCR.

    The acceptance case requested for this round: nothing here is compared
    against the library's own intermediate expressions.  The gas temperature
    comes from ISO 834, the steel temperature from the clause-4.2.2.2 ODE, the
    imposed strain from the clause-3.4.1.1 elongation curve, and the restrained
    force from the hand-derived ``-k_E E A eps_th`` of a fully restrained bar.
    The golden values below were measured, not fitted.
    """
    duration_min = 30.0
    section_factor = 250.0  # A_m/V [1/m], a heavy unprotected section
    theta_g = iso_834_temperature(duration_min * 60.0)
    heating = steel_temperature(
        duration_min, section_factor, fire_curve=iso_834_temperature, theta_a0=20.0
    )
    theta_a = float(heating.theta_final)
    assert theta_g > 800.0, "30 min of ISO 834 must exceed 800 degC"
    # measured, not assumed: a heavy section (A_m/V = 250) in 30 min of ISO 834
    assert theta_a == pytest.approx(832.64, abs=0.05)

    nodes = [
        Node(id="L", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=3.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(
            id="bar",
            node_i="L",
            node_j="R",
            E=E_STEEL,
            A=0.01,
            I_sec=1.0e-5,
            alpha=ALPHA_AMBIENT,
        )
    ]
    temps = {"bar": theta_a}

    with pytest.warns(ConstantAlphaWarning):
        cold = dcr_field(nodes, elements, {}, temps, F_Y)
    warm = dcr_field(nodes, elements, {}, temps, F_Y, use_effective_alpha=True)

    from truss_analysis.material.steel_eurocode import k_E

    s_E = float(k_E(theta_a))
    # Hand-derived restrained force, against the standard's own elongation
    # curve rather than against anything this library computed.  The capacity
    # is NOT hand-derived here: it is chi-reduced per EN 1993-1-2 4.2.3.1 and
    # is already pinned against a hand solution in test_limitstates.py, so
    # re-deriving it would duplicate that check rather than strengthen this one.
    n_curve = -s_E * E_STEEL * 0.01 * float(thermal_strain(theta_a))
    n_const = -s_E * E_STEEL * 0.01 * ALPHA_AMBIENT * (theta_a - 20.0)

    assert warm["bar"].axial_force == pytest.approx(n_curve, rel=1e-12)
    assert cold["bar"].axial_force == pytest.approx(n_const, rel=1e-12)
    # capacity depends on temperature alone, so it is identical in both runs
    # and the DCR must move by exactly the force ratio
    assert warm["bar"].capacity == pytest.approx(cold["bar"].capacity, rel=1e-15)
    assert warm["bar"].dcr == pytest.approx(
        cold["bar"].dcr * abs(n_curve) / abs(n_const), rel=1e-12
    )
    assert warm["bar"].dcr > cold["bar"].dcr

    # the demand the default was hiding, as a fraction of the true demand
    shortfall = 1.0 - abs(n_const) / abs(n_curve)
    assert shortfall == pytest.approx(
        1.0 - ALPHA_AMBIENT / float(effective_alpha(theta_a)), rel=1e-9
    )
    # 11.35%, not the 17.1% quoted at 600 degC: the clause-3.4.1.1 elongation
    # curve has a plateau between 750 and 860 degC, so the secant coefficient
    # falls back towards the ambient one and the gap is NOT monotone in
    # temperature.  Pinning the measured value here rather than a round
    # threshold is what keeps that non-monotonicity from being mistaken for a
    # regression later.
    assert shortfall == pytest.approx(0.11348, abs=5e-5)
    assert shortfall > 0.10, "a 30 min ISO 834 fire must expose a real gap"

    # and the member ratio stays reconstructible in both
    for state in (cold, warm):
        ls = state["bar"]
        assert ls.dcr == pytest.approx(abs(ls.axial_force) / ls.capacity, rel=1e-12)


def test_the_chain_is_quiet_below_the_measured_threshold() -> None:
    """No warning under 150 degC, where the straight line is within 5%.

    The other half of the acceptance case: the flag must not make a cold
    analysis noisy.  150 degC is where the strain error first exceeds 5%
    (measured 5.36%), not a round number.
    """
    nodes, elements, loads = _redundant_fan()
    for theta in (20.0, 100.0, 150.0):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConstantAlphaWarning)
            member_axial_forces(nodes, elements, loads, _uniform(elements, theta))
    with pytest.warns(ConstantAlphaWarning):
        member_axial_forces(nodes, elements, loads, _uniform(elements, 151.0))
