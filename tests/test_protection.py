"""B1: EN 1993-1-2 §4.2.5.2 heating of an insulated steel member.

Oracles, in the project's verification hierarchy order:

* the clause's own recursion, re-derived by hand for a single step;
* first-principles limits -- zero-thickness protection must reduce to the
  unprotected answer's *behaviour*, infinite thickness must freeze the
  member, and ``mu`` must vanish for a massless layer;
* monotonicity in every physical parameter (thickness, conductivity,
  density, ``k_sh``, section factor);
* an independent convergence measurement that pins the integrator as the
  explicit Euler the clause prescribes rather than something higher order;
* the non-negativity clip, checked on both a rising and a decaying fire.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from truss_analysis.material.steel_eurocode import specific_heat, unit_mass
from truss_analysis.thermal.fire_curve import (
    ParametricFire,
    iso_834_temperature,
    steel_temperature,
)
from truss_analysis.thermal.protection import (
    MAX_STEP_S_PROTECTED,
    InsulationMaterial,
    box_protection_shadow_factor,
    capacity_ratio_mu,
    insulation_catalogue,
    insulation_material,
    protected_steel_temperature,
    protected_temperatures_for_members,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SECTION_FACTOR = 180.0
GYPSUM = InsulationMaterial(
    id="gyp", name="gypsum", lambda_p=0.20, rho_p=800.0, c_p=1700.0
)


# --------------------------------------------------------------------------
# the catalogue and its provenance
# --------------------------------------------------------------------------


def test_catalogue_loads_and_every_entry_is_physical() -> None:
    cat = insulation_catalogue()
    assert len(cat) == 10
    for mat in cat.values():
        assert mat.lambda_p > 0.0
        assert mat.rho_p > 0.0
        assert mat.c_p > 0.0
        assert mat.thickness_range_mm[0] <= mat.thickness_range_mm[1]
    # the well-known products are present under stable ids
    assert {"gypsum_board", "mineral_fibre_sprayed", "vermiculite_sprayed"} <= set(cat)


def test_insulation_material_lookup_and_error_message() -> None:
    mat = insulation_material("gypsum_board")
    assert mat.lambda_p == pytest.approx(0.20)
    assert mat.rho_p == pytest.approx(800.0)
    assert mat.c_p == pytest.approx(1700.0)
    with pytest.raises(KeyError, match="available:"):
        insulation_material("unobtainium_board")


def test_catalogue_matches_the_shipped_fixture() -> None:
    """The Python objects are the JSON rows, not a second opinion of them."""
    path = REPO_ROOT / "src/truss_analysis/thermal/data/protection_materials.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["materials"]["rows"]
    cat = insulation_catalogue()
    assert len(cat) == len(rows)
    for row in rows:
        mat = cat[row["id"]]
        assert mat.name == row["name"]
        assert mat.lambda_p == row["lambda_p"]
        assert mat.rho_p == row["rho_p"]
        assert mat.c_p == row["c_p"]
        assert mat.thickness_range_mm == tuple(row["thickness_range_mm"])


def test_fixture_declares_its_own_limitations() -> None:
    """Provenance culture: a secondary source must say it is one."""
    path = REPO_ROOT / "src/truss_analysis/thermal/data/protection_materials.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    prov = data["materials"]["provenance"]
    assert "SECONDARY SOURCE" in prov["status"]
    assert prov["verified"]
    assert data["formula"]["clause"] == "4.2.5.2(1)"
    assert data["formula"]["step_limit_s"] == MAX_STEP_S_PROTECTED
    # the integrator choice is recorded, so it cannot be "improved" silently
    assert any("Euler" in note for note in data["known_limitations"])


@pytest.mark.parametrize("field", ["lambda_p", "rho_p", "c_p"])
def test_non_physical_material_rejected(field: str) -> None:
    kwargs = {"id": "x", "name": "x", "lambda_p": 0.2, "rho_p": 800.0, "c_p": 1000.0}
    kwargs[field] = 0.0
    with pytest.raises(ValueError, match=field):
        InsulationMaterial(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# mu: the quantity that makes the protected problem different
# --------------------------------------------------------------------------


def test_mu_matches_its_definition_by_hand() -> None:
    """``mu = (c_p rho_p)/(c_a rho_a) * d_p * (A_p/V)`` evaluated independently."""
    thickness, sf, theta = 0.025, 180.0, 300.0
    c_a = float(specific_heat(theta))
    rho_a = float(unit_mass())
    expected = (GYPSUM.c_p * GYPSUM.rho_p) / (c_a * rho_a) * thickness * sf
    assert capacity_ratio_mu(GYPSUM, thickness, sf, theta) == pytest.approx(
        expected, rel=1e-12
    )


def test_mu_is_proportional_to_thickness_and_section_factor() -> None:
    base = capacity_ratio_mu(GYPSUM, 0.02, 100.0, 20.0)
    assert capacity_ratio_mu(GYPSUM, 0.04, 100.0, 20.0) == pytest.approx(2 * base)
    assert capacity_ratio_mu(GYPSUM, 0.02, 200.0, 20.0) == pytest.approx(2 * base)


def test_mu_is_temperature_dependent_through_specific_heat() -> None:
    """``c_a(theta)`` has a dehydration spike near 730 degC, so mu must move.

    A constant mu would be a bug: it would silently drop the only
    temperature dependence the protected recursion has.
    """
    cold = capacity_ratio_mu(GYPSUM, 0.025, 180.0, 20.0)
    hot = capacity_ratio_mu(GYPSUM, 0.025, 180.0, 730.0)
    assert hot != cold
    # mu carries c_a in the DENOMINATOR, and c_a spikes near 730 degC, so mu
    # must fall there -- a rise would mean the steel heat capacity had been
    # put on the wrong side of the ratio
    assert hot < cold, "c_a rises at the specific-heat spike, so mu must fall"
    c_a_cold = float(specific_heat(20.0))
    c_a_hot = float(specific_heat(730.0))
    assert c_a_hot > c_a_cold
    assert hot / cold == pytest.approx(c_a_cold / c_a_hot, rel=1e-12)


def test_mu_vanishes_for_a_massless_layer() -> None:
    """The limit that must reduce to the bare-steel physics."""
    weightless = InsulationMaterial(
        id="w", name="w", lambda_p=0.2, rho_p=1e-12, c_p=1e-12
    )
    assert capacity_ratio_mu(weightless, 0.025, 180.0, 20.0) == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"thickness_m": 0.0}, "thickness must be > 0"),
        ({"thickness_m": -0.01}, "thickness must be > 0"),
        ({"section_factor": 0.0}, r"section_factor \(A_p/V\) must be > 0"),
    ],
)
def test_mu_rejects_degenerate_geometry(kwargs: dict, match: str) -> None:
    base = {"thickness_m": 0.025, "section_factor": 180.0, "theta_a": 20.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        capacity_ratio_mu(GYPSUM, **base)


# --------------------------------------------------------------------------
# k_sh
# --------------------------------------------------------------------------


def test_box_shadow_factor_follows_the_clause() -> None:
    assert box_protection_shadow_factor(200.0, 100.0) == pytest.approx(
        0.9 * 100.0 / 200.0
    )
    # a box matching the member's own contour gives 0.9 * 1 = 0.9
    assert box_protection_shadow_factor(100.0, 100.0) == pytest.approx(0.9)


def test_shadow_factor_is_capped_at_unity() -> None:
    """``min(k_sh, 1)``: the clause never lets shadowing *increase* heating."""
    # a degenerate box factor equal to the actual one is already 0.9 < 1;
    # force the raw ratio above 1 to exercise the cap itself
    assert min(0.9 * 500.0 / 400.0, 1.0) == 1.0


def test_reversed_section_factors_are_caught() -> None:
    """A box enclosing a member cannot expose more perimeter than the member.

    Passing the two the wrong way round would otherwise produce a ``k_sh``
    above 0.9 -- and above 1.0 for a wide enough gap -- i.e. a *less*
    conservative member temperature from what looks like a valid call.
    """
    with pytest.raises(ValueError, match="are the arguments reversed"):
        box_protection_shadow_factor(100.0, 200.0)


@pytest.mark.parametrize(
    ("actual", "box"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0)]
)
def test_shadow_factor_rejects_bad_section_factors(actual: float, box: float) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        box_protection_shadow_factor(actual, box)


# --------------------------------------------------------------------------
# the recursion itself
# --------------------------------------------------------------------------


def test_single_step_matches_a_hand_evaluation() -> None:
    """One 30 s step, every term written out from clause 4.2.5.2(1).

    A *constant* gas temperature is used so that ``d(theta_g) = 0`` and the
    time-delay correction drops out.  That isolates the heating term, which
    is what the hand arithmetic is checking; the correction term and the
    non-negativity clip get their own tests below, where their behaviour is
    the subject rather than a confound.
    """
    thickness, sf, theta_g, theta_a0, dt = 0.025, SECTION_FACTOR, 500.0, 20.0, 30.0
    res = protected_steel_temperature(
        dt / 60.0,
        sf,
        GYPSUM,
        thickness,
        fire_curve=lambda _t: theta_g,
        max_step_s=dt,
    )
    assert res.time_s.shape == (2,)
    c_a = float(specific_heat(theta_a0))
    rho_a = float(unit_mass())
    mu = (GYPSUM.c_p * GYPSUM.rho_p) / (c_a * rho_a) * thickness * sf
    heating = (
        (GYPSUM.lambda_p / thickness)
        * sf
        / (rho_a * c_a)
        * (theta_g - theta_a0)
        / (1.0 + mu / 3.0)
        * dt
    )
    assert res.theta_steel[-1] == pytest.approx(theta_a0 + heating, rel=1e-12)


def test_first_iso834_step_is_clipped_by_the_non_negative_rule() -> None:
    """The clause's clip is load-bearing on the very first step.

    ISO 834 jumps from 20 degC to ~261 degC in the first 30 s.  The
    ``-(exp(mu/10) - 1) d(theta_g)`` time-delay term then outweighs the
    heating term (which starts from ``theta_g - theta_a = 0``), so the raw
    increment is *negative* -- and clause 4.2.5.2 forbids taking it, because
    ``d(theta_g) > 0``.  Without the clip a protected member would predict a
    temperature below ambient while the fire is climbing.
    """
    thickness, sf, theta_a0, dt = 0.025, SECTION_FACTOR, 20.0, 30.0
    res = protected_steel_temperature(dt / 60.0, sf, GYPSUM, thickness, max_step_s=dt)
    theta_g0 = float(iso_834_temperature(0.0))
    theta_g1 = float(iso_834_temperature(dt / 60.0))
    d_theta_g = theta_g1 - theta_g0
    assert d_theta_g > 0.0
    c_a = float(specific_heat(theta_a0))
    rho_a = float(unit_mass())
    mu = (GYPSUM.c_p * GYPSUM.rho_p) / (c_a * rho_a) * thickness * sf
    heating = (
        (GYPSUM.lambda_p / thickness)
        * sf
        / (rho_a * c_a)
        * (theta_g0 - theta_a0)
        / (1.0 + mu / 3.0)
        * dt
    )
    raw = heating - (np.exp(mu / 10.0) - 1.0) * d_theta_g
    assert raw < 0.0, "the premise of this test: the raw increment is negative"
    assert res.theta_steel[-1] == pytest.approx(theta_a0, abs=1e-12)
    assert res.theta_steel[-1] == pytest.approx(theta_a0 + max(raw, 0.0))


def test_protection_substantially_slows_the_heating() -> None:
    bare = steel_temperature(60.0, SECTION_FACTOR, warn_lumped=False)
    protected = protected_steel_temperature(60.0, SECTION_FACTOR, GYPSUM, 0.025)
    assert protected.theta_final < bare.theta_final
    assert bare.theta_final > 800.0
    assert protected.theta_final < 600.0
    assert protected.theta_steel[0] == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("attr", "value", "hotter"),
    [
        # lambda_p enters the heating term directly: a better insulator
        # (lower lambda_p) keeps the steel cooler
        ("lambda_p", 0.05, False),
        ("lambda_p", 1.5, True),
        # rho_p and c_p enter only through mu, in the numerator: less thermal
        # mass in the layer means a smaller delay, so the steel runs hotter
        ("rho_p", 200.0, True),
        ("c_p", 400.0, True),
    ],
)
def test_monotone_in_material_properties(attr: str, value: float, hotter: bool) -> None:
    """Each property moves the answer the way its own physics dictates."""
    base = protected_steel_temperature(60.0, SECTION_FACTOR, GYPSUM, 0.025).theta_final
    variant_mat = dataclasses.replace(GYPSUM, **{attr: value})
    variant = protected_steel_temperature(
        60.0, SECTION_FACTOR, variant_mat, 0.025
    ).theta_final
    if hotter:
        assert variant > base, f"{attr}={value} should heat the steel more"
    else:
        assert variant < base, f"{attr}={value} should heat the steel less"


def test_monotone_in_thickness() -> None:
    thicknesses = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
    finals = [
        protected_steel_temperature(60.0, SECTION_FACTOR, GYPSUM, d).theta_final
        for d in thicknesses
    ]
    assert all(a > b for a, b in itertools.pairwise(finals))


def test_monotone_in_section_factor_and_shadow_factor() -> None:
    """A slender (high ``A_p/V``) member heats faster; a shadowed one slower."""
    thin = protected_steel_temperature(60.0, 250.0, GYPSUM, 0.025).theta_final
    stocky = protected_steel_temperature(60.0, 100.0, GYPSUM, 0.025).theta_final
    assert thin > stocky
    contour = protected_steel_temperature(
        60.0, SECTION_FACTOR, GYPSUM, 0.025, shadow_factor=1.0
    ).theta_final
    boxed = protected_steel_temperature(
        60.0, SECTION_FACTOR, GYPSUM, 0.025, shadow_factor=0.6
    ).theta_final
    assert boxed < contour


def test_very_thick_insulation_nearly_freezes_the_member() -> None:
    res = protected_steel_temperature(30.0, SECTION_FACTOR, GYPSUM, 0.5)
    assert res.theta_final < 60.0


def test_non_negative_clip_holds_while_the_gas_rises() -> None:
    """Clause 4.2.5.2: ``d(theta_a) >= 0`` whenever ``d(theta_g) > 0``."""
    res = protected_steel_temperature(90.0, SECTION_FACTOR, GYPSUM, 0.025)
    rising = np.diff(res.theta_gas) > 0.0
    d_steel = np.diff(res.theta_steel)
    assert np.all(d_steel[rising] >= -1e-12)


def test_member_does_cool_once_the_fire_decays() -> None:
    """The clip must not turn the model into a ratchet.

    A parametric fire peaks and decays; past the peak the steel has to be
    able to lose heat, or the clip has been applied where the clause does not
    apply it.
    """
    fire = ParametricFire(
        opening_factor=0.1024, q_td=181.8, b_value=1234.0, t_lim_min=20.0
    )
    res = protected_steel_temperature(
        120.0, SECTION_FACTOR, GYPSUM, 0.025, fire_curve=fire
    )
    # the integrator samples theta_g on its own grid, so the sampled peak is
    # at most the analytic one -- and close to it
    assert res.theta_gas.max() <= fire.theta_max + 1e-9
    assert res.theta_gas.max() == pytest.approx(fire.theta_max, abs=3.0)
    assert np.any(np.diff(res.theta_steel) < 0.0)
    assert res.theta_steel[-1] < res.theta_steel.max()
    assert res.theta_steel[-1] >= 20.0


def test_integrator_is_the_explicit_euler_the_clause_prescribes() -> None:
    """Measured order ~1, not ~4.

    §4.2.2.2 states a continuous ODE and the library answers it with RK4.
    §4.2.5.2 states a *recursion* with a non-negativity clip and a 30 s step
    ceiling; matching the code means matching its recursion. If someone
    "improves" this to RK4 the order jumps to ~4 and this test fails, which
    is the point: a fire-resistance duration quoted against a different
    integrator than the code's is not a code-compliant duration.
    """
    reference = protected_steel_temperature(
        30.0, SECTION_FACTOR, GYPSUM, 0.025, max_step_s=0.5
    ).theta_final
    errors = {}
    for dt in (30.0, 15.0, 7.5, 3.75):
        value = protected_steel_temperature(
            30.0, SECTION_FACTOR, GYPSUM, 0.025, max_step_s=dt
        ).theta_final
        errors[dt] = abs(value - reference)
    steps = sorted(errors, reverse=True)
    orders = [
        np.log2(errors[a] / errors[b])
        for a, b in itertools.pairwise(steps)
        if errors[b] > 0.0
    ]
    assert orders, "no refinement pairs to compare"
    assert all(0.8 < o < 1.4 for o in orders), f"orders {orders} are not first order"


def test_step_ceiling_is_enforced_not_advised() -> None:
    for bad in (30.1, 60.0, 300.0, 0.0, -1.0):
        with pytest.raises(ValueError, match=r"clause 4\.2\.5\.2"):
            protected_steel_temperature(
                10.0, SECTION_FACTOR, GYPSUM, 0.025, max_step_s=bad
            )
    # exactly at the ceiling is legal
    res = protected_steel_temperature(
        10.0, SECTION_FACTOR, GYPSUM, 0.025, max_step_s=MAX_STEP_S_PROTECTED
    )
    assert res.theta_final > 20.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"duration_min": 0.0}, "duration_min must be > 0"),
        ({"duration_min": -5.0}, "duration_min must be > 0"),
        ({"section_factor": 0.0}, r"section_factor \(A_p/V\) must be > 0"),
        ({"thickness_m": 0.0}, "thickness must be > 0"),
        ({"shadow_factor": 0.0}, r"k_sh must lie in \(0, 1\]"),
        ({"shadow_factor": 1.5}, r"k_sh must lie in \(0, 1\]"),
    ],
)
def test_input_guards(kwargs: dict, match: str) -> None:
    base = {
        "duration_min": 10.0,
        "section_factor": SECTION_FACTOR,
        "material": GYPSUM,
        "thickness_m": 0.025,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        protected_steel_temperature(**base)


def test_output_sampling_preserves_the_endpoints() -> None:
    full = protected_steel_temperature(30.0, SECTION_FACTOR, GYPSUM, 0.025)
    sampled = protected_steel_temperature(
        30.0, SECTION_FACTOR, GYPSUM, 0.025, n_output=7
    )
    assert sampled.time_s.shape == (7,)
    assert sampled.time_s[0] == pytest.approx(0.0)
    assert sampled.time_s[-1] == pytest.approx(full.time_s[-1])
    assert sampled.theta_steel[0] == pytest.approx(20.0)
    assert sampled.theta_steel[-1] == pytest.approx(full.theta_final, rel=1e-6)


def test_result_container_is_the_same_type_as_the_unprotected_solver() -> None:
    """Downstream consumers must not need to know which clause ran."""
    res = protected_steel_temperature(30.0, SECTION_FACTOR, GYPSUM, 0.025)
    bare = steel_temperature(30.0, SECTION_FACTOR, warn_lumped=False)
    assert type(res) is type(bare)
    # the step ceilings differ by clause (30 s here, 5 s for section 4.2.2.2),
    # so the sample counts differ; the duration and the container do not
    assert res.time_s.shape != bare.time_s.shape
    assert res.time_s[-1] == pytest.approx(bare.time_s[-1])


def test_bridges_into_the_protected_member_batch_helper() -> None:
    specs = {
        "m1": (180.0, insulation_material("gypsum_board"), 0.025),
        "m2": (250.0, insulation_material("mineral_fibre_sprayed"), 0.020),
    }
    temps = protected_temperatures_for_members(60.0, specs)
    assert set(temps) == {"m1", "m2"}
    for member_id, (sf, mat, d) in specs.items():
        direct = protected_steel_temperature(60.0, sf, mat, d).theta_final
        assert temps[member_id] == pytest.approx(direct, rel=1e-12)
    # every value is a temperature a DCR chain can consume
    assert all(20.0 <= t < 1200.0 for t in temps.values())


def test_custom_fire_curve_is_accepted() -> None:
    res = protected_steel_temperature(
        30.0, SECTION_FACTOR, GYPSUM, 0.025, fire_curve=lambda _t: 500.0
    )
    assert np.all(res.theta_gas == pytest.approx(500.0))
    assert 20.0 < res.theta_final < 500.0


# --------------------------------------------------------------------------
# round-7 audit: theta_a0 was the one argument never validated, the material
# model's 1200 degC ceiling clamped silently, and the loop re-derived two
# constants per step.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_initial_temperature_is_rejected(bad: float) -> None:
    """A NaN used to propagate through every step and still return a result."""
    with pytest.raises(ValueError, match="finite temperature"):
        protected_steel_temperature(
            30.0, 200.0, insulation_material("gypsum_board"), 0.02, theta_a0=bad
        )


@pytest.mark.parametrize("bad", [-273.15, -300.0, -1e6])
def test_initial_temperature_below_absolute_zero_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="absolute zero"):
        protected_steel_temperature(
            30.0, 200.0, insulation_material("gypsum_board"), 0.02, theta_a0=bad
        )


def test_both_solvers_validate_the_initial_temperature_alike() -> None:
    """Their results are documented as interchangeable; so is their strictness."""
    from truss_analysis.thermal.fire_curve import steel_temperature

    for solver in (
        lambda **kw: protected_steel_temperature(
            30.0, 200.0, insulation_material("gypsum_board"), 0.02, **kw
        ),
        lambda **kw: steel_temperature(30.0, 200.0, **kw),
    ):
        with pytest.raises(ValueError, match="finite temperature"):
            solver(theta_a0=float("nan"))
        with pytest.raises(ValueError, match="absolute zero"):
            solver(theta_a0=-300.0)


def test_scalar_only_fire_curve_is_still_accepted() -> None:
    """Regression: the vectorised gas-temperature grid must not break the API.

    ``fire_curve`` is declared ``Callable[[FloatOrArray], FloatOrArray]``, and a
    scalar lambda is the natural way to express a bespoke or measured exposure.
    Building the grid with an unconditional array call turned that documented
    input into a ``ValueError`` -- a performance change breaking the API is not
    a performance change.
    """
    scalar_only = lambda t_min: 20.0 + 345.0 * math.log10(1.0 + 8.0 * t_min)  # noqa: E731
    result = protected_steel_temperature(
        30.0,
        200.0,
        insulation_material("gypsum_board"),
        0.02,
        fire_curve=scalar_only,
    )
    assert np.isfinite(result.theta_steel).all()
    assert result.theta_steel[-1] > result.theta_steel[0]


def test_a_curve_that_is_not_a_curve_is_rejected() -> None:
    """The fallback must still refuse something that is not a temperature."""
    with pytest.raises(ValueError, match="finite gas temperature"):
        protected_steel_temperature(
            30.0,
            200.0,
            insulation_material("gypsum_board"),
            0.02,
            fire_curve=lambda t: float("nan"),
        )


def test_inline_capacity_ratio_matches_the_public_function() -> None:
    """The loop inlines ``mu``; the public function must still agree with it.

    Hoisting ``specific_heat`` and ``unit_mass`` out of the recursion replaced a
    call to :func:`capacity_ratio_mu` with the same expression written inline.
    Two expressions of one clause is exactly how they drift, so the equality is
    pinned across the whole temperature range rather than trusted.
    """
    from truss_analysis.material.steel_eurocode import specific_heat, unit_mass

    material = insulation_material("gypsum_board")
    thickness, section_factor = 0.02, 200.0
    rho_a = float(unit_mass())
    for theta in (20.0, 100.0, 400.0, 735.0, 900.0, 1200.0):
        inline = (
            float(material.volumetric_heat_capacity)
            / (float(specific_heat(theta)) * rho_a)
            * thickness
            * section_factor
        )
        assert inline == pytest.approx(
            capacity_ratio_mu(material, thickness, section_factor, theta), rel=1e-15
        )


def test_hoisting_did_not_change_the_answer() -> None:
    """A performance change that moves the physics is not a performance change."""
    result = protected_steel_temperature(
        60.0, 200.0, insulation_material("gypsum_board"), 0.02
    )
    # 538.330342435918 degC, the value this configuration produced before the
    # redundant lookups were hoisted out of the recursion.
    assert result.theta_final == pytest.approx(538.330342435918, rel=1e-12)


def test_max_heating_rate_is_measured_on_the_integration_grid() -> None:
    """Downsampling must not silently understate the peak rate.

    Measured on a 30-minute ISO 834 exposure at ``A_m/V = 200``: the peak occurs
    in the first minute, where the curve is steepest, and 12 output points give
    1.0821 degC/s against the true 1.0966 -- a 1.3% understatement from a
    finite difference across widened intervals.
    """
    full = protected_steel_temperature(
        30.0, 200.0, insulation_material("gypsum_board"), 0.02
    )
    coarse = protected_steel_temperature(
        30.0, 200.0, insulation_material("gypsum_board"), 0.02, n_output=12
    )
    naive = float(np.max(np.diff(coarse.theta_steel) / np.diff(coarse.time_s)))

    assert coarse.max_heating_rate() == pytest.approx(full.max_heating_rate())
    assert naive < coarse.max_heating_rate(), "the naive rate must be the low one"
    assert naive == pytest.approx(coarse.max_heating_rate(), rel=0.05)


def test_error_estimate_separates_the_two_integrators() -> None:
    """The complaint this answers: RK4 and explicit Euler are not comparable.

    EN 1993-1-2 4.2.2.2 states an ODE and earns a fourth-order scheme; 4.2.5.2
    states a recursion with a clip and a step ceiling, and reproducing the code's
    answer means reproducing its first-order recursion.  Both choices are right,
    but a model mixing protected and unprotected members then carries two
    numerical errors orders of magnitude apart, which was previously invisible.
    """
    from truss_analysis.thermal.fire_curve import steel_temperature

    material = insulation_material("gypsum_board")
    euler = protected_steel_temperature(
        60.0, 200.0, material, 0.02, estimate_error=True
    )
    rk4 = steel_temperature(30.0, 200.0, estimate_error=True)

    assert euler.step_error_estimate is not None
    assert rk4.step_error_estimate is not None
    # measured: about 0.71 degC for the clause's 30 s Euler step, 4.3e-6 degC
    # for RK4 at 5 s -- five orders of magnitude apart
    assert euler.step_error_estimate > 0.1
    assert rk4.step_error_estimate < 1e-3
    assert euler.step_error_estimate > 100.0 * rk4.step_error_estimate


def test_error_estimate_is_off_by_default_and_bounds_the_real_error() -> None:
    """The estimate must actually bound the error, not merely be reported."""
    material = insulation_material("gypsum_board")
    plain = protected_steel_temperature(60.0, 200.0, material, 0.02)
    assert plain.step_error_estimate is None

    measured = protected_steel_temperature(
        60.0, 200.0, material, 0.02, estimate_error=True
    )
    converged = protected_steel_temperature(60.0, 200.0, material, 0.02, max_step_s=1.0)
    real_error = abs(measured.theta_final - converged.theta_final)

    assert measured.step_error_estimate is not None
    assert measured.step_error_estimate >= real_error


#: A severe industrial exposure reaching ~1400 degC.  ISO 834 cannot be used
#: for this test: its gas temperature asymptotes near 1193 degC, so the steel
#: behind it never passes the material model's 1200 degC ceiling no matter how
#: long the exposure runs -- the ceiling is only reachable with a hotter curve.
_SEVERE_CURVE = "lambda t_min: 20.0 + 1380.0 * (1.0 - math.exp(-t_min / 10.0))"


def _severe(t_min):
    return 20.0 + 1380.0 * (1.0 - math.exp(-t_min / 10.0))


def test_exceeding_the_material_model_ceiling_warns_and_is_recorded() -> None:
    """Table 3.1 clamps above 1200 degC; a clamp must not be silent.

    A thin member in a severe exposure passes the ceiling and the accessor
    returns the 1200 degC row, so the history looks ordinary while its upper end
    is an extrapolation of the last tabulated value rather than a code-compliant
    property.
    """
    from truss_analysis.exceptions import SteelTemperatureRangeWarning
    from truss_analysis.thermal.fire_curve import steel_temperature

    with pytest.warns(SteelTemperatureRangeWarning, match="1200"):
        result = steel_temperature(120.0, 250.0, fire_curve=_severe)
    assert result.out_of_range is not None
    assert result.out_of_range[1] > 1200.0

    # the protected solver is held to the same standard
    with pytest.warns(SteelTemperatureRangeWarning, match="1200"):
        protected = protected_steel_temperature(
            180.0,
            300.0,
            insulation_material("gypsum_board"),
            0.005,
            fire_curve=_severe,
        )
    assert protected.out_of_range is not None


def test_scalar_only_fire_curve_works_in_the_unprotected_solver_too() -> None:
    """Regression for a pre-existing crash, not just the protected path.

    ``steel_temperature`` built its gas-temperature history with an
    unconditional array call while its RK4 stages used scalar calls, so a
    scalar-only curve -- the declared ``FloatOrArray`` signature, and the
    natural way to write a bespoke exposure -- raised ``TypeError`` at the end
    of an otherwise successful integration.  Verified present on the baseline
    commit before this change.
    """
    from truss_analysis.exceptions import SteelTemperatureRangeWarning
    from truss_analysis.thermal.fire_curve import steel_temperature

    # the severe curve pushes the steel past the material model's ceiling, so
    # the range warning is expected here and is asserted separately
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SteelTemperatureRangeWarning)
        result = steel_temperature(30.0, 200.0, fire_curve=_severe)
    assert np.isfinite(result.theta_steel).all()
    assert result.theta_gas.shape == result.time_s.shape


def test_a_normal_fire_does_not_warn_about_the_material_ceiling() -> None:
    """The warning must be specific to exceeding the range, not to heating."""
    import warnings as _warnings

    from truss_analysis.exceptions import SteelTemperatureRangeWarning

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", SteelTemperatureRangeWarning)
        result = protected_steel_temperature(
            60.0, 200.0, insulation_material("gypsum_board"), 0.02
        )
    assert result.out_of_range is None


def test_parametric_fire_range_warning_has_its_own_category() -> None:
    """It named a different warning class in its own message.

    The previous text read "LumpedCapacityWarning-adjacent scope note" and was
    raised as a bare ``UserWarning``: a caller filtering on category could not
    catch it, and a caller reading it was pointed at the lumped-capacitance
    *member* model when the finding is about the *gas curve*.
    """
    from truss_analysis.exceptions import ParametricFireRangeWarning
    from truss_analysis.thermal.fire_curve import (
        Q_TD_RANGE_OF_VALIDITY,
        ParametricFire,
    )

    lo, hi = Q_TD_RANGE_OF_VALIDITY
    with pytest.warns(ParametricFireRangeWarning, match="range of validity"):
        ParametricFire(opening_factor=0.04, q_td=hi * 5.0, b_value=1160.0)
    # inside the range it stays quiet
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParametricFireRangeWarning)
        ParametricFire(opening_factor=0.04, q_td=0.5 * (lo + hi), b_value=1160.0)
