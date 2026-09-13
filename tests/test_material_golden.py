"""Golden, physical and negative-regression tests for the EN 1993-1-2
single-source material module.

Every expected number here is either (a) read from the provenance fixture
(exact equality: every Table 3.1 point must match the fixture to the last
digit) or (b) recomputed inside the test from the fixture's own
coefficients using the standard's closed-form equations (figure 3.1,
eq. 3.1a-c, 3.2a-d, 3.3a-b).  No material number is hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from truss_analysis.material import steel_eurocode as ss
from truss_analysis.thermal.material import get_eurocode_k_E, get_eurocode_k_y

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TEST = REPO_ROOT / "tests" / "fixtures" / "en1993_1_2_table3_1.json"
FIXTURE_PKG = (
    REPO_ROOT
    / "src"
    / "truss_analysis"
    / "material"
    / "data"
    / "en1993_1_2_table3_1.json"
)

GRID = np.arange(20.0, 1200.01, 1.0)
TEMPS: np.ndarray


def setup_module(_: object) -> None:
    global TEMPS
    data = json.loads(FIXTURE_TEST.read_text())["table"]["temperatures_c"]
    TEMPS = np.asarray(data, float)


def _fixture() -> dict:
    return json.loads(FIXTURE_TEST.read_text())


# --------------------------------------------------------------------------
# provenance / integrity
# --------------------------------------------------------------------------
def test_fixture_copies_byte_identical() -> None:
    assert FIXTURE_PKG.read_bytes() == FIXTURE_TEST.read_bytes()


def test_fixture_carries_provenance_fields() -> None:
    data = _fixture()
    for field in (
        "source",
        "clause",
        "table",
        "extracted_at",
        "extracted_by",
        "cross_checked_against",
    ):
        assert field in data, field
    assert data["standard"]["primary_source_sha256"]
    assert data["table"]["clause"] == "3.2.1(3)"
    assert any(c["agreement"] == "exact" for c in data["cross_checked_against"])


# --------------------------------------------------------------------------
# golden: exact table points
# --------------------------------------------------------------------------
@pytest.mark.parametrize("idx", range(13))
def test_table_points_exact(idx: int) -> None:
    data = _fixture()["table"]
    theta = data["temperatures_c"][idx]
    assert ss.k_y(theta) == data["k_y"][idx]
    assert ss.k_p(theta) == data["k_p"][idx]
    assert ss.k_E(theta) == data["k_E"][idx]
    assert ss.k_s(theta) == data["k_E"][idx]  # alias: same tabulated quantity


def test_gate_values_from_prompt() -> None:
    assert ss.k_y(200) == 1.000
    assert ss.k_E(600) == 0.310
    assert ss.k_E(700) == 0.130


def test_k_y_unity_up_to_400() -> None:
    for theta in (20.0, 100.0, 200.0, 300.0, 400.0):
        assert ss.k_y(theta) == 1.0


def test_linear_interpolation_midpoints() -> None:
    data = _fixture()["table"]
    for key, fn in (("k_y", ss.k_y), ("k_p", ss.k_p), ("k_E", ss.k_E)):
        vals = data[key]
        for i in range(len(TEMPS) - 1):
            mid = (TEMPS[i] + TEMPS[i + 1]) / 2.0
            assert fn(mid) == pytest.approx((vals[i] + vals[i + 1]) / 2.0, abs=1e-15)


# --------------------------------------------------------------------------
# physical invariants
# --------------------------------------------------------------------------
def test_kE_le_kS_everywhere() -> None:
    assert np.all(ss.k_E(GRID) <= ss.k_s(GRID) + 0.0)


def test_monotonic_non_increasing() -> None:
    for fn in (ss.k_y, ss.k_p, ss.k_E):
        values = np.asarray(fn(GRID), float)
        assert np.all(np.diff(values) <= 0.0)


def test_bounds_zero_one() -> None:
    for fn in (ss.k_y, ss.k_p, ss.k_E):
        values = np.asarray(fn(GRID), float)
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)


def test_clamping_not_extrapolation_not_silent_zero() -> None:
    assert ss.k_E(-100.0) == ss.k_E(20.0) == 1.0
    assert ss.k_E(5000.0) == ss.k_E(1200.0) == 0.0
    # legacy bug regression: between 1000 and 1200 the old library returned 0.0
    assert ss.k_E(1100.0) == 0.0225
    assert ss.k_E(1050.0) == pytest.approx((0.045 + 0.0225) / 2.0)


def test_vector_scalar_equivalence() -> None:
    thetas = np.linspace(20.0, 1200.0, 97)
    for fn in (
        ss.k_y,
        ss.k_p,
        ss.k_E,
        ss.k_s,
        ss.alpha,
        ss.specific_heat,
        ss.thermal_conductivity,
        ss.eps_y,
        ss.eps_t,
        ss.eps_u,
    ):
        vec = np.asarray(fn(thetas), float)
        scalars = [fn(float(t)) for t in thetas]
        assert isinstance(scalars[0], float)
        np.testing.assert_allclose(vec, scalars, rtol=0.0, atol=0.0)
    # scalar inputs return floats
    assert isinstance(ss.k_E(600.0), float)
    assert isinstance(ss.alpha(600.0), float)


# --------------------------------------------------------------------------
# strain limits / derived quantities
# --------------------------------------------------------------------------
def test_strain_limits_are_figure31_constants() -> None:
    assert ss.eps_y(600.0) == 0.02
    assert ss.eps_t(600.0) == 0.15
    assert ss.eps_u(600.0) == 0.20
    vec = ss.eps_y(GRID)
    assert np.all(vec == 0.02)


def test_eps_p_derived_from_table() -> None:
    fy, ea = 235.0, 210000.0
    for theta in (20.0, 400.0, 600.0, 900.0):
        expected = ss.k_p(theta) * fy / (ss.k_E(theta) * ea)
        assert ss.eps_p(theta, f_y=fy, E_a=ea) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# thermal property closed forms (recomputed from fixture coefficients)
# --------------------------------------------------------------------------
def _piece_eval(piece: dict, t: float) -> float:
    coeffs = piece["coeffs"]
    if "const" in coeffs:
        return float(coeffs["const"])
    if "poly" in coeffs:
        return sum(float(c) * t**i for i, c in enumerate(coeffs["poly"]))
    den = coeffs["den"]
    return float(coeffs["c0"]) + float(coeffs["num"]) / (den[0] + den[1] * t)


@pytest.mark.parametrize(
    ("section", "fn"),
    [
        ("thermal_elongation", ss.alpha),
        ("specific_heat", ss.specific_heat),
        ("thermal_conductivity", ss.thermal_conductivity),
    ],
)
def test_thermal_pieces_match_fixture_formulas(section: str, fn) -> None:
    pieces = _fixture()["thermal"][section]["pieces"]
    for piece in pieces:
        lo, hi = piece["range_c"]
        lo_inc, hi_inc = piece["range_inclusive"]
        probes = [
            x
            for x in (
                lo + (0.0 if lo_inc else 0.5),
                (lo + hi) / 2.0,
                hi - (0.0 if hi_inc else 0.5),
            )
        ]
        for t in probes:
            expected = _piece_eval(piece, t)
            assert fn(t) == pytest.approx(expected, rel=1e-12), (section, t)


def test_elongation_piecewise_shape_and_standard_jump() -> None:
    # (3.1b) plateau
    assert ss.alpha(750.0) == 1.1e-2
    assert ss.alpha(860.0) == 1.1e-2
    # (3.1c) end point
    assert ss.alpha(1200.0) == pytest.approx(2.0e-5 * 1200.0 - 6.2e-3)
    # the standard itself is discontinuous at 750 degC by ~8.4e-6
    # ((3.1a) limit from below = 0.0110084 vs (3.1b) = 0.0110): measured, kept as-is
    below = 1.2e-5 * 750.0 + 0.4e-8 * 750.0**2 - 2.416e-4
    assert ss.alpha(749.9999999) == pytest.approx(below, rel=1e-9)
    assert below != pytest.approx(1.1e-2, abs=1e-9)
    # (3.1b)->(3.1c) IS continuous at 860
    assert ss.alpha(860.0000001) == pytest.approx(2.0e-5 * 860.0 - 6.2e-3, rel=1e-9)


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
def test_unit_mass_and_poisson() -> None:
    assert ss.unit_mass() == 7850.0
    assert ss.poisson_ratio() == 0.3


# --------------------------------------------------------------------------
# constitutive law (figure 3.1)
# --------------------------------------------------------------------------
def _law(theta: float, fy: float = 235.0, ea: float = 210000.0) -> dict:
    fy_t = ss.k_y(theta) * fy
    fp_t = ss.k_p(theta) * fy
    ea_t = ss.k_E(theta) * ea
    ep = fp_t / ea_t
    ey, et, eu = 0.02, 0.15, 0.20
    c = (fy_t - fp_t) ** 2 / ((ey - ep) * ea_t - 2.0 * (fy_t - fp_t))
    a = np.sqrt((ey - ep) * (ey - ep + c / ea_t))
    b = np.sqrt(c * (ey - ep) * ea_t + c**2)
    return {
        "fy": fy_t,
        "fp": fp_t,
        "ea": ea_t,
        "ep": ep,
        "ey": ey,
        "et": et,
        "eu": eu,
        "a": a,
        "b": b,
        "c": c,
    }


def test_stress_strain_branches_at_600() -> None:
    law = _law(600.0)
    # elastic
    eps = law["ep"] / 2.0
    assert ss.stress_strain(eps, 600.0) == pytest.approx(law["ea"] * eps, rel=1e-12)
    # elliptical midpoint recomputed from the standard's own formula
    eps_mid = (law["ep"] + law["ey"]) / 2.0
    expected = (
        law["fp"]
        - law["c"]
        + (law["b"] / law["a"]) * np.sqrt(law["a"] ** 2 - (law["ey"] - eps_mid) ** 2)
    )
    assert ss.stress_strain(eps_mid, 600.0) == pytest.approx(expected, rel=1e-12)
    # plateau and softening and end
    assert ss.stress_strain(0.05, 600.0) == pytest.approx(law["fy"], rel=1e-12)
    assert ss.stress_strain(0.175, 600.0) == pytest.approx(law["fy"] * 0.5, rel=1e-12)
    assert ss.stress_strain(0.2, 600.0) == 0.0


def test_stress_strain_continuity_at_branch_joins() -> None:
    for theta in (200.0, 600.0, 900.0):
        law = _law(theta)
        for join in (law["ep"], law["ey"], law["et"]):
            at = ss.stress_strain(join, theta)
            just_after = ss.stress_strain(float(np.nextafter(join, np.inf)), theta)
            assert at == pytest.approx(just_after, abs=1e-8)
        # branch values at the joins equal the physical anchors
        assert ss.stress_strain(law["ep"], theta) == pytest.approx(law["fp"], abs=1e-8)
        assert ss.stress_strain(law["ey"], theta) == pytest.approx(law["fy"], abs=1e-8)
        assert ss.stress_strain(law["et"], theta) == pytest.approx(law["fy"], abs=1e-8)


def test_stress_strain_inverse_roundtrip() -> None:
    for theta in (300.0, 600.0, 800.0):
        law = _law(theta)
        for eps in (
            law["ep"] * 0.5,
            law["ep"] * 0.999,
            (law["ep"] + law["ey"]) / 2.0,
            law["ey"] * 0.999,
        ):
            sigma = ss.stress_strain(eps, theta)
            back = ss.stress_strain(sigma, theta, branch="eps_of_sigma")
            assert back == pytest.approx(eps, rel=1e-9)
        # plateau inverse returns eps_y (documented convention)
        plateau_inv = ss.stress_strain(law["fy"], theta, branch="eps_of_sigma")
        assert plateau_inv == pytest.approx(law["ey"])


def test_stress_strain_degenerate_and_errors() -> None:
    # 20 degC: k_p == k_y -> elliptical branch collapses, sigma == f_y there
    law20 = _law(20.0)
    mid = (law20["ep"] + law20["ey"]) / 2.0
    assert ss.stress_strain(mid, 20.0) == pytest.approx(law20["fy"], rel=1e-12)
    # dead material at 1200 degC
    assert ss.stress_strain(0.01, 1200.0) == 0.0
    with pytest.raises(ValueError, match="zero strength"):
        ss.stress_strain(1.0, 1200.0, branch="eps_of_sigma")
    # above yield: no ascending-path pre-image
    with pytest.raises(ValueError, match=r"figure 3\.1"):
        ss.stress_strain(1e9, 600.0, branch="eps_of_sigma")
    with pytest.raises(ValueError, match="branch must be"):
        ss.stress_strain(0.01, 600.0, branch="nonsense")


def test_stress_strain_vectorised() -> None:
    eps = np.linspace(0.0, 0.2, 41)
    vec_out = ss.stress_strain(eps, 600.0)
    assert isinstance(vec_out, np.ndarray)
    vec = vec_out
    assert vec.shape == eps.shape
    scalars = [ss.stress_strain(float(e), 600.0) for e in eps]
    np.testing.assert_allclose(vec, scalars, rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------
# negative regression: the fabricated curves must NOT be reproduced
# --------------------------------------------------------------------------
FABRICATED = [
    # (function, theta, forbidden value, origin)
    (ss.k_E, 600.0, 0.10, "old library thermal/material.py"),
    (ss.k_E, 700.0, 0.05, "old library thermal/material.py"),
    (ss.k_E, 800.0, 0.02, "old library thermal/material.py"),
    (ss.k_E, 1000.0, 0.0, "old library thermal/material.py (truncation)"),
    (ss.k_E, 700.0, 0.23, "retrofit script fabricated k_E list"),
    (ss.k_E, 800.0, 0.11, "retrofit script fabricated k_E list"),
    (ss.k_E, 1000.0, 0.05, "retrofit script fabricated k_E list"),
    (ss.k_y, 200.0, 0.95, "retrofit script fabricated k_y list"),
    (ss.k_y, 400.0, 0.80, "retrofit script fabricated k_y list"),
    (ss.k_y, 600.0, 0.45, "retrofit script fabricated k_y list"),
    (ss.k_y, 1000.0, 0.02, "old library _EUROCODE_K_Y_VALUES tail"),
]


@pytest.mark.parametrize(("fn", "theta", "forbidden", "origin"), FABRICATED)
def test_fabricated_values_not_reproduced(
    fn, theta: float, forbidden: float, origin: str
) -> None:
    assert fn(theta) != forbidden, f"{origin} value resurrected at {theta} C"


def test_old_fake_vectors_differ_substantially() -> None:
    old_k_E = np.array([1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.0])
    new_k_E = np.asarray(ss.k_E(TEMPS[:11]), float)
    assert int(np.sum(~np.isclose(old_k_E, new_k_E))) >= 8


# --------------------------------------------------------------------------
# compatibility layer
# --------------------------------------------------------------------------
def test_compat_layer_delegates_with_warning() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert get_eurocode_k_E(600.0) == ss.k_E(600.0)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert get_eurocode_k_y(600.0) == ss.k_y(600.0)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert get_eurocode_k_E(1100.0) == 0.0225  # legacy 0.0 is gone
