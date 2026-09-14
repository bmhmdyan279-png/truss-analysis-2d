"""Section model tests: exact formulas, catalog hook, P_cr.

Hand-computed reference values are written as independent numeric literals
(hand solutions), not by re-calling the library formulas.
"""

from __future__ import annotations

import math

import pytest

from truss_analysis.sections import (
    SectionCatalog,
    SquareHSS,
    euler_buckling_load,
    idealised_square_hss,
)
from truss_analysis.topology_generator import TrussConfig, TrussFamily


def test_square_hss_exact_hand_values() -> None:
    sec = SquareHSS(name="hand", b=0.2, t=0.01)
    # A = 4 t (b - t) = 4 * 0.01 * 0.19
    assert sec.area == pytest.approx(0.0076, rel=1e-12)
    # I = (b^4 - (b-2t)^4)/12 = (0.0016 - 0.00104976)/12
    assert sec.i_sec == pytest.approx(4.585333333333333e-05, rel=1e-12)
    assert sec.width_to_thickness == pytest.approx(20.0, rel=1e-12)
    assert sec.radius_of_gyration == pytest.approx(
        math.sqrt(4.585333333333333e-05 / 0.0076)
    )


def test_square_hss_rejects_degenerate_shapes() -> None:
    with pytest.raises(ValueError, match="wall thickness must be > 0"):
        SquareHSS(name="bad", b=0.2, t=0.0)
    with pytest.raises(ValueError, match="leaves no void"):
        SquareHSS(name="bad", b=0.2, t=0.1)  # 2t == b -> solid/void-less
    with pytest.raises(ValueError, match="outer width must be > 0"):
        SquareHSS(name="bad", b=-0.2, t=0.01)


def test_idealised_roundtrip_area_and_ratio() -> None:
    sec = idealised_square_hss(0.01, thickness_ratio=25.0)
    assert sec.area == pytest.approx(0.01, rel=1e-12)
    assert sec.width_to_thickness == pytest.approx(25.0, rel=1e-12)
    # b = (r/2) sqrt(A/(r-1)) = 12.5 * sqrt(0.01/24)
    assert sec.b == pytest.approx(12.5 * math.sqrt(0.01 / 24.0), rel=1e-12)
    with pytest.raises(ValueError, match=r"thickness_ratio must be >= 2\.0"):
        idealised_square_hss(0.01, thickness_ratio=1.5)
    with pytest.raises(ValueError, match="area must be > 0"):
        idealised_square_hss(-0.01)


def test_config_i_sec_uses_idealised_hss_not_solid_square() -> None:
    cfg = TrussConfig(TrussFamily.WARREN, n_panels=4, span=16.0, height=3.0)
    assert cfg.i_sec == pytest.approx(cfg.section.i_sec, rel=1e-12)
    # the legacy solid-square formula must NOT be the value anymore
    assert cfg.i_sec != pytest.approx(cfg.area**2 / 12.0, rel=1e-3)
    assert cfg.section.area == pytest.approx(cfg.area, rel=1e-12)
    override = TrussConfig(
        TrussFamily.WARREN, n_panels=4, span=16.0, height=3.0, moment_of_inertia=2e-6
    )
    assert override.i_sec == 2e-6


def test_catalog_nearest_area() -> None:
    rows = [
        SquareHSS("s1", 0.1, 0.005),
        SquareHSS("s2", 0.2, 0.01),
        SquareHSS("s3", 0.3, 0.012),
    ]
    cat = SectionCatalog.from_rows("test", rows)
    assert cat.nearest_to_area(rows[1].area).name == "s2"
    assert cat.nearest_to_area(rows[2].area * 1.02).name == "s3"
    empty = SectionCatalog.from_rows("empty", [])
    with pytest.raises(ValueError, match="is empty"):
        empty.nearest_to_area(0.01)


def test_euler_buckling_against_hand_solution() -> None:
    # P_cr = pi^2 * 210e9 * 1e-6 / 2^2 = 518154.2310571913 N (hand arithmetic)
    assert euler_buckling_load(1e-6, 2.0, 210.0e9) == pytest.approx(
        518154.2310571913, rel=1e-9
    )
    # k = 0.5 halves the effective length -> four times the load
    assert euler_buckling_load(1e-6, 2.0, 210.0e9, 0.5) == pytest.approx(
        4 * 518154.2310571913, rel=1e-9
    )
    with pytest.raises(ValueError, match="length must be > 0"):
        euler_buckling_load(1e-6, 0.0, 210.0e9)


def test_no_solid_square_hollow_naming_anywhere() -> None:
    from pathlib import Path

    topo = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "truss_analysis"
        / "topology_generator.py"
    ).read_text(encoding="utf-8")
    assert "A**2 / 12" not in topo
    assert "square hollow-section approximation" not in topo
    sections = (
        Path(__file__).resolve().parents[1] / "src" / "truss_analysis" / "sections.py"
    ).read_text(encoding="utf-8")
    assert "idealised" in sections


# ---------------------------------------------------------------------
# Fire imperfection factor is defined for curve c only
# (EN 1993-1-2:2005 4.2.3.1(3); audit round 2, finding 5)
# ---------------------------------------------------------------------


def test_fire_reduction_on_curve_c_does_not_warn() -> None:
    import warnings

    from truss_analysis.sections import buckling_reduction_factor

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chi = buckling_reduction_factor(0.8, "c", fire=True)
    assert 0.0 < chi <= 1.0


@pytest.mark.parametrize("curve", ["a0", "a", "b", "d"])
def test_fire_reduction_off_curve_c_warns(curve: str) -> None:
    from truss_analysis.exceptions import BucklingCheckWarning
    from truss_analysis.sections import buckling_reduction_factor

    with pytest.warns(BucklingCheckWarning, match="extrapolation beyond the code"):
        chi = buckling_reduction_factor(0.8, curve, fire=True)
    assert 0.0 < chi <= 1.0


def test_ambient_curves_never_warn() -> None:
    import warnings

    from truss_analysis.sections import buckling_reduction_factor

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for curve in ("a0", "a", "b", "c", "d"):
            buckling_reduction_factor(0.8, curve, fire=False)
