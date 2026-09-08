"""EN 1993-1-2:2005 steel material properties — the single source of truth.

All tabulated values, equation coefficients and constants are loaded at import
time from the provenance-carrying fixture
``data/en1993_1_2_table3_1.json`` (a byte-identical copy lives in
``tests/fixtures/`` and a CI test enforces the identity).  This module contains
no hard-coded material numbers.

Provenance (measured, see fixture ``cross_checked_against``):
  * Table 3.1 (clause 3.2.1(3), printed page 22): k_y, k_p, k_E at
    20..1200 degC in 100 K steps; linear interpolation per the table NOTE.
  * Figure 3.1 (clause 3.2.1, printed page 21): five-branch stress-strain
    relationship and strain limits eps_y=0.02, eps_t=0.15, eps_u=0.20.
  * Clause 3.4.1.1 (eq. 3.1a-c): relative thermal elongation dl/l.
  * Clause 3.4.1.2 (eq. 3.2a-d): specific heat.
  * Clause 3.4.1.3 (eq. 3.3a-b): thermal conductivity.
  * Clause 3.2.2: unit mass (7850 kg/m3, temperature independent).

Behaviour outside the tabulated range [20, 1200] degC: values are CLAMPED to
the range endpoints (documented, conservative); never extrapolated, never a
silent zero.

Notation notes (verified against the printed standard):
  * Table 3.1 defines exactly one elastic reduction factor, k_E,theta =
    E_a,theta / E_a (slope of the linear elastic range).  ``k_s`` is exposed as
    a documented alias of ``k_E`` for compatibility with notations that
    distinguish the two; they are the same tabulated quantity.
  * ``alpha(theta)`` returns the relative thermal elongation dl/l of eq.
    3.1a-c (the quantity the standard defines), not a constant coefficient.
  * Poisson's ratio is NOT defined by EN 1993-1-2:2005; ``poisson_ratio()``
    returns the EN 1993-1-1 engineering constant with explicit attribution.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "alpha",
    "eps_p",
    "eps_t",
    "eps_u",
    "eps_y",
    "k_E",
    "k_p",
    "k_s",
    "k_y",
    "poisson_ratio",
    "specific_heat",
    "stress_strain",
    "table",
    "table_temperatures",
    "thermal_conductivity",
    "unit_mass",
]

FloatOrArray = float | NDArray[np.float64]

_DATA_PATH = Path(__file__).resolve().parent / "data" / "en1993_1_2_table3_1.json"

_BRANCH_SIGMA_OF_EPS = "sigma_of_eps"
_BRANCH_EPS_OF_SIGMA = "eps_of_sigma"


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    """Load and cache the provenance fixture (immutable for the process)."""
    data: dict[str, Any] = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return data


def table() -> dict[str, Any]:
    """Return a deep copy of the Table 3.1 block of the fixture."""
    block: dict[str, Any] = json.loads(json.dumps(_data()["table"]))
    return block


def table_temperatures() -> NDArray[np.float64]:
    """Tabulated steel temperatures [degC] (20..1200 step 100)."""
    temps: list[float] = _data()["table"]["temperatures_c"]
    return np.asarray(temps, dtype=float)


def _is_scalar(value: FloatOrArray) -> bool:
    return np.ndim(value) == 0


def _clamped_theta(theta: FloatOrArray) -> NDArray[np.float64]:
    """Clamp temperature into the standard's validity range (documented)."""
    temps = table_temperatures()
    arr = np.atleast_1d(np.asarray(theta, dtype=float))
    clamped: NDArray[np.float64] = np.clip(arr, float(temps[0]), float(temps[-1]))
    return clamped


def _interp_column(theta: FloatOrArray, column: str) -> FloatOrArray:
    temps = table_temperatures()
    raw: list[float] = _data()["table"][column]
    vals = np.asarray(raw, dtype=float)
    arr = _clamped_theta(theta)
    out = np.interp(arr, temps, vals)  # np.interp clamps outside the range
    return float(out[0]) if _is_scalar(theta) else out


def k_y(theta: FloatOrArray) -> FloatOrArray:
    """Effective-yield-strength reduction factor k_y,theta (Table 3.1)."""
    return _interp_column(theta, "k_y")


def k_p(theta: FloatOrArray) -> FloatOrArray:
    """Proportional-limit reduction factor k_p,theta (Table 3.1)."""
    return _interp_column(theta, "k_p")


def k_E(theta: FloatOrArray) -> FloatOrArray:
    """Slope-of-linear-elastic-range reduction factor k_E,theta (Table 3.1).

    This is the factor that scales Young's modulus in structural fire
    analysis: E_a,theta = k_E,theta * E_a.  It is the quantity the truss
    stiffness matrix must be scaled with (used by the criticality engine).
    """
    return _interp_column(theta, "k_E")


def k_s(theta: FloatOrArray) -> FloatOrArray:
    """Alias of :func:`k_E`.

    EN 1993-1-2:2005 Table 3.1 tabulates a single elastic reduction factor
    (slope of the linear elastic range, k_E,theta = E_a,theta / E_a); there is
    no separate k_s column.  This alias exists only for notation
    compatibility and returns exactly ``k_E(theta)``.
    """
    return k_E(theta)


def _strain_limit(theta: FloatOrArray, key: str) -> FloatOrArray:
    value = float(_data()["strain_limits"]["values"][key])
    if _is_scalar(theta):
        return value
    return np.full_like(np.asarray(theta, dtype=float), value)


def eps_y(theta: FloatOrArray) -> FloatOrArray:
    """Yield strain eps_y,theta (figure 3.1 constant, temperature independent)."""
    return _strain_limit(theta, "eps_y")


def eps_t(theta: FloatOrArray) -> FloatOrArray:
    """Limiting strain for yield strength eps_t,theta (figure 3.1 constant)."""
    return _strain_limit(theta, "eps_t")


def eps_u(theta: FloatOrArray) -> FloatOrArray:
    """Ultimate strain eps_u,theta (figure 3.1 constant)."""
    return _strain_limit(theta, "eps_u")


def eps_p(
    theta: FloatOrArray,
    f_y: float | None = None,
    E_a: float | None = None,
) -> FloatOrArray:
    """Strain at proportional limit: eps_p,theta = f_p,theta / E_a,theta.

    ``f_y`` / ``E_a`` are the 20 degC values in consistent units (defaults:
    S235 / EN 1993-1-1 values from the fixture).  At temperatures where
    k_E == 0 the elastic slope vanishes and eps_p is infinite.
    """
    fy, ea = _default_constants(f_y, E_a)
    with np.errstate(divide="ignore"):
        val = (k_p(theta) * fy) / (k_E(theta) * ea)
    if _is_scalar(theta):
        return float(val)
    return val


def _default_constants(f_y: float | None, E_a: float | None) -> tuple[float, float]:
    ref = _data()["reference_constants_for_default_arguments"]
    return (
        float(ref["f_y_s235_mpa"]) if f_y is None else float(f_y),
        float(ref["E_a_20c_mpa"]) if E_a is None else float(E_a),
    )


def _poly(coeffs: list[float], t: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros_like(t)
    for power, coeff in enumerate(coeffs):
        out = out + coeff * t**power
    return out


def _piece_value(piece: dict[str, Any], t: NDArray[np.float64]) -> NDArray[np.float64]:
    coeffs = piece["coeffs"]
    if "const" in coeffs:
        return np.full_like(t, float(coeffs["const"]))
    if "poly" in coeffs:
        return _poly([float(c) for c in coeffs["poly"]], t)
    den = coeffs["den"]
    return np.full_like(t, float(coeffs["c0"])) + float(coeffs["num"]) / (
        float(den[0]) + float(den[1]) * t
    )


def _piecewise(theta: FloatOrArray, section: str) -> FloatOrArray:
    pieces = _data()["thermal"][section]["pieces"]
    arr = _clamped_theta(theta)
    conditions = []
    choices = []
    for piece in pieces:
        lo, hi = (float(x) for x in piece["range_c"])
        lo_inc, hi_inc = piece["range_inclusive"]
        cond = (arr >= lo) if lo_inc else (arr > lo)
        cond = cond & ((arr <= hi) if hi_inc else (arr < hi))
        conditions.append(cond)
        choices.append(_piece_value(piece, arr))
    out = np.select(conditions, choices, default=np.nan)
    if np.isnan(out).any():
        bad = arr[np.isnan(out)]
        msg = f"temperature {bad.ravel()[:3]!r} outside any piece of {section}"
        raise ValueError(msg)
    return float(out[0]) if _is_scalar(theta) else out


def alpha(theta: FloatOrArray) -> FloatOrArray:
    """Relative thermal elongation dl/l of carbon steel (clause 3.4.1.1).

    Piecewise per eq. (3.1a)-(3.1c); this is the strain quantity the standard
    defines (the framework's ``alpha(theta)`` API), not a constant expansion
    coefficient.
    """
    return _piecewise(theta, "thermal_elongation")


def specific_heat(theta: FloatOrArray) -> FloatOrArray:
    """Specific heat c_a [J/(kg K)] of carbon steel (clause 3.4.1.2)."""
    return _piecewise(theta, "specific_heat")


def thermal_conductivity(theta: FloatOrArray) -> FloatOrArray:
    """Thermal conductivity lambda_a [W/(m K)] (clause 3.4.1.3)."""
    return _piecewise(theta, "thermal_conductivity")


def unit_mass() -> float:
    """Temperature-independent unit mass rho_a [kg/m3] (clause 3.2.2)."""
    return float(_data()["thermal"]["unit_mass"]["value_kg_m3"])


def poisson_ratio() -> float:
    """Poisson's ratio nu [-].

    NOT defined by EN 1993-1-2:2005 (the standard contains no occurrence of
    'Poisson'); value taken from EN 1993-1-1 with explicit attribution in the
    fixture.  Provided for completeness of the material layer only.
    """
    return float(_data()["thermal"]["poisson_ratio"]["value"])


def _law_coefficients(
    theta: FloatOrArray,
    f_y: float,
    E_a: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return (fy_t, fp_t, ea_t, ep, a, b, c, ey, et, eu) at theta's shape."""
    t = _clamped_theta(theta)
    fy_t = np.atleast_1d(np.asarray(k_y(t), dtype=float)) * f_y
    fp_t = np.atleast_1d(np.asarray(k_p(t), dtype=float)) * f_y
    ea_t = np.atleast_1d(np.asarray(k_E(t), dtype=float)) * E_a
    ey = np.atleast_1d(np.asarray(eps_y(t), dtype=float))
    et = np.atleast_1d(np.asarray(eps_t(t), dtype=float))
    eu = np.atleast_1d(np.asarray(eps_u(t), dtype=float))
    dead = ea_t == 0.0
    safe_ea = np.where(dead, 1.0, ea_t)
    ep = fp_t / safe_ea
    denom = (ey - ep) * ea_t - 2.0 * (fy_t - fp_t)
    if bool(np.any((denom <= 0.0) & (fy_t > fp_t) & ~dead)):
        msg = "degenerate constitutive-law parameters (c denominator <= 0)"
        raise ValueError(msg)
    c = np.where(
        fy_t > fp_t, (fy_t - fp_t) ** 2 / np.where(denom > 0.0, denom, 1.0), 0.0
    )
    a = np.sqrt(np.clip((ey - ep) * (ey - ep + c / safe_ea), 0.0, None))
    b = np.sqrt(np.clip(c * (ey - ep) * ea_t + c**2, 0.0, None))
    return fy_t, fp_t, ea_t, ep, a, b, c, ey, et, eu


def stress_strain(
    x: FloatOrArray,
    theta: FloatOrArray,
    branch: str = _BRANCH_SIGMA_OF_EPS,
    f_y: float | None = None,
    E_a: float | None = None,
) -> FloatOrArray:
    """Five-branch stress-strain relationship of figure 3.1 (clause 3.2.1).

    Parameters
    ----------
    x:
        strain (dimensionless) when ``branch="sigma_of_eps"``, or stress (in
        the same unit as ``f_y``) when ``branch="eps_of_sigma"``.
    theta:
        steel temperature [degC]; clamped into [20, 1200].
    branch:
        ``"sigma_of_eps"`` (default) or ``"eps_of_sigma"``.
    f_y, E_a:
        20 degC yield strength and Young's modulus in consistent units
        (defaults from the fixture: S235 / EN 1993-1-1).

    Inverse convention (documented, the mapping is multi-valued):
    ``eps_of_sigma`` returns the strain on the ASCENDING path: elastic for
    sigma <= f_p,theta, elliptical for f_p < sigma < f_y, and eps_y on the
    plateau (sigma == f_y).  The descending (softening) pre-image is not
    returned; sigma > f_y,theta has no ascending-path pre-image and raises.
    At temperatures where all reduction factors vanish (k_E == 0) the material
    carries no stress: ``sigma_of_eps`` returns 0 while ``eps_of_sigma``
    raises ValueError.
    """
    fy, ea = _default_constants(f_y, E_a)
    if branch not in (_BRANCH_SIGMA_OF_EPS, _BRANCH_EPS_OF_SIGMA):
        msg = f"branch must be one of {(_BRANCH_SIGMA_OF_EPS, _BRANCH_EPS_OF_SIGMA)}"
        raise ValueError(msg)
    scalar = _is_scalar(x) and _is_scalar(theta)
    fy_t, fp_t, ea_t, ep, a, b, c, ey, et, eu = _law_coefficients(theta, fy, ea)
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    shape = np.broadcast(xv, fy_t).shape
    xv = np.broadcast_to(xv, shape)
    fy_t = np.broadcast_to(fy_t, shape)
    fp_t = np.broadcast_to(fp_t, shape)
    ea_t = np.broadcast_to(ea_t, shape)
    ep = np.broadcast_to(ep, shape)
    a = np.broadcast_to(a, shape)
    b = np.broadcast_to(b, shape)
    c = np.broadcast_to(c, shape)
    ey = np.broadcast_to(ey, shape)
    et = np.broadcast_to(et, shape)
    eu = np.broadcast_to(eu, shape)
    dead = ea_t == 0.0

    if branch == _BRANCH_SIGMA_OF_EPS:
        eps = xv
        safe_a = np.where(a > 0.0, a, 1.0)
        inner = np.clip(a**2 - (ey - eps) ** 2, 0.0, None)
        conds = [
            eps <= ep,
            (eps > ep) & (eps < ey),
            (eps >= ey) & (eps <= et),
            (eps > et) & (eps < eu),
            eps >= eu,
        ]
        choices = [
            ea_t * eps,
            fp_t - c + (b / safe_a) * np.sqrt(inner),
            fy_t,
            fy_t * (1.0 - (eps - et) / (eu - et)),
            np.zeros(shape),
        ]
        out = np.select(conds, choices, default=np.nan)
        out = np.where(dead, 0.0, out)
    else:
        if bool(np.any(dead)):
            msg = "material has zero strength at this temperature; inverse undefined"
            raise ValueError(msg)
        sig = xv
        safe_b = np.where(b > 0.0, b, 1.0)
        plateau = np.isclose(sig, fy_t, rtol=1e-12, atol=0.0)
        inner = np.clip(a**2 - (a * (sig - fp_t + c) / safe_b) ** 2, 0.0, None)
        conds = [
            (sig <= fp_t) & ~plateau,
            (sig > fp_t) & (sig < fy_t),
            plateau,
        ]
        choices = [sig / ea_t, ey - np.sqrt(inner), ey]
        out = np.select(conds, choices, default=np.nan)
    if bool(np.isnan(out).any()):
        bad = xv[np.isnan(out)]
        msg = f"input outside the figure 3.1 relationship: {bad.ravel()[:3]!r}"
        raise ValueError(msg)
    return float(out[0]) if scalar else out
