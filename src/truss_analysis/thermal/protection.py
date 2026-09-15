"""EN 1993-1-2:2005 §4.2.5.2 -- heating of an *insulated* steel member.

:func:`truss_analysis.thermal.fire_curve.steel_temperature` covers the
unprotected member of §4.2.2.2. This module covers the other half of the
clause set: a member wrapped in fire protection, where the insulation's
thickness, conductivity and -- unlike the bare-steel case -- its *heat
capacity* all enter, because the protection layer itself has to be heated
before the steel behind it warms.

The standard's recursion (clause 4.2.5.2(1)) is

.. code-block:: text

                 k_sh * lambda_p      A_p/V    theta_g,t - theta_a,t
    d(theta_a) = -------------- * --------- * --------------------- * dt
                     d_p          rho_a c_a        1 + mu/3

                 - (exp(mu/10) - 1) * d(theta_g)

    mu = (c_p rho_p) / (c_a rho_a) * d_p * (A_p/V)

with ``d(theta_a) >= 0`` imposed whenever ``d(theta_g) > 0`` -- the steel
never cools while the fire is still heating -- and a step limit of
``dt <= 30 s``.

Three things about this implementation are deliberate, and each is the kind
of choice a reviewer should be able to check rather than take on trust:

* **The integrator is explicit Euler, not RK4.** The unprotected solver in
  :mod:`~truss_analysis.thermal.fire_curve` uses RK4 because §4.2.2.2 states
  a continuous ODE. §4.2.5.2 states a *recursion*, complete with a
  non-negativity clip and a step ceiling; reproducing the code's answer means
  reproducing its recursion. Substituting a higher-order scheme would be more
  accurate and less correct -- it would no longer be what EN 1993-1-2 asks
  for, and a fire-resistance duration quoted against a different integrator
  than the code's is not a code-compliant duration.
* **``c_a = c_a(theta_a)`` and the steel properties come from the library's
  own EN 1993-1-2 single source of truth**
  (:mod:`truss_analysis.material.steel_eurocode`), so the protected and
  unprotected paths cannot disagree about what steel is.
* **``k_sh`` defaults to 1.0.** It multiplies the heating term, so the
  contour-protection value is the conservative one; a box-protected member
  without a cavity may reduce it per §4.2.5.2(2), which
  :func:`box_protection_shadow_factor` computes.

Scope -- stated plainly, per the project's physics-boundary policy:

* One homogeneous insulation layer. No cavity, no multi-layer build-up, no
  intumescent coating (whose effective thickness is a function of
  temperature and time and cannot be expressed by a constant ``d_p``).
* Constant ``lambda_p``, ``rho_p``, ``c_p``. In particular the endothermic
  dehydration plateau of gypsum near 100-200 degC is *not* represented, so a
  gypsum-protected member will be predicted to heat faster through that
  range than it really does. That is the safe direction, but it is an
  approximation and the material catalogue says so.
* The material catalogue shipped with the module is representative
  literature data, **not** product data. Real design uses the manufacturer's
  certified properties.

The output is directly interchangeable with the unprotected solver's: feed
``theta_final`` into the temperature field consumed by
:func:`~truss_analysis.limitstates.dcr_field` and the fire-resistance chain
is complete either way.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..material.steel_eurocode import (
    FloatOrArray,
    specific_heat,
    unit_mass,
)
from .fire_curve import SteelHeatingResult, iso_834_temperature

__all__ = [
    "MAX_STEP_S_PROTECTED",
    "InsulationMaterial",
    "box_protection_shadow_factor",
    "capacity_ratio_mu",
    "insulation_catalogue",
    "insulation_material",
    "protected_steel_temperature",
    "protected_temperatures_for_members",
]

_DATA_PATH = Path(__file__).resolve().parent / "data" / "protection_materials.json"

#: Step ceiling of clause 4.2.5.2: the recursion is only validated by the
#: standard for ``dt <= 30 s``. Enforced rather than advised, because a larger
#: step silently degrades a fire-resistance duration.
MAX_STEP_S_PROTECTED = 30.0


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    """Load the provenance-carrying protection-material fixture once."""
    raw: dict[str, Any] = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return raw


@dataclass(frozen=True)
class InsulationMaterial:
    """Thermal properties of one fire-protection material.

    Attributes
    ----------
    id : str
        Stable identifier (the catalogue key).
    name : str
        Human-readable name.
    lambda_p : float
        Thermal conductivity ``lambda_p`` [W/(m K)].
    rho_p : float
        Unit mass ``rho_p`` [kg/m^3].
    c_p : float
        Specific heat ``c_p`` [J/(kg K)].
    thickness_range_mm : tuple[float, float]
        Informative applied-thickness range from the source table; not a
        validity bound of the heating model.

    Notes
    -----
    These are representative literature values. Product-certified properties
    differ, are temperature dependent, and are what a real design must use --
    see the module docstring and the fixture's ``provenance.status``.
    """

    id: str
    name: str
    lambda_p: float
    rho_p: float
    c_p: float
    thickness_range_mm: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Reject a non-physical protection layer."""
        if self.lambda_p <= 0.0:
            msg = f"{self.id}: lambda_p must be > 0 W/(m K), got {self.lambda_p}"
            raise ValueError(msg)
        if self.rho_p <= 0.0:
            msg = f"{self.id}: rho_p must be > 0 kg/m^3, got {self.rho_p}"
            raise ValueError(msg)
        if self.c_p <= 0.0:
            msg = f"{self.id}: c_p must be > 0 J/(kg K), got {self.c_p}"
            raise ValueError(msg)

    @property
    def thermal_resistance_per_mm(self) -> float:
        """Return ``1 / lambda_p`` [(m K)/W per m of thickness]."""
        return 1.0 / float(self.lambda_p)

    @property
    def volumetric_heat_capacity(self) -> float:
        """Return ``rho_p * c_p`` [J/(m^3 K)], the numerator of ``mu``."""
        return float(self.rho_p) * float(self.c_p)


@lru_cache(maxsize=1)
def insulation_catalogue() -> dict[str, InsulationMaterial]:
    """Return the shipped protection-material catalogue.

    Returns
    -------
    dict[str, InsulationMaterial]
        Keyed by material id. Loaded from
        ``thermal/data/protection_materials.json``, which carries the
        provenance and the stated limitations.
    """
    rows = _data()["materials"]["rows"]
    out: dict[str, InsulationMaterial] = {}
    for row in rows:
        lo, hi = row.get("thickness_range_mm", [0.0, 0.0])
        out[str(row["id"])] = InsulationMaterial(
            id=str(row["id"]),
            name=str(row["name"]),
            lambda_p=float(row["lambda_p"]),
            rho_p=float(row["rho_p"]),
            c_p=float(row["c_p"]),
            thickness_range_mm=(float(lo), float(hi)),
        )
    return out


def insulation_material(material_id: str) -> InsulationMaterial:
    """Look one material up in the catalogue.

    Parameters
    ----------
    material_id : str
        Catalogue key, e.g. ``"gypsum_board"``.

    Returns
    -------
    InsulationMaterial
        The matching material.

    Raises
    ------
    KeyError
        If the id is unknown.  The message lists the available keys, because
        guessing an insulation product's conductivity from a similar-sounding
        name is exactly the mistake this lookup exists to prevent.
    """
    catalogue = insulation_catalogue()
    try:
        return catalogue[material_id]
    except KeyError:
        available = ", ".join(sorted(catalogue))
        msg = (
            f"unknown protection material {material_id!r}; available: "
            f"{available}. Supply an InsulationMaterial directly for a "
            "product not in the catalogue."
        )
        raise KeyError(msg) from None


def box_protection_shadow_factor(
    section_factor: float, section_factor_box: float
) -> float:
    """Shadow factor ``k_sh`` for box protection without a cavity.

    ``k_sh = 0.9 * (A_p/V)_box / (A_p/V)``, EN 1993-1-2 §4.2.5.2(2): a box
    encasement presents a smaller heated perimeter to the fire than the
    member's own contour, so the member heats more slowly than a contour-
    protected one of the same insulation.

    Parameters
    ----------
    section_factor : float
        Actual ``A_p/V`` of the protected member [1/m].
    section_factor_box : float
        ``A_p/V`` of the box formed by the protection [1/m]; by construction
        no larger than ``section_factor``.

    Returns
    -------
    float
        ``k_sh`` in (0, 1].

    Raises
    ------
    ValueError
        If either section factor is non-positive, or if the box factor
        exceeds the actual one -- which would mean the box has *more* heated
        perimeter than the member it encloses, i.e. the two were passed the
        wrong way round.
    """
    if section_factor <= 0.0:
        msg = f"section_factor must be > 0, got {section_factor}"
        raise ValueError(msg)
    if section_factor_box <= 0.0:
        msg = f"section_factor_box must be > 0, got {section_factor_box}"
        raise ValueError(msg)
    if section_factor_box > section_factor:
        msg = (
            f"box section factor {section_factor_box:g} exceeds the actual "
            f"{section_factor:g}; a box encasement cannot expose more heated "
            "perimeter than the member it encloses -- are the arguments "
            "reversed?"
        )
        raise ValueError(msg)
    k_sh = 0.9 * float(section_factor_box) / float(section_factor)
    return float(min(k_sh, 1.0))


def capacity_ratio_mu(
    material: InsulationMaterial,
    thickness_m: float,
    section_factor: float,
    theta_a: float,
) -> float:
    """Insulation-to-steel heat-capacity ratio ``mu`` [-], clause 4.2.5.2.

    .. code-block:: text

        mu = (c_p rho_p) / (c_a rho_a) * d_p * (A_p/V)

    ``mu`` is what makes the protected problem different from the bare-steel
    one: it is the ratio of the heat the insulation layer stores to the heat
    the steel behind it stores. Small ``mu`` means a light board that barely
    delays the steel; large ``mu`` means a heavy encasement that soaks up the
    fire, and the ``1 + mu/3`` denominator is precisely the term that
    accounts for it.

    ``mu`` is temperature dependent through ``c_a(theta_a)``, so it is
    recomputed at every step rather than evaluated once.

    Parameters
    ----------
    material : InsulationMaterial
        Protection layer properties.
    thickness_m : float
        Insulation thickness ``d_p`` [m]; must be positive.
    section_factor : float
        ``A_p/V`` [1/m]; must be positive.
    theta_a : float
        Current steel temperature [degC]; selects ``c_a``.

    Returns
    -------
    float
        The dimensionless capacity ratio.

    Raises
    ------
    ValueError
        If ``thickness_m`` or ``section_factor`` is not positive.
    """
    if thickness_m <= 0.0:
        msg = f"insulation thickness must be > 0 m, got {thickness_m}"
        raise ValueError(msg)
    if section_factor <= 0.0:
        msg = f"section_factor (A_p/V) must be > 0, got {section_factor}"
        raise ValueError(msg)
    c_a = float(specific_heat(theta_a))
    rho_a = float(unit_mass())
    if c_a <= 0.0 or rho_a <= 0.0:  # pragma: no cover - the table is positive
        return float("inf")
    return float(
        (material.volumetric_heat_capacity / (c_a * rho_a))
        * thickness_m
        * section_factor
    )


def protected_steel_temperature(
    duration_min: float,
    section_factor: float,
    material: InsulationMaterial,
    thickness_m: float,
    fire_curve: Callable[[FloatOrArray], FloatOrArray] = iso_834_temperature,
    theta_a0: float = 20.0,
    shadow_factor: float = 1.0,
    max_step_s: float = MAX_STEP_S_PROTECTED,
    n_output: int = 0,
) -> SteelHeatingResult:
    """Steel temperature history of an insulated member (clause 4.2.5.2).

    Parameters
    ----------
    duration_min : float
        Fire exposure duration [minutes]; must be > 0.
    section_factor : float
        ``A_p/V`` [1/m] of the protected member: heated perimeter of the
        protection over steel volume. Must be positive.
    material : InsulationMaterial
        Protection properties (``lambda_p``, ``rho_p``, ``c_p``).
    thickness_m : float
        Insulation thickness ``d_p`` [m]; must be positive.
    fire_curve : Callable, default ``iso_834_temperature``
        Maps time [minutes] to gas temperature [degC]. Any curve works, so
        :class:`~truss_analysis.thermal.fire_curve.ParametricFire` and
        measured data are both acceptable.
    theta_a0 : float, default 20.0
        Initial steel temperature [degC].
    shadow_factor : float, default 1.0
        ``k_sh`` in (0, 1]; see :func:`box_protection_shadow_factor`.
    max_step_s : float, default 30.0
        Integration step [s]; the clause's ceiling is 30 s and a larger value
        is rejected rather than silently accepted.
    n_output : int, default 0
        Number of evenly spaced output samples (including ``t = 0`` and the
        end); ``0`` keeps every step.

    Returns
    -------
    SteelHeatingResult
        Time, steel-temperature and gas-temperature histories -- the same
        container the unprotected solver returns, so downstream consumers do
        not need to know which clause produced the field.

    Raises
    ------
    ValueError
        If ``duration_min`` <= 0, ``section_factor`` <= 0, ``thickness_m``
        <= 0, ``shadow_factor`` is outside (0, 1], or ``max_step_s`` is
        outside (0, 30].

    Notes
    -----
    The recursion is explicit Euler with the clause's non-negativity clip:
    ``d(theta_a)`` is floored at zero whenever the gas temperature is rising,
    so a protected member never predicts cooling during the heating phase.
    During a *decaying* fire (a parametric curve past its peak) the clip does
    not apply and the member does cool, which is physical.
    """
    if duration_min <= 0.0:
        msg = f"duration_min must be > 0, got {duration_min}"
        raise ValueError(msg)
    if section_factor <= 0.0:
        msg = f"section_factor (A_p/V) must be > 0, got {section_factor}"
        raise ValueError(msg)
    if thickness_m <= 0.0:
        msg = f"insulation thickness must be > 0 m, got {thickness_m}"
        raise ValueError(msg)
    if not 0.0 < shadow_factor <= 1.0:
        msg = f"shadow_factor k_sh must lie in (0, 1], got {shadow_factor}"
        raise ValueError(msg)
    if not 0.0 < max_step_s <= MAX_STEP_S_PROTECTED:
        msg = (
            f"max_step_s must lie in (0, {MAX_STEP_S_PROTECTED:g}] s -- the "
            f"step ceiling of EN 1993-1-2 clause 4.2.5.2; got {max_step_s}"
        )
        raise ValueError(msg)

    rho_a = float(unit_mass())
    duration_s = float(duration_min) * 60.0
    n_steps = max(1, int(np.ceil(duration_s / max_step_s)))
    dt = duration_s / n_steps

    ts = np.linspace(0.0, duration_s, n_steps + 1)
    thetas = np.empty(n_steps + 1, dtype=float)
    theta_gas = np.empty(n_steps + 1, dtype=float)

    theta_a = float(theta_a0)
    theta_g = float(fire_curve(0.0))
    thetas[0] = theta_a
    theta_gas[0] = theta_g

    # Constant part of the heating coefficient; c_a(theta_a) is the only
    # temperature-dependent factor and is evaluated per step.
    lam_over_d = float(material.lambda_p) / float(thickness_m)
    for i in range(n_steps):
        t_next_min = ts[i + 1] / 60.0
        theta_g_next = float(fire_curve(t_next_min))
        d_theta_g = theta_g_next - theta_g

        c_a = float(specific_heat(theta_a))
        mu = capacity_ratio_mu(material, thickness_m, section_factor, theta_a)
        heating = (
            shadow_factor
            * lam_over_d
            * section_factor
            / (rho_a * c_a)
            * (theta_g - theta_a)
            / (1.0 + mu / 3.0)
            * dt
        )
        d_theta_a = heating - (np.exp(mu / 10.0) - 1.0) * d_theta_g

        # clause 4.2.5.2: the steel does not cool while the fire is heating
        if d_theta_g > 0.0 and d_theta_a < 0.0:
            d_theta_a = 0.0

        theta_a += float(d_theta_a)
        theta_g = theta_g_next
        thetas[i + 1] = theta_a
        theta_gas[i + 1] = theta_g

    if n_output > 0:
        idx = (
            np.linspace(0, n_steps, min(int(n_output), n_steps + 1)).round().astype(int)
        )
        ts, thetas, theta_gas = ts[idx], thetas[idx], theta_gas[idx]

    return SteelHeatingResult(
        time_s=np.asarray(ts, dtype=float),
        theta_steel=np.asarray(thetas, dtype=float),
        theta_gas=np.asarray(theta_gas, dtype=float),
    )


def protected_temperatures_for_members(
    duration_min: float,
    specs: Mapping[str, tuple[float, InsulationMaterial, float]],
    fire_curve: Callable[[FloatOrArray], FloatOrArray] = iso_834_temperature,
    theta_a0: float = 20.0,
    shadow_factor: float = 1.0,
) -> dict[str, float]:
    """Return the final steel temperature per member, ready for the DCR chain.

    Convenience wrapper turning a per-member protection specification into
    the ``{member_id: theta_a}`` mapping that
    :func:`~truss_analysis.limitstates.dcr_field` and the criticality engine
    consume, so a protected fire design reaches the structural side without
    hand-assembling a dictionary.

    Parameters
    ----------
    duration_min : float
        Common exposure duration [minutes].
    specs : Mapping[str, tuple[float, InsulationMaterial, float]]
        ``member_id -> (A_p/V [1/m], material, thickness [m])``.
    fire_curve : Callable, optional
        Gas-temperature curve; see :func:`protected_steel_temperature`.
    theta_a0 : float, default 20.0
        Initial steel temperature [degC].
    shadow_factor : float, default 1.0
        Common ``k_sh`` for every member.

    Returns
    -------
    dict[str, float]
        Final steel temperature [degC] per member id.
    """
    return {
        member_id: protected_steel_temperature(
            duration_min,
            sf,
            material,
            thickness,
            fire_curve=fire_curve,
            theta_a0=theta_a0,
            shadow_factor=shadow_factor,
        ).theta_final
        for member_id, (sf, material, thickness) in specs.items()
    }
