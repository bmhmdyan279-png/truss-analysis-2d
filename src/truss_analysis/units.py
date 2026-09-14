"""Unit-system handling and conversion to SI.

The library solves every problem internally in SI base units (m, N, Pa, K).
Input files may declare ``"units": "Imperial"``; :func:`to_si` converts each
recognised quantity with fixed, exact-ratio factors.

Imperial input convention
-------------------------
The Imperial system in everyday structural use is **mixed**: lengths in feet,
moduli and stresses in psi, forces in lbf. Because those choices are not
self-consistent for mass, the density convention must be stated explicitly --
guessing produces a silent factor-of-32 error in self-weight.

This library reads ``density`` as **slug/ft^3**, the mass unit coherent with
``force = lbf`` and ``length = ft`` (``1 slug = 1 lbf s^2 / ft``). Structural
steel is therefore ``15.23 slug/ft^3``, *not* ``490`` -- ``490`` is its
specific weight in ``lbf/ft^3`` (pcf), and the two differ by exactly
``g = 32.174 ft/s^2``.

The alternative convention -- specific weight in ``lbf/ft^3`` (pcf), where
steel is ``490 pcf`` -- is equally common and differs by exactly standard
gravity. It is accepted under the separate key ``weight_density``, which
divides by ``g`` in both unit systems. Passing ``490`` as ``density`` is
detected and reported: see :func:`to_si`.

Supported quantity keys
-----------------------
``L`` (length), ``L2`` (area-like length^2), ``L4`` (second moment of area),
``A`` (cross-section area), ``I_sec`` (second moment of area), ``E``
(Young's modulus), ``F`` (force), ``delta_T`` (temperature difference),
``alpha`` (thermal expansion coefficient), ``density`` (mass density),
``weight_density`` (specific weight, converted to mass density).

Field-by-field Imperial input units
-----------------------------------
===============  ====================  ==========================  ==========
Input field      Imperial unit read    SI unit produced             Factor
===============  ====================  ==========================  ==========
``x``, ``y``     ft                    m                           0.3048
``A``            ft^2                  m^2                         0.092903
``I_sec``        ft^4                  m^4                         0.0086309
``E``            psi (lbf/in^2)        Pa                          6894.757
loads ``Fx/Fy``  lbf                   N                           4.44822
``delta_T``      deg F difference      K difference                5/9
``alpha``        1/deg F               1/K                         1.8
``density``      slug/ft^3             kg/m^3                      515.379
``weight_density`` lbf/ft^3 (pcf)      kg/m^3 (mass)               16.0185

===============  ====================  ==========================  ==========

``delta_T`` is a temperature *difference*, so no offset is applied: a 100 deg F
rise is a 55.56 K rise. Absolute temperatures are always supplied in degrees
Celsius and are not converted.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import overload

from .exceptions import UnitAmbiguityWarning, UnitConversionError

__all__ = ["UnitConversionError", "UnitSystem", "to_si"]

#: Standard gravity [m/s^2]. Used to turn a specific weight into a mass
#: density, which is the only form the self-weight loader consumes.
G_STANDARD = 9.80665

#: Plausible range of *mass* density [kg/m^3]. Air at sea level is 1.2,
#: timber ~400-900, structural steel 7850, and even osmium only reaches
#: 22 600. The upper bound is what catches the pcf trap: supplying 490 lbf/ft^3
#: as a mass density converts to 252 536 kg/m^3, which is 32x too large and
#: physically impossible. The lower bound only guards against nonsense.
_DENSITY_PLAUSIBLE_MIN = 0.5
_DENSITY_PLAUSIBLE_MAX = 30_000.0


class UnitSystem(Enum):
    """Recognised input unit systems."""

    SI = "SI"
    IMPERIAL = "Imperial"


_CONVERSION_FACTORS: dict[UnitSystem, dict[str, float]] = {
    UnitSystem.SI: {
        "L": 1.0,
        "L2": 1.0,
        "L4": 1.0,
        "A": 1.0,
        "I_sec": 1.0,
        "E": 1.0,
        "F": 1.0,
        "delta_T": 1.0,
        "alpha": 1.0,
        "density": 1.0,
        # N/m^3 -> kg/m^3 by dividing through by g.
        "weight_density": 1.0 / G_STANDARD,
    },
    UnitSystem.IMPERIAL: {
        "L": 0.3048,
        "L2": 0.092903,
        "L4": 0.0086309,
        "A": 0.092903,
        "I_sec": 0.0086309,
        "E": 6894.757,
        "F": 4.44822,
        "delta_T": 5.0 / 9.0,
        "alpha": 1.8,
        # slug/ft^3 -> kg/m^3. 1 slug = 14.5939 kg, 1 ft^3 = 0.0283168 m^3.
        "density": 515.379,
        # lbf/ft^3 (pcf) -> kg/m^3 mass density: multiply by 4.44822/0.0283168
        # to reach N/m^3, then divide by g. 1 lbm/ft^3 = 16.0185 kg/m^3.
        "weight_density": 16.0185,
    },
}


@overload
def to_si(value: float, unit_system: str, quantity: str) -> float: ...


@overload
def to_si(value: None, unit_system: str, quantity: str) -> None: ...


def to_si(value: float | None, unit_system: str, quantity: str) -> float | None:
    """Convert a scalar input value to SI units.

    Parameters
    ----------
    value : float or None
        Numeric value to convert. ``None`` passes through unchanged, which
        keeps optional input fields ergonomic for callers.
    unit_system : str
        Name of the source unit system (``"SI"`` or ``"Imperial"``).
    quantity : str
        Quantity key identifying the conversion factor (see the module
        docstring for the supported keys).

    Returns
    -------
    float or None
        ``value`` multiplied by the conversion factor, or ``None`` when
        ``value`` is ``None``.

    Raises
    ------
    UnitConversionError
        If ``unit_system`` or ``quantity`` is not recognised.

    Warns
    -----
    UnitAmbiguityWarning
        When an Imperial ``density`` converts to a mass density outside the
        plausible range for structural materials. The usual cause is
        supplying a specific weight in ``lbf/ft^3`` (pcf, steel = 490) where
        a mass density in ``slug/ft^3`` (steel = 15.23) is expected; the two
        differ by a factor of ``g`` and the error is otherwise silent.
    """
    if value is None:
        return None
    try:
        sys_enum = UnitSystem(unit_system)
    except ValueError as exc:
        raise UnitConversionError(f"Unknown unit system: {unit_system}") from exc
    factors = _CONVERSION_FACTORS.get(sys_enum)
    if not factors or quantity not in factors:
        raise UnitConversionError(f"Unknown quantity '{quantity}' for {unit_system}")
    converted = float(value) * factors[quantity]

    if (
        quantity == "density"
        and sys_enum is UnitSystem.IMPERIAL
        and converted > 0.0
        and not _DENSITY_PLAUSIBLE_MIN <= converted <= _DENSITY_PLAUSIBLE_MAX
    ):
        warnings.warn(
            f"Imperial density {value!r} converts to {converted:.6g} kg/m^3, "
            f"outside the plausible structural range "
            f"[{_DENSITY_PLAUSIBLE_MIN:g}, {_DENSITY_PLAUSIBLE_MAX:g}]. "
            f"'density' is read as slug/ft^3 (steel = 15.23); if you meant a "
            f"specific weight in lbf/ft^3 (steel = 490 pcf) use the "
            f"'weight_density' quantity instead.",
            UnitAmbiguityWarning,
            stacklevel=2,
        )
    return converted
