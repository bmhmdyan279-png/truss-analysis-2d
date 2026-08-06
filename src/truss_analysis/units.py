from __future__ import annotations

import enum

from .exceptions import UnitConversionError


class UnitSystem(enum.Enum):
    SI = "SI"
    IMPERIAL = "IMPERIAL"
    SI_MM = "SI_MM"


_CONVERSION_FACTORS = {
    UnitSystem.SI: {
        "L": 1.0,
        "A": 1.0,
        "I": 1.0,
        "E": 1.0,
        "delta_T": 1.0,
    },
    UnitSystem.IMPERIAL: {
        "L": 0.0254,  # in to m
        "A": 0.00064516,  # in² to m²
        "I": 4.162314e-7,  # in⁴ to m⁴
        "E": 6894.757,  # psi to Pa
        "delta_T": 5 / 9,  # °F to °C (change)
    },
    UnitSystem.SI_MM: {
        "L": 0.001,  # mm to m
        "A": 1e-6,  # mm² to m²
        "I": 1e-12,  # mm⁴ to m⁴
        "E": 1.0,
        "delta_T": 1.0,
    },
}


def to_si(value: float, system: str, quantity: str) -> float:
    try:
        sys_enum = UnitSystem(system)
    except ValueError:
        raise UnitConversionError(f"Unsupported unit system: {system}") from None
    if sys_enum not in _CONVERSION_FACTORS:
        raise UnitConversionError(f"No conversion factors for system: {system}")
    factors = _CONVERSION_FACTORS[sys_enum]
    if quantity not in factors:
        raise UnitConversionError(
            f"Unsupported quantity '{quantity}' for system {system}"
        )
    return value * factors[quantity]
