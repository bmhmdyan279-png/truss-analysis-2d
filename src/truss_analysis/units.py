from __future__ import annotations

from enum import Enum


class UnitSystem(Enum):
    SI = "SI"
    IMPERIAL = "Imperial"


_CONVERSION_FACTORS = {
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
    },
}


class UnitConversionError(Exception):
    pass


def to_si(value, unit_system: str, quantity: str):
    if value is None:
        return None
    try:
        sys_enum = UnitSystem(unit_system)
    except ValueError:
        raise UnitConversionError(f"Unknown unit system: {unit_system}")
    factors = _CONVERSION_FACTORS.get(sys_enum)
    if not factors or quantity not in factors:
        raise UnitConversionError(f"Unknown quantity '{quantity}' for {unit_system}")
    return float(value) * factors[quantity]
