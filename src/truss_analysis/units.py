from __future__ import annotations

from enum import Enum

from .exceptions import UnitConversionError


class UnitSystem(Enum):
    SI = "SI"
    SI_MM = "SI-mm"
    IMPERIAL = "Imperial"


FACTORS = {
    UnitSystem.SI: {"L": 1.0, "L2": 1.0, "L4": 1.0, "F": 1.0, "dT": 1.0},
    UnitSystem.SI_MM: {"L": 1e-3, "L2": 1e-6, "L4": 1e-12, "F": 1.0, "dT": 1.0},
    UnitSystem.IMPERIAL: {
        "L": 0.3048,
        "L2": 0.092903,
        "L4": 0.008630,
        "F": 4.44822,
        "dT": 5.0 / 9.0,
    },
}


def to_si(value: float | None, system: str | UnitSystem, qty: str) -> float | None:
    if value is None:
        return None
    if isinstance(system, str):
        system = UnitSystem(system)
    try:
        return float(value) * FACTORS[system][qty]
    except (KeyError, ValueError) as e:
        raise UnitConversionError(f"تبدیل واحد نامعتبر: {e}") from None
