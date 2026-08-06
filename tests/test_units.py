from __future__ import annotations

import pytest
from truss_analysis.exceptions import UnitConversionError
from truss_analysis.units import to_si


def test_to_si_length():
    assert to_si(1.0, "SI", "L") == 1.0
    assert to_si(1000.0, "SI_MM", "L") == 1.0


def test_to_si_imperial_length():
    assert abs(to_si(1.0, "Imperial", "L") - 0.3048) < 1e-6


def test_to_si_imperial_dt():
    assert abs(to_si(1.0, "Imperial", "dT") - (5.0 / 9.0)) < 1e-6


def test_to_si_imperial_l4():
    assert abs(to_si(1.0, "Imperial", "L4") - 0.00863) < 1e-4


def test_to_si_none():
    assert to_si(None, "SI", "L") is None


def test_invalid_system():
    with pytest.raises(UnitConversionError):
        to_si(1.0, "Martian", "L")
