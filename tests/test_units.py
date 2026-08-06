import pytest
from truss_analysis.units import UnitConversionError, to_si


def test_to_si_imperial_length():
    assert abs(to_si(1.0, "Imperial", "L") - 0.3048) < 1e-6


def test_to_si_none():
    assert to_si(None, "SI", "L") is None


def test_to_si_invalid_quantity():
    with pytest.raises(UnitConversionError):
        to_si(1.0, "SI", "INVALID")
