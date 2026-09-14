import pytest

from truss_analysis.units import UnitConversionError, to_si


def test_to_si_imperial_length():
    assert abs(to_si(1.0, "Imperial", "L") - 0.3048) < 1e-6


def test_to_si_none():
    assert to_si(None, "SI", "L") is None


def test_to_si_invalid_quantity():
    with pytest.raises(UnitConversionError):
        to_si(1.0, "SI", "INVALID")


# ---------------------------------------------------------------------------
# Density convention: the slug/ft^3 vs lbf/ft^3 trap
# ---------------------------------------------------------------------------


def test_imperial_density_is_slug_per_cubic_foot():
    """Steel's mass density is 15.23 slug/ft^3, converting to ~7850 kg/m^3."""
    converted = to_si(15.2315, "Imperial", "density")
    assert converted == pytest.approx(7850.0, rel=1e-4)


def test_imperial_weight_density_is_pcf():
    """Steel's specific weight is 490 pcf, converting to the same 7850 kg/m^3."""
    converted = to_si(490.0, "Imperial", "weight_density")
    assert converted == pytest.approx(7849.0, rel=1e-3)


def test_density_conventions_differ_by_gravity():
    """The two conventions are the same physical quantity scaled by g.

    The meaningful invariant is on the *conversion factors*: slug/ft^3 and
    lbf/ft^3 describe the same matter, so their factors must differ by
    exactly standard gravity expressed in ft/s^2 (32.1740).
    """
    factor_mass = to_si(1.0, "Imperial", "density")
    factor_weight = to_si(1.0, "Imperial", "weight_density")
    assert factor_mass / factor_weight == pytest.approx(32.1740, rel=1e-4)

    # and therefore steel gives the same kg/m^3 under either convention
    mass = to_si(490.0 / 32.1740, "Imperial", "density")
    weight = to_si(490.0, "Imperial", "weight_density")
    assert mass == pytest.approx(weight, rel=1e-4)
    assert weight == pytest.approx(7849.0, rel=1e-3)


def test_pcf_supplied_as_density_warns():
    """Passing 490 as a mass density is a silent 32x error, so it is flagged."""
    import warnings

    from truss_analysis.exceptions import UnitAmbiguityWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = to_si(490.0, "Imperial", "density")
    assert any(issubclass(w.category, UnitAmbiguityWarning) for w in caught)
    # and the number is indeed absurd: 250 tonnes per cubic metre
    assert value > 2e5


def test_plausible_imperial_density_does_not_warn():
    """A correct slug/ft^3 value must not raise a false alarm."""
    import warnings

    from truss_analysis.exceptions import UnitAmbiguityWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_si(15.2315, "Imperial", "density")  # steel, 7850 kg/m^3
        to_si(5.2389, "Imperial", "density")  # aluminium, 2700 kg/m^3
        to_si(0.00238, "Imperial", "density")  # air at sea level, 1.23 kg/m^3
    assert not any(issubclass(w.category, UnitAmbiguityWarning) for w in caught)


def test_si_weight_density_divides_by_gravity():
    """In SI, a specific weight in N/m^3 becomes a mass density in kg/m^3."""
    converted = to_si(78500.0, "SI", "weight_density")
    assert converted == pytest.approx(8005.0, rel=1e-3)


def test_si_density_is_identity():
    assert to_si(7850.0, "SI", "density") == pytest.approx(7850.0)


# ---------------------------------------------------------------------------
# Round-trip consistency for every quantity key
# ---------------------------------------------------------------------------

_EXPECTED_SI = {
    "L": 0.3048,
    "L2": 0.092903,
    "L4": 0.0086309,
    "A": 0.092903,
    "I_sec": 0.0086309,
    "E": 6894.757,
    "F": 4.44822,
    "delta_T": 5.0 / 9.0,
    "alpha": 1.8,
    "density": 515.379,
    "weight_density": 16.0185,
}


@pytest.mark.parametrize(("quantity", "factor"), sorted(_EXPECTED_SI.items()))
def test_imperial_factors_match_documented_values(quantity, factor):
    """Each factor must equal the value published in the module docstring."""
    assert to_si(1.0, "Imperial", quantity) == pytest.approx(factor, rel=1e-12)


@pytest.mark.parametrize("quantity", sorted(set(_EXPECTED_SI) - {"weight_density"}))
def test_si_conversion_is_identity_for_every_quantity(quantity):
    """SI input must pass through untouched for every dimension-preserving key.

    ``weight_density`` is excluded: it converts a specific weight to a mass
    density by dividing through by g, so it is not the identity even in SI.
    """
    assert to_si(3.25, "SI", quantity) == pytest.approx(3.25)


def test_delta_t_is_a_difference_not_an_absolute_temperature():
    """A 100 degF rise is a 55.56 K rise: no 32 degF offset may be applied."""
    assert to_si(100.0, "Imperial", "delta_T") == pytest.approx(55.5556, rel=1e-4)
    # An offset-based conversion would give (100-32)*5/9 = 37.78
    assert to_si(100.0, "Imperial", "delta_T") != pytest.approx(37.78, rel=1e-2)


def test_unknown_unit_system_raises():
    with pytest.raises(UnitConversionError):
        to_si(1.0, "Metric", "L")
