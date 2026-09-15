"""Tests for the heterogeneity module."""

import numpy as np
import pytest

from truss_analysis.heterogeneity import compute_heterogeneity


def test_heterogeneity_basic() -> None:
    margins = {
        "1": np.array([10.0, 10.0, 10.0]),
        "2": np.array([5.0, 5.0, 5.0]),
    }
    scf = {"1": 2.0, "2": 1.0}

    result = compute_heterogeneity(margins, scf, n_bootstrap=100, bootstrap_seed=42)

    # U = max(SRC)/min(SRC) = (2.0 * 1) / (1.0 * 1) = 2.0
    assert result.u_empirical_mean == pytest.approx(2.0)
    assert result.u_boot_mean_lower_95 == pytest.approx(2.0)
    assert result.h1_accepted is True
    assert len(result.unstable_members) == 0


def test_heterogeneity_unstable_member() -> None:
    margins = {
        "1": np.array([10.0, 10.0]),
        "2": np.array([-5.0, -5.0]),  # Mean < 0
    }
    scf = {"1": 1.0, "2": 1.0}

    result = compute_heterogeneity(margins, scf, n_bootstrap=100, bootstrap_seed=42)

    # Member 2 is unstable, uses absolute values. U = 1.0 / 1.0 = 1.0
    assert result.u_empirical_mean == pytest.approx(1.0)
    assert result.h1_accepted is False
    assert result.unstable_members == ["2"]


def test_heterogeneity_nan_handling() -> None:
    """Test that NaN values are handled gracefully without warnings.

    Member 2 has all NaN values, which should be filtered out and reported
    in warnings. The computation should proceed without numpy RuntimeWarnings
    by using ddof=0 when fewer than 2 valid samples exist.
    """
    import warnings

    margins = {
        "1": np.array([10.0, np.nan, 10.0]),
        "2": np.array([np.nan, np.nan, np.nan]),
    }
    scf = {"1": 1.0, "2": 1.0}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compute_heterogeneity(margins, scf, n_bootstrap=10, bootstrap_seed=42)

        runtime_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, RuntimeWarning)
            and (
                "Degrees of freedom" in str(warning.message)
                or "invalid value encountered" in str(warning.message)
            )
        ]
        msgs = [str(rw.message) for rw in runtime_warnings]
        assert len(runtime_warnings) == 0, f"Unexpected RuntimeWarnings: {msgs}"

    assert "Member 2" in result.warnings[0]
    assert result.unstable_members == []


def test_beta_hat_is_a_deprecated_alias_of_beta_mom() -> None:
    """C8: the rename warns rather than silently changing meaning.

    ``beta_hat`` read as a Hasofer-Lind / FORM reliability index; the
    quantity is a method-of-moments ``mu_g / std_g``.  The alias keeps old
    call sites alive for one release while telling them the name was wrong.
    """
    import warnings

    margins = {
        "1": np.array([10.0, 12.0, 8.0, 11.0]),
        "2": np.array([5.0, 4.0, 6.0, 5.5]),
    }
    result = compute_heterogeneity(
        margins, {"1": 1.0, "2": 1.0}, n_bootstrap=10, bootstrap_seed=7
    )
    with pytest.warns(DeprecationWarning, match="renamed to beta_mom"):
        legacy = result.beta_hat
    assert legacy == result.beta_mom
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert result.beta_mom == legacy
