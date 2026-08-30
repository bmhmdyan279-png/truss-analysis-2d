"""Tests for Phase 6 Heterogeneity module."""

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
    margins = {
        "1": np.array([10.0, np.nan, 10.0]),
        "2": np.array([np.nan, np.nan, np.nan]),
    }
    scf = {"1": 1.0, "2": 1.0}

    result = compute_heterogeneity(margins, scf, n_bootstrap=10, bootstrap_seed=42)

    assert "Member 2" in result.warnings[0]
    assert result.unstable_members == []
