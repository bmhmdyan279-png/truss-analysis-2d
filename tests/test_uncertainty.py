"""
Tests for Phase 1: Uncertainty Layer
Validates statistical convergence of empirical samples against theoretical parameters.
"""

import numpy as np
import pytest
from truss_analysis.uncertainty import (
    GumbelRV,
    LognormalRV,
    NormalRV,
    RandomVariable,
    load_distributions_config,
)

# Tolerances based on Phase 1 requirements (2% for mean, 5% for std)
MEAN_TOL = 0.02
STD_TOL = 0.05
N_SAMPLES = 10000
SEED = 42


def check_convergence(
    rv: RandomVariable, theoretical_mean: float, theoretical_std: float
) -> np.ndarray:
    samples = rv.sample(N_SAMPLES)
    emp_mean = np.mean(samples)
    emp_std = np.std(samples, ddof=1)

    # Avoid division by zero if theoretical_mean is 0
    if theoretical_mean != 0:
        assert abs(emp_mean - theoretical_mean) / abs(theoretical_mean) < MEAN_TOL
    else:
        assert abs(emp_mean - theoretical_mean) < 1e-5

    assert abs(emp_std - theoretical_std) / theoretical_std < STD_TOL
    return samples


def test_normal_rv() -> None:
    mean = 100.0
    std = 5.0
    rv = NormalRV(mean=mean, std=std, seed=SEED)
    check_convergence(rv, mean, std)


def test_lognormal_rv() -> None:
    mean = 200e9
    cov = 0.05
    rv = LognormalRV(mean=mean, cov=cov, seed=SEED)
    std = mean * cov
    samples = check_convergence(rv, mean, std)
    assert np.all(samples > 0), "Lognormal samples must be strictly positive"


def test_gumbel_rv() -> None:
    mean = 10000.0
    cov = 0.15
    rv = GumbelRV(mean=mean, cov=cov, seed=SEED)
    std = mean * cov
    check_convergence(rv, mean, std)


def test_zero_mean_normal() -> None:
    # Crucial for parameters like delta_T or delta_L0 that can be zero
    mean = 0.0
    std = 0.001
    rv = NormalRV(mean=mean, std=std, seed=SEED)
    samples = rv.sample(1000)
    assert abs(np.mean(samples) - mean) < 1e-4
    assert abs(np.std(samples, ddof=1) - std) / std < STD_TOL


def test_cov_vs_std_initialization() -> None:
    rv1 = NormalRV(mean=100.0, std=5.0, seed=SEED)
    rv2 = NormalRV(mean=100.0, cov=0.05, seed=SEED)

    samples1 = rv1.sample(100)
    samples2 = rv2.sample(100)

    assert np.allclose(samples1, samples2)


def test_missing_std_and_cov_raises() -> None:
    with pytest.raises(ValueError):
        NormalRV(mean=100.0, seed=SEED)


def test_yaml_config_loading(tmp_path) -> None:
    config_content = """
parameters:
  E:
    type: lognormal
    mean: 200.0e9
    cov: 0.05
  delta_T:
    type: normal
    mean: 0.0
    std: 5.0
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")

    config = load_distributions_config(config_file)
    assert "parameters" in config
    assert config["parameters"]["E"]["type"] == "lognormal"
    assert config["parameters"]["delta_T"]["std"] == 5.0
