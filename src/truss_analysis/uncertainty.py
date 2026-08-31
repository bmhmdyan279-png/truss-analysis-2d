"""
Phase 1: Uncertainty Layer
Defines RandomVariable classes for Monte Carlo sampling.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml  # type: ignore[import-untyped]
from scipy import stats


class RandomVariable:
    """Base class for random variables with controlled seeds."""

    def __init__(
        self,
        mean: float,
        std: Optional[float] = None,
        cov: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.mean = mean

        # Handle parameters where mean can be zero (e.g., delta_T, delta_L0)
        if std is not None:
            self.std = std
            self.cov = std / mean if mean != 0 else float("inf")
        elif cov is not None:
            self.cov = cov
            self.std = abs(mean) * cov
        else:
            raise ValueError("Either 'std' or 'cov' must be provided.")

        self.rng = np.random.default_rng(seed)
        self._setup_distribution()

    def _setup_distribution(self) -> None:
        raise NotImplementedError("Subclasses must implement _setup_distribution")

    def sample(self, size: int) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement sample")


class NormalRV(RandomVariable):
    """Normal (Gaussian) distribution."""

    def _setup_distribution(self) -> None:
        self.dist = stats.norm(loc=self.mean, scale=self.std)

    def sample(self, size: int) -> np.ndarray:
        return self.dist.rvs(size=size, random_state=self.rng)


class LognormalRV(RandomVariable):
    """Lognormal distribution. Strictly positive (used for E)."""

    def _setup_distribution(self) -> None:
        # Mathematical mapping from (Mean, CoV) to SciPy's (s, scale)
        # zeta (sigma) = sqrt(ln(1 + cov^2))
        # lambda (mu) = ln(mean) - 0.5 * zeta^2
        zeta = np.sqrt(np.log(1 + self.cov**2))
        lambda_ = np.log(self.mean) - 0.5 * zeta**2
        self.dist = stats.lognorm(s=zeta, scale=np.exp(lambda_))

    def sample(self, size: int) -> np.ndarray:
        return self.dist.rvs(size=size, random_state=self.rng)


class GumbelRV(RandomVariable):
    """Gumbel (Type I Extreme Value) distribution for maxima (used for Loads)."""

    def _setup_distribution(self) -> None:
        # Mathematical mapping from (Mean, CoV) to SciPy's (loc, scale)
        # beta_n (scale) = std * sqrt(6) / pi
        # mu_n (loc) = mean - beta_n * euler_gamma
        beta_n = self.std * np.sqrt(6) / np.pi
        mu_n = self.mean - beta_n * np.euler_gamma
        self.dist = stats.gumbel_r(loc=mu_n, scale=beta_n)

    def sample(self, size: int) -> np.ndarray:
        return self.dist.rvs(size=size, random_state=self.rng)


def load_distributions_config(config_path: Union[str, Path]) -> dict:
    """Loads the YAML configuration for distribution mappings."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
