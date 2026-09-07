"""Random variables with recorded, verification-checked citations.

Legacy classes (``NormalRV``, ``LognormalRV``, ``GumbelRV``) keep their exact
pre-prompt-6 behaviour; new additions:

* ``TruncatedNormalRV`` and ``DeterministicRV``;
* ``DistributionSpec`` + :func:`proposal_rv_specs`: the proposal §5.4 random
  variable table with a **citation status** for every entry.  Citation
  verification was performed on 2026-09-07 (see
  ``vault/reports/06_citation_checks.md``); unverified clause numbers are NOT
  asserted in code — they are downgraded to proposal-internal screening
  choices, per the project honesty rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml  # type: ignore[import-untyped]
from scipy import stats

__all__ = [
    "SPECIES_CITATION_STATUS",
    "DeterministicRV",
    "DistributionSpec",
    "GumbelRV",
    "LognormalRV",
    "NormalRV",
    "TruncatedNormalRV",
    "load_distributions_config",
    "proposal_rv_specs",
]


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


class TruncatedNormalRV(RandomVariable):
    """Normal distribution truncated to ``[low, high]`` (fire intensity)."""

    def __init__(
        self,
        mean: float,
        std: float,
        low: float,
        high: float,
        seed: Optional[int] = None,
    ) -> None:
        if not low < high:
            msg = f"require low < high, got [{low}, {high}]"
            raise ValueError(msg)
        self.low = low
        self.high = high
        super().__init__(mean=mean, std=std, seed=seed)

    def _setup_distribution(self) -> None:
        a = (self.low - self.mean) / self.std
        b = (self.high - self.mean) / self.std
        self.dist = stats.truncnorm(a=a, b=b, loc=self.mean, scale=self.std)

    def sample(self, size: int) -> np.ndarray:
        return self.dist.rvs(size=size, random_state=self.rng)


class DeterministicRV:
    """Degenerate 'random' variable: a fixed value (e.g. E per EN 1993-1-1)."""

    def __init__(self, value: float, seed: Optional[int] = None) -> None:
        del seed  # determinism needs no randomness
        self.value = float(value)
        self.mean = self.value
        self.std = 0.0
        self.cov = 0.0

    def sample(self, size: int) -> np.ndarray:
        return np.full(size, self.value)


def load_distributions_config(config_path: Union[str, Path]) -> dict:
    """Loads the YAML configuration for distribution mappings."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class DistributionSpec:
    """One row of the proposal §5.4 random-variable table, with provenance."""

    name: str
    family: str
    parameters: dict
    citation: str
    #: one of: verified | verified-corrected | family-verified |
    #: proposal-internal | not-found (see 06_citation_checks.md)
    citation_status: str
    note: str


#: Citation statuses recorded during the 2026-09-07 verification session.
#: Full evidence trail: ``vault/reports/06_citation_checks.md``.
SPECIES_CITATION_STATUS = {
    "live_load": "family-verified / clause-not-found",
    "f_y": "family-verified / value-proposal",
    "fire_intensity": "proposal-internal",
    "E": "verified-corrected",
}


def proposal_rv_specs(fire_scenario_temperature: float = 600.0) -> tuple:
    """The proposal §5.4 random-variable table with verified citations.

    Verification outcomes (2026-09-07, see ``06_citation_checks.md``):

    * live load — Gumbel family for imposed loads is confirmed by secondary
      literature on the JCSS Probabilistic Model Code; the exact clause
      "4.1.2" and COV 0.20 could not be verified verbatim, so the COV is
      recorded as a proposal screening choice (NOT attributed to JCSS).
    * f_y — lognormal family is supported by EN 1990:2002 (which discusses
      lognormal resistance variables but gives no COV); COV 0.05 is a proposal
      screening choice consistent with JCSS resistance-model ranges.
    * fire intensity — truncated normal is an internal modelling choice
      (proposal §5.4); no external claim is made.
    * E — EN 1993-1-1:2005 clause 3.2.6 verified verbatim: E = 210 000
      N/mm²; the proposal's "200 GPa" is corrected here.
    """
    return (
        DistributionSpec(
            name="live_load",
            family="gumbel",
            parameters={"cov": 0.20},
            citation=(
                "Gumbel family for imposed loads: JCSS Probabilistic "
                "Model Code (load models), secondary-verified "
                "2026-09-07; clause '4.1.2' and COV value NOT "
                "verbatim-verifiable -> COV = proposal §5.4 "
                "screening choice"
            ),
            citation_status=SPECIES_CITATION_STATUS["live_load"],
            note="multiplicative load scaler, mean 1.0",
        ),
        DistributionSpec(
            name="f_y",
            family="lognormal",
            parameters={"cov": 0.05},
            citation=(
                "lognormal family: EN 1990:2002 (lognormal suited to resistance "
                "variables; no COV given there); COV 0.05 = proposal §5.4 screening "
                "choice (JCSS resistance models: ~0.05-0.07 for steel yield)"
            ),
            citation_status=SPECIES_CITATION_STATUS["f_y"],
            note="yield strength [Pa], mean = grade nominal",
        ),
        DistributionSpec(
            name="fire_intensity",
            family="truncated_normal",
            parameters={
                "mu": fire_scenario_temperature,
                "sigma": 50.0,
                "low": 20.0,
                "high": 1000.0,
            },
            citation="proposal §5.4 (internal modelling choice; no external claim)",
            citation_status=SPECIES_CITATION_STATUS["fire_intensity"],
            note="member/scenario temperature [degC]",
        ),
        DistributionSpec(
            name="E",
            family="deterministic",
            parameters={"value": 210.0e9},
            citation=(
                "EN 1993-1-1:2005 clause 3.2.6: E = 210 000 N/mm2 "
                "(verified verbatim 2026-09-07; proposal value "
                "200 GPa corrected)"
            ),
            citation_status=SPECIES_CITATION_STATUS["E"],
            note="elastic modulus [Pa], deterministic",
        ),
    )
