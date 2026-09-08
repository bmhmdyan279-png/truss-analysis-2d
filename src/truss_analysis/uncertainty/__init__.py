"""Uncertainty layer: random variables, sampling, streaming statistics.

The legacy public API of the former single-module interface (``NormalRV``,
``LognormalRV``, ``GumbelRV``, ``load_distributions_config``) is preserved
verbatim in :mod:`.random_variables` and re-exported here, alongside the
sampling utilities (:mod:`.sampling`), the memory-bounded streaming
accumulator (:mod:`.streaming`) and the probabilistic ranking operator
(:mod:`.probabilistic_ranking`).
"""

from __future__ import annotations

from .probabilistic_ranking import ProbabilisticRanking, probabilistic_ranking
from .random_variables import (
    SPECIES_CITATION_STATUS,
    DeterministicRV,
    DistributionSpec,
    GumbelRV,
    LognormalRV,
    NormalRV,
    RandomVariable,
    TruncatedNormalRV,
    default_rv_specs,
    load_distributions_config,
)
from .sampling import (
    gaussian_copula_correlate,
    latin_hypercube,
    sample_spec_matrix,
)
from .streaming import RunningStat

__all__ = [
    "SPECIES_CITATION_STATUS",
    "DeterministicRV",
    "DistributionSpec",
    "GumbelRV",
    "LognormalRV",
    "NormalRV",
    "ProbabilisticRanking",
    "RandomVariable",
    "RunningStat",
    "TruncatedNormalRV",
    "default_rv_specs",
    "gaussian_copula_correlate",
    "latin_hypercube",
    "load_distributions_config",
    "probabilistic_ranking",
    "sample_spec_matrix",
]
