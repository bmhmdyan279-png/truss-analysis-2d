"""Uncertainty layer: random variables, sampling, streaming statistics.

Package evolution (prompt-06): the former single module ``uncertainty.py``
became this package.  The legacy public API (``NormalRV``, ``LognormalRV``,
``GumbelRV``, ``load_distributions_config``) is preserved verbatim in
:mod:`random_variables` and re-exported here.
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
    load_distributions_config,
    proposal_rv_specs,
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
    "gaussian_copula_correlate",
    "latin_hypercube",
    "load_distributions_config",
    "probabilistic_ranking",
    "proposal_rv_specs",
    "sample_spec_matrix",
]
