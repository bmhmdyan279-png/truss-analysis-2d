"""Sampling: Latin Hypercube, Gaussian-copula correlation, spec-driven draws.

Prompt-06 part B: LHS (screening, 100 samples) + Monte Carlo (validation,
2000 samples) with **correlation support** via a Gaussian copula (Cholesky of
the target rank-correlation matrix), deterministic seeds, and generation that
is independent of call order (every draw is a pure function of ``seed``).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm

from .random_variables import (
    DeterministicRV,
    DistributionSpec,
    GumbelRV,
    LognormalRV,
    TruncatedNormalRV,
)

__all__ = ["gaussian_copula_correlate", "latin_hypercube", "sample_spec_matrix"]


def latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    """Stratified uniform matrix ``(n, dim)``: one sample per stratum per dim.

    Column ``d`` is a random permutation of the strata midpoints
    ``(k + U_k)/n``; fully determined by ``seed`` (order-independent).
    """
    if n < 1 or dim < 1:
        msg = f"require n,dim >= 1, got n={n}, dim={dim}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    u = rng.random((n, dim))
    perms = np.argsort(rng.random((n, dim)), axis=0)
    strata = (perms + u) / n
    return strata


def gaussian_copula_correlate(u: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    """Map independent uniforms to uniforms with the target rank correlation.

    ``z = Phi^-1(u)``, ``z_c = L z`` with ``L = cholesky(correlation)``,
    ``u_c = Phi(z_c)``.  The target matrix is the correlation in normal space
    (Spearman-type rank correlation of the output, approximately).
    """
    m = correlation.shape[0]
    if u.shape[1] != m:
        msg = f"u has {u.shape[1]} columns, correlation is {m}x{m}"
        raise ValueError(msg)
    z = norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
    chol = np.linalg.cholesky(correlation)
    z_c = z @ chol.T
    return np.asarray(norm.cdf(z_c))


def _build_rv(spec: DistributionSpec, mean: float, seed: int):
    if spec.family == "gumbel":
        return GumbelRV(mean=mean, cov=spec.parameters["cov"], seed=seed)
    if spec.family == "lognormal":
        return LognormalRV(mean=mean, cov=spec.parameters["cov"], seed=seed)
    if spec.family == "truncated_normal":
        return TruncatedNormalRV(
            mean=spec.parameters["mu"],
            std=spec.parameters["sigma"],
            low=spec.parameters["low"],
            high=spec.parameters["high"],
            seed=seed,
        )
    if spec.family == "deterministic":
        return DeterministicRV(spec.parameters["value"], seed=seed)
    msg = f"unknown family {spec.family!r}"
    raise ValueError(msg)


def sample_spec_matrix(
    specs: Sequence[DistributionSpec],
    means: Mapping[str, float],
    n: int,
    seed: int,
    correlation: np.ndarray | None = None,
) -> dict:
    """Draw ``(n, len(specs))`` samples keyed by spec name.

    ``correlation`` (optional, len(specs) x len(specs)) couples the draws via
    :func:`gaussian_copula_correlate`; deterministic specs ignore coupling.
    The result depends only on ``(specs, means, n, seed)`` — never on call
    order or history.
    """
    u = latin_hypercube(n, len(specs), seed)
    if correlation is not None:
        u = gaussian_copula_correlate(u, correlation)
    out: dict = {}
    for d, spec in enumerate(specs):
        if spec.family == "deterministic":
            rv = _build_rv(
                spec, means.get(spec.name, spec.parameters["value"]), seed + d
            )
            out[spec.name] = rv.sample(n)
            continue
        rv = _build_rv(spec, means[spec.name], seed + d)
        # invert the rv's distribution through the (possibly correlated) uniforms
        dist = rv.dist
        out[spec.name] = np.asarray(dist.ppf(u[:, d]))
    return out
