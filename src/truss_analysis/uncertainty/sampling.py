"""Sampling: Latin hypercube, Gaussian-copula correlation, spec-driven draws.

Provides the stochastic input layer for screening (small Latin-hypercube
designs) and Monte Carlo studies (larger sample counts) with **correlation
support** via a Gaussian copula (the target *rank* correlation is mapped to
its normal-space Pearson equivalent through Kruskal's relation before the
Cholesky factorisation), deterministic seeds, and generation that is
independent of call order (every draw is a pure function of ``seed``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from scipy.stats import norm

from .random_variables import (
    DeterministicRV,
    DistributionSpec,
    GumbelRV,
    LognormalRV,
    RandomVariable,
    TruncatedNormalRV,
)

__all__ = ["gaussian_copula_correlate", "latin_hypercube", "sample_spec_matrix"]


def latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    """Draw a stratified uniform matrix ``(n, dim)``.

    One sample per stratum per dimension: column ``d`` is a random
    permutation of the strata midpoints ``(k + U_k)/n``; fully determined
    by ``seed`` (order-independent).

    Parameters
    ----------
    n : int
        Number of samples (strata per dimension); must be >= 1.
    dim : int
        Number of dimensions; must be >= 1.
    seed : int
        Generator seed.

    Returns
    -------
    np.ndarray
        Uniform ``(n, dim)`` matrix in ``[0, 1)``.

    Raises
    ------
    ValueError
        If ``n`` or ``dim`` is below 1.
    """
    if n < 1 or dim < 1:
        msg = f"require n,dim >= 1, got n={n}, dim={dim}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    u = rng.random((n, dim))
    perms = np.argsort(rng.random((n, dim)), axis=0)
    strata: np.ndarray = (perms + u) / n
    return strata


def gaussian_copula_correlate(u: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    """Map independent uniforms to uniforms with a target rank correlation.

    ``correlation`` is the target **Spearman (rank) correlation** of the
    output. By Kruskal's theorem a bivariate normal with Pearson correlation
    ``rho_P`` has Spearman correlation ``rho_S = (6/pi) arcsin(rho_P / 2)``,
    so achieving a target ``rho_S`` requires the inverse mapping

    .. code-block:: text

        rho_P = 2 sin((pi/6) rho_S)

    applied elementwise **before** the Cholesky factorisation. Feeding the
    rank-correlation target to the Cholesky directly (the pre-2.7 behaviour)
    produced normals with ``rho_P = rho_S`` and therefore an output rank
    correlation of ``(6/pi) arcsin(rho_S/2)`` — a systematic contraction
    (target 0.70 realised 0.682, target 0.90 realised 0.891).

    Pipeline: ``z = Phi^-1(u)``, ``z_c = L z`` with ``L = cholesky(rho_P)``,
    ``u_c = Phi(z_c)``.

    Parameters
    ----------
    u : np.ndarray
        Independent uniform matrix ``(n, m)``.
    correlation : np.ndarray
        Symmetric positive-definite target **Spearman** correlation matrix
        ``(m, m)`` with unit diagonal and off-diagonals in ``(-1, 1)``.

    Returns
    -------
    np.ndarray
        Correlated uniform matrix ``(n, m)`` whose column rank correlations
        reproduce ``correlation``.

    Raises
    ------
    ValueError
        If the column count of ``u`` does not match ``correlation``, or if
        the Kruskal-mapped matrix is not positive definite.
    """
    m = correlation.shape[0]
    if u.shape[1] != m:
        msg = f"u has {u.shape[1]} columns, correlation is {m}x{m}"
        raise ValueError(msg)
    # Kruskal inverse: target Spearman -> Pearson correlation in normal space.
    pearson = 2.0 * np.sin((np.pi / 6.0) * np.asarray(correlation, dtype=float))
    # 2*sin(pi/6) is 1 up to round-off; pin the diagonal exactly so the
    # Cholesky sees a proper correlation matrix.
    np.fill_diagonal(pearson, 1.0)
    try:
        chol = np.linalg.cholesky(pearson)
    except np.linalg.LinAlgError as exc:
        msg = (
            "target rank-correlation matrix is not positive definite after "
            "the Kruskal mapping rho_P = 2 sin((pi/6) rho_S)"
        )
        raise ValueError(msg) from exc
    z = norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
    z_c = z @ chol.T
    return np.asarray(norm.cdf(z_c))


def _build_rv(
    spec: DistributionSpec, mean: float, seed: int
) -> RandomVariable | DeterministicRV:
    """Instantiate the random variable described by ``spec``."""
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
) -> dict[str, np.ndarray]:
    """Draw ``(n, len(specs))`` samples keyed by spec name.

    ``correlation`` (optional, ``len(specs) x len(specs)``) couples the
    draws via :func:`gaussian_copula_correlate`; deterministic specs ignore
    the coupling. The result depends only on ``(specs, means, n, seed)`` -
    never on call order or history.

    Parameters
    ----------
    specs : Sequence[DistributionSpec]
        Variable specifications (e.g. from ``default_rv_specs``).
    means : Mapping[str, float]
        Variable name -> mean value (unused for deterministic specs, whose
        value comes from ``spec.parameters["value"]``).
    n : int
        Number of samples.
    seed : int
        Base seed; spec ``d`` uses ``seed + d``.
    correlation : np.ndarray or None
        Optional target rank-correlation matrix.

    Returns
    -------
    dict[str, np.ndarray]
        Variable name -> sample vector of length ``n``.

    Raises
    ------
    ValueError
        On unknown distribution families (see :func:`_build_rv`).
    """
    u = latin_hypercube(n, len(specs), seed)
    if correlation is not None:
        u = gaussian_copula_correlate(u, correlation)
    out: dict[str, np.ndarray] = {}
    for d, spec in enumerate(specs):
        if spec.family == "deterministic":
            # deterministic specs always take their spec value (the ``means``
            # entry, when present, is documentation only - historic behaviour)
            det = DeterministicRV(spec.parameters["value"], seed=seed + d)
            out[spec.name] = det.sample(n)
            continue
        rv = cast(RandomVariable, _build_rv(spec, means[spec.name], seed + d))
        # invert the rv's distribution through the (possibly correlated) uniforms
        out[spec.name] = np.asarray(rv.dist.ppf(u[:, d]))
    return out
