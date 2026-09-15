"""Sampling: Latin hypercube, rank-correlated draws, spec-driven samples.

Provides the stochastic input layer for screening (small Latin-hypercube
designs) and Monte Carlo studies (larger sample counts) with **correlation
support** in two variants that share one target semantics (the *rank*
correlation, mapped to its normal-space Pearson equivalent through
Kruskal's relation before factorisation):

* :func:`iman_conover_correlate` -- rank reordering; preserves each input
  column exactly (a Latin hypercube stays a Latin hypercube). This is what
  :func:`sample_spec_matrix` uses.
* :func:`gaussian_copula_correlate` -- linear mix in normal space; achieves
  the same target rank correlation but does not preserve stratification,
  so it is for arbitrary (non-LHS) uniform inputs.

Seeds are deterministic and generation is independent of call order (every
draw is a pure function of ``seed``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from scipy.stats import norm, rankdata

from .random_variables import (
    DeterministicRV,
    DistributionSpec,
    GumbelRV,
    LognormalRV,
    RandomVariable,
    TruncatedNormalRV,
)

__all__ = [
    "gaussian_copula_correlate",
    "iman_conover_correlate",
    "latin_hypercube",
    "sample_spec_matrix",
]


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


def _kruskal_inverse(correlation: np.ndarray) -> np.ndarray:
    """Validate a target Spearman matrix and map it to normal-space Pearson.

    By Kruskal's theorem a bivariate normal with Pearson correlation
    ``rho_P`` has Spearman correlation ``rho_S = (6/pi) arcsin(rho_P / 2)``,
    so a target ``rho_S`` requires the inverse ``rho_P = 2 sin((pi/6) rho_S)``
    elementwise before any factorisation.

    Raises
    ------
    ValueError
        If the target is not a symmetric correlation matrix with unit
        diagonal and off-diagonals in ``[-1, 1]``, or if the mapped matrix
        is not positive definite. Asymmetric input previously disappeared
        silently into the lower triangle of the Cholesky (round-5 audit).
    """
    c = np.asarray(correlation, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        msg = f"correlation must be a square matrix, got shape {c.shape}"
        raise ValueError(msg)
    if not np.allclose(c, c.T, rtol=0.0, atol=1e-12):
        msg = "correlation matrix is not symmetric"
        raise ValueError(msg)
    if not np.allclose(np.diag(c), 1.0, rtol=0.0, atol=1e-12):
        msg = "correlation matrix diagonal must be exactly 1"
        raise ValueError(msg)
    off = c[~np.eye(c.shape[0], dtype=bool)]
    if off.size and (np.max(np.abs(off)) > 1.0):
        msg = "correlation off-diagonals must lie in [-1, 1]"
        raise ValueError(msg)
    pearson = 2.0 * np.sin((np.pi / 6.0) * c)
    # 2*sin(pi/6) is 1 up to round-off; pin the diagonal exactly so the
    # Cholesky sees a proper correlation matrix.
    np.fill_diagonal(pearson, 1.0)
    return pearson


def _cholesky_or_raise(pearson: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(pearson)
    except np.linalg.LinAlgError as exc:
        msg = (
            "target rank-correlation matrix is not positive definite after "
            "the Kruskal mapping rho_P = 2 sin((pi/6) rho_S)"
        )
        raise ValueError(msg) from exc


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

    .. warning::
        The linear mix ``z @ L.T`` recombines **all** columns, so a
        stratified input (e.g. from :func:`latin_hypercube`) does *not*
        stay stratified: each output column is a mixture, and the LHS
        property "exactly one sample per stratum per dimension" is lost
        (marginals stay uniform, variance reduction does not survive).
        Use :func:`iman_conover_correlate` when the input is an LHS design
        and stratification must be preserved -- it reorders instead of
        mixing, so every column of the output is an exact permutation of
        the corresponding input column.

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
        If the column count of ``u`` does not match ``correlation``, if the
        target is not a symmetric unit-diagonal correlation matrix, or if
        the Kruskal-mapped matrix is not positive definite.
    """
    m = correlation.shape[0]
    if u.shape[1] != m:
        msg = f"u has {u.shape[1]} columns, correlation is {m}x{m}"
        raise ValueError(msg)
    pearson = _kruskal_inverse(correlation)
    chol = _cholesky_or_raise(pearson)
    z = norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
    z_c = z @ chol.T
    return np.asarray(norm.cdf(z_c))


def iman_conover_correlate(u: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    """Impose a target rank correlation on ``u`` **without touching marginals**.

    Iman--Conover rank reordering: build a score matrix with the target
    normal-space correlation, then permute each column of ``u`` so its ranks
    match the corresponding score column. Every output column is an exact
    permutation of the input column, so a Latin-hypercube design stays a
    Latin-hypercube design (one sample per stratum per dimension), while the
    realised Spearman correlation reproduces ``correlation`` within sampling
    noise. The Gaussian-copula path (:func:`gaussian_copula_correlate`)
    achieves the same target but destroys stratification by linear mixing.

    Deterministic: the score matrix is built from the normal scores of
    ``u`` itself, so the output is a pure function of ``(u, correlation)``
    -- no extra seed, consistent with the library's reproducibility rule.

    Parameters
    ----------
    u : np.ndarray
        Input uniform matrix ``(n, m)`` (typically a Latin hypercube).
    correlation : np.ndarray
        Symmetric positive-definite target **Spearman** correlation matrix
        ``(m, m)`` (same contract as :func:`gaussian_copula_correlate`).

    Returns
    -------
    np.ndarray
        Reordered ``(n, m)`` matrix: identical column *values* (hence
        identical marginals and stratification), target rank correlation.

    Raises
    ------
    ValueError
        Same validation as :func:`gaussian_copula_correlate`.
    """
    n, m = u.shape
    if m != correlation.shape[0]:
        msg = (
            f"u has {m} columns, correlation is "
            f"{correlation.shape[0]}x{correlation.shape[1]}"
        )
        raise ValueError(msg)
    pearson = _kruskal_inverse(correlation)
    chol = _cholesky_or_raise(pearson)
    # Normal scores of each input column, mixed to carry the target
    # correlation. Ties get average ranks (an LHS design has none, but a
    # user-supplied u might).
    scores = norm.ppf((rankdata(u, axis=0) - 0.5) / n)
    z_target = scores @ chol.T
    out = np.array(u, dtype=float, copy=True)
    for d in range(m):
        # Textbook Iman--Conover placement: the position holding the i-th
        # smallest score receives the i-th smallest value, so the output
        # column's ranks equal the score column's ranks exactly.
        out[np.argsort(z_target[:, d], kind="stable"), d] = np.sort(u[:, d])
    return out


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

    ``correlation`` (optional, ``len(specs) x len(specs)``) couples the draws
    via :func:`iman_conover_correlate`, which imposes the target *rank*
    correlation by **reordering** the hypercube rather than mixing it, so
    every marginal keeps exactly one sample per stratum. (Pre-2.8 this used
    the Gaussian-copula linear mix, which achieved the same rank correlation
    but destroyed the stratification the LHS exists to provide -- the
    round-5 audit finding; :func:`gaussian_copula_correlate` remains
    available for non-LHS inputs.) Deterministic specs ignore the coupling.
    The result depends only on ``(specs, means, n, seed)`` - never on call
    order or history.

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
        u = iman_conover_correlate(u, correlation)
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
