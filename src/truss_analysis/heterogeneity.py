"""Heterogeneity indices of member demand with bootstrap confidence bounds.

Given per-member stochastic demand samples (safety margins), this module
quantifies how unevenly demand is distributed across members:

* bounded inequality metrics (Gini coefficient, log-ratio) computed on
  absolute values, robust to sign changes and singularities;
* an empirical heterogeneity index ``U = max/min`` per sample, reported
  without clamping and filtered for non-finite values;
* a one-sided bootstrap test that the mean of ``U`` exceeds 1, i.e. that
  demand is genuinely non-uniform across the ensemble.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class HeterogeneityResult:
    """Results of the heterogeneity and bootstrap analysis.

    Attributes
    ----------
    member_ids : list[str]
        Sorted member identifiers.
    scf_values : dict[str, float]
        Per-member importance factors applied when forming the demand
        ratio (defaults to 1.0 for members absent from the input map).
    mu_g : dict[str, float]
        Per-member mean safety margin.
    beta_hat : dict[str, float]
        Per-member reliability-like index ``mu_g / std_g`` (signed;
        ``inf``/``-inf``/``nan`` for degenerate dispersion).
    u_empirical_mean, u_empirical_std : float
        Mean and sample standard deviation of the heterogeneity index
        over the valid samples.
    u_empirical_quantiles : dict[str, float]
        ``2.5%``/``50%``/``97.5%`` percentiles of the index.
    u_boot_mean_lower_95 : float
        Lower 95% bootstrap confidence bound of the index mean.
    cov_empirical : float
        Mean coefficient of variation across members and samples.
    gini_empirical : float
        Mean Gini coefficient across the valid samples.
    h1_accepted : bool
        ``True`` when the bootstrap confidence bound of the index mean
        lies above 1 (demand is significantly non-uniform).
    unstable_members : list[str]
        Members whose mean safety margin is non-positive.
    warnings : list[str]
        Human-readable diagnostics collected during the computation.
    """

    member_ids: list[str]
    scf_values: dict[str, float]
    mu_g: dict[str, float]
    beta_hat: dict[str, float]
    u_empirical_mean: float
    u_empirical_std: float
    u_empirical_quantiles: dict[str, float]
    u_boot_mean_lower_95: float
    cov_empirical: float
    gini_empirical: float
    h1_accepted: bool
    unstable_members: list[str]
    warnings: list[str]


def compute_bounded_metrics(values: npt.ArrayLike) -> dict[str, float]:
    """Compute bounded heterogeneity metrics (Gini coefficient and log-ratio).

    Unlike the raw ratio ``U = max/min``, both metrics stay bounded under
    singularities: non-finite entries (e.g. from ``beta -> 0``) are dropped,
    the Gini coefficient is evaluated on absolute values, and the log-ratio
    ignores entries below ``1e-12``.

    Parameters
    ----------
    values : npt.ArrayLike
        1-D sample of member demand values.

    Returns
    -------
    dict[str, float]
        ``{"gini", "log_ratio", "u_raw"}``; ``u_raw`` is the unclamped
        ``max/min`` ratio retained for logging (``inf`` when the minimum
        vanishes, ``nan`` when fewer than two finite values remain).
    """
    arr = np.asarray(values)

    # Filter out NaNs and Infs resulting from physical singularities (e.g., beta -> 0)
    arr = arr[~np.isnan(arr) & ~np.isinf(arr)]

    if len(arr) < 2:
        return {"gini": 0.0, "log_ratio": 0.0, "u_raw": np.nan}

    # 1. Gini Coefficient (requires non-negative values, use absolute)
    arr_abs = np.abs(arr)
    arr_sorted = np.sort(arr_abs)
    n = len(arr_sorted)
    sum_sorted = np.sum(arr_sorted)
    if sum_sorted == 0:
        gini = 0.0
    else:
        index = np.arange(1, n + 1)
        gini = np.sum((2 * index - n - 1) * arr_sorted) / (n * sum_sorted)

    # 2. Log-Ratio: ln(max) - ln(min)
    valid_arr = arr_abs[arr_abs > 1e-12]
    if len(valid_arr) < 2:
        log_ratio = 0.0
    else:
        log_ratio = np.log(np.max(valid_arr)) - np.log(np.min(valid_arr))

    # 3. Raw U (retained for historical logging, NO CLAMPING)
    min_val = np.min(arr_abs)
    max_val = np.max(arr_abs)
    u_raw = max_val / min_val if min_val > 1e-12 else np.inf

    return {
        "gini": float(gini),
        "log_ratio": float(log_ratio),
        "u_raw": float(u_raw),
    }


def compute_heterogeneity(
    margins: Mapping[str, npt.NDArray[np.float64]],
    scf_values: Mapping[str, float],
    n_bootstrap: int = 5000,
    bootstrap_seed: int = 2026,
) -> HeterogeneityResult:
    """Compute the heterogeneity index, CoV and Gini, with a bootstrap test.

    Parameters
    ----------
    margins : Mapping[str, npt.NDArray[np.float64]]
        Per-member safety-margin samples (equal-length 1-D arrays).
    scf_values : Mapping[str, float]
        Per-member importance factors multiplying the demand ratio.
    n_bootstrap : int, default 5000
        Number of bootstrap resamples of the index mean.
    bootstrap_seed : int, default 2026
        Seed of the bootstrap generator (reproducibility).

    Returns
    -------
    HeterogeneityResult
        Empirical and bootstrap statistics; see the dataclass fields.

    Raises
    ------
    ValueError
        If ``margins`` is empty.
    """
    member_ids = sorted(margins.keys())
    if not member_ids:
        raise ValueError("No margins provided.")

    n_samples = len(next(iter(margins.values())))
    n_members = len(member_ids)

    warnings_list: list[str] = []
    unstable_members: list[str] = []
    mu_g_dict: dict[str, float] = {}
    beta_hat_dict: dict[str, float] = {}
    src_matrix = np.zeros((n_samples, n_members))

    for idx, mid in enumerate(member_ids):
        gc = margins[mid]
        valid_gc = gc[~np.isnan(gc)]

        if len(valid_gc) == 0:
            mu_g = float("nan")
            beta = float("nan")
            warnings_list.append(
                f"Member {mid}: No valid safety margin samples (all missing)."
            )
            mu_g_dict[mid] = mu_g
            beta_hat_dict[mid] = beta
            src_matrix[:, idx] = np.nan
            continue

        mu_g = float(np.mean(valid_gc))
        if len(valid_gc) > 1:
            std_g = float(np.std(valid_gc, ddof=1))
        else:
            std_g = 0.0
        if std_g > 1e-12:
            beta = float(mu_g / std_g)
        else:
            if mu_g > 0:
                beta = float("inf")
            elif mu_g < 0:
                beta = float("-inf")
            else:
                beta = float("nan")

        mu_g_dict[mid] = mu_g
        beta_hat_dict[mid] = beta

        scf = scf_values.get(mid, 1.0)

        if mu_g <= 0:
            unstable_members.append(mid)
            warnings_list.append(
                f"Member {mid}: Mean margin <= 0 ({mu_g:.2e}). "
                "Using absolute values for SRC computation."
            )
            num = abs(mu_g)
            # Avoid division by zero: replace zeros in den with nan
            den = np.where(np.abs(gc) > 1e-12, np.abs(gc), np.nan)
        else:
            num = mu_g
            # Avoid division by zero: replace zeros in den with nan
            den = np.where(np.abs(gc) > 1e-12, gc, np.nan)

        src_k = scf * (num / den)
        # Replace any remaining inf/nan from division with nan to avoid downstream warnings
        src_k = np.where(np.isfinite(src_k), src_k, np.nan)
        src_matrix[:, idx] = src_k

    u_arr = np.zeros(n_samples)
    cov_arr = np.zeros(n_samples)
    gini_arr = np.zeros(n_samples)

    for k in range(n_samples):
        src_k = src_matrix[k, :]
        valid_src = src_k[~np.isnan(src_k)]

        if len(valid_src) == 0:
            u_arr[k] = np.nan
            cov_arr[k] = np.nan
            gini_arr[k] = np.nan
            continue

        # No clamping: let division by zero produce inf, filter later
        max_val = np.max(valid_src)
        min_val = np.min(valid_src)
        u_arr[k] = max_val / min_val  # may be inf if min_val == 0

        mean_src = np.mean(valid_src)
        if mean_src != 0 and len(valid_src) > 1:
            cov_arr[k] = float(np.std(valid_src, ddof=1) / mean_src)
        else:
            cov_arr[k] = np.inf if mean_src != 0 else np.nan

        # Use bounded metrics for Gini (based on absolute values)
        metrics = compute_bounded_metrics(valid_src)
        gini_arr[k] = metrics["gini"]

    # Filter out inf and nan from U, CoV, Gini
    valid_u = u_arr[~np.isnan(u_arr) & ~np.isinf(u_arr)]
    valid_cov = cov_arr[~np.isnan(cov_arr) & ~np.isinf(cov_arr)]
    valid_gini = gini_arr[~np.isnan(gini_arr) & ~np.isinf(gini_arr)]

    # One-sided bootstrap test on the mean of U (bound above 1 => non-uniform)
    if len(valid_u) > 0:
        rng_boot = np.random.default_rng(bootstrap_seed)
        boot_means = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            resample = rng_boot.choice(valid_u, size=len(valid_u), replace=True)
            boot_means[b] = np.mean(resample)
        ci_lower = float(np.percentile(boot_means, 2.5))
        h1_accepted = bool(ci_lower > 1.0)
    else:
        ci_lower = float("nan")
        h1_accepted = False

    # Quantiles of U (raw)
    quantiles = {
        "2.5%": float(np.percentile(valid_u, 2.5))
        if len(valid_u) > 0
        else float("nan"),
        "50%": (
            float(np.percentile(valid_u, 50)) if len(valid_u) > 0 else float("nan")
        ),
        "97.5%": float(np.percentile(valid_u, 97.5))
        if len(valid_u) > 0
        else float("nan"),
    }

    return HeterogeneityResult(
        member_ids=member_ids,
        scf_values={mid: scf_values.get(mid, 1.0) for mid in member_ids},
        mu_g=mu_g_dict,
        beta_hat=beta_hat_dict,
        u_empirical_mean=float(np.mean(valid_u)) if len(valid_u) > 0 else float("nan"),
        u_empirical_std=float(np.std(valid_u, ddof=1)) if len(valid_u) > 1 else 0.0,
        u_empirical_quantiles=quantiles,
        u_boot_mean_lower_95=ci_lower,
        cov_empirical=float(np.mean(valid_cov)) if len(valid_cov) > 0 else float("nan"),
        gini_empirical=float(np.mean(valid_gini))
        if len(valid_gini) > 0
        else float("nan"),
        h1_accepted=h1_accepted,
        unstable_members=unstable_members,
        warnings=warnings_list,
    )


def gini_normalized(values: npt.ArrayLike) -> float:
    """Return the bias-corrected Gini coefficient ``gini * n / (n - 1)``.

    Known-distribution checks: all-equal values give 0; a single holder of
    everything gives 1.0 (the raw ``Gini = (n-1)/n`` scaled up). Fewer than
    two finite values give 0.0.

    Parameters
    ----------
    values : npt.ArrayLike
        1-D sample; non-finite entries are dropped, absolute values used.

    Returns
    -------
    float
        Bias-corrected Gini coefficient in ``[0, 1]``.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
    n = len(arr)
    if n < 2:
        return 0.0
    arr_abs = np.abs(arr)
    total = float(np.sum(arr_abs))
    if total == 0.0:
        return 0.0
    order = np.sort(arr_abs)
    idx = np.arange(1, n + 1)
    raw = float(np.sum((2 * idx - n - 1) * order) / (n * total))
    return raw * n / (n - 1)
