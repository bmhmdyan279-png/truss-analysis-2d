"""Phase 6: Heterogeneity Index and H1 Test with Bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class HeterogeneityResult:
    """Results of the Phase 6 heterogeneity and bootstrap analysis."""

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


def compute_heterogeneity(
    margins: Mapping[str, npt.NDArray[np.float64]],
    scf_values: Mapping[str, float],
    n_bootstrap: int = 5000,
    bootstrap_seed: int = 2026,
) -> HeterogeneityResult:
    """Compute heterogeneity index U, CoV, Gini, and perform H1 test."""
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

        # Fix: Handle completely missing data separately from instability
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
        std_g = float(np.std(valid_gc, ddof=1)) if len(valid_gc) > 1 else 0.0
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
            den = np.abs(gc)
        else:
            num = mu_g
            den = gc

        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        src_k = scf * (num / den)
        src_matrix[:, idx] = src_k

    u_arr = np.zeros(n_samples)
    cov_arr = np.zeros(n_samples)
    gini_arr = np.zeros(n_samples)

    for k in range(n_samples):
        src_k = src_matrix[k, :]
        valid_src = src_k[~np.isnan(src_k)]
        if len(valid_src) == 0 or np.min(np.abs(valid_src)) < 1e-12:
            u_arr[k] = np.nan
            cov_arr[k] = np.nan
            gini_arr[k] = np.nan
            continue

        u_arr[k] = float(np.max(valid_src) / np.min(valid_src))

        mean_src = np.mean(valid_src)
        if mean_src != 0:
            cov_arr[k] = float(np.std(valid_src, ddof=1) / mean_src)
        else:
            cov_arr[k] = float("nan")

        sorted_src = np.sort(valid_src)
        n = len(sorted_src)
        index = np.arange(1, n + 1)
        sum_src = np.sum(sorted_src)
        if sum_src != 0:
            gini_arr[k] = float(
                np.sum((2 * index - n - 1) * sorted_src) / (n * sum_src)
            )
        else:
            gini_arr[k] = float("nan")

    valid_u = u_arr[~np.isnan(u_arr)]
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

    valid_cov = cov_arr[~np.isnan(cov_arr)]
    valid_gini = gini_arr[~np.isnan(gini_arr)]

    quantiles = {
        "2.5%": float(np.percentile(valid_u, 2.5))
        if len(valid_u) > 0
        else float("nan"),
        "50%": float(np.percentile(valid_u, 50)) if len(valid_u) > 0 else float("nan"),
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
