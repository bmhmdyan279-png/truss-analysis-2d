"""
Phase 8: Statically Indeterminate H1 Test.

Tests H1: Uniform member reliability does not imply uniform system-risk-contribution.
Uses uniform_beta_problem.json (from Phase 6.5).
Computes SCF via alpha-degradation (alpha=0.7) and evaluates heterogeneity
using bounded metrics (Gini, Log-Ratio) with Bootstrap Resampling.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from truss_analysis.main import run


def get_max_displacement(json_path: str) -> float:
    """Run analysis on a JSON file and return max absolute nodal displacement."""
    # quiet=True prevents printing the summary to console
    result = run(json_path, quiet=True)

    max_u = 0.0
    for node_disp in result.displacements.values():
        ux = float(node_disp.get("ux", 0.0))
        uy = float(node_disp.get("uy", 0.0))
        u_mag = np.sqrt(ux**2 + uy**2)
        if u_mag > max_u:
            max_u = u_mag
    return float(max_u)


def compute_scf_vector(base_json_path: str, alpha: float = 0.7) -> dict[str, float]:
    """Compute SCF for all members using alpha-degradation."""
    with open(base_json_path, encoding="utf-8") as f:
        base_data = json.load(f)

    u_base = get_max_displacement(base_json_path)
    if u_base < 1e-14:
        msg = "Baseline max displacement is zero. Cannot compute SCF."
        raise ValueError(msg)

    scf_dict: dict[str, float] = {}

    for elem in base_data.get("elements", []):
        elem_id = str(elem.get("id"))

        # Deepcopy the data to degrade this specific element safely
        deg_data = json.loads(json.dumps(base_data))

        for e in deg_data["elements"]:
            if str(e.get("id")) == elem_id:
                if "E" in e:
                    e["E"] = float(e["E"]) * alpha
                elif "A" in e:
                    e["A"] = float(e["A"]) * alpha
                break

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(deg_data, tf)
            temp_path = tf.name

        try:
            u_deg = get_max_displacement(temp_path)
            scf_dict[elem_id] = float(u_deg / u_base)
        except Exception:
            # Mechanism detected (singular matrix) or other solver error
            scf_dict[elem_id] = float("inf")
        finally:
            os.unlink(temp_path)

    return scf_dict


def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient. Bounded between 0 and 1."""
    if np.any(np.isinf(x)) or np.any(np.isnan(x)):
        return float("nan")
    if np.mean(x) == 0.0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    num = np.sum((2 * index - n - 1) * sorted_x)
    den = n * np.sum(sorted_x)
    return float(num / den)


def log_ratio(x: np.ndarray) -> float:
    """Compute Log-Ratio: ln(max) - ln(min). Bounded >= 0."""
    if np.any(np.isinf(x)) or np.any(np.isnan(x)) or np.any(x <= 0.0):
        return float("nan")
    return float(np.log(np.max(x)) - np.log(np.min(x)))


def bootstrap_metrics(data: np.ndarray, n_boot: int = 5000, seed: int = 2026) -> dict:
    """Perform bootstrap resampling to estimate 95% CI for metrics."""
    np.random.seed(seed)
    valid_data = data[~np.isinf(data) & ~np.isnan(data)]

    if len(valid_data) < 2:
        return {
            "gini_ci": (float("nan"), float("nan")),
            "log_ratio_ci": (float("nan"), float("nan")),
            "gini_mean": float("nan"),
            "log_ratio_mean": float("nan"),
        }

    gini_samples = []
    lr_samples = []
    n = len(valid_data)

    for _ in range(n_boot):
        sample = np.random.choice(valid_data, size=n, replace=True)
        gini_samples.append(gini_coefficient(sample))
        lr_samples.append(log_ratio(sample))

    return {
        "gini_ci": (
            float(np.percentile(gini_samples, 2.5)),
            float(np.percentile(gini_samples, 97.5)),
        ),
        "log_ratio_ci": (
            float(np.percentile(lr_samples, 2.5)),
            float(np.percentile(lr_samples, 97.5)),
        ),
        "gini_mean": float(np.mean(gini_samples)),
        "log_ratio_mean": float(np.mean(lr_samples)),
    }


def main() -> None:
    """Execute Phase 8 H1 test."""
    problem_file = "examples/uniform_beta_problem.json"
    if not Path(problem_file).exists():
        msg = f"ERROR: {problem_file} not found. Did you complete Phase 6.5?"
        print(msg)
        sys.exit(1)

    print(f"Running Phase 8 on {problem_file}...")
    scf_dict = compute_scf_vector(problem_file, alpha=0.7)

    member_ids = list(scf_dict.keys())
    scf_values = np.array(list(scf_dict.values()), dtype=float)

    has_inf = np.any(np.isinf(scf_values))
    if has_inf:
        print("WARNING: One or more members caused a mechanism (SCF=inf).")
        print("Bounded metrics will be NaN for the full vector.")

    gini_val = gini_coefficient(scf_values)
    lr_val = log_ratio(scf_values)
    boot_results = bootstrap_metrics(scf_values, n_boot=5000, seed=2026)

    h1_gini = (
        boot_results["gini_ci"][0] > 0.0
        if not np.isnan(boot_results["gini_ci"][0])
        else False
    )
    h1_lr = (
        boot_results["log_ratio_ci"][0] > 0.0
        if not np.isnan(boot_results["log_ratio_ci"][0])
        else False
    )
    h1_accepted = h1_gini or h1_lr

    timestamp = datetime.now(timezone.utc).isoformat()

    print("\n--- Phase 8 Results ---")
    print(f"Execution date: {timestamp}")
    print("Computed by: `scripts/compute_phase8_h1_test.py`")
    print(f"Problem: `{problem_file}`")
    print("Alpha: 0.7")
    print(f"H1 Accepted: {h1_accepted}")

    print("\n| Member ID | SCF(alpha=0.7) |")
    print("|---|---|")
    for mid, scf in zip(member_ids, scf_values):
        scf_str = "inf" if np.isinf(scf) else f"{scf:.4f}"
        print(f"| {mid} | {scf_str} |")

    print("\n| Metric | Point Estimate | 95% CI Lower | 95% CI Upper | H1 Cond |")
    print("|---|---|---|---|---|")
    print(
        f"| Gini | {gini_val:.4f} | "
        f"{boot_results['gini_ci'][0]:.4f} | "
        f"{boot_results['gini_ci'][1]:.4f} | "
        f"{h1_gini} |"
    )
    print(
        f"| Log-Ratio | {lr_val:.4f} | "
        f"{boot_results['log_ratio_ci'][0]:.4f} | "
        f"{boot_results['log_ratio_ci'][1]:.4f} | "
        f"{h1_lr} |"
    )

    output_dir = Path("PROJECT_DOCUMENTATION/raw_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "phase8_raw.json"

    payload = {
        "timestamp": timestamp,
        "problem_file": problem_file,
        "alpha": 0.7,
        "scf_dict": {k: ("inf" if np.isinf(v) else v) for k, v in scf_dict.items()},
        "gini": gini_val,
        "log_ratio": lr_val,
        "bootstrap": boot_results,
        "h1_accepted": h1_accepted,
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nRaw data saved to {output_file}")


if __name__ == "__main__":
    main()
