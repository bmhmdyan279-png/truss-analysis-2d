"""
Phase 9: Robustness Analysis
Tests the stability of Phase 8 results against:
1. Alpha degradation severity (0.7 -> 0.5)
2. Proxy definition (Max Displacement vs Global Compliance)
3. Bootstrap convergence (5000 -> 10000)
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np
from truss_analysis.main import run


def get_response(json_path: str, proxy: str = "max_disp") -> float:
    """Run analysis and return proxy response."""
    result = run(json_path, quiet=True)
    if proxy == "max_disp":
        max_u = 0.0
        for node_disp in result.displacements.values():
            ux = float(node_disp.get("ux", 0.0))
            uy = float(node_disp.get("uy", 0.0))
            u_mag = np.sqrt(ux**2 + uy**2)
            if u_mag > max_u:
                max_u = u_mag
        return float(max_u)
    if proxy == "compliance":
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        comp = 0.0
        for load in data.get("loads", []):
            nid = str(load.get("node_id"))
            if nid in result.displacements:
                ux = float(result.displacements[nid].get("ux", 0.0))
                uy = float(result.displacements[nid].get("uy", 0.0))
                fx = float(load.get("Fx", 0.0))
                fy = float(load.get("Fy", 0.0))
                comp += fx * ux + fy * uy
        return abs(comp)
    return 0.0


def compute_scf_vector(
    base_json_path: str,
    alpha: float,
    proxy: str,
) -> dict[str, float]:
    """Compute SCF for all members using alpha-degradation."""
    with open(base_json_path, encoding="utf-8") as f:
        base_data = json.load(f)

    u_base = get_response(base_json_path, proxy)
    if u_base < 1e-14:
        return {str(e["id"]): float("nan") for e in base_data.get("elements", [])}

    scf_dict: dict[str, float] = {}
    for elem in base_data.get("elements", []):
        elem_id = str(elem.get("id"))
        deg_data = json.loads(json.dumps(base_data))

        for e in deg_data["elements"]:
            if str(e.get("id")) == elem_id:
                if "E" in e:
                    e["E"] = float(e["E"]) * alpha
                elif "A" in e:
                    e["A"] = float(e["A"]) * alpha
                break

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as tf:
            json.dump(deg_data, tf)
            temp_path = tf.name

        try:
            u_deg = get_response(temp_path, proxy)
            scf_dict[elem_id] = float(u_deg / u_base)
        except Exception:
            scf_dict[elem_id] = float("inf")
        finally:
            os.unlink(temp_path)

    return scf_dict


def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient."""
    if np.any(np.isinf(x)) or np.any(np.isnan(x)) or np.mean(x) == 0.0:
        return float("nan")
    sorted_x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    num = np.sum((2 * index - n - 1) * sorted_x)
    den = n * np.sum(sorted_x)
    return float(num / den)


def log_ratio(x: np.ndarray) -> float:
    """Compute Log-Ratio."""
    if np.any(np.isinf(x)) or np.any(np.isnan(x)) or np.any(x <= 0.0):
        return float("nan")
    return float(np.log(np.max(x)) - np.log(np.min(x)))


def bootstrap_metrics(
    data: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Perform bootstrap resampling."""
    np.random.seed(seed)
    valid_data = data[~np.isinf(data) & ~np.isnan(data)]
    if len(valid_data) < 2:
        return {
            "gini_ci": (float("nan"), float("nan")),
            "log_ratio_ci": (float("nan"), float("nan")),
        }
    gini_samples, lr_samples = [], []
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
    }


def main() -> None:
    """Execute Phase 9 Robustness Analysis."""
    problem_file = "examples/uniform_beta_problem.json"
    print("--- Phase 9: Robustness Analysis ---")

    scenarios: dict[str, dict[str, Any]] = {
        "Baseline (Phase 8)": {
            "alpha": 0.7,
            "proxy": "max_disp",
            "n_boot": 5000,
        },
        "Alpha Severity": {
            "alpha": 0.5,
            "proxy": "max_disp",
            "n_boot": 5000,
        },
        "Proxy (Compliance)": {
            "alpha": 0.7,
            "proxy": "compliance",
            "n_boot": 5000,
        },
        "Bootstrap Conv.": {
            "alpha": 0.7,
            "proxy": "max_disp",
            "n_boot": 10000,
        },
    }

    results: list[dict[str, Any]] = []
    for name, params in scenarios.items():
        print(f"Running Scenario: {name}...")
        alpha = float(params["alpha"])
        proxy = str(params["proxy"])
        n_boot = int(params["n_boot"])

        scf_dict = compute_scf_vector(problem_file, alpha, proxy)
        scf_vals = np.array(list(scf_dict.values()))
        boot = bootstrap_metrics(scf_vals, n_boot, seed=2026)

        results.append(
            {
                "Scenario": name,
                "Alpha": alpha,
                "Proxy": proxy,
                "N_Boot": n_boot,
                "Gini_Point": gini_coefficient(scf_vals),
                "Gini_CI_Lower": boot["gini_ci"][0],
                "LR_Point": log_ratio(scf_vals),
                "LR_CI_Lower": boot["log_ratio_ci"][0],
            }
        )

    print(
        "\n| Scenario | Alpha | Proxy | N_Boot | Gini Point | "
        "Gini CI Lower | LR Point | LR CI Lower |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['Scenario']} | {r['Alpha']} | {r['Proxy']} | "
            f"{r['N_Boot']} | {r['Gini_Point']:.4f} | "
            f"{r['Gini_CI_Lower']:.4f} | {r['LR_Point']:.4f} | "
            f"{r['LR_CI_Lower']:.4f} |"
        )


if __name__ == "__main__":
    main()
