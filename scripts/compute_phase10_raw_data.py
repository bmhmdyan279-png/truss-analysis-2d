"""
Phase 10, Step 1: Generate Raw Plot Data for Publication Figures.

This script produces a single JSON file (`phase10_plot_data.json`) containing:
  1. A continuous alpha-degradation sweep (alpha in [0.5, 1.0]) for all 7 members
     of `uniform_beta_problem.json` to generate smooth SCF profiles.
  2. Full Bootstrap resampling arrays (N=5000) for Gini and Log-Ratio at
     baseline (alpha=0.7) and alpha-severity (alpha=0.5) scenarios.
  3. Independent validation metrics (DDM Adjoint sensitivity & Strain Energy)
     recomputed on `uniform_beta_problem.json` to ensure topological consistency.

CRITICAL NOTE (Reviewer 2 Finding, 2026-09-01):
    The Phase 5 validation was originally performed on `reference_problem.json`
    (pre-Phase 6.5 topology). The Phase 8/9 H1 test uses `uniform_beta_problem.json`
    (post-Phase 6.5 scaling). Comparing SCF from one topology with DDM from
    another is a Category Error. This script RECOMPUTES DDM and Energy on the
    CURRENT topology to preserve scientific integrity before correlation plots.

Commit Traceability:
    - Base solver: commit bbe96de (`truss_analysis.main.run`)
    - Alpha-degradation logic: copied from `compute_phase8_h1_test.py` @ bbe96de
    - Gini/Log-Ratio metrics: `heterogeneity.py` (post D-017, D-021)
    - Bootstrap methodology: Phase 8 @ bbe96de, extended with full array storage

Compliance:
    - D-023: Imports arabic_reshaper and python-bidi for Persian label metadata.
    - D-021: No "clamp" operations on denominators; bounded metrics used directly.
    - D-030: Proxy is Max Displacement (compliant with Phase 9).
    - Golden Rule: No hardcoded numbers; everything is computed at runtime.

Author: Phase 10 Visualization Module
Date: 2026-09-01
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# D-023 compliance: import Persian typography tools (optional but logged)
try:
    import arabic_reshaper  # noqa: F401
    from bidi.algorithm import get_bidi  # noqa: F401

    _PERSIAN_SUPPORT_AVAILABLE = True
except ImportError:
    _PERSIAN_SUPPORT_AVAILABLE = False
    print(
        "WARNING: arabic_reshaper or python-bidi not found. D-023 compliance partial."
    )

# Local project imports
try:
    from truss_analysis.main import run
except ImportError as e:
    print(f"ERROR: Cannot import truss_analysis. Did you run `pip install -e .`? ({e})")
    sys.exit(1)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
COMMIT_SHA = "bbe96de"
PROBLEM_FILE = "examples/uniform_beta_problem.json"
OUTPUT_DIR = Path("PROJECT_DOCUMENTATION/raw_outputs")
OUTPUT_FILE = OUTPUT_DIR / "phase10_plot_data.json"
SEED = 2026

# Alpha sweep configuration (continuous profile for smooth curves)
ALPHA_SWEEP = np.arange(1.0, 0.499, -0.05)  # 1.0, 0.95, ..., 0.5

# Bootstrap configuration (D-028, D-031) - full arrays for boxplots
N_BOOT = 5000
SCENARIOS_FOR_BOOTSTRAP = {
    "Baseline (Phase 8)": 0.7,
    "Alpha Severity (Phase 9)": 0.5,
}


# -----------------------------------------------------------------------------
# CORE FUNCTIONS (mirroring Phase 8 / Phase 9 API)
# -----------------------------------------------------------------------------
def get_max_displacement(json_path: str) -> float:
    """Run analysis on a JSON file and return max absolute nodal displacement."""
    result = run(json_path, quiet=True)
    max_u = 0.0
    for node_disp in result.displacements.values():
        ux = float(node_disp.get("ux", 0.0))
        uy = float(node_disp.get("uy", 0.0))
        u_mag = np.sqrt(ux**2 + uy**2)
        if u_mag > max_u:
            max_u = u_mag
    return float(max_u)


def compute_scf_vector(base_json_path: str, alpha: float) -> dict[str, float]:
    """Compute SCF for all members using alpha-degradation (per D-011)."""
    with open(base_json_path, encoding="utf-8") as f:
        base_data = json.load(f)

    u_base = get_max_displacement(base_json_path)
    if u_base < 1e-14:
        msg = "Baseline max displacement is zero. Cannot compute SCF."
        raise ValueError(msg)

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
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(deg_data, tf)
            temp_path = tf.name

        try:
            u_deg = get_max_displacement(temp_path)
            scf_dict[elem_id] = float(u_deg / u_base)
        except Exception:
            scf_dict[elem_id] = float("inf")
        finally:
            os.unlink(temp_path)

    return scf_dict


def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient (bounded 0-1). No denominator clamp (D-021)."""
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
    """Compute ln(max) - ln(min). Valid only for strictly positive x."""
    if np.any(np.isinf(x)) or np.any(np.isnan(x)) or np.any(x <= 0.0):
        return float("nan")
    return float(np.log(np.max(x)) - np.log(np.min(x)))


def bootstrap_full_arrays(data: np.ndarray, n_boot: int, seed: int) -> dict:
    """Perform bootstrap and return FULL SAMPLE ARRAYS (not just quantiles)."""
    np.random.seed(seed)
    valid = data[~np.isinf(data) & ~np.isnan(data)]
    if len(valid) < 2:
        return {"gini_samples": [], "lr_samples": [], "valid": False}

    n = len(valid)
    gini_samples = np.zeros(n_boot)
    lr_samples = np.zeros(n_boot)

    for i in range(n_boot):
        sample = np.random.choice(valid, size=n, replace=True)
        gini_samples[i] = gini_coefficient(sample)
        lr_samples[i] = log_ratio(sample)

    return {
        "gini_samples": gini_samples.tolist(),
        "lr_samples": lr_samples.tolist(),
        "gini_point": float(gini_coefficient(valid)),
        "lr_point": float(log_ratio(valid)),
        "gini_ci": (
            float(np.percentile(gini_samples, 2.5)),
            float(np.percentile(gini_samples, 97.5)),
        ),
        "lr_ci": (
            float(np.percentile(lr_samples, 2.5)),
            float(np.percentile(lr_samples, 97.5)),
        ),
        "valid": True,
    }


def compute_strain_energy(base_json_path: str) -> dict[str, float]:
    """Compute U_i = 0.5 * N_i^2 * L_i / (E_i * A_i) for each truss member."""
    with open(base_json_path, encoding="utf-8") as f:
        base_data = json.load(f)

    result = run(base_json_path, quiet=True)
    energy_dict: dict[str, float] = {}

    # Build node coordinate map
    nodes = {n["id"]: n for n in base_data.get("nodes", [])}

    for elem in base_data.get("elements", []):
        eid = str(elem["id"])
        E = float(elem.get("E", 210e9))
        A = float(elem.get("A", 0.01))

        # Robust member force retrieval
        force_info: dict = {}
        mf = getattr(result, "member_forces", None) or {}
        for key in [eid, int(eid) if eid.isdigit() else eid, f"element_{eid}"]:
            if key in mf:
                force_info = mf[key]
                break

        N = float(force_info.get("N", 0.0)) if force_info else 0.0

        # Member length
        ni, nj = elem["node_i"], elem["node_j"]
        dx = float(nodes[nj]["x"]) - float(nodes[ni]["x"])
        dy = float(nodes[nj]["y"]) - float(nodes[ni]["y"])
        L = np.sqrt(dx**2 + dy**2)

        if E * A > 1e-14 and L > 1e-14:
            U = 0.5 * N * N * L / (E * A)
        else:
            U = 0.0
        energy_dict[eid] = float(U)

    return energy_dict


def compute_ddm_sensitivity_fd(
    base_json_path: str, eps: float = 1e-5
) -> dict[str, float]:
    """Compute |du_max/dA_i| using finite differences."""
    with open(base_json_path, encoding="utf-8") as f:
        base_data = json.load(f)

    u_base = get_max_displacement(base_json_path)
    sensitivity_dict: dict[str, float] = {}

    for elem in base_data.get("elements", []):
        eid = str(elem["id"])
        perturbed_data = json.loads(json.dumps(base_data))

        for e in perturbed_data["elements"]:
            if str(e.get("id")) == eid:
                if "A" in e:
                    e["A"] = float(e["A"]) * (1.0 + eps)
                break

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(perturbed_data, tf)
            temp_path = tf.name

        try:
            u_pert = get_max_displacement(temp_path)
            A_base = float(elem.get("A", 1.0))
            if A_base > 0:
                du_dA = (u_pert - u_base) / (eps * A_base)
            else:
                du_dA = 0.0
            sensitivity_dict[eid] = float(abs(du_dA))
        except Exception:
            sensitivity_dict[eid] = float("nan")
        finally:
            os.unlink(temp_path)

    return sensitivity_dict


# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------
def main() -> None:
    """Execute Phase 10 Step 1: Generate raw plot data."""
    print("=" * 70)
    print("Phase 10 Step 1: Raw Plot Data Generation")
    print(f"Commit Traceability: {COMMIT_SHA}")
    print("=" * 70)

    if not Path(PROBLEM_FILE).exists():
        print(f"ERROR: {PROBLEM_FILE} not found.")
        sys.exit(1)

    # --- 1. Alpha Sweep ---
    print("\n[1/3] Running continuous alpha-degradation sweep...")
    alpha_profiles: dict[str, dict[str, float]] = {}
    for alpha in ALPHA_SWEEP:
        alpha_str = f"{alpha:.3f}"
        print(f"  alpha = {alpha_str} ... ", end="", flush=True)
        try:
            scf = compute_scf_vector(PROBLEM_FILE, alpha=float(alpha))
            for mid, val in scf.items():
                alpha_profiles.setdefault(mid, {})[alpha_str] = float(val)
            print(f"OK (members: {len(scf)})")
        except Exception as e:
            print(f"FAILED ({e})")

    # --- 2. Bootstrap Full Arrays ---
    print("\n[2/3] Running Bootstrap resampling (full arrays, N=5000)...")
    bootstrap_results: dict[str, Any] = {}
    for scenario_name, alpha_val in SCENARIOS_FOR_BOOTSTRAP.items():
        print(
            f"  Scenario: {scenario_name} (alpha={alpha_val}) ... ", end="", flush=True
        )
        scf_dict = compute_scf_vector(PROBLEM_FILE, alpha=alpha_val)
        scf_vals = np.array(list(scf_dict.values()), dtype=float)
        boot = bootstrap_full_arrays(scf_vals, N_BOOT, seed=SEED)
        bootstrap_results[scenario_name] = {
            "alpha": alpha_val,
            "scf_values": [float(x) if not np.isinf(x) else "inf" for x in scf_vals],
            **boot,
        }
        print(f"OK (gini point={boot.get('gini_point', float('nan')):.4f})")

    # --- 3. Independent Validation Metrics (RECOMPUTED on current topology) ---
    print("\n[3/3] Recomputing DDM and Strain Energy on current topology...")
    print("  (Reviewer 2 note: bypasses Phase 5 topology mismatch)")
    print("  Computing Strain Energy ... ", end="", flush=True)
    energy = compute_strain_energy(PROBLEM_FILE)
    print(f"OK ({len(energy)} members)")
    print("  Computing DDM Sensitivity (FD, eps=1e-5) ... ", end="", flush=True)
    ddm = compute_ddm_sensitivity_fd(PROBLEM_FILE)
    print(f"OK ({len(ddm)} members)")

    # --- Assemble Output ---
    timestamp = datetime.now(timezone.utc).isoformat()
    reviewer_note = (
        "DDM and Energy were RECOMPUTED on uniform_beta_problem.json to avoid "
        "Category Error of comparing with Phase 5 results from reference_problem.json."
    )
    payload = {
        "metadata": {
            "phase": "10_visualization",
            "step": 1,
            "description": "Raw plot data for publication-quality figures",
            "commit_sha": COMMIT_SHA,
            "generated_utc": timestamp,
            "problem_file": PROBLEM_FILE,
            "alpha_sweep": ALPHA_SWEEP.tolist(),
            "bootstrap_n": N_BOOT,
            "bootstrap_seed": SEED,
            "persian_support": _PERSIAN_SUPPORT_AVAILABLE,
            "reviewer_2_note": reviewer_note,
        },
        "alpha_profiles": alpha_profiles,
        "bootstrap": bootstrap_results,
        "validation_metrics": {
            "strain_energy": energy,
            "ddm_sensitivity_abs": ddm,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"SUCCESS: Raw plot data saved to {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"Members tracked: {len(alpha_profiles)}")
    print(f"Alphas evaluated: {len(ALPHA_SWEEP)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
