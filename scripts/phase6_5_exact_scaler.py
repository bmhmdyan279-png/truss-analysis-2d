"""
Phase 6.5: Exact One-Shot Area Scaling for Uniform Beta.
"""

import json
import math
import subprocess
import sys

TARGET_BETA = 3.5
OUTPUT_FILE = "examples/uniform_beta_problem.json"
INPUT_FILE = "examples/example1.json"
RESULTS_FILE = "PROJECT_DOCUMENTATION/phase2_results.json"


def run_monte_carlo():
    print("🔄 Running Monte Carlo to get current g_mean and g_std...")
    cmd = [sys.executable, "scripts/compute_phase2_metrics.py"]
    subprocess.run(cmd, check=True, capture_output=True)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {}
    for stat in data.get("statistics", []):
        if stat["limit_state"] == "buckling":
            stats[str(stat["target_id"])] = {
                "mean": stat["mean"],
                "std": stat["std"],
            }
    return stats


def main():
    print(f"🚀 Phase 6.5 Exact One-Shot Scaler (Target Beta: {TARGET_BETA})")

    mc_stats = run_monte_carlo()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = {n["id"]: (n["x"], n["y"]) for n in data.get("nodes", [])}

    scaled_count = 0
    for elem in data.get("elements", []):
        elem_id = str(elem.get("id", "Unknown"))

        if elem_id not in mc_stats:
            continue

        mean_g = mc_stats[elem_id]["mean"]
        std_g = mc_stats[elem_id]["std"]

        if std_g <= 0 or math.isnan(std_g):
            print(f"⚠️ Skipping Member {elem_id}: Invalid std_g ({std_g})")
            continue

        e_mod = elem.get("E", 210e9)
        i_old = elem.get("I_sec", 0)
        k_fac = elem.get("effective_length_factor", 1.0)

        n_i = elem.get("node_i")
        n_j = elem.get("node_j")
        if n_i in nodes and n_j in nodes:
            x1, y1 = nodes[n_i]
            x2, y2 = nodes[n_j]
            length = math.hypot(x2 - x1, y2 - y1)
        else:
            print(f"❌ Missing node coordinates for Member {elem_id}")
            continue

        p_cr_old = (math.pi**2 * e_mod * i_old) / ((k_fac * length) ** 2)

        g_target = TARGET_BETA * std_g
        p_cr_new = p_cr_old - mean_g + g_target

        if p_cr_new <= 0:
            msg = f"⚠️ Member {elem_id}: Required P_cr is negative. Cannot scale."
            print(msg)
            continue

        k_i = p_cr_new / p_cr_old
        k_a = math.sqrt(k_i)

        if "A" in elem and "I_sec" in elem:
            a_old = elem["A"]
            elem["A"] = a_old * k_a
            elem["I_sec"] = i_old * k_i
            scaled_count += 1
            msg = (
                f"✅ Member {elem_id}: Scaled A by {k_a:.4f}, "
                f"I by {k_i:.4f} (Expected Beta ≈ {TARGET_BETA})"
            )
            print(msg)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n🎉 SUCCESS! Scaled {scaled_count} members.")
    print(f"📁 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
