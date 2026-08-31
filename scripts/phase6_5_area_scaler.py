"""
Phase 6.5: Iterative Area Scaling for Uniform Beta.
This script scales member areas (A) and moments of inertia (I_sec) to adjust
buckling beta. Geometric Scaling Rule: I_new = I_old * (A_new / A_old)^2
"""

import json
from pathlib import Path


def main():
    input_file = Path("examples/uniform_beta_problem.json")
    if not input_file.exists():
        input_file = Path("examples/reference_problem.json")

    output_file = Path("examples/uniform_beta_problem.json")
    SCALE_FACTOR = 1.15

    print(f"🔄 Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scaled_count = 0
    for elem in data.get("elements", []):
        if "A" in elem and "I_sec" in elem:
            a_old = elem["A"]
            i_old = elem["I_sec"]

            a_new = a_old * SCALE_FACTOR
            elem["A"] = a_new

            i_new = i_old * (SCALE_FACTOR**2)
            elem["I_sec"] = i_new

            scaled_count += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Successfully scaled {scaled_count} members by {SCALE_FACTOR}.")
    print(f"📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()
