"""
Phase 6.5: Surgical Injection of Physical Moment of Inertia (D-010)
"""

import json
import sys
from pathlib import Path


def inject_physics():
    file_path = Path("examples/example1.json")
    if not file_path.exists():
        print("❌ File not found!")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"💉 Injecting physical I_sec into {file_path}...\n")

    # Standard physical I_sec for A = 0.01 m^2 (equivalent to r_g ~ 2.8 cm)
    # This represents a realistic steel section (e.g., small pipe or IPE)
    PHYSICAL_I_SEC = 8.33e-6

    for elem in data.get("elements", []):
        if "properties" not in elem:
            elem["properties"] = {}

        # Force overwrite to ensure physical realism (D-010)
        elem["properties"]["I_sec"] = PHYSICAL_I_SEC
        print(f"✅ Member {elem.get('id', '?')}: I_sec set to {PHYSICAL_I_SEC:.2e}")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("\n🎉 SUCCESS: Physical I_sec injected successfully.")
    print("👉 Next step: Run verify_physics.py and compute_phase2_metrics.py again.")


if __name__ == "__main__":
    inject_physics()
