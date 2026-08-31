"""
Phase 6.5 Pre-requisite: Verify Physical Realism of Cross-Sections (D-010)
"""

import json
import sys
from pathlib import Path


def verify_physics():
    file_path = Path("examples/example1.json")
    if not file_path.exists():
        print("❌ File not found!")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔍 Verifying physical realism of {file_path}:\n")
    all_ok = True

    PHYSICAL_THRESHOLD = 1e-7

    for elem in data.get("elements", []):
        elem_id = elem.get("id", "Unknown")
        area = elem.get("A", 0)
        i_sec = elem.get("I_sec", 0)

        ratio = i_sec / area if area > 0 else 0

        if ratio < PHYSICAL_THRESHOLD:
            msg = (
                f"❌ Member {elem_id}: I_sec/A = {ratio:.2e} "
                "(UNPHYSICAL! D-010 NOT APPLIED)"
            )
            print(msg)
            all_ok = False
        else:
            print(f"✅ Member {elem_id}: I_sec/A = {ratio:.2e} (Physical)")

    print("-" * 50)
    if all_ok:
        print("🎉 SUCCESS: All members have physical cross-sections.")
        print("👉 You can now trust the new Beta values from Phase 2.")
    else:
        print("⚠️ WARNING: Some members are still unphysical.")
        msg2 = "👉 Check if reference_problem.json was saved with I_sec = 8.33e-6."
        print(msg2)


if __name__ == "__main__":
    verify_physics()
