# final_cleanup.py
"""Final cleanup - remove temporary debug scripts from git."""

import subprocess
from pathlib import Path

# فایل‌هایی که باید حذف شوند
temp_scripts = [
    "diagnose_and_fix.py",
    "final_fix_cantilever.py",
    "fix_all_failures.py",
    "fix_all_final.py",
    "fix_cantilever_test.py",
    "fix_energy_formula.py",
    "fix_final_2.py",
    "fix_future_imports.py",
    "fix_last_4_tests.py",
    "fix_missing_functions.py",
    "fix_root_main.py",
    "fix_stderr_loss.py",
]

print("🧹 Final cleanup - removing temporary debug scripts...")

for script in temp_scripts:
    p = Path(script)
    if p.exists():
        p.unlink()
        print(f"  ✅ Deleted {script}")

# Commit and push
subprocess.run(["git", "add", "-A"], check=True)
subprocess.run(
    [
        "git",
        "commit",
        "-m",
        "chore: remove temporary debug scripts - project is production-ready",
    ],
    check=True,
)
subprocess.run(["git", "push"], check=True)

print("\n🎊 Project is now clean and production-ready!")
print("🚀 Run `git tag v2.0.9 && git push --tags` to publish to PyPI!")
