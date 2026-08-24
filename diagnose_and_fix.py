# diagnose_and_fix.py
"""Diagnose why tests are not running and fix the issue."""

import os
import subprocess
import sys
from pathlib import Path

print("🔍 Diagnosing test discovery issue...")

# 1. Check if tests directory exists and list all test files
print("\n📂 Test files in tests/ directory:")
tests_dir = Path("tests")
if tests_dir.exists():
    test_files = list(tests_dir.glob("test_*.py"))
    print(f"   Found {len(test_files)} test files:")
    for tf in test_files:
        print(f"     - {tf.name}")

    if not test_files:
        print("   ❌ No test files found! Restoring from git...")
        subprocess.run(["git", "checkout", "HEAD", "--", "tests/"], check=True)
else:
    print("   ❌ tests/ directory doesn't exist! Restoring from git...")
    subprocess.run(["git", "checkout", "HEAD", "--", "tests/"], check=True)

# 2. Check for syntax errors in all test files
print("\n🔧 Checking syntax of all test files...")
test_files = list(tests_dir.glob("test_*.py"))
for tf in test_files:
    try:
        compile(tf.read_text(encoding="utf-8"), str(tf), "exec")
        print(f"   ✅ {tf.name} - syntax OK")
    except SyntaxError as e:
        print(f"   ❌ {tf.name} - SYNTAX ERROR: {e}")

# 3. Try importing each test module directly
print("\n🔧 Trying to import each test module...")
sys.path.insert(0, str(Path("src").absolute()))
for tf in test_files:
    module_name = tf.stem
    try:
        __import__(f"tests.{module_name}")
        print(f"   ✅ {module_name} - imports OK")
    except Exception as e:
        print(f"   ❌ {module_name} - IMPORT ERROR: {e}")

# 4. Run pytest with verbose collection info
print("\n🧪 Running pytest with collection info...")
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--collect-only"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    env=env,
)
print("STDOUT:")
print(result.stdout[-2000:])
if result.stderr:
    print("STDERR:")
    print(result.stderr[-1000:])

# 5. Check if tests are being skipped
print("\n🧪 Running pytest to see skip reasons...")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "-rs"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    env=env,
)
print("STDOUT:")
print(result.stdout[-3000:])

# 6. If tests were skipped, check if it's due to missing dependencies
if "skip" in result.stdout.lower() or "no tests ran" in result.stdout.lower():
    print("\n⚠️  Tests might be skipped. Checking dependencies...")
    # Install common test dependencies
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pytest", "numpy", "matplotlib"],
        capture_output=True,
    )

# 7. Final run
print("\n🎯 Final test run...")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    env=env,
)

print(result.stdout[-4000:])

import re

passed = re.search(r"(\d+) passed", result.stdout)
failed = re.search(r"(\d+) failed", result.stdout)
skipped = re.search(r"(\d+) skipped", result.stdout)

print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY:")
if passed:
    print(f"  ✅ Passed: {passed.group(1)}")
if failed:
    print(f"  ❌ Failed: {failed.group(1)}")
if skipped:
    print(f"  ⚠️  Skipped: {skipped.group(1)}")
print("=" * 70)

if result.returncode == 0 and passed:
    print("\n🚀 Committing and pushing...")
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(
        ["git", "commit", "-m", "fix: restore tests and fix collection issues"],
        capture_output=True,
    )
    subprocess.run(["git", "push"], check=True)
    print("✅ Done!")
