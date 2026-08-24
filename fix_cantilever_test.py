# fix_cantilever_test.py
import subprocess
import sys
from pathlib import Path

print("🔧 Fixing Cantilever test with correct analytical values...")

test_path = Path("tests/test_golden_phase3.py")
if not test_path.exists():
    print(f"❌ File not found: {test_path}")
    sys.exit(1)

text = test_path.read_text(encoding="utf-8")

# اصلاح docstring
old_docstring = '''    """
    Cantilever Truss with load at free end.
    Analytical solution:
    - AB = P = 10 (tension)
    - BC = P\\u221a2 \\u2248 14.14 (tension)
    - AC = P = 10 (tension)
    - Reactions: Rx_A = 10, Ry_A = 10, Rx_C = -10, Ry_C = 0
    """'''

new_docstring = '''    """
    Cantilever Truss with load at free end.
    Analytical solution:
    - AB = -P = -10 (compression)
    - BC = P\\u221a2 \\u2248 14.14 (tension)
    - AC = -P = -10 (compression)
    - Reactions: Rx_A = 10, Ry_A = 10, Rx_C = -10, Ry_C = 0
    """'''

text = text.replace(old_docstring, new_docstring)

# اصلاح assertions برای AB و AC
old_assertions = """            assert abs(force_dict["AB"] - 10.0) < 1e-6
            assert abs(force_dict["BC"] - 10 * 2**0.5) < 1e-5
            assert abs(force_dict["AC"] - 10.0) < 1e-6"""

new_assertions = """            assert abs(force_dict["AB"] - (-10.0)) < 1e-6
            assert abs(force_dict["BC"] - 10 * 2**0.5) < 1e-5
            assert abs(force_dict["AC"] - (-10.0)) < 1e-6"""

text = text.replace(old_assertions, new_assertions)

test_path.write_text(text, encoding="utf-8")
print("✅ Test file updated with correct analytical values")

# اجرای تست
print("\n🧪 Running tests...")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_golden_phase3.py", "-v"],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.returncode == 0:
    print("\n🎉 All tests passed!")

    # کامیت و پوش
    print("\n🚀 Committing and pushing...")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "fix(tests): correct analytical values for cantilever truss",
        ],
        check=True,
    )
    subprocess.run(["git", "push"], check=True)
    print("✅ Successfully pushed to GitHub")
    print("\n🎊 Phase 3 is now COMPLETE!")
else:
    print(f"\n❌ Tests still failed:\n{result.stdout}")
    sys.exit(1)
