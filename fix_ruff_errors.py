# fix_ruff_errors.py
import os
import re

SRC = "src/truss_analysis"


def fix_solver_tolerances():
    print("🔧 رفع TOLERANCES duplicate در solver.py...")
    path = os.path.join(SRC, "solver.py")

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # حذف تعریف تکراری TOLERANCES
    new_lines = []
    seen_tolerances = False
    for line in lines:
        if "TOLERANCES = " in line or "TOLERANCES={" in line:
            if seen_tolerances:
                continue  # skip duplicate
            seen_tolerances = True
        new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("  ✅ solver.py fixed")


def fix_test_edge_cases():
    print("\n🔧 رفع خطاهای test_edge_cases.py...")
    path = "tests/test_edge_cases.py"

    with open(path, encoding="utf-8") as f:
        content = f.read()

    # حذف import pytest از وسط فایل
    content = re.sub(r"\nimport pytest\n", "\n", content)

    # رفع F841: حذف e1
    content = content.replace("e1 = Element(", "Element(")

    # رفع B011: تغییر assert False به raise
    content = content.replace(
        'assert False, "Should raise ValueError"',
        'raise AssertionError("Should raise ValueError")',
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  ✅ test_edge_cases.py fixed")


def fix_test_solver():
    print("\n🔧 رفع خطاهای test_solver.py...")
    path = "tests/test_solver.py"

    with open(path, encoding="utf-8") as f:
        content = f.read()

    # حذف import pytest از وسط فایل
    content = re.sub(r"\nimport pytest\n", "\n", content)

    # رفع E712: تغییر == False به is False
    content = content.replace("== False", "is False")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  ✅ test_solver.py fixed")


if __name__ == "__main__":
    fix_solver_tolerances()
    fix_test_edge_cases()
    fix_test_solver()
    print("\n✅ حالا ruff باید پاس شود")
