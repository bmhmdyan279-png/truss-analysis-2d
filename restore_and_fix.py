#!/usr/bin/env python3
"""
restore_and_fix.py — بازیابی test_solver.py و رفع نهایی
"""

import os
import subprocess

SRC = "src/truss_analysis"


def restore_test_solver():
    """بازیابی test_solver.py از git"""
    print("🔧 [1/3] بازیابی test_solver.py از git...")
    subprocess.run("git checkout HEAD -- tests/test_solver.py", shell=True)
    print("  ✅ Restored from git")


def fix_solver_tolerances():
    """حذف کامل تعریف تکراری TOLERANCES"""
    print("\n🔧 [2/3] رفع TOLERANCES تکراری در solver.py...")
    path = os.path.join(SRC, "solver.py")

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        # حذف کامنت قبل از تعریف
        if "# Fallback definition" in line:
            skip_next = True
            continue

        # حذف تعریف TOLERANCES اگر قبلاً import شده
        if skip_next and ("TOLERANCES = " in line or "TOLERANCES={" in line):
            skip_next = False
            continue

        # حذف هر تعریف TOLERANCES دیگر
        if (
            "TOLERANCES = " in line or "TOLERANCES={" in line
        ) and "from .constants import" not in "".join(lines[:i]):
            continue

        skip_next = False
        new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("  ✅ solver.py fixed")


def fix_test_solver_properly():
    """اصلاح صحیح test_solver.py"""
    print("\n🔧 [3/3] اصلاح صحیح test_solver.py...")
    path = "tests/test_solver.py"

    with open(path, encoding="utf-8") as f:
        content = f.read()

    # رفع assert is False
    content = content.replace(
        'assert element_result["buckling_warning"] is False',
        'assert not element_result["buckling_warning"]',
    )

    # اطمینان از اینکه importها در ابتدای فایل هستند
    lines = content.split("\n")

    # پیدا کردن docstring
    docstring_end = 0
    in_docstring = False
    for i, line in enumerate(lines):
        if '"""' in line:
            if not in_docstring:
                in_docstring = True
            else:
                in_docstring = False
                docstring_end = i + 1
                break

    # استخراج importها
    imports = []
    other_lines = []

    for i, line in enumerate(lines):
        if i < docstring_end:
            other_lines.append(line)
            continue

        stripped = line.strip()
        if (
            stripped.startswith("import ") or stripped.startswith("from ")
        ) and not stripped.startswith("    "):
            imports.append(line)
        else:
            other_lines.append(line)

    # بازسازی فایل
    new_content = "\n".join(other_lines[:docstring_end])
    new_content += "\n" + "\n".join(imports) + "\n"
    new_content += "\n".join(other_lines[docstring_end:])

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("  ✅ test_solver.py fixed")


def run_tools():
    """اجرای ابزارها"""
    print("\n🔧 اجرای ruff و pytest...")
    subprocess.run("ruff check --fix src/ tests/", shell=True)
    subprocess.run("ruff format src/ tests/", shell=True)
    subprocess.run("pytest -v", shell=True)


if __name__ == "__main__":
    print("═" * 60)
    print("🚀 بازیابی و رفع نهایی")
    print("═" * 60)

    restore_test_solver()
    fix_solver_tolerances()
    fix_test_solver_properly()
    run_tools()

    print("\n" + "═" * 60)
    print("✅ آماده commit!")
    print("═" * 60)
