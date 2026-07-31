#!/usr/bin/env python3
"""
final_fix.py — رفع نهایی تمام مشکلات v1.4.0
"""

import os
import re
import subprocess

SRC = "src/truss_analysis"
MODS = [
    "model",
    "utils",
    "constants",
    "exceptions",
    "assembly",
    "solver",
    "fileio",
    "postprocess",
]


def fix_src_imports():
    """تبدیل import مطلق به relative در تمام فایل‌های src/"""
    print("🔧 [1/4] رفع import مطلق در src/...")
    for fname in os.listdir(SRC):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(SRC, fname)
        with open(path, encoding="utf-8") as f:
            content = f.read()

        original = content
        for m in MODS:
            # from model import -> from .model import
            content = re.sub(
                rf"^from\s+{m}\b(\s+import)",
                rf"from .{m}\1",
                content,
                flags=re.MULTILINE,
            )
            # import model -> from . import model
            content = re.sub(
                rf"^import\s+{m}\s*$", f"from . import {m}", content, flags=re.MULTILINE
            )

        if content != original:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ {fname}")


def fix_solver_tolerances():
    """حذف تعریف تکراری TOLERANCES"""
    print("\n🔧 [2/4] رفع TOLERANCES تکراری در solver.py...")
    path = os.path.join(SRC, "solver.py")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    found_first = False
    for i, line in enumerate(lines):
        if "TOLERANCES = " in line or "TOLERANCES={" in line:
            if found_first:
                continue  # حذف تکراری
            found_first = True
        # حذف کامنت قبل از تعریف تکراری
        if (
            "# Fallback definition" in line
            and i + 1 < len(lines)
            and ("TOLERANCES = " in lines[i + 1] or "TOLERANCES={" in lines[i + 1])
        ):
            continue
        new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("  ✅ solver.py fixed")


def fix_test_files():
    """رفع مشکلات تست‌ها"""
    print("\n🔧 [3/4] رفع مشکلات تست‌ها...")

    # ─── test_edge_cases.py ───
    path = "tests/test_edge_cases.py"
    with open(path, encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    imports = [
        "import pytest",
        "import numpy as np",
        "from truss_analysis.model import Element, Node, TrussModel",
        "from truss_analysis.exceptions import InputDataError, SingularMatrixError",
    ]

    new_lines = imports.copy()
    new_lines.append("")

    for line in lines:
        stripped = line.strip()
        # حذف importهای قدیمی
        if any(stripped == imp for imp in imports):
            continue
        if stripped.startswith("from truss_analysis."):
            continue

        # 修复 e1/e2
        if "Element(1, n1, n2, E=-200e9, A=0.01)" in line and "e1 =" not in line:
            line = line.replace("Element(1,", "e1 = Element(1,")
        if "Element(2, n1, n2, E=200e9, A=-0.01)" in line and "e2 =" not in line:
            line = line.replace("Element(2,", "e2 = Element(2,")

        new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
    print("  ✅ test_edge_cases.py fixed")

    # ─── test_solver.py ───
    path = "tests/test_solver.py"
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # رفع assert is False
    content = content.replace(
        'assert element_result["buckling_warning"] is False',
        'assert not element_result["buckling_warning"]',
    )

    # انتقال importها به ابتدای فایل
    lines = content.split("\n")
    imports = []
    other_lines = []

    for line in lines:
        # فقط importهای سطح ماژول (بدون indent)
        if (
            line.startswith("import ") or line.startswith("from ")
        ) and not line.startswith("    "):
            imports.append(line)
        else:
            other_lines.append(line)

    new_content = "\n".join(imports) + "\n\n" + "\n".join(other_lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("  ✅ test_solver.py fixed")


def run_tools():
    """اجرای ابزارها"""
    print("\n🔧 [4/4] اجرای ruff و pytest...")

    cmds = ["ruff check --fix src/ tests/", "ruff format src/ tests/", "pytest -v"]

    for cmd in cmds:
        print(f"\n{'─' * 60}")
        print(f"▶ {cmd}")
        print(f"{'─' * 60}")
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    print("═" * 60)
    print("🚀 رفع نهایی تمام مشکلات v1.4.0")
    print("═" * 60)

    fix_src_imports()
    fix_solver_tolerances()
    fix_test_files()
    run_tools()

    print("\n" + "═" * 60)
    print("✅ تمام اصلاحات انجام شد!")
    print("\n📝 مراحل بعدی:")
    print("   git add .")
    print('   git commit -m "v1.4.0: Professional Polish"')
    print("   git push origin main")
    print("\n⚠️  خطای PyPI (403 Forbidden) طبیعی است — نیاز به API Token دارد.")
    print("   بعداً می‌توانید با دستور زیر آپلود کنید:")
    print("   twine upload dist/*")
    print("═" * 60)
