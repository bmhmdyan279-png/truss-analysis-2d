#!/usr/bin/env python3
"""
fix_all_absolute.py — رفع تمام importهای مطلق در src/
این بار بدون محدودیت ^ در regex، همه جا را می‌گیرد.
"""

import os
import re

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


def fix_file(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()

    original = content

    for m in MODS:
        # از assembly import -> from .assembly import (هر جا در خط)
        content = re.sub(rf"\bfrom\s+{m}\b(\s+import)", rf"from .{m}\1", content)
        # import assembly -> from . import assembly (هر جا در خط)
        content = re.sub(rf"\bimport\s+{m}\b", rf"from . import {m}", content)

    # اصلاح خاص __init__.py: from truss_analysis.main -> from .main
    if path.endswith("__init__.py"):
        content = content.replace(
            "from truss_analysis.main import main", "from .main import main"
        )

    if content != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def main():
    print("🔧 رفع تمام importهای مطلق در src/...")
    fixed = []
    for fname in os.listdir(SRC):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(SRC, fname)
        if fix_file(path):
            fixed.append(fname)
            print(f"  ✅ {fname}")

    if not fixed:
        print("  ⚠️  هیچ فایلی تغییر نکرد")
    else:
        print(f"\n✅ {len(fixed)} فایل اصلاح شد")

    # بررسی نهایی: جستجوی باقی‌مانده‌ها
    print("\n🔍 بررسی نهایی برای import مطلق باقی‌مانده...")
    found_any = False
    for fname in os.listdir(SRC):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(SRC, fname)
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                for m in MODS:
                    # الگوی import مطلق (نه relative)
                    if re.search(
                        rf"(?<!\.)(?:from\s+{m}\b\s+import|import\s+{m}\b)", line
                    ):
                        # استثنا: داخل کامنت یا string
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        print(f"  ⚠️  {fname}:{i}: {stripped}")
                        found_any = True

    if not found_any:
        print("  ✅ هیچ import مطلقی یافت نشد!")


if __name__ == "__main__":
    main()
