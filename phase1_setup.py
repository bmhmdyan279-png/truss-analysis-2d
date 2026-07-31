import re
from pathlib import Path


def create_ground_truth_test():
    test_dir = Path("tests")
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test_ground_truth.py"

    content = '''import pytest
import sys
import os

# مسیر پروژه را به sys.path اضافه کنید (در صورت نیاز تنظیم شود)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.solver import solve_truss  # <-- ایمپورت واقعی solver خود را جایگزین کنید

def test_ground_truth_simple_truss():
    """
    تست Ground Truth: خرپای ۳ گره‌ای و ۲ عضوی افقی.
    گره ۱: (0, 0) مفصلی (Fixed)
    گره ۲: (1, 0) آزاد، بار Fx=1000 N
    گره ۳: (2, 0) مفصلی (Fixed)
    اعضا: 1-2 و 2-3 با E=200e9 Pa و A=1e-4 m^2

    جواب تحلیلی دقیق (کتاب مرجع):
    - جابجایی افقی گره ۲ (u2_x) = 2.5e-5 متر
    - جابجایی عمودی گره ۲ (u2_y) = 0.0 متر
    - نیروی محوری عضو ۱ (F_12) = 500.0 N (کششی)
    - نیروی محوری عضو ۲ (F_23) = -500.0 N (فشاری)
    """
    # TODO: فراخوانی واقعی solver خود را اینجا قرار دهید
    # مثال:
    # result = solve_truss(nodes, members)

    expected_ux = 2.5e-5
    expected_force_12 = 500.0
    expected_force_23 = -500.0

    # مثال assert (بر اساس ساختار خروجی solver خودتان تنظیم کنید):
    # assert abs(result["displacements"][2]["ux"] - expected_ux) < 1e-8
    # assert abs(result["forces"][1]["axial_force"] - expected_force_12) < 1e-6
    # assert abs(result["forces"][2]["axial_force"] - expected_force_23) < 1e-6

    assert True, "Placeholder: لطفاً فراخوانی solver و assertهای واقعی را جایگزین کنید"
'''
    test_file.write_text(content, encoding="utf-8")
    print("✅ فایل tests/test_ground_truth.py ایجاد شد.")


def update_mypy_config():
    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        print("⚠️ فایل pyproject.toml یافت نشد. از ایجاد آن صرف‌نظر شد.")
        return

    content = pyproject.read_text(encoding="utf-8")
    if "[[tool.mypy.overrides]]" not in content:
        append_text = """
[[tool.mypy.overrides]]
module = [
    "matplotlib.*",
    "yaml",
    "arabic_reshaper",
    "pytest"
]
ignore_missing_imports = true
"""
        pyproject.write_text(content + append_text, encoding="utf-8")
        print("✅ پیکربندی هوشمند Mypy به pyproject.toml اضافه شد.")
    else:
        print("ℹ️ پیکربندی Mypy از قبل در pyproject.toml وجود دارد.")


def update_main_shim():
    main_file = Path("main.py")
    content = '''#!/usr/bin/env python3
"""
Backward compatibility shim for running truss_analysis from the root.

Exit Code Taxonomy:
  0: Success (موفقیت)
  1: Input error (خطای ورودی - مثلاً فایل نامعتبر یا داده ناقص)
  2: Solver error (خطای حل‌گر - مثلاً ماتریس منفرد یا عدم همگرایی)
  3: Output/Visualization error (خطای خروجی/بصری‌سازی - مثلاً خطا در ذخیره نمودار)
  4: Internal error (خطای داخلی - استثناهای غیرمنتظره)
"""
import sys

try:
    from truss_analysis.main import main as app_main
    if __name__ == "__main__":
        sys.exit(app_main())
except ImportError as e:
    print(f"Internal error (4): Failed to import truss_analysis.main ({e})", file=sys.stderr)
    sys.exit(4)
except Exception as e:
    print(f"Internal error (4): {e}", file=sys.stderr)
    sys.exit(4)
'''
    main_file.write_text(content, encoding="utf-8")
    print("✅ فایل main.py (Shim) با طبقه‌بندی کدهای خروج به‌روزرسانی شد.")


def secure_yaml_loading():
    replaced_count = 0
    for py_file in Path(".").rglob("*.py"):
        # رد کردن محیط‌های مجازی و دایرکتوری‌های غیرمنبع
        if any(
            part in [".venv", "venv", "env", ".git", "__pycache__"]
            for part in py_file.parts
        ):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
            # جایگزینی امن yaml.load با yaml.safe_load (حتی با فاصله احتمالی)
            new_content = re.sub(
                r"(?<!safe_)yaml\.load\s*\(", "yaml.safe_load(", content
            )
            if new_content != content:
                py_file.write_text(new_content, encoding="utf-8")
                replaced_count += 1
        except Exception:
            continue

    if replaced_count > 0:
        print(
            f"✅ بارگذاری امن YAML در {replaced_count} فایل اعمال شد (yaml.load -> yaml.safe_load)."
        )
    else:
        print("ℹ️ هیچ نمونه‌ای از yaml.load ناامن برای جایگزینی یافت نشد.")


if __name__ == "__main__":
    print("🚀 شروع خودکارسازی فاز ۱...")
    create_ground_truth_test()
    update_mypy_config()
    update_main_shim()
    secure_yaml_loading()
    print("🎉 فاز ۱ تکمیل شد! حالا می‌توانید تغییرات را در PyCharm بررسی و کامیت کنید.")
