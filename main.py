#!/usr/bin/env python3
import io
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

"""
Backward compatibility shim for running truss_analysis from the root.

Exit Code Taxonomy:
  0: Success (موفقیت)
  1: Input error (خطای ورودی - مثلاً فایل نامعتبر یا داده ناقص)
  2: Solver error (خطای حل‌گر - مثلاً ماتریس منفرد یا عدم همگرایی)
  3: Output/Visualization error (خطای خروجی/بصری‌سازی - مثلاً خطا در ذخیره نمودار)
  4: Internal error (خطای داخلی - استثناهای غیرمنتظره)
"""


# مدیریت TTY در ویندوز برای جلوگیری از Mojibake


try:
    from truss_analysis.main import main as app_main

    if __name__ == "__main__":
        sys.exit(app_main())
except ImportError as e:
    print(
        f"Internal error (4): Failed to import truss_analysis.main ({e})",
        file=sys.stderr,
    )
    sys.exit(4)
except Exception as e:
    print(f"Internal error (4): {e}", file=sys.stderr)
    sys.exit(4)
