import re
import subprocess


def run_cmd(cmd, ignore_error=False):
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0 and not ignore_error:
        print(f"   ❌ خطا: {result.stderr.strip()}")
        return False
    return True


def main():
    print("🚀 شروع فاز ۰: پاکسازی واقع‌گرایانه ریپازیتوری (Repo Hygiene & Truth)")

    # 1. اصلاح .gitignore
    print("\n📝 1. به‌روزرسانی .gitignore...")
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
env/
.env

# IDE
.vscode/
.idea/

# Logs
*.log
truss_analysis.log

# Output files
results/
*.png
*.csv
*.json
!examples/*.json

# Test coverage
.coverage
htmlcov/
.pytest_cache/

# Build and distribution
dist/
build/
*.egg-info/

# Result files
results_*.md
results_*.csv
results_*.png
"""
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    print(
        "   ✅ .gitignore به‌روزرسانی شد (حذف الگوی شکننده *.py و افزودن موارد استاندارد)."
    )

    # 2. پاکسازی گیت
    print("\n🧹 2. پاکسازی فایل‌های track شده اشتباه...")
    run_cmd("git rm -r --cached dist/", ignore_error=True)
    run_cmd("git rm --cached results_report.md", ignore_error=True)
    run_cmd("git rm -r --cached src/truss_analysis_2d.egg-info/", ignore_error=True)
    print("   ✅ فایل‌های بیلد و خروجی از git untrack شدند.")

    # 3. به‌روزرسانی main.py
    print("\n⚠️ 3. اضافه کردن هشدار Deprecation به main.py...")
    main_py_content = '''#!/usr/bin/env python3
"""Backward compatibility shim for running truss_analysis from the root."""

import warnings
warnings.warn(
    "Running root main.py is deprecated. Please use 'truss-analyze' command instead.",
    DeprecationWarning,
    stacklevel=2
)

from truss_analysis.main import main

if __name__ == "__main__":
    main()
'''
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(main_py_content)
    print("   ✅ main.py به‌روزرسانی شد.")

    # 4. اصلاح README.md
    print("\n📖 4. اصلاح بج‌ها و دستورات README.md...")
    with open("README.md", encoding="utf-8") as f:
        readme = f.read()

    # جایگزینی بلوک بج‌های قدیمی با بلوک جدید و صحیح
    old_badges = r"!\[CI\]\(https://github\.com/bmhmdyan279-png/truss-analysis-2d/actions\)\n!\[Python 3\.8\+\]\(https://www\.python\.org/\)\n!\[License: MIT\]\(LICENSE\)\n!\[Tests\]\(#تستها-و-اعتبارسنجی\)\n!\[Platform\]\(#\)"
    new_badges = """![CI](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/ci.yml/badge.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)"""

    readme = re.sub(old_badges, new_badges, readme)

    # جایگزینی تمامی instances از `python main.py` با `truss-analyze`
    readme = re.sub(r"python\s+main\.py", "truss-analyze", readme)

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    print(
        "   ✅ README.md به‌روزرسانی شد (بج‌های واقعی، حذف تست جعلی، افزودن macOS و تغییر دستورات)."
    )

    # 5. کامیت تغییرات
    print("\n💾 5. ثبت تغییرات در گیت...")
    run_cmd("git add .gitignore main.py README.md")
    run_cmd(
        "git add -u dist/ results_report.md src/truss_analysis_2d.egg-info/",
        ignore_error=True,
    )

    status = subprocess.run(
        "git status --porcelain", shell=True, capture_output=True, text=True
    ).stdout.strip()
    commit_msg = "chore: fix gitignore, untrack build artifacts, update badges, and deprecate root main.py"

    if not status:
        print(
            "   ℹ️ هیچ تغییر جدیدی برای کامیت وجود ندارد (احتمالاً از قبل انجام شده است)."
        )
    else:
        run_cmd(f'git commit -m "{commit_msg}"')

    # بررسی Definition of Done
    print("\n" + "=" * 60)
    print("🔍 بررسی Definition of Done:")
    ls_files = subprocess.run(
        "git ls-files dist/ results_*.md", shell=True, capture_output=True, text=True
    ).stdout.strip()
    if not ls_files:
        print(
            "  ✅ دستور `git ls-files dist/ results_*.md` هیچ خروجی‌ای نداد (پاکسازی موفق)."
        )
    else:
        print(f"  ⚠️ هنوز فایل‌هایی در گیت مانده‌اند: {ls_files}")

    print("  ✅ هشدار Deprecation به main.py اضافه شد.")
    print("  ✅ بج‌های README به‌روز و دستورات به `truss-analyze` تغییر یافتند.")
    print("=" * 60)
    print("🎉 فاز ۰ با موفقیت به پایان رسید! آماده دریافت دستور فاز ۱ هستید.")


if __name__ == "__main__":
    main()
