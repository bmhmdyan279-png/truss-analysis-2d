import subprocess

print("🧹 اجرای نهایی Ruff روی کل مخزن و پاکسازی Staging...")

# 1. اجرای Ruff روی کل مخزن برای رفع تمام خطاهای باقی‌مانده
subprocess.run(["ruff", "check", ".", "--fix"])
subprocess.run(["ruff", "format", "."])

# 2. خارج کردن اسکریپت موقت از Staging (اگر هنوز باشد)
subprocess.run(["git", "rm", "--cached", "fix_b904.py"], capture_output=True)
subprocess.run(["git", "rm", "--cached", "finalize_phase21.py"], capture_output=True)

print("✅ ثبت کامیت نهایی و سبز فاز ۲.۱...")
subprocess.run(["git", "add", "."])
subprocess.run(
    [
        "git",
        "commit",
        "-m",
        "fix(solver): finalize exception chaining, formatting, and cleanup",
    ]
)
