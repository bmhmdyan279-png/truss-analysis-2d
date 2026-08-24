import subprocess

print("🗑️ حذف .venv از گیت (بدون حذف از دیسک)...")
subprocess.run(["git", "rm", "-r", "--cached", ".venv"])

print("\n🗑️ حذف fix_ruff_errors.py از گیت...")
subprocess.run(["git", "rm", "--cached", "fix_ruff_errors.py"])

print("\n📝 ساخت کامیت پاکسازی...")
subprocess.run(
    ["git", "commit", "-m", "chore: remove .venv and temp scripts from tracking"]
)

print("\n🚀 پوش به گیت‌هاب...")
subprocess.run(["git", "push", "origin", "main"])

print("\n✅ مخزن شما سبک و تمیز شد!")
print("محیط مجازی شما سالم باقی ماند.")
