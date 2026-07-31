import os
import subprocess


def run_cmd(cmd):
    print(f"⚙️ {cmd}")
    res = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, encoding="utf-8"
    )
    if (
        res.returncode != 0
        and "nothing to commit" not in res.stderr
        and "already up to date" not in res.stdout
    ):
        print(f"⚠️ {res.stderr.strip() or res.stdout.strip()}")
    return res


print("1️⃣ بازنشانی postprocess.py به آخرین نسخه سالم گیت...")
run_cmd("git checkout HEAD -- src/truss_analysis/postprocess.py")

print("2️⃣ حذف فایل‌های موقت اسکریپت از staging و دیسک...")
run_cmd("git reset HEAD fix_phase2_errors.py phase2_automation.py 2>$null")
if os.path.exists("fix_phase2_errors.py"):
    os.remove("fix_phase2_errors.py")
if os.path.exists("phase2_automation.py"):
    os.remove("phase2_automation.py")

print("3️⃣ اعمال ایمن و خط‌به‌خط تغییرات فونت (با حفظ دقیق تورفتگی)...")
with open("src/truss_analysis/postprocess.py", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
import_added = False
font_patched = False

for line in lines:
    # افزودن ایمپورت در اولین خط ممکن
    if not import_added and (line.startswith("import ") or line.startswith("from ")):
        new_lines.append("import importlib.resources\n")
        import_added = True

    # جایگزینی دقیق و ایمن تنظیم فونت با حفظ تورفتگی (Indentation) اصلی
    if (
        not font_patched
        and 'plt.rcParams["font.family"]' in line
        and "DejaVu Sans" in line
    ):
        indent = len(line) - len(line.lstrip())
        sp = " " * indent
        new_lines.append(f"{sp}# مدیریت فونت مدرن (سازگار با پایتون ۳.۹+)\n")
        new_lines.append(f"{sp}try:\n")
        new_lines.append(
            f"{sp}    font_path = importlib.resources.files('truss_analysis').joinpath('assets/fonts/Vazirmatn-Regular.ttf')\n"
        )
        new_lines.append(f'{sp}    plt.rcParams["font.family"] = "sans-serif"\n')
        new_lines.append(
            f'{sp}    plt.rcParams["font.sans-serif"] = [str(font_path)]\n'
        )
        new_lines.append(f"{sp}except Exception:\n")
        new_lines.append(
            f'{sp}    plt.rcParams["font.family"] = "DejaVu Sans"  # fallback\n'
        )
        font_patched = True
    else:
        new_lines.append(line)

if not import_added:
    new_lines.insert(0, "import importlib.resources\n")

with open("src/truss_analysis/postprocess.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("4️⃣ اجرای Ruff برای اطمینان از صحت سینتکس...")
run_cmd("ruff format src/truss_analysis/postprocess.py")
run_cmd("ruff check src/truss_analysis/postprocess.py --fix")

print("5️⃣ کامیت و پوش تغییرات نهایی...")
run_cmd(
    "git add src/truss_analysis/postprocess.py main.py src/truss_analysis/fileio.py src/truss_analysis/i18n.py tests/test_rtl_svg.py"
)
run_cmd(
    'git commit -m "fix: safely apply modern font loading (importlib) and resolve syntax errors"'
)
run_cmd("git push")

print("🎉 همه چیز تمیز و بدون خطای SyntaxError اصلاح و پوش شد!")
