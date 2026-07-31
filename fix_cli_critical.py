# fix_cli_critical.py
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


def fix_all_imports():
    """تبدیل تمام importهای مطلق به relative در src/"""
    print("🔧 رفع importهای مطلق در تمام فایل‌های src/...")

    for fname in os.listdir(SRC):
        if not fname.endswith(".py"):
            continue

        path = os.path.join(SRC, fname)
        with open(path, encoding="utf-8") as f:
            content = f.read()

        original = content

        # تبدیل import مطلق به relative
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


if __name__ == "__main__":
    fix_all_imports()
    print("\n✅ حالا CLI باید کار کند!")
    print("\n📝 تست کن:")
    print("   truss-analyze examples/reference_problem.json")
    print("\n📝 اگر کار کرد، commit کن:")
    print("   git add .")
    print('   git commit -m "fix: convert to relative imports for CLI"')
    print("   git push origin main")
