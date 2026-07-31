# fix_main_direct.py
import re

path = "src/truss_analysis/main.py"
with open(path, encoding="utf-8") as f:
    content = f.read()

# چاپ 30 خط اول برای دیباگ
print("📄 محتوای 30 خط اول main.py:")
print("\n".join(content.split("\n")[:30]))
print("\n" + "=" * 60)

# یافتن try/except با import مطلق
mods = [
    "model",
    "utils",
    "constants",
    "exceptions",
    "assembly",
    "solver",
    "fileio",
    "postprocess",
]

# تبدیل همه importهای مطلق به relative
for m in mods:
    # داخل try/except هم باید تبدیل شود
    content = re.sub(rf"from\s+{m}\b(\s+import)", rf"from .{m}\1", content)
    content = re.sub(rf"import\s+{m}\b", rf"from . import {m}", content)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("\n✅ main.py اصلاح شد")
print("\n📝 حالا تست کن:")
print("   truss-analyze examples/reference_problem.json")
