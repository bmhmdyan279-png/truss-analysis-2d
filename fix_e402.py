#!/usr/bin/env python3
"""
fix_e402.py — رفع خطای E402 در test_solver.py
"""

path = "tests/test_solver.py"
with open(path, encoding="utf-8") as f:
    content = f.read()

# حذف docstring از ابتدای فایل
lines = content.split("\n")
new_lines = []
in_docstring = False
docstring_removed = False

for line in lines:
    if '"""' in line and not docstring_removed:
        if not in_docstring:
            in_docstring = True
            continue
        else:
            in_docstring = False
            docstring_removed = True
            continue
    if in_docstring:
        continue
    new_lines.append(line)

# اطمینان از اینکه import pytest در ابتدای فایل است
content = "\n".join(new_lines)
if not content.startswith("import pytest"):
    content = "import pytest\n\n" + content

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ test_solver.py fixed - docstring removed")
print("\n📝 حالا commit کنید:")
print("   git add .")
print('   git commit -m "v1.4.0: Professional Polish"')
print("   git push origin main")
