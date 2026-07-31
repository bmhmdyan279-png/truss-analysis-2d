# fix_pyproject.py
import os

path = "pyproject.toml"
if not os.path.exists(path):
    print("❌ pyproject.toml not found")
    exit(1)

with open(path, encoding="utf-8") as f:
    content = f.read()

# اصلاح بخش ruff.lint
old_ignore = """ignore = [
    "E501",  # line too long (handled by formatter)
    "B008",  # do not perform function calls in argument defaults
]"""

new_ignore = """ignore = [
    "E501",   # line too long (handled by formatter)
    "B008",   # do not perform function calls in argument defaults
    "N803",   # Argument name should be lowercase (engineering: K, F, E, A)
    "N806",   # Variable in function should be lowercase (engineering: K_global, F_f)
    "N802",   # Function name should be lowercase (engineering: test_element_delta_L_free)
    "UP006",  # Use dict instead of Dict (typing compatibility)
    "UP035",  # typing.Dict is deprecated (typing compatibility)
    "SIM105", # Use contextlib.suppress (style preference)
    "SIM108", # Use ternary operator (style preference)
    "SIM116", # Use dictionary instead of if-else (style preference)
    "SIM101", # Multiple isinstance calls (style preference)
]"""

content = content.replace(old_ignore, new_ignore)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ pyproject.toml updated - engineering conventions allowed")
