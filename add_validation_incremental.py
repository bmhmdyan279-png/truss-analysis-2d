# add_validation_incremental.py
import re

path = "src/truss_analysis/model.py"

with open(path, encoding="utf-8") as f:
    content = f.read()

# 1. اضافه کردن validation به __init__ در TrussModel
init_pattern = r'(class TrussModel.*?def __init__\(self.*?\):.*?"""[\s\S]*?""")'
match = re.search(init_pattern, content, re.DOTALL)

if match:
    old_init = match.group(1)

    # اضافه کردن validation بعد از docstring
    validation_code = """
        # ✅ Validation ورودی
        if not isinstance(input_data, dict):
            raise TypeError(
                f"input_data must be a dict, got {type(input_data).__name__}"
            )
"""

    new_init = old_init + validation_code
    content = content.replace(old_init, new_init)
    print("✅ Added validation to __init__")
else:
    print("⚠️  Could not find __init__ method")

# 2. اضافه کردن validation به _create_nodes
nodes_pattern = r'(def _create_nodes\(self.*?\):.*?"""[\s\S]*?""")'
match = re.search(nodes_pattern, content, re.DOTALL)

if match:
    old_method = match.group(1)

    validation_code = """
        # ✅ Validation نوع
        if not isinstance(nodes_data, list):
            raise TypeError(
                f"'nodes' must be a list, got {type(nodes_data).__name__}"
            )
"""

    new_method = old_method + validation_code
    content = content.replace(old_method, new_method)
    print("✅ Added validation to _create_nodes")

# 3. ذخیره
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Validation added incrementally")
