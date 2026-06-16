import os
print("✅ os import شد")

import sys
print("✅ sys import شد")

import json
print("✅ json import شد")

# حالا تست کنیم فایل وجود داره یا نه
if os.path.exists("examples/example1.json"):
    print("✅ فایل example1.json پیدا شد")
    with open("examples/example1.json", "r") as f:
        data = json.load(f)
    print(f"📁 فایل خوانده شد: {len(data.get('nodes', []))} گره")
else:
    print("❌ فایل example1.json پیدا نشد")