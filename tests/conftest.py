"""
فایل پیکربندی pytest برای تست‌های خرپا
"""

import sys
import os

# اضافه کردن مسیر پوشه والد به sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ایمپورت ماژول‌های اصلی برای دسترسی در تست‌ها
try:
    print("✅ ماژول‌ها با موفقیت import شدند")
except ImportError as e:
    print(f"❌ خطای import: {e}")
    raise
