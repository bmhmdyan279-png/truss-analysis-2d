"""
ثابت‌های پروژه تحلیل خرپا
"""

import numpy as np

# قرارداد علامت برای نیروها
FORCE_CONVENTION = {
    'thermal': 'compression',  # انبساط حرارتی → نیروی فشاری
    'mechanical': 'tension_positive'  # نیروی کششی مثبت
}

# آستانه‌های عددی
TOLERANCES = {
    'zero': 1e-12,      # صفر عددی
    'small': 1e-8,      # مقادیر خیلی کوچک
    'medium': 1e-6,     # دقت متوسط
    'large': 1e-4,      # دقت پایین
    'energy': 1e-6,     # دقت انرژی
    'force_balance': 1e-3,  # دقت تعادل نیرو
    'singular': 1e-12   # تشخیص ماتریس منفرد
}

# تبدیل واحدها
UNIT_CONVERSIONS = {
    'SI': {'length': 1.0, 'area': 1.0, 'force': 1.0},
    'SI-mm': {'length': 0.001, 'area': 1e-6, 'force': 1.0},
    'SI-cm': {'length': 0.01, 'area': 1e-4, 'force': 1.0},
    'Imperial': {'length': 0.3048, 'area': 0.0929, 'force': 4.44822}
}