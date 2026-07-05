# ثابت‌های پروژه تحلیل خرپا

# ضریب انبساط حرارتی پیش‌فرض (برای فولاد)
DEFAULT_ALPHA = 1.2e-5

# تلورانس‌های عددی
ZERO_LENGTH_TOLERANCE = 1e-12

# دیکشنری تلورانس‌ها (برای سازگاری با کد قدیمی)
TOLERANCES = {
    'zero': 1e-12,
    'small': 1e-6,
    'energy': 1e-6,
    'singular': 1e-10  # آستانه تشخیص ماتریس منفرد
}

# تنظیمات پیش‌فرض
DEFAULT_PENALTY_VALUE = 1e10
DEFAULT_DISPLACEMENT_SCALE = 50.0

# واحدهای پشتیبانی شده
SUPPORTED_UNITS = ['si', 'imperial']
