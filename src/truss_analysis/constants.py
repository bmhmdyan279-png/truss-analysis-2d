# ثابت‌های پروژه تحلیل خرپا

# ضریب انبساط حرارتی پیش‌فرض (برای فولاد)
DEFAULT_ALPHA = 1.2e-5

# تلورانس‌های عددی
ZERO_LENGTH_TOLERANCE = 1e-12

# دیکشنری تلورانس‌ها (برای سازگاری با کد قدیمی)
TOLERANCES = {
    "zero": 1e-12,
    "small": 1e-6,
    "energy": 1e-6,
    "singular": 1e-10,  # آستانه تشخیص ماتریس منفرد
}

# تنظیمات پیش‌فرض
DEFAULT_PENALTY_VALUE = 1e12
DEFAULT_DISPLACEMENT_SCALE = 50.0

# واحدهای پشتیبانی شده
SUPPORTED_UNITS = ["si", "imperial"]


# ── Magic Numbers (استخراج‌شده برای رفع PLR2004) ──────────────
# طول/جابجایی
LENGTH_THRESHOLD_MM = 0.001
LENGTH_THRESHOLD_MICRO = 1e-9
LENGTH_THRESHOLD_NANO = 1e-6
LENGTH_THRESHOLD_MICRO_M = 1e-3
LENGTH_THRESHOLD_KM = 1000.0
# نیرو
FORCE_THRESHOLD_MICRO_N = 0.001
FORCE_THRESHOLD_MILLI_N = 1e-3
FORCE_THRESHOLD_KN = 1e3
FORCE_THRESHOLD_MN = 1e6
# تنش
STRESS_THRESHOLD_KPA = 1e3
STRESS_THRESHOLD_MPA = 1e6
STRESS_THRESHOLD_GPA = 1e9
# انرژی
ENERGY_THRESHOLD_NJ = 1e-9
ENERGY_THRESHOLD_UJ = 1e-6
# ضرایب تبدیل
CONV_1E3 = 1e3
CONV_1E6 = 1e6
CONV_1E9 = 1e9
CONV_1E12 = 1e12
# تلرانس‌ها
ZERO_ENERGY_TOL = 1e-12
ZERO_DISP_TOL = 1e-10
MIN_NODES_SPAN = 2
# کمانش
BUCKLING_WARN = 0.5
BUCKLING_CRIT = 0.8
# مقیاس
MAX_SCALE = 1000.0
MIN_SCALE = 1.0
# اعتبارسنجی
VALIDATION_REL_ERR = 0.01
THERMAL_ENERGY_WARN = 0.1
MIN_SUPPORTS = 2
# نمودار
PIE_MIN_PCT = 5.0
FILTER_ENERGY_TOL = 1e-10
