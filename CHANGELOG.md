# Changelog

تمام تغییرات قابل توجه این پروژه در این فایل مستند می‌شوند.

فرمت بر اساس [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) است و این پروژه از [Semantic Versioning](https://semver.org/spec/v2.0.0.html) پیروی می‌کند.

## [1.7.0] - 2026-08-05

### 🎯 خلاصه
این نسخه شامل **بازنگری کامل در صحت محاسباتی، پایداری حل‌گر، و بهداشت مخزن** است. تمام ۱۸ ایراد بحرانی و بالایی که در سه نقد فنی مستقل شناسایی شده بودند، برطرف شدند. پروژه اکنون در سطح **Production-Ready** قرار دارد.

### 🔴 اصلاحات بحرانی (Critical Fixes)

#### صحت فیزیکی و ریاضی
- **باگ ۱۰۰ برابری مساحت**: `A_si = A_value**2` به `A_si = A * L_conv**2` اصلاح شد
- **تبدیل واحدهای ناقص**: `E`, `delta_L0`, `I`, `Fx`, `Fy` اکنون با `convert_to_si` کامل تبدیل می‌شوند
- **ایندکس‌گذاری گره‌ها**: استفاده از `node_id` واقعی به جای ایندکس ترتیبی در `load["id"]`

#### پایداری حل‌گر (Solver Robustness)
- **حذف Silent Failure**: `np.zeros` و `lstsq` برای ماتریس‌های منفرد حذف شدند
- **الگوی Fail-Fast**: پرتاب `TRUSS-3001` برای مکانیزم‌ها و ماتریس‌های منفرد
- **هماهنگ‌سازی ثابت‌ها**: `BUCKLING_WARN`, `THERMAL_ENERGY_WARN`, `DEFAULT_PENALTY_VALUE`

### 🟠 اصلاحات با اولویت بالا

#### یکپارچگی تست‌ها (Test Integrity)
- **`test_golden.py`**: تبدیل `if/else print` به `assert` واقعی
- **`test_energy_balance.py`**: سفت‌کردن تلورانس انرژی از ۱۰۰٪ به ۱٪ (استاندارد FEM)
- **`test_negative_validation.py`**: حذف `Exception` عمومی از لیست catches

#### بهداشت مخزن (Repo Hygiene)
- حذف ۱۴+ فایل موقت (`fix_*.py`, `final_*.py`, `phase*.py`, `$null`)
- حذف پوشه `dist/` از tracking
- اصلاح `.gitignore` برای build artifacts

#### معماری و نسخه‌بندی
- **Single Source of Truth**: `__version__` از `importlib.metadata` خوانده می‌شود
- **حذف Side-Effect**: `logging.basicConfig` از `__init__.py` به `main()` منتقل شد
- **هماهنگ‌سازی CI Matrix**: تست در پایتون 3.9 تا 3.12 در تمام jobها
- **سازگاری پایتون ۳.۹**: اضافه‌کردن `from __future__ import annotations`

### 🟡 بهبودهای کیفیت کد

#### پوشش تست واقعی
- حذف `main.py`, `fileio.py`, `postprocess.py` از لیست `omit`
- افزایش `fail_under` از ۷۰٪ به ۹۰٪

#### Code Style و Linting
- حذف تمام `# noqa: E402` با مرتب‌سازی صحیح importها
- اصلاح سایه‌سازی builtin `id` به `node_id`
- حذف `.toarray()` روی ماتریس‌های sparse (حفظ مزیت حافظه)
- رفع خطاهای Ruff (B904, B007, E402)

#### وابستگی‌ها
- حذف `arabic-reshaper` و `python-bidi` از dependencies اجباری

### 📋 تغییرات فنی دقیق

| فایل | تغییر |
|------|-------|
| `model.py` | تبدیل کامل واحدها، اصلاح `node_id`، حذف `# noqa: E402` |
| `solver.py` | Fail-Fast، حذف `.toarray()`، هماهنگ‌سازی ثابت‌ها |
| `__init__.py` | حذف Side-Effect لاگر، اتصال به `importlib.metadata` |
| `main.py` | انتقال `logging.basicConfig` به داخل تابع، اصلاح shebang |
| `constants.py` | هماهنگ‌سازی `BUCKLING_WARN`, `THERMAL_ENERGY_WARN`, `DEFAULT_PENALTY_VALUE` |
| `.github/workflows/ci.yml` | هماهنگ‌سازی matrix برای 3.9-3.12 |
| `pyproject.toml` | حذف `omit`ها، افزایش `fail_under` به ۹۰٪ |
| `tests/*.py` | تبدیل به `assert` واقعی، سفت‌کردن تلورانس‌ها |

### 🎯 تأثیر بر کاربران
- **کاربران Imperial units**: تبدیل واحدها اکنون کاملاً صحیح است
- **مهندسین سازه**: مکانیزم‌ها با کد خطای صریح متوقف می‌شوند (نه نتایج صفر)
- **توسعه‌دهندگان**: تست‌ها واقعاً کار می‌کنند و پوشش ۹۰٪ واقعی است
- **مصرف‌کنندگان کتابخانه**: `import truss_analysis` دیگر لاگ نمی‌سازد

### 🔧 شکستن سازگاری (Breaking Changes)
هیچ تغییر API عمومی وجود ندارد. اما:
- `solver.solve_displacements` اکنون به جای بازگرداندن `np.zeros` در خطا، `ValueError` با کد `TRUSS-3001` پرتاب می‌کند
- `logging.basicConfig` دیگر در زمان `import` اجرا نمی‌شود

## [1.6.0] - 2026-08-01

### اضافه شده
- اعتبارسنجی با Pydantic v2
- کدهای خطای ساختاریافته (TRUSS-1xxx تا TRUSS-4xxx)
- پشتیبانی از چندین دستگاه واحد (SI, SI-mm, SI-cm, Imperial)
- اثرات حرارتی و خطای ساخت
- کمانش اویلر

### اصلاح شده
- بهبود مدیریت استثناها
- بهینه‌سازی ماتریس‌های sparse

### تغییر یافته
- بازنویسی کامل معماری ماژولار

## [1.5.0] - 2026-07-15

### اضافه شده
- لاگ‌نویسی با چرخش خودکار
- اعتبارسنجی تعادل انرژی
- رسم نمودارهای نتایج

## [1.4.0] - 2026-07-01

### اضافه شده
- نسخه اولیه تحلیل‌گر خرپای 2D
- پشتیبانی از شرایط مرزی حذف و پنالتی
- محاسبه نیروهای اعضا

[1.7.0]: https://github.com/bmhmdyan279-png/truss-analysis-2d/releases/tag/v1.7.0
[1.6.0]: https://github.com/bmhmdyan279-png/truss-analysis-2d/releases/tag/v1.6.0
[1.5.0]: https://github.com/bmhmdyan279-png/truss-analysis-2d/releases/tag/v1.5.0
[1.4.0]: https://github.com/bmhmdyan279-png/truss-analysis-2d/releases/tag/v1.4.0
