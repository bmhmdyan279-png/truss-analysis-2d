# 🏗️ تحلیل سازه‌های خرپایی دوبعدی (Truss Analysis 2D)

<div align="center">

![Python Version](https://img.shields.io/badge/python-≥3.9-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-2.0.4-orange.svg)
[![CI Status](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/ci.yml/badge.svg)](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions)

**یک ابزار تحلیل خرپای دوبعدی کاملاً پایتون، با معماری Fail-Fast، اعتبارسنجی سخت‌گیرانه انرژی و پشتیبانی کامل از شرایط مرزی مستقل**

</div>

---

## 📋 فهرست مطالب

- [ویژگی‌های کلیدی](#-ویژگی‌های-کلیدی)
- [معماری](#-معماری)
- [نصب](#-نصب)
- [استفاده سریع](#-استفاده-سریع)
- [مثال کامل](#-مثال-کامل)
- [سیستم واحد](#-سیستم-واحد)
- [اعتبارسنجی انرژی](#-اعتبارسنجی-انرژی)
- [تست‌ها](#-تست‌ها)
- [ساختار پروژه](#-ساختار-پروژه)
- [تغییرات نسخه](#-تغییرات-نسخه)
- [مجوز](#-مجوز)

---

## ✨ ویژگی‌های کلیدی

### 🎯 معماری مدرن و تمیز
- **Pure DTO Architecture**: جداسازی کامل داده (`model.py`)، اسمبل (`assembly.py`) و حل (`solver.py`)
- **Fail-Fast Validation**: تشخیص فوری خطاهای هندسی، ماتریس‌های تکین و عدم تعادل انرژی
- **Zero Side-Effects**: تمام محاسبات در توابع خالص، بدون تغییر وضعیت پنهان
- **Centralized Exceptions**: مدیریت متمرکز خطاها در `exceptions.py`

### 🔧 قابلیت‌های فنی
- **پشتیبانی کامل از تکیه‌گاه غلتکی (Roller)**: شرایط مرزی مستقل در راستاهای X و Y
- **اثرات حرارتی و خطای ساخت**: محاسبه دقیق نیروهای ناشی از تغییر دما (α·ΔT) و خطای مونتاژ (`delta_L_free`)
- **سیستم واحد متمرکز**: تبدیل خودکار SI ↔ Imperial با یک منبع حقیقت
- **اعتبارسنجی انرژی مکانیکی**: بررسی تعادل انرژی بر اساس تغییر شکل مکانیکی خالص (قضیه کلپیرون)

### 🛡️ ایمنی و قابلیت اطمینان
- **Fail-Fast Rank Check**: تشخیص صریح ماتریس‌های تکین (بدون حذف صامت ردیف‌های صفر)
- **Zero-Length Element Detection**: خطای صریح برای المان‌های با طول صفر یا منفی
- **Strict JSON Schema Validation**: اعتبارسنجی ساختار فایل ورودی و فیلد `loads` قبل از پردازش
- **Comprehensive Test Suite**: تست‌های Golden با دقت تحلیلی و پوشش ۱۰۰٪ APIها

---

## 🏛️ معماری

```text
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│  (ورودی JSON → تبدیل واحد → اسمبل → حل → پس‌پردازش → خروجی)  │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
  ┌──────────┐      ┌──────────────┐     ┌────────────┐
  │ fileio.py │      │  assembly.py │     │  solver.py │
  │(بارگذاری)│      │  (اسمبل K,F)  │     │ (حل KU=F)  │
  └──────────┘      └──────────────┘     └────────────┘
        ↓                   ↓                   ↓
  ┌──────────────────────────────────────────────────┐
  │              model.py (Pure DTOs)                │
  │  @dataclass Node: id, x, y, support_dx, support_dy │
  │  @dataclass Element: id, node_i, node_j, E, A    │
  └──────────────────────────────────────────────────┘
        ↓                   ↓                   ↓
  ┌──────────┐      ┌──────────────┐     ┌────────────────┐
  │ units.py  │      │postprocess.py│     │ exceptions.py  │
  │(SI↔Imperial)│    │  (نیروها، انرژی)│    │ (مدیریت خطاها) │
  └──────────┘      └──────────────┘     └────────────────┘
```

### اصول طراحی

1. **Single Responsibility**: هر ماژول فقط یک مسئولیت دارد
2. **Pure Functions**: توابع بدون side-effect، قابل تست و پیش‌بینی
3. **Fail-Fast**: خطاها در اولین فرصت گزارش می‌شوند، نه در مراحل بعدی
4. **Type Safety**: استفاده از `@dataclass` و Type Hints برای ساختارهای داده

---

## 📦 نصب

### روش ۱: نصب از PyPI (به زودی)
```bash
pip install truss-analysis-2d
```

### روش ۲: نصب از سورس
```bash
git clone https://github.com/bmhmdyan279-png/truss-analysis-2d.git
cd truss-analysis-2d
pip install .
```

### روش ۳: نصب برای توسعه
```bash
git clone https://github.com/bmhmdyan279-png/truss-analysis-2d.git
cd truss-analysis-2d
pip install -e ".[dev]"
pre-commit install
```

### وابستگی‌ها
- `numpy>=1.20`: محاسبات برداری و ماتریسی (تنها وابستگی اجرایی پروژه)

---

## 🚀 استفاده سریع

### از خط فرمان
```bash
python -m truss_analysis.main examples/example1.json SI
```

### از کد پایتون
```python
from truss_analysis import solve, Node, Element
from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.postprocess import calculate_element_forces

# تعریف گره‌ها
nodes = [
    Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    Node(id="2", x=3.0, y=0.0, is_support=False),
    Node(id="3", x=0.0, y=4.0, is_support=True, support_dx=False, support_dy=True),
]

# تعریف المان‌ها
elements = [
    Element(id="1", node_i="1", node_j="2", E=200e9, A=0.001),
    Element(id="2", node_i="2", node_j="3", E=200e9, A=0.002),
    Element(id="3", node_i="1", node_j="3", E=200e9, A=0.0015),
]

# اسمبل ماتریس‌ها
K, F, fixed_dofs = assemble_global_matrices(nodes, elements)

# اعمال بار
F[2] += 10000.0  # 10 kN در راستای X گره 2

# حل
U = solve(K, F, fixed_dofs)

# محاسبه نیروهای داخلی
results, strain_energy = calculate_element_forces(nodes, elements, U)

# نمایش نتایج
for r in results:
    print(f"Element {r['element']}: Force = {r['force']:.2f} N")
```

---

## 📝 مثال کامل

### فایل ورودی (`examples/example1.json`)
```json
{
  "nodes": [
    {"id": "1", "x": 0.0, "y": 0.0, "is_support": true, "support_dx": true, "support_dy": true},
    {"id": "2", "x": 3.0, "y": 0.0, "is_support": false},
    {"id": "3", "x": 0.0, "y": 4.0, "is_support": true, "support_dx": false, "support_dy": true}
  ],
  "elements": [
    {"id": "1", "node_i": "1", "node_j": "2", "E": 200e9, "A": 0.001, "alpha": 1.2e-5, "delta_T": 50},
    {"id": "2", "node_i": "2", "node_j": "3", "E": 200e9, "A": 0.002},
    {"id": "3", "node_i": "1", "node_j": "3", "E": 200e9, "A": 0.0015}
  ],
  "loads": [
    {"node_id": "2", "Fx": 10000.0, "Fy": -5000.0}
  ]
}
```

### توضیحات
- **گره ۱**: تکیه‌گاه ثابت کامل (گیردار در X و Y)
- **گره ۲**: گره آزاد (بدون تکیه‌گاه)
- **گره ۳**: تکیه‌گاه غلتکی (آزاد در X، مقید در Y)
- **المان ۱**: دارای اثر حرارتی (α=1.2e-5, ΔT=50°C)
- **بارگذاری**: ۱۰ kN در راستای X و -۵ kN در راستای Y روی گره ۲

---

## 📏 سیستم واحد

### واحد‌های پشتیبانی شده

| کمیت | SI | Imperial |
|------|-----|----------|
| طول (L) | متر | فوت |
| مساحت (A) | متر² | فوت² |
| ممان اینرسی (I) | متر⁴ | فوت⁴ |
| مدول الاستیسیته (E) | پاسکال | psi |
| نیرو (F) | نیوتن | پوند-نیرو |
| ضریب انبساط حرارتی (α) | ۱/°C | ۱/°F |
| تغییر دما (ΔT) | °C | °F |

---

## ⚡ اعتبارسنجی انرژی

### قضیه کلپیرون (تعمیم‌یافته)

برای یک سیستم خطی الاستیک، انرژی کرنشی تنها بر اساس **تغییر شکل مکانیکی** (`delta_l_mech`) محاسبه می‌شود تا از خطاهای Double-Counting حرارتی جلوگیری گردد:
```text
کار خارجی = انرژی کرنشی
W = ½ · U^T · F = Σ (½ · k_axial · ΔL_mech²)
```

### پیاده‌سازی
```python
from truss_analysis.solver import check_energy

# بررسی تعادل انرژی با تلورانس ۰.۰۱٪
check_energy(U, F_ext, strain_energy, tol=0.01)

# در صورت عدم تعادل:
# EnergyValidationError: Energy validation failed: Error 0.0234% exceeds tolerance 1.00%
```

---

## 🧪 تست‌ها

### اجرای تست‌ها
```bash
# اجرای تمام تست‌ها
pytest

# با پوشش کد
pytest --cov=truss_analysis --cov-report=term-missing
```

---

## 📁 ساختار پروژه

```text
truss-analysis-2d/
├── src/
│   └── truss_analysis/
│       ├── __init__.py          # Public API
│       ├── model.py             # Pure DTOs (Node, Element)
│       ├── assembly.py          # Assembly logic
│       ├── solver.py            # Linear solver + energy check
│       ├── postprocess.py       # Element forces & strain energy
│       ├── units.py             # Unit conversion (SI ↔ Imperial)
│       ├── fileio.py            # JSON loader with schema validation
│       ├── exceptions.py        # Centralized custom exceptions
│       ├── main.py              # CLI entry point
│       └── logger.py            # Logging utilities
├── tests/                       # Comprehensive Pytest suite
├── examples/                    # JSON input examples
├── pyproject.toml               # Project metadata & dependencies
├── CHANGELOG.md                 # Version history
└── README.md                    # This file
```

---

## 📊 تغییرات نسخه

### نسخه ۲.۰.۴ (۲۰۲۶-۰۸-۰۷) - Documentation Sync
#### 📚 مستندات
- همگام‌سازی کامل README با معماری ضدگلوله v2.0.3
- به‌روزرسانی Badgeها به Python ≥3.9
- اصلاح ساختار JSON مثال برای فیلد `loads`
- به‌روزرسانی دیاگرام معماری با `exceptions.py`

### نسخه ۲.۰.۳ (۲۰۲۶-۰۸-۰۷) - Critic-Proof Architecture
#### 🚀 دستاوردهای بزرگ
- **اصلاح باگ فیزیکی Double-Counting**: انرژی کرنشی اکنون صرفاً بر اساس تغییر شکل مکانیکی محاسبه می‌شود.
- **یکپارچگی مطلق API**: همگام‌سازی کامل ساختار `list` بین تمام ماژول‌ها و تست‌ها.
- **اعتبارسنجی Schema**: بررسی سخت‌گیرانه فیلد `loads` در `fileio.py` برای جلوگیری از پردازش JSONهای ناقص.
- **اصلاح واحدهای Imperial**: رفع باگ تبدیل `delta_T` (اعمال ضریب 5/9).
- **مدیریت خطای متمرکز**: انتقال تمام Exceptionها به `exceptions.py` و حذف تعاریف محلی.
- **حذف وابستگی‌های مرده**: حذف `scipy` از `pyproject.toml` و پارامتر `use_sparse` از `solver.py`.
- **استانداردسازی کد**: ارتقا به Python `>=3.9`، اضافه شدن `ruff` به CI و اصلاح متغیر مبهم `I` به `I_sec`.

### نسخه ۲.۰.۱ (۲۰۲۶-۰۸-۰۶)
#### ✅ اضافه شده
- **معماری Pure DTO**: جداسازی کامل داده و منطق
- **Fail-Fast Rank Check**: تشخیص صریح ماتریس‌های تکین
- **پشتیبانی از Roller**: شرایط مرزی مستقل در X و Y

برای تاریخچه کامل، [CHANGELOG.md](CHANGELOG.md) را مشاهده کنید.

---

## 🤝 مشارکت

مشارکت‌ها خوش‌آمد هستند! لطفاً:
1. Fork کنید
2. Branch ویژگی بسازید (`git checkout -b feature/amazing-feature`)
3. Commit کنید (`git commit -m 'feat: add amazing feature'`)
4. Push کنید (`git push origin feature/amazing-feature`)
5. Pull Request باز کنید

</div>
