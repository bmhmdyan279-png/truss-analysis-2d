# 🏗️ تحلیل سازه‌های خرپایی دوبعدی (Truss Analysis 2D)


![CI Pipeline](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-90%25%2B-success)
![PyPI Version](https://img.shields.io/pypi/v/truss-analysis-2d)


<div align="center">

![Python Version](https://img.shields.io/badge/python-≥3.9-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-2.0.9-orange.svg)
[![CI Status](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/ci.yml/badge.svg)](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions)

**یک ابزار تحلیل خرپای دوبعدی کاملاً پایتون، با معماری استاندارد، اعتبارسنجی ترمودینامیکی تعمیم‌یافته و پایداری سینماتیکی تضمین‌شده**

</div>

---

## 📋 فهرست مطالب

- [ویژگی‌های کلیدی](#-ویژگی‌های-کلیدی)
- [معماری](#-معماری)
- [نصب](#-نصب)
- [استفاده سریع](#-استفاده-سریع)
- [مثال کامل](#-مثال-کامل)
- [سیستم واحد](#-سیستم-واحد)
- [اعتبارسنجی ترمودینامیکی](#-اعتبارسنجی-ترمودینامیکی)
- [تست‌ها](#-تست‌ها)
- [ساختار پروژه](#-ساختار-پروژه)
- [تغییرات نسخه](#-تغییرات-نسخه)
- [مجوز](#-مجوز)

---

## ✨ ویژگی‌های کلیدی

### 🎯 معماری استاندارد
- **Pure DTO Architecture**: جداسازی کامل داده (`model.py`)، اسمبل (`assembly.py`) و حل (`solver.py`)
- **Centralized Exceptions**: سلسله‌مراتب متمرکز `TrussError` در `exceptions.py`
- **Fail-Fast Validation**: تشخیص فوری خطاهای هندسی، ماتریس‌های تکین و ناپایداری سینماتیکی

### 🔧 قابلیت‌های فنی پیشرفته
- **Thermodynamic Consistency**: تفکیک بردارهای نیرو (`F_ext` vs `F_mechanical`) برای محاسبه صحیح تعادل انرژی در حضور بارهای حرارتی
- **Prestress Work Tracking**: محاسبه کار پیش‌تنیدگی (`W_prestress = Σ k·ΔL_thermal·ΔL_mech`) برای قضیه کلپیرون تعمیم‌یافته
- **Kinematic Stability**: اعتبارسنجی پایداری سینماتیکی تکیه‌گاه‌ها (جلوگیری از مکانیزم‌های ناپایدار)
- **Roller Support**: شرایط مرزی مستقل در X و Y
- **Thermal + Fabrication**: پشتیبانی همزمان از `α·ΔT` و `delta_L_free`

### 🛡️ ایمنی و قابلیت اطمینان
- **Fail-Fast Rank Check**: تشخیص صریح ماتریس‌های تکین (`SingularMatrixError`)
- **Zero-Length Element Detection**: `AssemblyError` برای المان‌های با طول صفر
- **Strict JSON Schema Validation**: اعتبارسنجی ساختار فایل ورودی و فیلد `loads`
- **Division-by-Zero Guards**: محافظت در برابر مسائل خودمتعادل (`abs(W) < 1e-12`)

---

## 🏛️ معماری

```text
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│  (JSON → Units → Assemble → Solve → Postprocess → Energy)    │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
  ┌──────────┐      ┌──────────────┐     ┌────────────┐
  │ fileio.py │      │  assembly.py │     │  solver.py │
  │(Schema✓) │      │ (K,F_ext,F_mech)  │ (KU=F + Energy) │
  └──────────┘      └──────────────┘     └────────────┘
        ↓                   ↓                   ↓
  ┌──────────────────────────────────────────────────┐
  │              model.py (Pure DTOs)                │
  │  @dataclass Node: id, x, y, support_dx, support_dy │
  │  @dataclass Element: id, node_i, node_j, E, A, ... │
  └──────────────────────────────────────────────────┘
        ↓                   ↓                   ↓
  ┌──────────┐      ┌──────────────┐     ┌────────────────┐
  │ units.py  │      │postprocess.py│     │ exceptions.py  │
  │(SI↔Imperial)│    │ (forces, energy,│    │ (TrussError    │
  │  5/9 ✓    │      │  prestress)   │    │  hierarchy)    │
  └──────────┘      └──────────────┘     └────────────────┘
```

### اصول طراحی

1. **Single Responsibility**: هر ماژول فقط یک مسئولیت دارد
2. **Pure Functions**: توابع بدون side-effect، قابل تست و پیش‌بینی
3. **Fail-Fast**: خطاها در اولین فرصت گزارش می‌شوند
4. **Type Safety**: `@dataclass` و Type Hints کامل
5. **Thermodynamic Consistency**: فرمول‌های صحیح انرژی حتی در حضور بارهای حرارتی

---

## 📦 نصب

```bash
# نصب از سورس
git clone https://github.com/bmhmdyan279-png/truss-analysis-2d.git
cd truss-analysis-2d
pip install .

# نصب برای توسعه
pip install -e ".[dev]"
pre-commit install
```

### وابستگی‌ها
- `numpy>=1.20`: محاسبات برداری و ماتریسی (تنها وابستگی)

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
from truss_analysis.solver import check_energy

# تعریف گره‌ها
nodes = [
    Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    Node(id="2", x=3.0, y=0.0, is_support=False),
    Node(id="3", x=0.0, y=4.0, is_support=True, support_dx=True, support_dy=True),
]

# تعریف المان‌ها (با اثر حرارتی)
elements = [
    Element(id="1", node_i="1", node_j="2", E=200e9, A=0.001, alpha=1.2e-5, delta_T=50),
    Element(id="2", node_i="2", node_j="3", E=200e9, A=0.002),
    Element(id="3", node_i="1", node_j="3", E=200e9, A=0.0015),
]

# اسمبل ماتریس‌ها - API جدید: 4 خروجی
K, F_ext, F_mech, fixed_dofs = assemble_global_matrices(nodes, elements)

# اعمال بار مکانیکی به هر دو بردار
F_ext[2] += 10000.0
F_mech[2] += 10000.0

# حل
U = solve(K, F_ext, fixed_dofs)

# محاسبه نیروها + انرژی + کار پیش‌تنیدگی
results, strain_energy, prestress_work = calculate_element_forces(nodes, elements, U)

# اعتبارسنجی ترمودینامیکی با فرمول تعمیم‌یافته
check_energy(U, F_mech, strain_energy, prestress_work)

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
    {"id": "3", "x": 0.0, "y": 4.0, "is_support": true, "support_dx": true, "support_dy": true}
  ],
  "elements": [
    {"id": "1", "node_i": "1", "node_j": "2", "E": 200e9, "A": 0.001,
     "alpha": 1.2e-5, "delta_T": 50},
    {"id": "2", "node_i": "2", "node_j": "3", "E": 200e9, "A": 0.002},
    {"id": "3", "node_i": "1", "node_j": "3", "E": 200e9, "A": 0.0015}
  ],
  "loads": [
    {"node_id": "2", "Fx": 10000.0, "Fy": -5000.0}
  ]
}
```

### توضیحات
- **گره ۱**: تکیه‌گاه مفصلی کامل
- **گره ۲**: گره آزاد
- **گره ۳**: تکیه‌گاه مفصلی (پایداری سینماتیکی تضمین‌شده)
- **المان ۱**: دارای اثر حرارتی (α=1.2e-5, ΔT=50°C)
- **بارگذاری**: ۱۰ kN در X و -۵ kN در Y روی گره ۲

---

## 📏 سیستم واحد

| کمیت | SI | Imperial |
|------|-----|----------|
| طول (L) | متر | فوت |
| مساحت (A) | متر² | فوت² |
| مدول الاستیسیته (E) | پاسکال | psi |
| نیرو (F) | نیوتن | پوند-نیرو |
| ضریب انبساط حرارتی (α) | ۱/°C | ۱/°F |
| تغییر دما (ΔT) | °C | °F |

**نکته مهم**: ضریب تبدیل `delta_T` از °F به °C برابر با `5/9 ≈ 0.5556` است (نه 1.0)، و `alpha` برابر با `1.8` است (چرا که `1/°F = 1.8 × 1/°C`). این دو ضریب مکمل یکدیگرند و از نظر فیزیکی صحیح هستند.

---

## ⚡ اعتبارسنجی ترمودینامیکی

### قضیه کلپیرون تعمیم‌یافته

در حضور بارهای حرارتی، فرمول ساده `W = U_strain` دیگر برقرار نیست. این پروژه از فرمول تعمیم‌یافته استفاده می‌کند:

```text
W_mech = U_strain + W_prestress
```

که:
- **`W_mech = ½·Uᵀ·F_mechanical`**: کار بارهای خارجی مکانیکی (نه حرارتی)
- **`U_strain = Σ (½·k·ΔL_mech²)`**: انرژی کرنشی مکانیکی
- **`W_prestress = Σ (k·ΔL_thermal·ΔL_mech)`**: کار پیش‌تنیدگی حرارتی

### پیاده‌سازی
```python
from truss_analysis.solver import check_energy

# بررسی تعادل انرژی ترمودینامیکی
check_energy(U, F_mech, strain_energy, prestress_work, tol=0.01)

# در صورت عدم تعادل:
# EnergyValidationError: W_mech=..., U_strain=..., W_prestress=..., Error ...%
```

### مزایا
- **تشخیص باگ‌های Double-Counting**: اگر اثر حرارتی دو بار محاسبه شود، تعادل انرژی برقرار نخواهد شد
- **حفاظت از مسائل خودمتعادل**: Guard در برابر `abs(W) < 1e-12`
- **اعتبارسنجی خودکار هر تحلیل**: هر run به صورت خودکار بررسی می‌شود

---

## 🧪 تست‌ها

### اجرای تست‌ها
```bash
# تمام تست‌ها (29 تست)
pytest tests/

# با پوشش کد
pytest --cov=truss_analysis --cov-report=term-missing
```

### انواع تست‌ها

| تست | هدف |
|-----|------|
| `test_assembly.py` | اسمبل، شرایط مرزی، نیروهای حرارتی |
| `test_golden.py` | پایداری سینماتیکی، انبساط آزاد، حالت مقید |
| `test_solver.py` | حل، ماتریس تکین، تعادل انرژی ترمودینامیکی |
| `test_postprocess.py` | نیروها، انرژی، درصدها، Scale Factor |
| `test_model.py` | DTOها، `validate_inputs` |
| `test_fileio.py` | Schema validation برای JSON |
| `test_exceptions.py` | سلسله‌مراتب `TrussError` |
| `test_units.py` | تبدیل واحدها، `None` guard |
| `test_dof_mapping.py` | نگاشت صحیح درجات آزادی |

---

## 📁 ساختار پروژه

```text
truss-analysis-2d/
├── src/truss_analysis/
│   ├── __init__.py          # Public API
│   ├── model.py             # Pure DTOs (Node, Element)
│   ├── assembly.py          # Assembly (K, F_ext, F_mech)
│   ├── solver.py            # KU=F + Energy check
│   ├── postprocess.py       # Forces, energy, prestress
│   ├── units.py             # SI ↔ Imperial (5/9 for ΔT)
│   ├── fileio.py            # JSON loader + schema
│   ├── exceptions.py        # TrussError hierarchy
│   ├── main.py              # CLI entry
│   └── logger.py            # Lazy logging
├── tests/                   # 29 comprehensive tests
├── examples/                # JSON inputs
├── pyproject.toml           # Python ≥3.9, numpy only
├── CHANGELOG.md             # Version history
└── README.md                # This file
```

---

## 📊 تغییرات نسخه

### نسخه ۲.۰.۹ (۲۰۲۶-۰۸-۰۷) - پایدار Release
#### 🎯 تثبیت نهایی
- همگام‌سازی کامل مستندات با معماری استاندارد
- پوشش کامل تمام دستاوردهای v2.0.5 تا v2.0.7 در مستندات

### نسخه ۲.۰.۷ (۲۰۲۶-۰۸-۰۷) - Kinematic Stability
#### 🔧 اصلاحات
- **test_golden_simple_truss**: گره ۳ به تکیه‌گاه مفصلی تبدیل شد (جلوگیری از مکانیزم ناشی از المان عمودی)
- **test_golden_thermal_loading**: گره ۲ در Y مقید شد (جلوگیری از مود چرخشی جسم صلب)
- تمام ۲۹ تست با ماتریس‌های سختی پایدار ریاضی پاس می‌شوند

### نسخه ۲.۰.۶ (۲۰۲۶-۰۸-۰۷) - API Alignment
#### 🔧 اصلاحات
- **test_dof_mapping**: به‌روزرسانی برای unpack چهار خروجی `assemble_global_matrices`
- رفع ناسازگاری‌های باقی‌مانده در suite تست

### نسخه ۲.۰.۵ (۲۰۲۶-۰۸-۰۷) - Thermodynamic Consistency
#### 🚀 دستاورد بزرگ
- **رفع باگ Double-Counting حرارتی**: قضیه کلپیرون تعمیم‌یافته
- **تفکیک بردارهای نیرو**: `F_ext` (کل) در برابر `F_mechanical` (خارجی مکانیکی)
- **محاسبه `W_prestress`**: کار پیش‌تنیدگی حرارتی در `postprocess.py`
- **فرمول جدید `check_energy`**: `W_mech = U_strain + W_prestress`
- **تست جدید**: `test_golden_thermal_constrained` برای میله مقید
- **تست جدید**: `test_check_energy_pass_with_thermal` برای اثبات درستی فرمول

### نسخه ۲.۰.۴ (۲۰۲۶-۰۸-۰۷) - Documentation Sync
- همگام‌سازی README با معماری v2.0.9
- اصلاح ساختار JSON مثال (`loads` به صورت flat list)
- به‌روزرسانی Badgeها و دیاگرام معماری

### نسخه ۲.۰.۹ (۲۰۲۶-۰۸-۰۷) - استاندارد Architecture
- یکپارچگی مطلق API (list-based)
- مدیریت خطای متمرکز (`exceptions.py`)
- اعتبارسنجی Schema برای `loads`
- اصلاح Imperial `delta_T` به `5/9`
- حذف `scipy` و `use_sparse`
- ارتقا به Python `>=3.9` + Ruff در CI
- اصلاح متغیر مبهم `I` به `I_sec`
- حذف side-effect در `logger.py`

### نسخه ۲.۰.۱ (۲۰۲۶-۰۸-۰۶)
- معماری Pure DTO
- Fail-Fast Rank Check
- پشتیبانی از Roller Support
- Zero-Length Element Detection

برای تاریخچه کامل، [CHANGELOG.md](CHANGELOG.md) را مشاهده کنید.

---

## 🤝 مشارکت

مشارکت‌ها خوش‌آمد هستند!
1. Fork کنید
2. Branch ویژگی بسازید (`git checkout -b feature/...`)
3. Commit کنید (`git commit -m 'feat: ...'`)
4. Push کنید
5. Pull Request باز کنید

### دستورالعمل‌ها
- از `ruff` برای linting استفاده کنید
- تست بنویسید (`pytest`)
- Type hints اضافه کنید
- Docstring بنویسید
- تمام ۲۹ تست باید پاس شوند

---

<div align="center">

**اگر این پروژه برای شما مفید بود، لطفاً ⭐ بدهید!**

[![GitHub stars](https://img.shields.io/github/stars/bmhmdyan279-png/truss-analysis-2d.svg?style=social)](https://github.com/bmhmdyan279-png/truss-analysis-2d/stargazers)

</div>
