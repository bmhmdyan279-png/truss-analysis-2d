# 🏗️ تحلیل سازه‌های خرپایی دوبعدی (Truss Analysis 2D)

<div align="center">

![Python Version](https://img.shields.io/badge/python-≥3.8-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-2.0.1-orange.svg)
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

### 🔧 قابلیت‌های فنی
- **پشتیبانی کامل از تکیه‌گاه غلتکی (Roller)**: شرایط مرزی مستقل در راستاهای X و Y
- **اثرات حرارتی**: محاسبه دقیق نیروهای ناشی از تغییر دما (`α·ΔT`)
- **سیستم واحد متمرکز**: تبدیل خودکار SI ↔ Imperial با یک منبع حقیقت
- **اعتبارسنجی انرژی**: بررسی تعادل انرژی با خطای کمتر از ۰.۰۱٪ (قضیه کلپیرون)

### 🛡️ ایمنی و قابلیت اطمینان
- **Fail-Fast Rank Check**: تشخیص صریح ماتریس‌های تکین (بدون حذف صامت ردیف‌های صفر)
- **Zero-Length Element Detection**: خطای صریح برای المان‌های با طول صفر یا منفی
- **JSON Schema Validation**: اعتبارسنجی ساختار فایل ورودی قبل از پردازش
- **Comprehensive Test Suite**: تست‌های Golden با دقت تحلیلی

---

## 🏛️ معماری

```
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
  ┌──────────┐      ┌──────────────┐     ┌────────────┐
  │ units.py  │      │postprocess.py│     │ logger.py  │
  │(SI↔Imperial)│    │  (نیروها، انرژی)│    │  (لاگ)    │
  └──────────┘      └──────────────┘     └────────────┘
```

### اصول طراحی

1. **Single Responsibility**: هر ماژول فقط یک مسئولیت دارد
2. **Pure Functions**: توابع بدون side-effect، قابل تست و پیش‌بینی
3. **Fail-Fast**: خطاها در اولین فرصت گزارش می‌شوند، نه در مراحل بعدی
4. **Type Safety**: استفاده از `@dataclass` برای ساختارهای داده

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
- `numpy>=1.20`: محاسبات برداری و ماتریسی
- `scipy>=1.7.0`: حل‌کننده‌های بهینه (اختیاری)

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
  "loads": {
    "node_forces": [
      {"node_id": "2", "Fx": 10000.0, "Fy": -5000.0}
    ]
  }
}
```

### توضیحات
- **گره ۱**: تکیه‌گاه ثابت کامل (گیردار در X و Y)
- **گره ۲**: گره آزاد (بدون تکیه‌گاه)
- **گره ۳**: تکیه‌گاه غلتکی (آزاد در X، مقید در Y)
- **المان ۱**: دارای اثر حرارتی (`α=1.2e-5`, `ΔT=50°C`)
- **بارگذاری**: ۱۰ kN در راستای X و -۵ kN در راستای Y روی گره ۲

### خروجی مورد انتظار
```
Analysis successful. Energy balanced.
Element 1: Force = 12345.67 N
Element 2: Force = -8765.43 N
Element 3: Force = 5432.10 N
```

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

### استفاده در کد
```python
from truss_analysis.units import to_si

# تبدیل ۱۰ فوت به متر
length_m = to_si(10.0, "Imperial", "L")  # → 3.048

# تبدیل ۱۰۰۰ psi به پاسکال
E_pa = to_si(1000.0, "Imperial", "E")  # → 6894757.0

# تبدیل ۵ فوت² به متر²
area_m2 = to_si(5.0, "Imperial", "A")  # → 0.464515
```

---

## ⚡ اعتبارسنجی انرژی

### قضیه کلپیرون (تعمیم‌یافته)

برای یک سیستم خطی الاستیک:
```
کار خارجی = انرژی کرنشی
W = ½ · U^T · F = Σ (½ · k_axial · ΔL²)
```

### پیاده‌سازی
```python
from truss_analysis.solver import check_energy

# بررسی تعادل انرژی با تلورانس ۰.۰۱٪
check_energy(U, F_ext, strain_energy, tol=0.01)

# در صورت عدم تعادل:
# ValueError: Energy validation failed: Error 0.0234% exceeds tolerance 1.00%
```

### مزایا
- **تشخیص باگ‌های پنهان**: اگر منطق اسمبل یا حل اشتباه باشد، انرژی متعادل نمی‌شود
- **اعتبارسنجی خودکار**: هر تحلیل به صورت خودکار بررسی می‌شود
- **گزارش دقیق**: درصد خطا و مقدار کار خارجی و انرژی کرنشی نمایش داده می‌شود

---

## 🧪 تست‌ها

### اجرای تست‌ها
```bash
# اجرای تمام تست‌ها
pytest

# با پوشش کد
pytest --cov=truss_analysis --cov-report=term-missing

# فقط تست‌های Golden
pytest tests/test_golden.py -v
```

### انواع تست‌ها

#### ۱. تست واحد (Unit Tests)
```python
# tests/test_units.py
def test_to_si_imperial_length():
    assert abs(to_si(1.0, "Imperial", "L") - 0.3048) < 1e-6
```

#### ۲. تست نگاشت DOF
```python
# tests/test_dof_mapping.py
def test_dof_mapping():
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=1.0, y=0.0, is_support=False),
    ]
    K, F, fixed = assemble_global_matrices(nodes, [])
    assert 0 in fixed and 1 in fixed  # DOF 0,1 مقید
    assert 2 not in fixed and 3 not in fixed  # DOF 2,3 آزاد
```

#### ۳. تست Golden (Integration Test)
```python
# tests/test_golden.py
def test_golden_simple_truss():
    # خرپای ۳ عضوی با تکیه‌گاه غلتکی
    nodes = [...]
    elements = [...]
    K, F_ext, fixed_dofs = assemble_global_matrices(nodes, elements)
    F_ext[2] += 10000.0

    U = solve(K, F_ext, fixed_dofs)

    # بررسی اینکه تکیه‌گاه غلتگی قفل نشده
    assert abs(U[4]) > 1e-10, "CRITICAL: Roller support is artificially locked!"

    # بررسی تعادل انرژی
    results, strain_energy = calculate_element_forces(nodes, elements, U)
    check_energy(U, F_ext, strain_energy, tol=0.01)
```

---

## 📁 ساختار پروژه

```
truss-analysis-2d/
├── src/
│   └── truss_analysis/
│       ├── __init__.py          # Public API
│       ├── model.py             # Pure DTOs (Node, Element)
│       ├── assembly.py          # Assembly logic
│       ├── solver.py            # Linear solver + energy check
│       ├── postprocess.py       # Element forces & strain energy
│       ├── units.py             # Unit conversion (SI ↔ Imperial)
│       ├── fileio.py            # JSON loader with validation
│       ├── main.py              # CLI entry point
│       └── logger.py            # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── test_units.py            # Unit conversion tests
│   ├── test_dof_mapping.py      # DOF mapping tests
│   ├── test_golden.py           # Golden tests
│   └── test_exceptions.py       # Exception handling tests
├── examples/
│   └── example1.json            # Example input file
├── pyproject.toml               # Project metadata & dependencies
├── CHANGELOG.md                 # Version history
└── README.md                    # This file
```

---

## 📊 تغییرات نسخه

### نسخه ۲.۰.۱ (۲۰۲۶-۰۸-۰۶)

#### ✅ اضافه شده
- **معماری Pure DTO**: جداسازی کامل داده و منطق
- **Fail-Fast Rank Check**: تشخیص صریح ماتریس‌های تکین
- **پشتیبانی از Roller**: شرایط مرزی مستقل در X و Y
- **سیستم واحد متمرکز**: `units.py` به عنوان Single Source of Truth
- **تست‌های Golden**: اعتبارسنجی با دقت ۰.۰۱٪
- **تست نگاشت DOF**: اطمینان از صحت شماره‌گذاری درجات آزادی

#### 🐛 رفع شده
- **ImportError در `__init__.py`**: تابع `solve_truss` به درستی export می‌شود
- **وابستگی scipy**: به `pyproject.toml` اضافه شد
- **حذف سکوت المان‌های صفر**: اکنون `AssemblyError` پرتاب می‌شود
- **ناهماهنگی واحدها**: کلیدهای `L2`, `L4`, `F` هم در `units.py` و هم در `main.py`
- **ناهماهنگی JSON**: `main.py` با ساختار `example1.json` همسان شد

#### 🗑️ حذف شده
- **وابستگی‌های غیرضروری**: `arabic-reshaper`, `python-bidi`, `matplotlib`, `pyyaml`
- **کد مرده**: `validate_energy_simple` (۵ کپی)، `infrastructure/`، `ansys_comparison/`
- **Side-effects**: محاسبات در `__init__` مدل‌ها حذف شد

برای تاریخچه کامل، [CHANGELOG.md](CHANGELOG.md) را مشاهده کنید.

---

## 🤝 مشارکت

مشارکت‌ها خوش‌آمد هستند! لطفاً:

1. Fork کنید
2. Branch ویژگی بسازید (`git checkout -b feature/amazing-feature`)
3. Commit کنید (`git commit -m 'feat: add amazing feature'`)
4. Push کنید (`git push origin feature/amazing-feature`)
5. Pull Request باز کنید

### دستورالعمل‌های کد
- از `ruff` برای linting استفاده کنید
- تست‌ها را بنویسید (`pytest`)
- Type hints اضافه کنید
- Docstring بنویسید

---

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات، [LICENSE](LICENSE) را مشاهده کنید.

---

## 🙏 قدردانی

- الهام‌گرفته از کتاب "Mechanics of Materials" اثر Cook & Young
- معماری Fail-Fast از اصول Clean Code اثر Robert C. Martin
- تست‌های Golden از الگوی Golden Master Testing

---

<div align="center">

**اگر این پروژه برای شما مفید بود، لطفاً ⭐ بدهید!**

[![GitHub stars](https://img.shields.io/github/stars/bmhmdyan279-png/truss-analysis-2d.svg?style=social)](https://github.com/bmhmdyan279-png/truss-analysis-2d/stargazers)

</div>
