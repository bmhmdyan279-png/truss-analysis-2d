```markdown
# 🏗️ Truss Analysis 2D

> **ابزار تحلیل علمی خرپای دوبعدی با اعتبارسنجی ترمودینامیکی.**

[![CI Pipeline](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/ci.yml/badge.svg)](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-90.5%25-brightgreen)](https://github.com/bmhmdyan279-png/truss-analysis-2d)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/truss-analysis.svg)](https://pypi.org/project/truss-analysis/)
[![PyPI downloads](https://img.shields.io/pypi/dm/truss-analysis)](https://pypi.org/project/truss-analysis/)

---

## 📚 فهرست مطالب

- [📦 نصب](#-نصب)
- [🚀 شروع سریع](#-شروع-سریع)
- [🖼️ نمونه خروجی تصویری](#️-نمونه-خروجی-تصویری)
- [📖 فرمت ورودی JSON](#-فرمت-ورودی-json)
- [📤 فرمت خروجی](#-فرمت-خروجی)
- [🎯 ویژگی‌های کلیدی](#-ویژگی‌های-کلیدی)
- [🧪 تست](#-تست)
- [📁 ساختار پروژه](#-ساختار-پروژه)
- [🤝 مشارکت](#-مشارکت)
- [📚 استناد](#-استناد)
- [📄 مجوز](#-مجوز)
- [🙏 قدردانی](#-قدردانی)

---

## 📦 نصب

### از PyPI (پیشنهادی)

```bash
pip install truss-analysis
```

> **توجه:** نام بسته در PyPI `truss-analysis` است (بدون `-2d`). نام مخزن `truss-analysis-2d` می‌باشد.

### از سورس (برای توسعه یا آخرین تغییرات)

```bash
git clone https://github.com/bmhmdyan279-png/truss-analysis-2d.git
cd truss-analysis-2d
pip install -e ".[dev]"
```

### با استفاده از requirements.txt

```bash
# فقط برای اجرا
pip install -r requirements.txt

# برای توسعه (شامل ابزارهای تست و لینت)
pip install -r requirements-dev.txt
```

### بررسی سریع

```bash
# اجرای همه تست‌ها
pytest

# اجرای یک مثال سریع
truss-analysis examples/example1.json
```

---

## 🚀 شروع سریع

### استفاده از CLI

```bash
# تحلیل پایه
truss-analysis input.json

# با خروجی‌های چندگانه
truss-analysis input.json --output result.json --csv forces.csv --report report.md

# با رسم شکل و بررسی کمانش
truss-analysis input.json --plot --check-buckling --plot-path diagram.png
```

### API پایتون

```python
from truss_analysis.main import run

# اجرای تحلیل
result = run("input.json", check_buckling=True)

# دسترسی به نتایج
print(result.summary())
print(f"Displacements: {result.displacements}")
print(f"Element forces: {result.element_forces}")
print(f"Reactions: {result.reactions}")
print(f"Equilibrium valid: {result.equilibrium['is_valid']}")
```

---

## 🖼️ نمونه خروجی تصویری

در زیر یک نمونه تصویر تولید شده با استفاده از پرچم `--plot` نشان داده شده است. خطوط چین نشان‌دهنده شکل تغییر شکل یافته و خطوط توپر هندسه اصلی خرپا را نشان می‌دهند. رنگ‌ها وضعیت تنش اعضا را مشخص می‌کنند (کشش آبی، فشار قرمز).

![Truss Analysis Result](docs/images/example_output.png)

*شکل تغییر شکل یافته در برابر شکل اصلی برای `examples/example1.json`*

---

## 📖 فرمت ورودی JSON

```json
{
  "units": "SI",
  "nodes": [
    {"id": 1, "x": 0.0, "y": 0.0, "is_support": true, "support_dx": true, "support_dy": true},
    {"id": 2, "x": 3.0, "y": 0.0, "is_support": false},
    {"id": 3, "x": 0.0, "y": 4.0, "is_support": true, "support_dx": true, "support_dy": true}
  ],
  "elements": [
    {"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "A": 0.001},
    {"id": 2, "node_i": 2, "node_j": 3, "E": 200e9, "A": 0.002},
    {"id": 3, "node_i": 1, "node_j": 3, "E": 200e9, "A": 0.0015}
  ],
  "loads": [
    {"node_id": 2, "Fx": 10000.0, "Fy": -5000.0}
  ]
}
```

### پارامترهای اختیاری المان

| پارامتر | توضیح | واحد |
|-----------|-------------|------|
| `alpha` | ضریب انبساط حرارتی | 1/°C |
| `delta_T` | تغییر دما | °C |
| `delta_L_free` | تغییر طول آزاد | m |
| `I_sec` | ممان اینرسی (برای کمانش) | m⁴ |
| `rho` | چگالی (برای وزن خودی) | kg/m³ |

---

## 📤 فرمت خروجی

### ساختار خروجی JSON

```json
{
  "status": "converged",
  "displacements": {
    "2": {"dx": 0.00234, "dy": -0.00567}
  },
  "element_forces": [
    {
      "id": "1",
      "N": 15234.5,
      "status": "tension",
      "sigma": 15.23e6,
      "strain": 7.6e-5
    }
  ],
  "reactions": {
    "1": {"Fx": -8000.0, "Fy": 3500.0}
  },
  "equilibrium": {
    "sum_fx": 1.2e-10,
    "sum_fy": 3.4e-11,
    "is_valid": true
  }
}
```

### واحدها

تمام خروجی‌ها در واحد SI هستند:

| کمیت | واحد |
|----------|------|
| جابجایی | m |
| نیرو | N |
| تنش | Pa |

---

## 🎯 ویژگی‌های کلیدی

### ۱. دقت علمی

- **قضیه کلاپیرون تعمیم‌یافته:** `W_mech = U_strain + 0.5 * W_prestress`
- **تفکیک اثرات:** `δL_mech = δL_total - δL_prestress`
- **تعادل استاتیکی:** ΣFx=0، ΣFy=0، ΣM=0
- **تشخیص مکانیزم:** خطای ماتریس منفرد

### ۲. قابلیت‌های مهندسی

- **کمانش اویلری:** `P_cr = π²EI/L²` برای اعضای فشاری
- **نسبت لاغری:** `λ = L/r` با `r = √(I/A)`
- **وزن خودی:** از چگالی المان
- **واحدهای SI/Imperial:** تبدیل خودکار

### ۳. فرمت‌های خروجی

- **JSON:** نتایج ساختاریافته کامل
- **CSV:** نیروهای المان‌ها
- **Markdown:** گزارش قابل خواندن برای انسان
- **Plot:** خرپای اصلی و تغییر شکل یافته

---

## 🧪 تست

### اجرای تست‌ها

```bash
# اجرای همه تست‌ها
pytest

# با گزارش پوشش
pytest --cov=src/truss_analysis --cov-report=term-missing

# با اعمال آستانه پوشش (مانند CI)
pytest --cov=src/truss_analysis --cov-report=term-missing --cov-fail-under=90
```

**وضعیت فعلی:** ۳۹ تست موفق، پوشش ۹۰.۵٪ (آستانه: ۹۰٪)

### لینت و بررسی نوع

```bash
# اجرای ruff (لینتر + فرمت‌دهنده)
ruff check .
ruff format --check .

# اجرای mypy
mypy src/
```

---

## 📁 ساختار پروژه

```
truss-analysis-2d/
├── src/
│   └── truss_analysis/
│       ├── __init__.py
│       ├── _version.py
│       ├── assembly.py
│       ├── exceptions.py
│       ├── fileio.py
│       ├── main.py
│       ├── model.py
│       ├── postprocess.py
│       ├── solver.py
│       ├── units.py
│       └── visualization.py
├── tests/
│   ├── test_analytical.py
│   ├── test_assembly.py
│   ├── test_dof_mapping.py
│   ├── test_e2e_cli.py
│   ├── test_exceptions.py
│   ├── test_fileio.py
│   ├── test_golden.py
│   ├── test_golden_phase3.py
│   ├── test_model.py
│   ├── test_phase2_cli_coverage.py
│   ├── test_solver.py
│   └── test_units.py
├── docs/
│   ├── theory.md
│   ├── error_codes.md
│   └── images/
│       └── example_output.png
├── examples/
│   ├── example1.json
│   ├── example2.json
│   ├── reference_problem.json
│   └── example_analysis.py
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── publish.yml
│       └── release.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── CONTRIBUTING.fa.md
├── LICENSE
└── README.md
```

---

## 🤝 مشارکت

مشارکت‌ها خوش‌آمدند! لطفاً برای راهنمایی به [CONTRIBUTING.md](CONTRIBUTING.md) مراجعه کنید.

### راه‌اندازی توسعه

```bash
git clone https://github.com/bmhmdyan279-png/truss-analysis-2d.git
cd truss-analysis-2d
pip install -e ".[dev]"
pre-commit install
pytest
```

---

## 📚 استناد

اگر از این نرم‌افزار در پژوهش خود استفاده می‌کنید، لطفاً به شکل زیر استناد دهید:

```bibtex
@software{truss_analysis_2d,
  author = {bmhmdyan279-png},
  title = {Truss Analysis 2D: Scientific Truss Solver},
  year = {2026},
  url = {https://github.com/bmhmdyan279-png/truss-analysis-2d},
  version = {2.4.0}
}
```

---

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است — برای جزئیات به [LICENSE](LICENSE) مراجعه کنید.

---

## 🙏 قدردانی

- **NumPy** برای محاسبات ماتریسی
- **Matplotlib** برای مصورسازی
- **SciPy** برای عملیات ماتریس اسپارس
- **Pytest** برای چارچوب تست
- **Ruff** برای لینت و فرمت‌دهی
- **setuptools_scm** برای نسخه‌بندی خودکار
- **arabic-reshaper** و **python-bidi** برای پشتیبانی از رندر متن فارسی
```
