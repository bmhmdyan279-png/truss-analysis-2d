# 🏗️ Truss Analysis 2D

[![CI](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/publish.yml/badge.svg)](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions)
[![PyPI](https://img.shields.io/pypi/v/truss_analysis)](https://pypi.org/project/truss_analysis/)
[![Python](https://img.shields.io/pypi/pyversions/truss_analysis)](https://pypi.org/project/truss_analysis/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

یک ابزار تحلیل خرپای دوبعدی مهندسی و علمی با:
- ✅ اعتبارسنجی ترمودینامیکی (قضیه کلپیرون تعمیم‌یافته)
- ✅ کنترل تعادل استاتیکی
- ✅ هشدار کمانش اویلر
- ✅ پشتیبانی از بار حرارتی و پیش‌تنیدگی
- ✅ خروجی JSON/CSV/Markdown و مصورسازی

## 📦 نصب

```bash
pip install truss_analysis
```

یا از سورس:

```bash
git clone https://github.com/bmhmdyan279-png/truss-analysis-2d.git
cd truss-analysis-2d
pip install .
```

## 🚀 استفاده سریع

### CLI (خط فرمان)

```bash
# تحلیل ساده
truss-analysis input.json

# با خروجی‌های مختلف
truss-analysis input.json --output result.json --csv forces.csv --report report.md

# با مصورسازی و بررسی کمانش
truss-analysis input.json --plot --check-buckling --plot-path diagram.png
```

### Python API

```python
from truss_analysis import run

# تحلیل خرپا
result = run("input.json", check_buckling=True)

# دسترسی به نتایج
print(result.summary())
print(f"Displacements: {result.displacements}")
print(f"Element forces: {result.element_forces}")
print(f"Reactions: {result.reactions}")
print(f"Equilibrium valid: {result.equilibrium['is_valid']}")
```

## 📖 قالب ورودی JSON

```json
{
  "nodes": [
    {"id": "1", "x": 0.0, "y": 0.0, "is_support": true, "support_dx": true, "support_dy": true},
    {"id": "2", "x": 3.0, "y": 0.0, "is_support": true, "support_dy": true},
    {"id": "3", "x": 1.5, "y": 2.0, "is_support": false}
  ],
  "elements": [
    {"id": "1", "node_i": "1", "node_j": "3", "E": 200e9, "A": 0.001},
    {"id": "2", "node_i": "2", "node_j": "3", "E": 200e9, "A": 0.001},
    {"id": "3", "node_i": "1", "node_j": "2", "E": 200e9, "A": 0.001}
  ],
  "loads": [
    {"node_id": "3", "Fx": 0, "Fy": -10000}
  ]
}
```

### پارامترهای اختیاری المان

- `alpha`: ضریب انبساط حرارتی (1/°C)
- `delta_T`: تغییر دما (°C)
- `delta_L_free`: تغییر طول اولیه (m)
- `I_sec`: ممان اینرسی برای بررسی کمانش (m⁴)
- `density`: چگالی برای وزن خودی (kg/m³)

## 🎯 ویژگی‌های کلیدی

### 1. صحت علمی
- **قضیه کلپیرون تعمیم‌یافته**: `W_mech = U_strain + 0.5 * W_prestress`
- **جداسازی اثرات**: `δL_mech = δL_total - δL_prestress`
- **کنترل تعادل**: ΣFx=0, ΣFy=0, ΣM=0
- **تشخیص مکانیزم**: خطای ماتریس منفرد

### 2. امکانات مهندسی
- **کمانش اویلر**: `P_cr = π²EI/L²` برای المان‌های تحت فشار
- **نسبت لاغری**: `λ = L/r` با `r = √(I/A)`
- **وزن خودی**: از چگالی و حجم المان
- **واحدهای SI و Imperial**: تبدیل خودکار

### 3. خروجی‌ها
- **JSON**: نتایج کامل ساختاریافته
- **CSV**: نیروهای المان‌ها
- **Markdown**: گزارش خوانا
- **Plot**: خرپای اولیه و تغییرشکل‌یافته

## 🧪 مثال‌ها

مثال‌های کامل در پوشه `examples/`:

```bash
# اجرای مثال
truss-analysis examples/example1.json

# یا با Python
python examples/example_analysis.py
```

## 🔧 API Reference

### توابع اصلی

#### `run(filepath, unit_sys="SI", plot=False, check_buckling=False, output=None, csv_path=None, report_path=None, plot_path=None)`

تحلیل کامل خرپا و برگرداندن `AnalysisResult`.

**پارامترها:**
- `filepath` (str): مسیر فایل JSON ورودی
- `unit_sys` (str): سیستم واحد ("SI" یا "Imperial")
- `plot` (bool): نمایش نمودار
- `check_buckling` (bool): بررسی کمانش
- `output` (str): مسیر خروجی JSON
- `csv_path` (str): مسیر خروجی CSV
- `report_path` (str): مسیر خروجی Markdown
- `plot_path` (str): مسیر ذخیره نمودار

**برگشتی:** `AnalysisResult` با فیلدهای:
- `status` (str): "SUCCESS" یا خطا
- `displacements` (dict): جابه‌جایی گره‌ها
- `element_forces` (list): نیروهای المان‌ها
- `reactions` (dict): عکس‌العمل‌های تکیه‌گاهی
- `equilibrium` (dict): کنترل تعادل
- `buckling` (list): نتایج کمانش

#### `AnalysisResult.summary()`

تولید خلاصه متنی از نتایج.

### توابع سطح پایین

#### `assemble_global_matrices(nodes, elements)`

مونتاژ ماتریس سختی و بردارهای نیرو.

**برگشتی:** `(K, F_ext, F_mechanical, fixed_dofs)`

#### `solve(K, F_ext, fixed_dofs)`

حل دستگاه `KU = F` با شرایط مرزی.

**برگشتی:** `U` (numpy array)

#### `calculate_element_forces(nodes, elements, U)`

محاسبه نیروهای المان، انرژی کرنشی و کار پیش‌تنیدگی.

**برگشتی:** `(results, strain_energy, prestress_work)`

#### `calculate_reactions(nodes, K, U, F_ext, fixed_dofs)`

محاسبه عکس‌العمل‌های تکیه‌گاهی: `R = KU - F_ext`

#### `check_equilibrium(nodes, reactions, applied_loads, tol=1e-6)`

کنترل تعادل استاتیکی.

**برگشتی:** `{"sum_fx": float, "sum_fy": float, "sum_m": float, "is_valid": bool}`

#### `calculate_buckling(nodes, elements, results, tol=1e-12)`

بررسی کمانش اویلر برای المان‌های تحت فشار.

**برگشتی:** لیست دیکشنری با `{"id", "N", "P_cr", "ratio", "slenderness", "safe"}`

## 🧪 تست‌ها

```bash
# اجرای همه تست‌ها
pytest

# با پوشش کد
pytest --cov=truss_analysis --cov-report=term-missing
```

**پوشش فعلی:** 90% (37 تست)

## 🤝 مشارکت

مشارکت‌ها خوش‌آمد هستند! لطفاً:

1. Fork کنید
2. Branch بسازید: `git checkout -b feature/amazing-feature`
3. Commit کنید: `git commit -m 'Add amazing feature'`
4. Push کنید: `git push origin feature/amazing-feature`
5. Pull Request باز کنید

### توسعه محلی

```bash
git clone https://github.com/bmhmdyan279-png/truss-analysis-2d.git
cd truss-analysis-2d
pip install -e ".[dev]"
pre-commit install
pytest
```

## 📝 تاریخچه تغییرات

برای تاریخچه کامل، [CHANGELOG.md](CHANGELOG.md) را ببینید.

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات، [LICENSE](LICENSE) را ببینید.

## 🙏 تشکر

- **NumPy** برای محاسبات ماتریسی
- **Matplotlib** برای مصورسازی
- **Pytest** برای فریمورک تست
- **Ruff** برای linting و formatting
- **setuptools_scm** برای versioning خودکار

## 📞 پشتیبانی

- **Issues**: [GitHub Issues](https://github.com/bmhmdyan279-png/truss-analysis-2d/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bmhmdyan279-png/truss-analysis-2d/discussions)
