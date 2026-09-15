# Truss Analysis 2D

**این صفحه به:** [English](README.md) | [فارسی](README.fa.md)

کتابخانهٔ پایتون برای **تحلیل استاتیکی خطی خرپاهای دوبعدی صفحه‌ای** با روش
سختی مستقیم — به‌همراه مدل مصالح وابسته به دما طبق EN 1993-1-2، تولید
پارامتریک توپولوژی، شاخص بحرانیّت عضو، تحلیل عدم‌قطعیت و ابزارهای
اولویت‌بندی مقاوم‌سازی.

[![CI](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/ci.yml/badge.svg)](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/coverage-۹۴.۲%25-brightgreen)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![PyPI version](https://img.shields.io/pypi/v/truss-analysis.svg)](https://pypi.org/project/truss-analysis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## فهرست

- [مرور کلی](#مرور-کلی)
- [قابلیت‌ها](#قابلیت‌ها)
- [نصب](#نصب)
- [شروع سریع](#شروع-سریع)
- [واسط خط فرمان](#واسط-خط-فرمان)
- [API پایتون](#api-پایتون)
- [اسکیمای JSON ورودی](#اسکیمای-json-ورودی)
- [اسکیمای JSON خروجی](#اسکیمای-json-خروجی)
- [بررسی‌های درونی](#بررسی‌های-درونی)
- [آزمون و توسعه](#آزمون-و-توسعه)
- [ساختار پروژه](#ساختار-پروژه)
- [مشارکت](#مشارکت)
- [استناد](#استناد)
- [مجوز](#مجوز)

---

## مرور کلی

`truss_analysis` مدل‌های خرپای دوبعدی (گره + اعضای محوری) را در حالت
استاتیکی خطی حل می‌کند: جابه‌جایی گره‌ها، نیروی محوری اعضا، عکس‌العمل‌های
تکیه‌گاهی، باقی‌مانده‌های تعادل و تراز انرژی کرنشی. فراتر از حل‌گر کلاسیک،
یک جعبه‌ابزار مهندسی برای خرپاهای فولادی در دمای بالا فراهم می‌کند:

* ضرایب کاهش مقاومت/سختی و قانون ساختاری **EN 1993-1-2** (جدول 3.1)، به‌همراه
  **منحنی آتش ISO 834** و **حل‌گر گرمایش ظرفیت متمرکز** همان استاندارد
  (آتش ← دمای فولاد ← پاسخ سازه‌ای در یک زنجیره)،
* **حالات حدی آتش** (میدان‌های DCR، دماهای بحرانی با برچسب حالت خرابی) و
  **پایداری خطی‌شدهٔ سامانه** (سختی هندسی، ضریب بار دوشاخگی حول حالت پایهٔ
  پیش‌تنیده)،
* **تولیدکنندهٔ پارامتریک** توپولوژی برای سه خانوادهٔ Warren، Pratt و Howe،
* **شاخص بحرانیّت عضو** با موتور اختلال رتبه‑۱ دقیق (یک فاکتورگیری برای
  همهٔ اعضا)،
* **تحلیل عدم‌قطعیت** (نمونه‌برداری لاتین هایپرکیوب و مونت‌کارلو با
  همبستگی رتبه‌ای ایمان–کونوورِ حافظِ لایه‌بندی) و موتور مونت‌کارلو
  قابلیت اطمینان با همان مدل ظرفیتِ زنجیرهٔ DCR،
* **تریاز مقاوم‌سازی** با بودجه (راهبردهای حریصانه، شمارش کامل، مقاوم و
  آگاه از افزونگی + سه سناریوی هزینه)،
* **اعتبارسنجی گراف سازه** (همبندی، مکانیزم، رتبه و عدد حالتی ماتریس
  سختی گره‌های آزاد).

همه‌چیز با ورودی‌های داده‌شده قطعی است، تایپ کامل دارد
(`mypy --strict` پاک) و با ۸۳۸ آزمون پوشش ۹۴.۲٪ (گیت ۹۰٪) گرفته است.
نتایج همیشه SI هستند (متر، نیوتن، پاسکال) — صرف‌نظر از دستگاه واحد ورودی.

## قابلیت‌ها

| حوزه | آنچه می‌گیرید |
|---|---|
| حل‌گر | روش سختی مستقیم؛ شرایط مرزی حذفی (پیش‌فرض) یا جریمه‌ای؛ مونتاژ تنک یا چگال؛ تشخیص مکانیزم/تکینگی |
| حرارتی | تغییر دمای عضو‌به‌عضو `delta_T`، تغییر طول آزاد `delta_L_free` (خطای ساخت)؛ تفکیک کار پیش‌تنیدگی |
| آتش | ضرایب کاهش `k_E(T)`، `k_y(T)`، `k_s(T)`، `k_p(T)`، کران‌های کرنش و قانون کامل تنش–کرنش EN 1993-1-2؛ مقاومت محاسباتی و میدان نسبت تقاضا/ظرفیت (DCR)؛ دمای بحرانی سامانه `theta_sys` با جاروی DCR و برچسب **حالت خرابی** (حدِّ مصالح در برابر فروپاشی سختی)؛ منحنی آتش ISO 834 و حل‌گر **گرمایش ظرفیت متمرکز** بند ۴.۲.۲.۲ (قرارگیری در آتش ← دمای فولاد ← زنجیرهٔ سازه‌ای)؛ کرنش حرارتی قوسی `effective_alpha(T)` از منحنی طویل‌شدگی استاندارد |
| بحرانیّت | موتور اختلال رتبه‑۱ دقیق (Sherman–Morrison): CI و CI نرمال‌شده برای هر عضو، رتبه‌ها، مجموعهٔ ۵ عضو بحرانی، پایداری رتبه با تاو‑b کندال در برابر حالت دمای محیط |
| عدم‌قطعیت | جدول متغیرهای تصادفی (بار زندهٔ گامبل، مقاومت تسلیم لوگ‌نرمال، شدت آتش نرمالِ کرانه‌دار، E قطعی) با وضعیت مستندِ استنادها؛ LHS + مونت‌کارلو؛ همبستگی رتبه‌ای با **بازچینش ایمان–کونوور** (لایه‌بندی LHS در کوپلینگ حفظ می‌شود) به‌همراه جایگزین کوپولای گاوسی؛ آمار جریان‌یافته؛ رتبه‌بندی احتمالاتی؛ حاشیه‌های قابلیت اطمینان بر **همان مدل ظرفیت `chi` زنجیرهٔ DCR**؛ احتمال خرابی تجربی با بازهٔ دقیق کلوپر–پیرسون در کنار تقریب نرمال `Φ(−β̂)` |
| مقاوم‌سازی | ارتقای عضو با بودجه؛ راهبردهای حریصانه/شمارش کامل/مقاوم/افزونگی‌محور؛ سناریوهای هزینهٔ خطی، درجه‑۲ و پله‌ای؛ معیارها: هزینه، `u_max`، `theta_sys`، تعداد اعضای DCR ≥ ۱ |
| اعتبارسنجی | بررسی‌های گراف (عضو یتیم، عضو تکراری، حلقهٔ خودی، همبندی، رتبه، عدد حالتی)؛ تراز انرژی با قضیهٔ کلی کلپرو؛ باقی‌ماندهٔ تعادل؛ پل مرجع اختیاری OpenSeesPy (extra با نام `validation`) |
| کمانش | بار بحرانی اویلر با ضریب طول مؤثر؛ نسبت بهره‌وری برای هر عضو فشاری |
| پایداری | سختی هندسی `K_G = Σ (N/L) g gᵀ` و **ضریب بار دوشاخگی خطی‌شده** `λ_cr` حول حالت پایهٔ پیش‌تنیده (حرارتی/ساخت)، همراه با شکل مود — همزادِ سطحِ سامانه برای بررسی‌های سطحِ عضو؛ اعتبارسنجی‌شده با حل بستهٔ خرپای کم‌عمق و حل ویژهٔ مستقل QZ |
| واحدها | ورودی SI و امپریال (تبدیل خودکار)؛ خروجی همیشه SI |
| خروجی | خلاصهٔ کنسول، JSON (با بلوک بازتولیدپذیری `solver_metadata`: نسخه‌ها، تجزیهٔ استفاده‌شده، رتبه/عدد حالتی `K_ff`، سیاست تلورانس)، جدول CSV نیروها، گزارش Markdown، شکل PNG (برچسب‌های فارسی با extra با نام `viz`) |
| بسته‌بندی | نشانگر `py.typed`، تایپ سخت‌گیرانه، زنجیرهٔ ابزارِ پین‌شده، داکرفایل چندمرحله‌ای، CI ماتریسی (۳ سیستم‌عامل × ۳ نسخهٔ پایتون) |

## نصب

### از PyPI

```bash
pip install truss-analysis
```

نام توزیع `truss-analysis` و نام ایمپورت `truss_analysis` است.

extraهای اختیاری:

```bash
pip install "truss-analysis[viz]"         # رسم شکل + حروف‌چینی فارسی/دوطرفه
pip install "truss-analysis[validation]"  # پل مرجع OpenSeesPy
pip install "truss-analysis[dev]"         # ابزار آزمون، لینتر، تایپ و بسته‌بندی
```

### از سورس

```bash
git clone https://github.com/bmhmdyan279-png/truss-analysis-2d.git
cd truss-analysis-2d
pip install -e ".[dev]"
pre-commit install
```

### با داکر

```bash
docker build -t truss-analysis .

# دستور پیش‌فرض، مثالِ همراه مخزن را سرتاسر تحلیل می‌کند (آزمون دود)
docker run --rm truss-analysis

# تحلیل مدل خودتان
docker run --rm -v "$PWD:/data" truss-analysis \
    truss-analysis analyze /data/my_model.json
```

ایمیج چندمرحله‌ای است، با ایمیج پایهٔ `python:3.11-slim` پین‌شده با digest
و بک‌اند بدون نمایشگر matplotlib.

## شروع سریع

یک اجرای کامل روی مثال همراه مخزن `examples/example1.json`
(۵ گره، ۷ عضو، دو گرهٔ بارگذاری‌شده — برشی از ورودی):

```json
{
  "units": "SI",
  "nodes": [
    {"id": 1, "x": 0.0, "y": 0.0, "is_support": true, "support_dx": true, "support_dy": true},
    {"id": 2, "x": 4.0, "y": 0.0, "is_support": false},
    "... ۳ گره دیگر ..."
  ],
  "elements": [
    {"id": 1, "node_i": 1, "node_j": 2, "A": 0.0561, "E": 210e9, "alpha": 1.2e-5},
    "... ۶ عضو دیگر ..."
  ],
  "loads": [
    {"node_id": 4, "Fx": 20000.0, "Fy": -30000.0},
    {"node_id": 5, "Fx": -20000.0, "Fy": -30000.0}
  ]
}
```

تحلیل همراه با بررسی کمانش:

```console
$ truss-analysis analyze examples/example1.json --check-buckling
==============================================================
 TRUSS ANALYSIS RESULT
==============================================================
Element 1: N =         -0.000  [Zero]
Element 2: N =          0.000  [Zero]
Element 3: N =     -36055.513  [Compression]
Element 4: N =          0.000  [Zero]
Element 5: N =         -0.000  [Zero]
Element 6: N =     -36055.513  [Compression]
Element 7: N =     -40000.000  [Compression]
Reaction @ 1: Fx =    20000.000  Fy =    30000.000
Reaction @ 3: Fx =   -20000.000  Fy =    30000.000
Equilibrium: dFx=3.64e-12 dFy=1.46e-11 dM=2.91e-11 -> OK
Buckling 1: ratio=0.000 -> OK
Buckling 3: ratio=2.164 -> BUCKLING RISK
Buckling 5: ratio=0.000 -> OK
Buckling 6: ratio=1.219 -> BUCKLING RISK
Buckling 7: ratio=1.219 -> BUCKLING RISK
Status: SUCCESS
```

تولید همهٔ خروجی‌ها در یک فراخوانی:

```bash
truss-analysis analyze examples/example1.json --check-buckling \
    -o result.json --csv forces.csv --report report.md --plot-path diagram.png
```

محتوای `forces.csv` (عیناً):

```csv
element_id,axial_force,status
1,-2.771055810051527e-12,Zero
2,4.935565360740278e-13,Zero
3,-36055.512754639894,Compression
4,4.9950858176817626e-12,Zero
5,-3.637805361141061e-12,Zero
6,-36055.51275463989,Compression
7,-40000.00000000001,Compression
```

`report.md` جدول نیروها/عکس‌العمل‌ها و باقی‌مانده‌های تعادل را دارد و
`result.json` از [اسکیمای خروجی](#اسکیمای-json-خروجی) پیروی می‌کند.

شکل تغییرشکلیافته در برابر شکل اصلی با `--plot-path`
(خط‌چین = تغییرشکلیافته؛ رنگ = کشش/فشار):

![نتیجهٔ تحلیل خرپا](docs/images/example_output.png)

## واسط خط فرمان

```console
$ truss-analysis --help
usage: truss-analysis [-h] {analyze,validate,generate,version} ...

2D truss analysis: linear statics, thermal loads, buckling checks, input
validation and parametric model generation.

positional arguments:
  {analyze,validate,generate,version}
    analyze             run the full analysis on one input JSON file
    validate            validate an input JSON file without solving it
    generate            generate a parametric truss model as canonical JSON
    version             print the package version

options:
  -h, --help            show this help message and exit
```

کدهای خروج: `۰` موفق · `۱` شکست تحلیل/اعتبارسنجی (جزئیات روی stderr یا
در گزارش validate) · `۲` خطای کاربرد یا فایل ورودیِ خوانده‌نشده.
فراخوانی قدیمی هم پشتیبانی می‌شود:
`truss-analysis model.json` ≡ `truss-analysis analyze model.json`.

### `analyze`

`truss-analysis analyze INPUT [options]`

| گزینه | اثر |
|---|---|
| `--units {SI,Imperial}` | دستگاه واحدِ جایگزین وقتی فایل هیچ‌کدام را اعلام نکرده (پیش‌فرض `SI`) |
| `-o, --output PATH` | نوشتن نتیجهٔ JSON |
| `--csv PATH` | نوشتن جدول CSV نیروها |
| `--report PATH` | نوشتن گزارش Markdown |
| `--plot` | نمایش شکل تعاملی (extra با نام `viz`) |
| `--plot-path PATH` | ذخیرهٔ شکل با فرمت PNG |
| `--check-buckling` | افزودن بهره‌وری کمانش اویلر برای اعضای فشاری |
| `--quiet` | حذف خلاصهٔ کنسول |

خروجی واقعی: در [شروع سریع](#شروع-سریع) بالا.

### `validate`

اسکیمای ورودی، واحدها، تکیه‌گاه‌ها و توپولوژی را **بدون حل** بررسی می‌کند:

```console
$ truss-analysis validate examples/example1.json
{
  "file": "examples/example1.json",
  "valid": true,
  "errors": [],
  "units": "SI",
  "n_nodes": 5,
  "n_elements": 7,
  "topology": {
    "n_nodes": 5,
    "n_members": 7,
    "n_reactions": 4,
    "n_dof_free": 6,
    "indeterminacy": 1,
    "connected": true,
    "orphan_nodes": [],
    "zero_length_members": [],
    "duplicate_members": [],
    "self_loops": [],
    "mechanism": false,
    "rank_k_ff": 6,
    "cond_k_ff": 40.66278238719741,
    "cond_warning": false,
    "symmetric": true
  }
}
```

مدل نامعتبر با ذکر دلیل و کد خروج ۱ گزارش می‌شود (اجرای واقعی):

```console
$ truss-analysis validate bad.json
{
  "file": "bad.json",
  "valid": false,
  "errors": [
    "input validation: Insufficient constraints for stability: 0 < 3"
  ],
  "units": "SI"
}
$ echo $?
1
```

### `generate`

مدل پارامتریک (Warren / Pratt / Howe) را به‌صورت JSON استاندارد می‌سازد:

```console
$ truss-analysis generate --family warren --panels 4 --span 24 --height 3.6 -o warren4.json
$ head -c 240 warren4.json
{"elements":[{"A":0.01,"E":210000000000.0,"I_sec":0.0001001736111111111,"alpha":1.2e-05,"delta_L0":0.0,"delta_T":0.0,"effective_length_factor":1.0,"id":1,"node_i":1,"node_j":2,"section_type":"idealised_square_hss"},{"A":0.01,"E":21000000000…
```

(مدل تولیدشده یک خط JSON است؛ برش بالا عیناً ۲۴۰ بایت نخست است.
۹ گره، ۱۵ عضو، ۷ گرهٔ بارگذاری‌شده.)

| گزینه | پیش‌فرض | معنا |
|---|---|---|
| `--family {warren,pratt,howe}` | الزامی | خانوادهٔ خرپا |
| `--panels N` | الزامی | تعداد دهانه |
| `--span L` | الزامی | دهانهٔ کل [m] |
| `--height H` | الزامی | ارتفاع خرپا [m] |
| `--area A` | `0.01` | سطح مقطع [m²] |
| `--youngs-modulus E` | `210e9` | مدول یانگ [Pa] |
| `--thermal-expansion` | `1.2e-5` | ضریب انبساط حرارتی [1/°C] |
| `--total-load P` | `100e3` | بار قائم کل [N] — مستقل از خانواده روی گره‌های باربر تقسیم می‌شود |
| `-o, --output PATH` | stdout | فایل مقصد |

### `version`

```console
$ truss-analysis version
2.5.0
```

(نصب از خروجیِ بدون git سورس. در یک checkout گیت، رشتهٔ دقیق
setuptools-scm همان درخت کاری چاپ می‌شود.)

## API پایتون

همهٔ قطعات زیر روی همین مخزن اجرا شده‌اند؛ مقادیر چاپ‌شده خروجی واقعی‌اند.

### ۱. تحلیل یک فایل مدل

```python
from truss_analysis import run

result = run("examples/example1.json", check_buckling=True, quiet=True)

print(result.status)  # SUCCESS
print(result.equilibrium["is_valid"])  # True
print(max(abs(e["N"]) for e in result.element_forces))  # 40000.00000000001
print([b["id"] for b in result.buckling if b["ratio"] > 1.0])  # ['3', '6', '7']
```

فیلدهای `AnalysisResult`: `status`، `displacements` (برای هر گره `ux` و
`uy` برحسب متر)، `element_forces` (نیروی محوری `N` برحسب نیوتن + برچسب
`status` برای هر عضو)، `reactions` (`Fx` و `Fy` برحسب نیوتن)،
`equilibrium` (باقی‌مانده‌ها + اعتبار) و `buckling` (بهره‌وری هر عضو
فشاری؛ خالی مگر اینکه درخواست شود).

### ۲. توپولوژی پارامتریک + بحرانیّت وابسته به دما

```python
from truss_analysis import Element, Node, compute_ci_for_topology, generate_topology
from truss_analysis.material import k_E, k_y

model = generate_topology("warren", n_panels=4, span=16.0, height=3.0)
nodes = [
    Node(
        id=str(n["id"]),
        x=n["x"],
        y=n["y"],
        is_support=n.get("is_support", False),
        support_dx=n.get("support_dx", False),
        support_dy=n.get("support_dy", False),
    )
    for n in model["nodes"]
]
elements = [
    Element(
        id=str(e["id"]),
        node_i=str(e["node_i"]),
        node_j=str(e["node_j"]),
        E=e["E"],
        A=e["A"],
    )
    for e in model["elements"]
]
loads = {str(ld["node_id"]): {"Fx": ld["Fx"], "Fy": ld["Fy"]} for ld in model["loads"]}

print(k_E(600.0), k_y(600.0))  # 0.31 0.47   (EN 1993-1-2 Table 3.1)

res = compute_ci_for_topology(nodes, elements, loads, {}, "uniform", 600.0)
print(res.top_5)  # ['6', '2', '3', '5', '7']
# میدان دمای یکنواخت، اثبات‌پذیر، هیچ عضوی را جابه‌جا نمی‌کند — تا وقتی هیچ عضویی
# کرنش تحمیلی حرارتی نداشته باشد (اینجا alpha=0؛ با alpha>0 روی سازهٔ مهارشده،
# حالت پایه خودش وابسته به دما می‌شود — docs/theory.md §5.4):
print(res.tau_vs_base)  # 1.0
```

سناریوهای حرارتی `compute_ci_for_topology`: `uniform`، `local_left`،
`local_mid`، `local_right` و `linear_gradient` (نگاه کنید به
`truss_analysis.criticality.SCENARIOS`). میدان‌های موضعی ترتیب اعضا را
**تغییر می‌دهند** — آمارهٔ `tau_vs_base` همین را کمی می‌کند.

### ۳. تریاز مقاوم‌سازی با بودجه

```python
from truss_analysis.retrofit import greedy, make_context

ctx = make_context(
    nodes,
    elements,
    loads,
    "uniform",
    600.0,
    f_y=235.0e6,
    alpha=0.7,
    budget_fraction=0.2,
)
outcome = greedy(ctx)

print(
    [m for m, a in zip(outcome.decision.member_ids, outcome.decision.actions) if a > 0]
)  # ['6']
print(outcome.metrics.cost)  # 6000.0
print(outcome.metrics.u_max)  # 0.002673174…
```

راهبردهای دیگر (`exhaustive`، `robust_strategy`، `redundant_strategy`،
`stress_based_strategy`) دقیقاً همان context را مصرف می‌کنند، پس معیارهایشان
بی‌واسطه قابل مقایسه است.

### ۴. تحلیل عدم‌قطعیت

```python
from truss_analysis.uncertainty import default_rv_specs, sample_spec_matrix

specs = default_rv_specs(fire_scenario_temperature=600.0)
means = {"live_load": 1.0, "f_y": 235.0e6, "fire_intensity": 600.0, "E": 210.0e9}
samples = sample_spec_matrix(specs, means, n=2000, seed=42)

print(samples["f_y"].mean() / 235.0e6)  # ≈ 1.0   (لوگ‌نرمال، COV 0.05)
print(samples["fire_intensity"].mean())  # ≈ 600.0 (نرمال کرانه‌دار، σ 50)
```

نمونه‌برداری فقط به `(specs, means, n, seed)` وابسته است — نه به ترتیب
فراخوانی‌ها. `latin_hypercube`، `gaussian_copula_correlate`،
`RunningStat` (گشتاورهای جریان‌یافته) و `probabilistic_ranking` نیز از
همین زیربسته صادر می‌شوند.

### ۵. اعتبارسنجی انرژی

هر فراخوانی `run()` ترازِ کلیِ کلپرون
`W_mech = U_strain + ½·W_prestress` را بررسی می‌کند و در صورت ناسازگاری
`EnergyValidationError` می‌دهد؛ `check_energy` برای خطوط محاسباتی سفارشی
عمومی است. سطح عمومی کتابخانه همچنین شامل `assemble_global_matrices`،
`solve`، `calculate_element_forces`، `structural_report`،
`validate_topology`، `system_critical_temperature`، `dcr_field`،
`ci_two_component` و `euler_buckling_load` است
(`truss_analysis.__all__` را ببینید).

## اسکیمای JSON ورودی

شیء سطح بالا:

| کلید | نوع | الزامی | اعتبارسنجی / توضیح |
|---|---|---|---|
| `units` | `"SI"` یا `"Imperial"` | خیر (پیش‌فرض `SI`) | دستگاه ناشناخته → `UnitConversionError`؛ وقتی کلید غایب است، پرچم `--units` در CLI جایگزین می‌شود |
| `nodes` | آرایه | **بله** | جدول زیر؛ در مجموع حداقل ۳ درجهٔ آزادیِ مقید (وگرنه خطای `Insufficient constraints`) |
| `elements` | آرایه | **بله** | جدول زیر؛ `E > 0` و `A > 0`؛ ارجاع به گره‌های موجود الزامی است |
| `loads` | آرایه | خیر | نیروهای گره‌ای؛ شناسهٔ گرهٔ ناشناخته نادیده گرفته می‌شود |
| `temperature_change` | float | خیر | بخشی از قالب استانداردِ تولیدکننده است؛ اثرات حرارتی در خط تحلیل از `delta_T` هر عضو می‌آیند |
| `options` | object | خیر | برای سازگاری با مدل‌های تولیدشده پذیرفته می‌شود (`use_sparse`، `bc_method`، `penalty_value`، `plot_results`، `displacement_scale`)؛ خط تحلیل فعلی این تنظیمات را از API/CLI می‌گیرد |

`nodes[i]`:

| کلید | نوع | الزامی | توضیح |
|---|---|---|---|
| `id` | int یا string | **بله** | به رشته تبدیل می‌شود؛ باید یکتا باشد |
| `x`, `y` | float | **بله** | مختصات [m] (در حالت امپریال از ft تبدیل می‌شود) |
| `is_support` | bool | خیر (پیش‌فرض `false`) | |
| `support_dx`, `support_dy` | bool | خیر (پیش‌فرض `false`) | درجات آزادی مقید؛ مفصل = هر دو true، غلتک = یکی true |

`elements[i]`:

| کلید | نوع | الزامی | توضیح |
|---|---|---|---|
| `id` | int یا string | **بله** | به رشته تبدیل می‌شود؛ باید یکتا باشد |
| `node_i`, `node_j` | int یا string | **بله** | باید به گره موجود ارجاع دهند؛ حلقهٔ خودی و عضو تکراری در `validate` رد می‌شوند |
| `E` | float | **بله** | مدول یانگ [Pa]؛ باید مثبت باشد |
| `A` | float | **بله** | سطح مقطع [m²]؛ باید مثبت باشد |
| `I_sec` (نام قدیمی `I`) | float | خیر (پیش‌فرض `0`) | ممان دوم سطح [m⁴] برای بررسی کمانش |
| `alpha` | float | خیر (پیش‌فرض `0`) | ضریب انبساط حرارتی [1/°C] |
| `delta_T` | float | خیر (پیش‌فرض `0`) | تغییر دمای عضو [°C] (اختلاف دمای فارنهایت با ۵/۹ تبدیل می‌شود) |
| `delta_L_free` (نام قدیمی `delta_L0`) | float | خیر (پیش‌فرض `0`) | تغییر طول آزادِ تحمیلی (خطای ساخت) [m] |
| `rho` | float | خیر (پیش‌فرض `0`) | چگالی [kg/m³]؛ اگر مثبت باشد، وزن عضو نصف-نصف به دو گرهٔ انتها اعمال می‌شود |
| `section_type`, `effective_length_factor`, `properties` | — | خیر | بخشی از قالب استانداردِ تولیدکننده؛ در ورودی پذیرفته می‌شوند — کمانش فعلی با ضریب طول مؤثر پیش‌فرض `K = 1.0` کار می‌کند |

`loads[i]`:

| کلید | نوع | الزامی | توضیح |
|---|---|---|---|
| `node_id` (نام قدیمی `id`) | int یا string | **بله** | گرهٔ هدف |
| `Fx`, `Fy` | float | خیر (پیش‌فرض `0`) | مؤلفه‌های نیرو [N] (در حالت امپریال از lbf تبدیل می‌شود) |

سازگاری با فایل‌های قدیمی: کلیدهایی با فاصلهٔ اضافی ابتدا/انتها پذیرفته و
نرمال‌سازی می‌شوند (`"E "` ← `"E"`).

## اسکیمای JSON خروجی

`analyze --output result.json` این ساختار را می‌نویسد (آینهٔ dataclass
به نام `AnalysisResult`؛ همهٔ مقادیر SI):

```json
{
  "status": "SUCCESS",
  "displacements": {"<node_id>": {"ux": 0.0, "uy": 0.0}},
  "element_forces": [{"id": "<element_id>", "N": 0.0,
                      "status": "Tension|Compression|Zero",
                      "delta_L_mech": 0.0, "delta_L_prestress": 0.0}],
  "reactions": {"<node_id>": {"Fx": 0.0, "Fy": 0.0}},
  "equilibrium": {"sum_fx": 0.0, "sum_fy": 0.0, "sum_m": 0.0,
                  "is_valid": true},
  "buckling": [{"id": "<element_id>", "N": 0.0, "length": 0.0,
                "P_cr": 0.0, "ratio": 0.0, "slenderness": 0.0,
                "safe": true}]
}
```

`buckling` خالی است مگر اینکه `--check-buckling` داده شود؛ برای اعضای بدون
`I_sec` قابل استفاده، `P_cr` برابر `null` است.

## بررسی‌های درونی

* **اعتبارسنجی گراف/توپولوژی** — گره یتیم، عضو با طول صفر، عضو تکراری،
  حلقهٔ خودی، همبندی، درجهٔ نامعینی، رتبهٔ SVD و عدد حالتی ماتریس سختی
  کاهش‌یافته (`structural_report`؛ خروجیِ `truss-analysis validate`).
* **اعتبارسنجی انرژی** — قضیهٔ کلی کلپرون
  `W_mech = U_strain + ½·W_prestress` باید در هر حل با تلورانس نسبی سخت
  برقرار بماند (وگرنه `EnergyValidationError`).
* **تعادل** — باقی‌ماندهٔ ΣFx، ΣFy و ΣM بین عکس‌العمل‌ها و بارهای اعمالی.
* **بی‌تغییری میدان دمای یکنواخت** — مقیاس یکسانِ سختی همهٔ اعضا نمی‌تواند
  ترتیب بحرانیّت را عوض کند **تا وقتی هیچ عضویی کرنش تحمیلی حرارتی نداشته
  باشد**؛ موتور این ویژگی را **اندازه‌گیری** می‌کند (تاو‑b کندال = ۱) نه اینکه
  فرض کند، و `tests/test_thermal_demand.py` صورت دقیق‌تر آن را برای سازه‌های
  مهارشدهٔ گرم‌شده میخکوب می‌کند (docs/theory.md §5.4).
* **پل مرجع (اختیاری)** — نصب `"truss-analysis[validation]"` ابزارهای
  مقایسه با چارچوب اجزای محدود OpenSees را فراهم می‌کند
  (`truss_analysis.validation`).

## آزمون و توسعه

```bash
make install        # وابستگی‌های اجرا + توسعه + هوک‌های pre-commit
make test           # pytest با گزارش پوشش
make test-cov       # pytest با گیت پوشش ≥ ۹۰٪
make lint           # ruff check + بررسی قالب
make type-check     # mypy (سخت‌گیرانه) روی src/
make check-all      # lint + type-check + test-cov (همان چیزی که CI اجرا می‌کند)
make stats          # اندازه‌گیری مجدد و به‌روزرسانی اعداد آزمون/پوشش در README
make build          # sdist + wheel + twine check
```

وضعیت فعلی این شاخه: **۸۳۸ آزمون پاس، پوشش ۹۴.۱۹٪** (`pytest tests/`)،
`ruff` پاک با مجموعه‌قواعد گسترش‌یافته (E, F, I, W, UP, B, SIM, RUF, PT,
N, D) و `mypy --strict` پاک روی هر ۴۶ ماژول کتابخانه.

زنجیرهٔ پین‌شدهٔ `pre-commit` شامل ruff، mypy (فقط src/)، detect-secrets
(با baseline)، اسکنر بهداشت مخزن (کلیدواژه/آرتیفکت/مسیر/اندازهٔ فایل) و
هوک‌های معمولِ بهداشت فایل است. CI (GitHub Actions) لینتر، تایپ
سخت‌گیرانه، ماتریس تست (ubuntu/windows/macos × ۳.۱۰/۳.۱۱/۳.۱۲)، گیت
پوشش، بررسی‌های بسته‌بندی و build داکر را اجرا می‌کند.

## ساختار پروژه

```text
src/truss_analysis/
├── assembly.py, solver.py, postprocess.py   # خط لولهٔ سختی
├── model.py, fileio.py, units.py            # قرارداد داده + ورودی/خروجی + واحدها
├── main.py                                  # CLI (analyze/validate/generate/version)
├── graph_validation.py                      # بررسی‌های توپولوژی
├── limitstates.py, degradation.py           # DCR، ظرفیت، دمای بحرانی + حالت خرابی
├── sections.py, heterogeneity.py            # مدل‌های مقطع
├── material/                                # داده و میان‌یاب‌های EN 1993-1-2 (+ آلفای قوسی)
├── thermal/                                 # منحنی ISO 834 + گرمایش ظرفیت متمرکز + لایهٔ سازگاری
├── stability.py                             # سختی هندسی K_G + ضریب دوشاخگی خطی‌شده
├── criticality/                             # موتور CI رتبه‑۱، شاخص‌ها، رتبه‌بندی، سناریوها
├── uncertainty/                             # متغیرهای تصادفی، نمونه‌برداری ایمان–کونوور/کوپولا، آمار جریان‌یافته
├── retrofit/                                # اقدام‌ها، هزینه‌ها، راهبردها
├── validation/                              # معیارها، پل مرجع، سورگیت
├── sensitivity.py, reliability.py           # گرادیان‌ها + توابع قابلیت اطمینان (حاشیه‌های هم‌تراز با chi)
└── visualization.py                         # رسم شکل (matplotlib با بارگذاری تنبل)
tests/                                       # ۸۳۸ آزمون شامل tests/validation/
docs/theory.md, docs/error_codes.md          # فرمول‌بندی + مرجع کدهای خطا
examples/                                    # مدل‌های مثالِ قابل اجرا
scripts/                                     # ابزارهای نگهداری مخزن
```

## مشارکت

[CONTRIBUTING.md](CONTRIBUTING.md) ([فارسی](CONTRIBUTING.fa.md)) را ببینید.
خلاصه: fork کنید ← `pip install -e ".[dev]"` ← `pre-commit install` ←
تغییر را با آزمون بیاورید ← `make check-all` سبز ← Pull Request باز کنید.

## استناد

فرادادهٔ استناد در [CITATION.cff](CITATION.cff) نگهداری می‌شود:

```bibtex
@software{truss_analysis,
  author  = {bmhmdyan279-png},
  title   = {truss\_analysis: linear 2D truss finite-element analysis for Python},
  year    = {2026},
  version = {2.5.0},
  license = {MIT},
  url     = {https://github.com/bmhmdyan279-png/truss-analysis-2d}
}
```

## مجوز

MIT — فایل [LICENSE](LICENSE) را ببینید.
