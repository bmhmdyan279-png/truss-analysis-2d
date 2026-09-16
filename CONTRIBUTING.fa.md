# راهنمای مشارکت

از مشارکت شما در پروژه Truss Analysis 2D استقبال می‌کنیم! این راهنما به شما کمک می‌کند تا به راحتی در توسعه پروژه شرکت کنید.

## 🚀 شروع سریع

1. **Fork** کنید
2. **Clone** کنید:
   ```bash
   git clone https://github.com/YOUR_USERNAME/truss-analysis-2d.git
   cd truss-analysis-2d
   ```
3. **محیط توسعه** را آماده کنید:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## 🧪 تست‌ها

قبل از ارسال PR، مطمئن شوید:

```bash
# همه تست‌ها پاس شوند
pytest

# پوشش کد >= ۹۰٪ (همان گیت CI)
pytest tests/ --cov=src/truss_analysis --cov-report=term-missing --cov-fail-under=90

# Pre-commit hooks پاس شوند
pre-commit run --all-files

# یا همهٔ گیت‌های محلی یک‌جا (lint + mypy --strict + تست با گیت پوشش)
make check-all
```

## 📝 قوانین کد

### استایل کد
- **Ruff**: برای linting و formatting استفاده می‌شود
- **Line length**: حداکثر 88 کاراکتر
- **Type hints**: برای همه توابع عمومی
- **Docstrings**: به فرمت NumPy (طبق قواعد pydocstyle در ruff با `convention = "numpy"`)

### مثال تابع خوب

```python
def calculate_element_forces(
    nodes: list[Node],
    elements: list[Element],
    U: np.ndarray,
) -> tuple[list[dict], float, float]:
    """Calculate axial forces, strain energy, and prestress work.

    Parameters
    ----------
    nodes : list[Node]
        List of Node objects.
    elements : list[Element]
        List of Element objects.
    U : np.ndarray
        Displacement vector (numpy array).

    Returns
    -------
    tuple[list[dict], float, float]
        - results: List of element force dictionaries
        - strain_energy: Total mechanical strain energy (J)
        - prestress_work: Total prestress work (J)

    Examples
    --------
    >>> results, U_strain, W_prestress = calculate_element_forces(
    ...     nodes, elements, U
    ... )
    """
    # Implementation...
```

## 🔄 فرآیند Pull Request

1. **Branch** بسازید:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **تغییرات** را اعمال کنید

3. **تست** کنید:
   ```bash
   pytest
   pre-commit run --all-files
   ```

4. **Commit** کنید:
   ```bash
   git add -A
   git commit -m "feat: add amazing feature"
   ```

   **فرمت commit message:**
   - `feat:` ویژگی جدید
   - `fix:` رفع باگ
   - `docs:` تغییرات مستندات
   - `test:` اضافه کردن تست
   - `chore:` تغییرات ساختاری
   - `refactor:` بازنویسی کد

5. **Push** کنید:
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Pull Request** باز کنید

## 🔏 منشأ کامیت و کمک ابزار

ارزش محوری این پروژه حساب‌پذیری است: نتایج یک بلوک `solver_metadata` با
نسخه‌ها و تجزیهٔ استفاده‌شده حمل می‌کنند، مرز فیزیکی یک artefact با هشِ
pin‌شده است نه متن، و هر مقدار مرجع در `benchmarks/` نام oracle خود را می‌گوید.
تاریخچهٔ کامیت بخشی از همان زنجیره است و با همان معیار سنجیده می‌شود.

**کار با کمک ابزار اعلام می‌شود، پنهان نمی‌شود.** کامیت‌هایی که با کمک یک
عامل یا دستیار کدنویس تولید شده‌اند یک trailer از نوع `Co-authored-by:` با نام
همان ابزار دارند. این تشریفات نیست: یک بازبین حق دارد بداند کدام کامیت‌ها
دستی نوشته شده‌اند و کدام تولید شده‌اند، چون این دو حالتِ شکستِ متفاوتی دارند و
سطح متفاوتی از موشکافی می‌طلبند. کامیت تولیدشده‌ای که از همهٔ گیت‌ها پاس شود
همچنان کامیتی است که کسی خط‌به‌خط نخوانده، مگر اینکه کسی صریحاً بگوید.

**کامیت‌هایتان را امضا کنید.** `git commit -S` وقتی کلیدتان روی GitHub ثبت شده
باشد، تا نشان «Verified» به این معنا باشد که دارندهٔ کلید کامیت را تولید کرده،
نه صرفاً اینکه یک نشانی ایمیل پیکربندی شده بود. تاریخچه‌ای که انتسابش قابل
راستی‌آزمایی نباشد، در پروژه‌ای که موضوعش همان منشأ است، زنجیرهٔ سستی است.

**یک هویت برای هر نویسنده.** تاریخچهٔ اولیهٔ مخزن چند هویت برای یک مشارکت‌کننده
انباشت، از جمله پیش‌فرض یک ماشین لوکال (`dev@local`) و یک هویت عمومیِ دستیار بدون
trailer. آن کامیت‌ها دست‌نخورده می‌مانند — بازنویسی تاریخچهٔ منتشرشده هزینه‌اش
از فایده‌اش بیشتر است — ولی کار جدید از یک هویت پیکربندی‌شده استفاده می‌کند.
بررسی با:

```bash
git log --format='%an <%ae>' | sort -u
```

**یافته‌ها ثبت می‌شوند حتی وقتی نادرست‌اند.** چند دور ممیزی بیرونی یافته‌هایی
تولید کرد که بازتولید نشدند. آن‌ها درست پیاده‌سازی شده‌اند و اختلافشان در خودِ
پیام کامیت نوشته شده، نه اینکه بی‌صدا حذف شوند — تا خوانندهٔ بعدی هم ادعا را
ببیند هم اندازه‌گیری را. دو نمونه از دور هفتم: یک `ValueError` خامِ گزارش‌شده که
ادعا می‌شد از `brentq` در حدِ مقطع solid فرار می‌کند (scipy مقدار `f(a) == 0` را
تحمل می‌کند، پس بازتولید نمی‌شود — ولی گارد همچنان به‌اندازهٔ یک rounding از
اشتباه بودن فاصله داشت و حالا صریح است)، و یک تست همگرایی مرتبهٔ چهارم که
اصلاً مرتبه را تست نمی‌کرد چون انتگرال‌سازِ مرجعش از چیزی که اندازه می‌گرفت
دقیق‌تر نبود.

## 🐛 گزارش باگ

لطفاً شامل این موارد باشید:

- **شرح مشکل**: چه اتفاقی افتاد؟
- **مراحل بازتولید**: چگونه می‌توان باگ را بازتولید کرد؟
- **خروجی مورد انتظار**: چه باید می‌شد؟
- **محیط**: نسخه Python، OS
- **فایل ورودی**: اگر ممکن است

## 💡 پیشنهاد ویژگی

لطفاً شامل این موارد باشید:

- **شرح ویژگی**: چه می‌خواهید اضافه شود؟
- **مورد استفاده**: چرا این ویژگی مفید است؟
- **پیاده‌سازی پیشنهادی**: (اختیاری) چگونه می‌توان پیاده کرد؟

## 📚 اضافه کردن تست

برای هر ویژگی جدید، تست بنویسید:

```python
def test_amazing_feature():
    """Test that amazing feature works correctly."""
    # Arrange
    nodes = [Node(id="1", x=0.0, y=0.0)]

    # Act
    result = amazing_function(nodes)

    # Assert
    assert result == expected_value
```

## 🔢 Versioning

ما از [Semantic Versioning](https://semver.org/) استفاده می‌کنیم:

- **MAJOR.MINOR.PATCH**
- `2.1.0` -> `2.2.0`: ویژگی جدید backward-compatible
- `2.1.0` -> `2.1.1`: رفع باگ
- `2.1.0` -> `3.0.0`: تغییر breaking

## ❓ سوالات؟

- **GitHub Discussions**: برای سوالات عمومی
- **GitHub Issues**: برای باگ‌ها و پیشنهادات
- **Email**: برای موارد خصوصی

## 🙏 تشکر

از همه مشارکت‌کنندگان متشکریم! 🎉

---

این راهنما بر اساس [Contributing Guide Template](https://github.com/nayafia/contributing-template) است.
