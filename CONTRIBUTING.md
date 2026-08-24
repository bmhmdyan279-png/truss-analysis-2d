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

# پوشش کد >= 85%
pytest --cov=truss_analysis --cov-report=term-missing

# Pre-commit hooks پاس شوند
pre-commit run --all-files
```

## 📝 قوانین کد

### استایل کد
- **Ruff**: برای linting و formatting استفاده می‌شود
- **Line length**: حداکثر 88 کاراکتر
- **Type hints**: برای همه توابع عمومی
- **Docstrings**: به فرمت Google

### مثال تابع خوب

```python
def calculate_element_forces(
    nodes: list[Node],
    elements: list[Element],
    U: np.ndarray,
) -> tuple[list[dict], float, float]:
    """Calculate axial forces, strain energy, and prestress work.

    Args:
        nodes: List of Node objects
        elements: List of Element objects
        U: Displacement vector (numpy array)

    Returns:
        Tuple of:
        - results: List of element force dictionaries
        - strain_energy: Total mechanical strain energy (J)
        - prestress_work: Total prestress work (J)

    Example:
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
