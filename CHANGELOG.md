## [1.5.0] - 2026-07-06

### Added
- All 65 tests now passing (0 skipped)

### Fixed
- Unskipped 9 tests by updating to new input_data API
- Fixed test_single_element.py: removed duplicate @pytest.mark.skip decorators
- Fixed test_solver.py: rewrote input_data from dict to list format
- Fixed calculate_total_energy() calls with F_global parameter

### Changed
- Improved test coverage from 56/65 to 65/65 (100% pass rate)
## [1.4.1] - 2026-07-06

### Fixed
- **Critical**: 11 skipped tests now passing
  - `test_edge_cases.py`: Rewritten to use list-based input_data API
  - `test_single_element.py`: Migrated from non-existent add_node/add_element API
  - `test_solver.py`: Fixed dict→list structure mismatch in nodes/elements

### Added
- `tests/test_api_contract.py`: Regression tests for API structure validation
- Validation in `TrussModel._create_nodes()` to reject dict input with clear error

### Documentation
- Added API migration guide in README.md
- Documented input_data schema in docs/schema.md
## [1.4.0] - 2026-07-06

### تغییرات عمده (Professional Polish)
- **ساختار استاندارد `src/`**: انتقال تمام ماژول‌ها به `src/truss_analysis/` برای بسته‌بندی حرفه‌ای.
- **CLI Entry Point**: اضافه شدن دستور `truss-analyze` از طریق `console_scripts`.
- **حذف فایل‌های اضافی**: حذف `setup.py` قدیمی و `__init__.py` از ریشه.
- **یکپارچه‌سازی Importها**: استفاده از relative imports در هسته پکیج و absolute imports در تست‌ها.
- **Shim backward compatibility**: حفظ `main.py` در ریشه برای اجرای آسان.

### بهبودهای زیرساختی
- تنظیم `pythonpath` در `pytest` برای اجرای تست‌ها بدون نیاز به نصب.
- به‌روزرسانی `pyproject.toml` با متادیتای کامل و نسخه 1.4.0.
- رفع تناقض‌های ظاهری در README (تأکید بر Python 3.8+).

## [1.3.1] - 2026-07-06

### Fixed
- Convert NumPy bool to Python bool in buckling warning to ensure consistent type checking
- Fix test assertion compatibility with NumPy boolean values

### Changed
- Improved type consistency in solver results
## [1.1.0] - 2026-06-30

### تغییرات عمده
- **ساماندهی ساختار مخزن**: انتقال فایل‌های تست به `tests/` و فایل‌های نمونه به `examples/` برای هماهنگی با مستندات.
- **بهبود خوانایی کد**: بازنویسی تابع `solve_displacements` در `solver.py` با استخراج توابع کمکی (`_solve_elimination`، `_solve_penalty`).
- **افزایش پایداری**: حذف تبدیل‌های خودکار ماتریس‌های تنک به متراکم در صورت خطا (اکنون خطا به‌صورت شفاف گزارش می‌شود).

### بهبودهای جزئی
- حذف فایل‌های اضافی (`model.py.bak`).
- اصلاح دستور نصب در README.
- افزودن فایل `pytest.ini` برای تنظیم خودکار مسیر تست‌ها.

### زیرساخت
- راه‌اندازی GitHub Actions برای اجرای خودکار تست‌ها روی پایتون ۳.۸ تا ۳.۱۱.
-
# گزارش تغییرات

## [1.0.0] - 2026-01-01
### اضافه شده
- تحلیل کامل خرپای ۲بعدی
- اثرات حرارتی و خطای ساخت
- تحلیل کمانش با فرمول اویلر
- پشتیبانی از واحدهای مختلف
- خروجی‌های گرافیکی و گزارش‌های CSV/JSON/Markdown
- ۶۵ تست کامل با پوشش ۱۰۰٪
## [v1.5.0] - 2026-07-06

### Added
- All 65 tests now passing (0 skipped)

### Fixed
- Unskipped 9 tests by updating to new input_data API
- Fixed test_single_element.py: removed duplicate @pytest.mark.skip decorators
- Fixed test_solver.py: rewrote input_data from dict to list format
- Fixed calculate_total_energy() calls with F_global parameter

### Changed
- Improved test coverage from 56/65 to 65/65 (100% pass rate)
