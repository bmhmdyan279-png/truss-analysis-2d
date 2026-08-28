# Release Notes

## v2.4.0 — Infrastructure Overhaul (Phase 0 Complete)

**Release Date:** 2026-08-29
**Tag:** `v2.4.0`
**Commit:** `9c56e17`

### 🇬🇧 English

This release completes **Phase 0 (Infrastructure & Repair)** of the research project:
*"Automated redundancy-based consequence factor (SCF) extraction for 2D truss members."*

#### What Changed
- **BREAKING:** Removed root-level `main.py` shim — use `python -m truss_analysis.main` or the `truss-analysis` CLI instead
- Unified CLI entry point at `src/truss_analysis/main.py`
- Migrated to `src/` layout for proper package structure
- Pinned all dependencies in `pyproject.toml`

#### What Was Added
- Comprehensive test suite (39 tests, 90.5% coverage)
- GitHub Actions CI/CD pipeline (pytest, ruff, mypy)
- Backward compatibility for legacy JSON schemas with space-padded keys
- Pre-commit hooks for code quality enforcement
- `PROJECT_DOCUMENTATION/` structure (git-ignored, local only)

#### What Was Fixed
- JSON schema tolerance for legacy files (e.g., `example1.json`)
- Coverage tracking for CLI execution paths
- Type hints and mypy compliance across all modules

### 🇮🇷 فارسی

این نسخه **فاز صفر (زیرساخت و ترمیم)** پروژه پژوهشی را تکمیل می‌کند.

#### تغییرات
- **شکننده:** فایل `main.py` ریشه حذف شد — از `python -m truss_analysis.main` یا دستور `truss-analysis` استفاده کنید
- یکپارچه‌سازی نقطه ورود CLI در `src/truss_analysis/main.py`
- مهاجرت به ساختار `src/` برای بسته‌بندی صحیح
- پین کردن تمام وابستگی‌ها در `pyproject.toml`

#### افزوده‌ها
- مجموعه تست جامع (۳۹ تست، پوشش ۹۰.۵٪)
- خط لوله CI/CD با GitHub Actions (pytest, ruff, mypy)
- سازگاری با فایل‌های JSON قدیمی با کلیدهای فاصله‌دار
- هوک‌های pre-commit برای کنترل کیفیت کد

#### اصلاحات
- تحمل خطای فایل‌های JSON قدیمی (مانند `example1.json`)
- ردیابی صحیح پوشش کد برای مسیرهای اجرایی CLI
- انطباق type hints و mypy در تمام ماژول‌ها

---
*Generated for release v2.4.0*
