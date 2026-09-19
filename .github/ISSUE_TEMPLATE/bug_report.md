---
name: Bug report / گزارش اشکال
about: Something computes a wrong value, crashes, or disagrees with the documentation
title: "bug: "
labels: bug
---

<!--
لطفاً همهٔ بخش‌ها را پر کنید. گزارشی که ورودیِ قابل بازتولید نداشته باشد،
قابل پیگیری نیست. اگر خروجی عددی «اشتباه» به نظر می‌رسد، ابتدا بررسی کنید که
در دامنهٔ اعتبار مدل باشد: docs/theory.md (بخش ۱۲) و
src/truss_analysis/data/physics_boundary.yaml
-->

## Description / شرح اشکال

<!-- One or two sentences: what did you run, what happened, what should have
happened? -->

## Reproduction / بازتولید

Minimal input file and command. Attach or inline the smallest JSON model that
shows the problem (a 3-node truss beats a 300-node bridge).

```bash
truss-analysis version
truss-analysis model.json
```

```json
{ "nodes": [], "elements": [], "loads": [] }
```

Or the equivalent Python snippet:

```python
from truss_analysis import run
```

## Expected vs actual / انتظار در برابر واقعیت

* **Expected:** (analytical value, documentation quote, or previous-version behaviour)
* **Actual:** (value/message; paste the **full traceback** if it crashed)

## Environment / محیط

| Item | Value |
|---|---|
| `truss-analysis version` | |
| Python (`python --version`) | |
| OS | |
| NumPy / SciPy (`python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"`) | |
| Install method (PyPI / source / Docker) | |

## Checklist / چک‌لیست

- [ ] I searched existing issues (open and closed) for this problem
- [ ] The behaviour is within the documented model scope (`docs/theory.md` §12);
      if it is a *scope* complaint, this is still the right place, but say so
- [ ] If an error code was raised, I checked `docs/error_codes.md`
- [ ] I have not modified the library source (or I can reproduce on a clean install)
