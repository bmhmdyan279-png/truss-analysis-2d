<!--
خلاصهٔ فارسی: قبل از ارسال PR، همهٔ دروازه‌های کیفیت زیر را محلی اجرا کنید
(`make check-all` معادل CI است). عنوان کامیت‌ها از قالب conventional استفاده
کند و کارِ انجام‌شده با کمک ابزار، با trailer مربوطه اعلام شود (CONTRIBUTING.md).
-->

## Summary / خلاصه

<!-- What does this PR change and why? Link the issue it closes, if any
("Closes #123"). -->

## Evidence / شواهد

<!-- For any numerical claim: which test/benchmark proves it, and against
which independent oracle? This project does not accept unmeasured claims
(see CONTRIBUTING.md, "Findings are recorded even when they are wrong"). -->

## Checklist / چک‌لیست

Quality gates (mirror CI — `make check-all` runs the first four):

- [ ] `make lint` — `ruff check` + `ruff format --check` clean over
      `src/ tests/ scripts/ benchmarks/ docs/`
- [ ] `make type-check` — `mypy src` clean under `--strict`
- [ ] `make test-cov` — full suite green, coverage gate ≥ 90 %
- [ ] `make stats` re-run **if** the test count, coverage or module count
      changed (the README statistics gate fails otherwise)

Project-specific rules:

- [ ] `python scripts/sync_requirements.py` run if `pyproject.toml`
      dependencies changed (`tests/test_packaging.py` enforces the mirrors)
- [ ] New features carry tests; reference values name their independent
      oracle — no fabricated numbers, no mocked physics
- [ ] EN 1993-1-2 material values are **not** re-typed anywhere
      (single source: `src/truss_analysis/material/data/en1993_1_2_table3_1.json`)
- [ ] All internal quantities are SI; conversion happens at the boundary
      (`units.to_si`)
- [ ] `docs/theory.md` updated if the change touches modelled physics or a
      claim in the verification matrix; `physics_boundary.yaml` updated if
      supported/not-supported status changed
- [ ] `CHANGELOG.md` — entry added under `[Unreleased]`

Process:

- [ ] Commit messages use the documented prefixes
      (`feat:` / `fix:` / `docs:` / `test:` / `chore:` / `refactor:`)
- [ ] Tool-assisted commits carry the `Co-authored-by:` trailer
      (see CONTRIBUTING.md, "Commit Provenance and Tool Assistance")
