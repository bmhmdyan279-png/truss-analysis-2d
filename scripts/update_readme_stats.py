#!/usr/bin/env python3
"""Regenerate the test/coverage/module statistics quoted in both READMEs.

The suite size and coverage percentage are quoted in four places per README
(English + Persian). Hand-maintained numbers drift — the round-5 audit found
the READMEs still advertising "356 tests / 95.44 %" long after the suite had
grown past 500. This script measures the real values (one full coverage run,
one ``mypy`` invocation) and patches every occurrence in place, so the
release checklist is ``python scripts/update_readme_stats.py`` and nothing
else.

Usage
-----
::

    python scripts/update_readme_stats.py           # measure + patch
    python scripts/update_readme_stats.py --dry-run # measure + report only
    python scripts/update_readme_stats.py --tests 539 --coverage 95.06 \
        --modules 44                                # patch from given values

Exit code is 0 on success, 1 if a pattern could not be found (the READMEs
changed shape and this script must be updated, not silently skipped).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README_EN = REPO_ROOT / "README.md"
README_FA = REPO_ROOT / "README.fa.md"

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def _fa(value: str) -> str:
    return value.translate(PERSIAN_DIGITS)


def measure() -> tuple[int, float, int]:
    """Return ``(n_tests, coverage_percent, n_modules)`` from real runs."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--cov=src",
            "--cov-report=term",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    out = proc.stdout
    m_tests = re.search(r"(\d+) passed", out)
    m_cov = re.search(r"Total coverage: ([\d.]+)%", out) or re.search(
        r"^TOTAL\s+\d+\s+\d+\s+([\d.]+)%", out, re.MULTILINE
    )
    if proc.returncode != 0 or not m_tests or not m_cov:
        msg = f"test run failed or unparseable (rc={proc.returncode})"
        raise RuntimeError(msg)

    proc_mypy = subprocess.run(
        [sys.executable, "-m", "mypy", "src/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    m_mod = re.search(r"in (\d+) source files", proc_mypy.stdout)
    modules = int(m_mod.group(1)) if m_mod else 0
    return int(m_tests.group(1)), float(m_cov.group(1)), modules


# (pattern, replacement template) — templates use {n}, {cov1}, {cov2}, {fa_*}.
# cov1 = one-decimal coverage (badge/prose), cov2 = two-decimal (status line).
EN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"badge/coverage-[\d.]+%25-brightgreen"),
        "badge/coverage-{cov1}%25-brightgreen",
    ),
    (
        re.compile(r"test suite of \d+ tests\n\([\d.]+ % coverage"),
        "test suite of {n} tests\n({cov1} % coverage",
    ),
    (
        re.compile(r"\*\*\d+ tests passing, [\d.]+ % coverage\*\*"),
        "**{n} tests passing, {cov2} % coverage**",
    ),
    (
        re.compile(r"tests/(\s+)# \d+ tests incl\."),
        r"tests/\1# {n} tests incl.",
    ),
    (
        re.compile(r"clean on all \d+ library modules"),
        "clean on all {modules} library modules",
    ),
]

FA_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"badge/coverage-[\d.۰-۹]+%25-brightgreen"),
        "badge/coverage-{fa_cov1}%25-brightgreen",
    ),
    (
        re.compile(r"با [\d۰-۹]+ آزمون پوشش [\d۰-۹.]+٪"),
        "با {fa_n} آزمون پوشش {fa_cov1}٪",
    ),
    (
        re.compile(r"\*\*[\d۰-۹]+ آزمون پاس، پوشش [\d۰-۹.]+٪\*\*"),
        "**{fa_n} آزمون پاس، پوشش {fa_cov2}٪**",
    ),
    (
        re.compile(r"tests/(\s+)# [\d۰-۹]+ آزمون شامل"),
        r"tests/\1# {fa_n} آزمون شامل",
    ),
]


def patch(
    path: Path, patterns: list[tuple[re.Pattern[str], str]], fmt: dict[str, str]
) -> int:
    text = path.read_text(encoding="utf-8")
    hits = 0
    for pattern, template in patterns:
        replacement = template.format(**fmt)
        text, count = pattern.subn(replacement, text)
        if count == 0:
            print(f"WARNING: pattern not found in {path.name}: {pattern.pattern}")
        hits += count
    path.write_text(text, encoding="utf-8")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=int, default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--modules", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.tests is not None and args.coverage is not None:
        n, cov = args.tests, args.coverage
        modules = args.modules or 0
    else:
        n, cov, modules = measure()

    cov1 = f"{cov:.1f}"
    cov2 = f"{cov:.2f}"
    fmt = {
        "n": str(n),
        "cov1": cov1,
        "cov2": cov2,
        "modules": str(modules),
        "fa_n": _fa(str(n)),
        "fa_cov1": _fa(cov1),
        "fa_cov2": _fa(cov2),
    }
    print(f"measured: tests={n} coverage={cov2}% modules={modules}")
    if args.dry_run:
        return 0
    if modules == 0:
        print("WARNING: module count unavailable; module-count line not patched")
        EN_PATTERNS_USE = [p for p in EN_PATTERNS if "{modules}" not in p[1]]
    else:
        EN_PATTERNS_USE = EN_PATTERNS

    hits_en = patch(README_EN, EN_PATTERNS_USE, fmt)
    hits_fa = patch(README_FA, FA_PATTERNS, fmt)
    print(f"patched {hits_en} occurrence(s) in README.md, {hits_fa} in README.fa.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
