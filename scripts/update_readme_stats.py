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
    python scripts/update_readme_stats.py --check    # fail if the READMEs are stale
    python scripts/update_readme_stats.py --tests 539 --coverage 95.06 \
        --modules 44                                # patch from given values

Exit code is 0 on success, 1 if a pattern could not be found (the READMEs
changed shape and this script must be updated, not silently skipped).

``--check`` is the CI/pre-commit mode: it measures the same values, renders
the same patches *in memory*, and exits 1 when either README differs from
what the numbers would produce.  It never writes.  That turns "remember to
run ``make stats`` before releasing" -- which the round-5 and round-6 audits
both caught being forgotten, leaving the READMEs advertising 356 tests /
95.44 % and then 611 tests / 95.3 % against a suite that had moved on -- into
a gate that fails the build instead.
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

def _opensees_ignores() -> list[str]:
    """Return the pytest --ignore flags for the reference-solver files.

    tests/validation/test_level4_opensees.py (and its two siblings) pull in
    truss_analysis.validation, which imports openseespy.  When that import
    fails -- either because openseespy is absent, or because its compiled
    extension cannot be loaded (the DLL-load failure seen on Windows/py3.14
    and on some Linux CI runners even with the ``validation`` extra
    installed) -- pytest aborts during collection with rc=2, and no
    statistics can be measured.

    The ``test`` and ``coverage`` CI jobs already pass the same three
    --ignore flags on the command line.  The ``stats`` job measures the same
    suite and must therefore reach the same count.  Probing the import at
    runtime, rather than hard-coding the flags, keeps the measured number
    honest on a machine where the reference solver really does load.
    """
    try:
        import openseespy.opensees  # type: ignore[import-not-found]  # noqa: F401
    except Exception:  # noqa: BLE001 -- DLL-load failures are not ImportError
        return [
            "--ignore=tests/validation/test_level4_opensees.py",
            "--ignore=tests/validation/test_level4_rho_branches.py",
            "--ignore=tests/validation/test_level5_crossval.py",
        ]
    return []


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
            *_opensees_ignores(),  # ← این خط
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


def render(
    text: str,
    name: str,
    patterns: list[tuple[re.Pattern[str], str]],
    fmt: dict[str, str],
) -> tuple[str, int, list[str]]:
    """Apply every pattern to ``text`` in memory.

    Returns ``(new_text, n_hits, missing_patterns)``.  Pure, so ``--check``
    can compare against the file on disk without ever writing to it.
    """
    hits = 0
    missing: list[str] = []
    for pattern, template in patterns:
        replacement = template.format(**fmt)
        text, count = pattern.subn(replacement, text)
        if count == 0:
            missing.append(pattern.pattern)
            print(f"WARNING: pattern not found in {name}: {pattern.pattern}")
        hits += count
    return text, hits, missing


def patch(
    path: Path, patterns: list[tuple[re.Pattern[str], str]], fmt: dict[str, str]
) -> int:
    """Render and write; returns the number of patched occurrences."""
    text, hits, _missing = render(
        path.read_text(encoding="utf-8"), path.name, patterns, fmt
    )
    path.write_text(text, encoding="utf-8")
    return hits


def check_stale(
    patterns_en: list[tuple[re.Pattern[str], str]],
    patterns_fa: list[tuple[re.Pattern[str], str]],
    fmt: dict[str, str],
) -> int:
    """Exit status for ``--check``: 0 when both READMEs already agree.

    Compares rendered-in-memory text against the files without writing, and
    reports *which* README is stale plus how many occurrences would change.
    A missing pattern is also a failure here: if the READMEs changed shape,
    the numbers are no longer being maintained at all.
    """
    stale = 0
    for path, patterns in ((README_EN, patterns_en), (README_FA, patterns_fa)):
        current = path.read_text(encoding="utf-8")
        rendered, hits, missing = render(current, path.name, patterns, fmt)
        if missing:
            print(f"STALE: {path.name}: {len(missing)} pattern(s) no longer match")
            stale += 1
        elif rendered != current:
            changed = sum(
                1
                for a, b in zip(
                    current.splitlines(), rendered.splitlines(), strict=False
                )
                if a != b
            )
            print(
                f"STALE: {path.name}: {changed} line(s) would change "
                f"({hits} occurrence(s)); run `make stats`"
            )
            stale += 1
        else:
            print(f"OK: {path.name} matches the measured statistics")
    return 1 if stale else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=int, default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--modules", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail (exit 1) if the READMEs do not already quote these numbers",
    )
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

    if args.check:
        return check_stale(EN_PATTERNS_USE, FA_PATTERNS, fmt)

    hits_en = patch(README_EN, EN_PATTERNS_USE, fmt)
    hits_fa = patch(README_FA, FA_PATTERNS, fmt)
    print(f"patched {hits_en} occurrence(s) in README.md, {hits_fa} in README.fa.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
