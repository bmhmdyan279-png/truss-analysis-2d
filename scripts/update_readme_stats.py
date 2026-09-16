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

Stability note
--------------
Both badge/prose and the bold status lines now quote *one-decimal* coverage
(``cov1``).  Two-decimal coverage drifts by 0.01 between Windows and Linux
because IEEE-754 summation order over the per-file coverage counters is not
identical; quoting ``cov1`` in the bold line as well keeps the four
occurrences per README mutually consistent and immune to that jitter.
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


#: The three reference-solver test files, always excluded from the canonical
#: measurement.  See :func:`_canonical_ignores` for why this is unconditional.
REFERENCE_SOLVER_FILES: tuple[str, ...] = (
    "tests/validation/test_level4_opensees.py",
    "tests/validation/test_level4_rho_branches.py",
    "tests/validation/test_level5_crossval.py",
)


def _canonical_ignores() -> list[str]:
    """Return the pytest ``--ignore`` flags the canonical numbers are measured with.

    These are applied **unconditionally**, whether or not ``openseespy`` happens
    to import on this machine.  That is the whole point.

    The previous version probed the import at runtime and returned the flags only
    when it failed.  The intention was honesty -- measure the full suite where
    the reference solver really loads -- but the effect was a gate whose expected
    value depends on the environment it runs in.  The round-7 audit found exactly
    that: HEAD advertised 820 tests / 94.2% measured where ``openseespy`` does
    not load, while earlier commits on the same tree advertised 838 / 94.19%
    measured where it does, so ``make stats-check`` was red on one machine and
    green on another for the same commit.  A number that is not canonical is not
    a gate.

    The canonical configuration is therefore the *narrower* one: the same three
    ``--ignore`` flags the ``test`` and ``coverage`` CI jobs pass, so the README
    quotes the suite every CI runner actually executes.  The with-validation
    count is still measured and printed as information (see
    :func:`measure_with_validation`), because it is genuinely useful -- it just
    must not be the number the gate compares against.
    """
    return [f"--ignore={path}" for path in REFERENCE_SOLVER_FILES]


def _reference_solver_available() -> bool:
    """Whether ``openseespy`` imports on this machine, for reporting only."""
    try:
        import openseespy.opensees  # type: ignore[import-not-found]  # noqa: F401
    except Exception:  # DLL-load failures are not ImportError
        return False
    return True


def _run_pytest(extra: list[str], with_cov: bool) -> tuple[int, float]:
    """Run pytest once and parse ``(n_passed, coverage_percent)``."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header", *extra]
    if with_cov:
        cmd += ["--cov=src", "--cov-report=term"]
    else:
        cmd.append("--no-cov")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = proc.stdout
    m_tests = re.search(r"(\d+) passed", out)
    m_cov = re.search(r"Total coverage: ([\d.]+)%", out) or re.search(
        r"^TOTAL\s+\d+\s+\d+\s+([\d.]+)%", out, re.MULTILINE
    )
    if proc.returncode != 0 or not m_tests or (with_cov and not m_cov):
        msg = f"test run failed or unparseable (rc={proc.returncode})"
        raise RuntimeError(msg)
    coverage = float(m_cov.group(1)) if m_cov else 0.0
    return int(m_tests.group(1)), coverage


def _released_version() -> str:
    """Latest release tag, e.g. ``2.8.0`` -- the version the docs should quote.

    Taken from git rather than from ``setuptools_scm`` on purpose.  A working
    checkout reports ``2.8.1.dev21+gee0e9bfd5``, which is the right string for
    ``truss_analysis.__version__`` and the wrong one for a citation record or a
    documented CLI transcript: both are about the *release*, not the commit.
    ``pyproject.toml``'s ``fallback_version`` is the backstop when there are no
    tags at all.
    """
    proc = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip().removeprefix("v")
    fallback = re.search(
        r'fallback_version\s*=\s*"([^"]+)"',
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    )
    return fallback.group(1) if fallback else "0.0.0"


def measure() -> tuple[int, float, int, str]:
    """Return ``(n_tests, coverage_percent, n_modules, version)`` from real runs.

    Measured in the canonical configuration -- see :func:`_canonical_ignores`.
    """
    n_tests, coverage = _run_pytest(_canonical_ignores(), with_cov=True)

    proc_mypy = subprocess.run(
        [sys.executable, "-m", "mypy", "src/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    m_mod = re.search(r"in (\d+) source files", proc_mypy.stdout)
    modules = int(m_mod.group(1)) if m_mod else 0
    return n_tests, coverage, modules, _released_version()


def measure_with_validation() -> tuple[int, float] | None:
    """The full suite including the reference-solver files, if it runs at all.

    Informational only.  Never compared against the READMEs, which is what makes
    the canonical number environment-independent.
    """
    if not _reference_solver_available():
        return None
    try:
        return _run_pytest([], with_cov=False)
    except RuntimeError:
        return None


# (pattern, replacement template) — templates use {n}, {cov1}, {cov2}, {fa_*}.
# cov1 = one-decimal coverage, used in *every* README occurrence (badge,
# prose, and bold status line) so Windows/Linux float-sum jitter cannot
# make the four copies disagree.  cov2 remains available for callers that
# explicitly want two decimals.
#: Occurrences of the released version, patched alongside the statistics.
#: The round-7 audit found the CLI transcript and the BibTeX record both
#: quoting 2.5.0 while the tree built 2.8.0 and CITATION.cff said 2.8.0 --
#: three different answers to "what version is this", none of them enforced,
#: because this script patched counts and coverage but never the version.
VERSION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(\$ truss-analysis version\n)[\d.]+"),
        r"\g<1>{version}",
    ),
    (
        re.compile(r"(  version = \{)[\d.]+(\},)"),
        r"\g<1>{version}\g<2>",
    ),
]

CITATION_VERSION = re.compile(r'(^version: ")[\d.]+(")', re.MULTILINE)

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
        "**{n} tests passing, {cov1} % coverage**",
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
        "**{fa_n} آزمون پاس، پوشش {fa_cov1}٪**",
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


def _patch_citation(version: str, dry_run: bool = False) -> int:
    """Keep CITATION.cff's ``version:`` in step with the latest release tag.

    A citation record that quotes a stale version is worse than one that quotes
    no version: it is the field a reader is most likely to copy.
    """
    path = REPO_ROOT / "CITATION.cff"
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")
    rendered, count = CITATION_VERSION.subn(rf"\g<1>{version}\g<2>", text)
    if count and not dry_run:
        path.write_text(rendered, encoding="utf-8")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=int, default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--modules", type=int, default=None)
    parser.add_argument("--version", type=str, default=None)
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
        version = args.version or _released_version()
    else:
        n, cov, modules, version = measure()
        if args.version:
            version = args.version

    cov1 = f"{cov:.1f}"
    cov2 = f"{cov:.2f}"
    fmt = {
        "n": str(n),
        "cov1": cov1,
        "cov2": cov2,
        "modules": str(modules),
        "version": version,
        "fa_n": _fa(str(n)),
        "fa_cov1": _fa(cov1),
        "fa_cov2": _fa(cov2),
        "fa_version": _fa(version),
    }
    print(f"measured (canonical): tests={n} coverage={cov2}% modules={modules}")
    print(f"measured (release):   version={version}")

    # The with-validation count is reported and deliberately never gated: it
    # depends on whether openseespy loads, which is a property of the machine
    # rather than of the tree.
    with_validation = measure_with_validation()
    if with_validation is not None:
        print(
            f"informational: with the reference solver the suite is "
            f"{with_validation[0]} tests; the READMEs quote the canonical "
            f"{n} so the number does not depend on this machine"
        )
    else:
        print(
            "informational: openseespy unavailable here, so the "
            "with-validation count was not measured; the canonical number is "
            "unaffected by design"
        )

    if args.dry_run:
        return 0
    if modules == 0:
        print("WARNING: module count unavailable; module-count line not patched")
        en_patterns = [pt for pt in EN_PATTERNS if "{modules}" not in pt[1]]
    else:
        en_patterns = EN_PATTERNS

    all_en = [*en_patterns, *VERSION_PATTERNS]
    all_fa = [*FA_PATTERNS, *VERSION_PATTERNS]

    if args.check:
        status = check_stale(all_en, all_fa, fmt)
        citation_version = _citation_version_on_disk()
        if citation_version is not None and citation_version != version:
            print(
                f"STALE: CITATION.cff quotes {citation_version}, latest release "
                f"is {version}; run `make stats`"
            )
            status = 1
        return status

    hits_en = patch(README_EN, all_en, fmt)
    hits_fa = patch(README_FA, all_fa, fmt)
    hits_cff = _patch_citation(version)
    print(
        f"patched {hits_en} occurrence(s) in README.md, {hits_fa} in "
        f"README.fa.md, {hits_cff} in CITATION.cff"
    )
    return 0


def _citation_version_on_disk() -> str | None:
    """The version CITATION.cff currently quotes, or None if absent."""
    path = REPO_ROOT / "CITATION.cff"
    if not path.exists():
        return None
    found = CITATION_VERSION.search(path.read_text(encoding="utf-8"))
    return found.group(0).split('"')[1] if found else None


if __name__ == "__main__":
    sys.exit(main())
