#!/usr/bin/env python3
"""Repository hygiene scanner for public-file cleanliness.

Guards the working branch against four classes of accidental commits:

1. **Narrative keywords** — internal project vocabulary that must never
   appear in public files (pattern list assembled from fragments so this
   scanner itself does not trip the scan).
2. **Build artifacts** — generated files that must stay out of version
   control (``PKG-INFO``, ``SOURCES.txt``, egg-info contents, ``_version.py``,
   performance baselines, logs, wheels).
3. **File size** — any staged file above 500 KB is rejected unless it is
   explicitly allowlisted.
4. **Forbidden paths** — private workspace directories and secret material.

Usage
-----
::

    python scripts/check_public_hygiene.py file1 file2 ...   # explicit list
    python scripts/check_public_hygiene.py --staged          # git staged files
    python scripts/check_public_hygiene.py --all             # git tracked files

Exit code is 0 when clean and 1 on any violation; every violation is
printed with file, line number and a bilingual (English/Persian) message.

Documented allowlists
---------------------
* ``gini`` — kept as a statistical eponym in code identifiers and in the
  phrase "Gini coefficient"; flagged only in narrative documents
  (top-level ``*.md`` and ``docs/*.md``).
* ``bootstrap`` — standard statistical resampling term; not scanned.
* This scanner file itself (patterns would otherwise match themselves).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Rule tables (fragments keep this file self-clean)
# --------------------------------------------------------------------------

_P = "Pha" + "se"
_H = "H" + "[1-4]"
_L = "Le" + "mma"
_UB = "uniform" + "_beta"
_G = "Gi" + "ni"
_KS = "kill" + ".?shot"
_MS = "manu" + "script"
_RV = "review" + "er"
_CL = "CONTEXT" + "_LOCK"
_N25 = "Nature" + "_2025"
_PC = "paper" + "_case"
_CIR = "CI" + "_results"
_SS = "SS" + "OT"
_DEC = "D" + "-0" + r"\d\d"
_DRL = r"D[RL]" + "-0" + r"\d\d"
_PR = "prompt" + r"-0*\d"

KEYWORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("internal project phase label", re.compile(_P + r"[ _-]?\d", re.IGNORECASE)),
    ("internal hypothesis label", re.compile(r"\b" + _H + r"\b")),
    ("internal proposition label", re.compile(_L, re.IGNORECASE)),
    ("internal example name", re.compile(_UB, re.IGNORECASE)),
    ("informal attack phrase", re.compile(_KS, re.IGNORECASE)),
    ("internal decision id", re.compile(r"\b" + _DEC + r"\d?\b")),
    ("internal log id", re.compile(r"\b" + _DRL + r"\d?\b")),
    ("internal context file", re.compile(_CL)),
    ("internal prompt reference", re.compile(_PR, re.IGNORECASE)),
    ("publication venue marker", re.compile(_N25)),
    ("internal case name", re.compile(_PC, re.IGNORECASE)),
    ("internal artifact name", re.compile(_CIR)),
    ("draft-document term", re.compile(_MS, re.IGNORECASE)),
    ("peer-referee term", re.compile(r"\b" + _RV + r"s?\b", re.IGNORECASE)),
    ("internal acronym", re.compile(r"\b" + _SS + r"\b")),
    ("Persian proposal term", re.compile("پرو" + "پوزال")),
    ("Persian paper term", re.compile("مقا" + "له")),
    ("Persian hypothesis term", re.compile("فرضی" + "ه")),
)

# "Gini" is a documented statistical eponym: flagged only in narrative docs.
GINI_PATTERN = re.compile(_G, re.IGNORECASE)
NARRATIVE_GLOBS = (
    "README*.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "CONTRIBUTING*.md",
    "docs/*.md",
)

ARTIFACT_NAME = re.compile(
    r"(^|/)("
    r"PKG-INFO|SOURCES\.txt|.*\.egg-info.*|_version\.py|\.baseline_perf|"
    r"structure\.txt|scm_version\.json|scm_file_list\.json|"
    r".*\.whl|.*\.log|test-report\.html|\.coverage(\..*)?|coverage\.xml|"
    r"dependency_links\.txt|entry_points\.txt|requires\.txt|top_level\.txt"
    r")$"
)

FORBIDDEN_PATH = re.compile(
    r"(^|/)(vault|STATE|uploads|PROJECT_DOCUMENTATION)(/|$)|"
    r"^docs/DECISIONS\.md$|\.pem$|(^|/)id_rsa|\.secrets$|credentials"
)

SIZE_LIMIT_BYTES = 500 * 1024
SIZE_ALLOWLIST: frozenset[str] = frozenset()

SELF_PATHS = frozenset({"scripts/check_public_hygiene.py"})

BINARY_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".pdf",
        ".ttf",
        ".woff",
        ".woff2",
        ".whl",
        ".gz",
        ".zip",
        ".bundle",
    }
)


def _git_files(flag: str) -> list[str]:
    cmd = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
        if flag == "--staged"
        else ["git", "ls-files"]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def _is_narrative(rel: str) -> bool:
    return any(Path(rel).match(g) for g in NARRATIVE_GLOBS)


def scan_file(rel: str) -> list[str]:
    """Return a list of violation messages for one repository-relative path."""
    problems: list[str] = []
    if rel in SELF_PATHS:
        return problems

    if ARTIFACT_NAME.search(rel):
        problems.append(
            f"{rel}: build artifact must not be committed / "
            "آرتیفکت بیلد نباید کامیت شود"
        )
    if FORBIDDEN_PATH.search(rel):
        problems.append(f"{rel}: forbidden private path / مسیر خصوصی ممنوع است")

    path = Path(rel)
    if not path.exists():
        return problems  # deleted in this commit — nothing to scan

    if path.stat().st_size > SIZE_LIMIT_BYTES and rel not in SIZE_ALLOWLIST:
        problems.append(
            f"{rel}: file larger than 500 KB "
            f"({path.stat().st_size} bytes) / فایل بزرگ‌تر از ۵۰۰ کیلوبایت"
        )

    if path.suffix.lower() in BINARY_SUFFIXES:
        return problems  # keyword scan is text-only

    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return problems

    for lineno, line in enumerate(text.splitlines(), start=1):
        for label, pattern in KEYWORD_PATTERNS:
            if pattern.search(line):
                problems.append(
                    f"{rel}:{lineno}: {label} [{pattern.pattern}] / "
                    "واژگان داخلی پروژه در فایل عمومی"
                )
        if _is_narrative(rel) and GINI_PATTERN.search(line):
            problems.append(
                f"{rel}:{lineno}: eponym reserved for code identifiers "
                f"[{_G}] / واژهٔ اختصاصی کد در سند روایی"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the scanner over staged, tracked or explicitly listed files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="explicit file list")
    parser.add_argument("--staged", action="store_true", help="scan git staged files")
    parser.add_argument("--all", action="store_true", help="scan all git tracked files")
    args = parser.parse_args(argv)

    if args.staged:
        targets = _git_files("--staged")
    elif args.all:
        targets = _git_files("--all")
    else:
        targets = list(args.files)

    violations: list[str] = []
    for rel in targets:
        violations.extend(scan_file(rel))

    if violations:
        print(
            f"hygiene scan FAILED — {len(violations)} violation(s) / "
            f"اسکن بهداشت ناموفق — {len(violations)} تخلف",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    print(f"hygiene scan PASSED — {len(targets)} file(s) checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
