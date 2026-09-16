#!/usr/bin/env python3
"""Repository hygiene scanner for public-file cleanliness.

Guards the working branch against four classes of accidental commits:

1. **Internal vocabulary** — project-private terms that must never appear in
   published files. The vocabulary lives in ``scripts/hygiene_terms.yaml``, not
   in this source, for a reason explained below.
2. **Build artifacts** — generated files that must stay out of version control
   (``PKG-INFO``, ``SOURCES.txt``, egg-info contents, ``_version.py``,
   performance baselines, logs, wheels).
3. **File size** — any staged file above 500 KB is rejected unless it is
   explicitly allowlisted.
4. **Forbidden paths** — private workspace directories and secret material.

Why the vocabulary is a data file
---------------------------------
An earlier version of this scanner declared its patterns as concatenated string
fragments — the first half of a word plus the second — so that its own source
would not match its own rules, and then exempted itself a second time through an
allowlist of paths. Both mechanisms existed for one purpose: to keep the checker
from catching itself.

That is not a quality control. It is unreadable (a maintainer cannot tell what
the forbidden vocabulary actually is without mentally reassembling eighteen
fragments), and it is unreviewable (the list cannot be diffed against the policy
it implements, because the policy is not written down anywhere as text). The
round-7 audit called this out as process theatre, and the criticism is correct.

Loading the vocabulary from ``hygiene_terms.yaml`` removes the need for both
evasions: this file contains none of the terms, so it passes its own scan on the
merits, and it carries no self-exemption. The single exemption that remains is
the data file, whose entire purpose is to hold the list.

A missing or malformed vocabulary file is a hard error rather than an empty rule
set. A scanner that silently degrades to checking nothing reports success while
guarding nothing, which is worse than not running.

Usage
-----
::

    python scripts/check_public_hygiene.py file1 file2 ...   # explicit list
    python scripts/check_public_hygiene.py --staged          # git staged files
    python scripts/check_public_hygiene.py --all             # git tracked files

Exit code is 0 when clean and 1 on any violation; every violation is printed
with file, line number and a bilingual (English/Persian) message.

Documented allowlists
---------------------
* ``gini`` — kept as a statistical eponym in code identifiers and in the phrase
  "Gini coefficient"; flagged only in narrative documents (top-level ``*.md``
  and ``docs/*.md``). Held in the scanner rather than the vocabulary file
  because its rule is path-conditional, not a plain match.
* ``bootstrap`` — standard statistical resampling term; not scanned.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VOCABULARY_PATH = Path(__file__).resolve().parent / "hygiene_terms.yaml"
VOCABULARY_SCHEMA = "truss-analysis/hygiene-terms/v2"


@dataclass(frozen=True)
class TermRule:
    """One forbidden-term rule, loaded from the vocabulary file.

    Attributes
    ----------
    rule_id : str
        Stable identifier, reported in violations so a rule can be looked up.
    label : str
        Human-readable description of what the term is.
    pattern : re.Pattern[str]
        Compiled matcher.
    matches : tuple[str, ...]
        Strings the pattern must match.  Declared beside the rule rather than in
        a test file, because this file is the one path exempt from scanning -- so
        the examples can be the real literals instead of paraphrases that happen
        to dodge the very rule they are meant to exercise.
    not_matches : tuple[str, ...]
        Strings the pattern must *not* match.  This is the half that catches a
        rule that is too wide, which is the failure mode that turns a control
        into something contributors work around.
    """

    rule_id: str
    label: str
    pattern: re.Pattern[str]
    matches: tuple[str, ...] = ()
    not_matches: tuple[str, ...] = ()


@dataclass(frozen=True)
class Vocabulary:
    """The loaded term list and its exemptions."""

    schema: str
    rules: tuple[TermRule, ...]
    exempt_paths: frozenset[str]

    def __post_init__(self) -> None:
        """Refuse a vocabulary that would silently check nothing."""
        if self.schema != VOCABULARY_SCHEMA:
            msg = (
                f"hygiene vocabulary schema mismatch: file declares "
                f"{self.schema!r}, scanner expects {VOCABULARY_SCHEMA!r}"
            )
            raise ValueError(msg)
        if not self.rules:
            msg = (
                "hygiene vocabulary declares no terms; a scanner with an empty "
                "rule set reports success while guarding nothing"
            )
            raise ValueError(msg)
        ids = [r.rule_id for r in self.rules]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            msg = f"hygiene vocabulary has duplicate term ids: {duplicates}"
            raise ValueError(msg)


_FLAG_MAP = {"ignorecase": re.IGNORECASE, "multiline": re.MULTILINE}


def _check_examples(rule: TermRule) -> None:
    """Refuse a rule whose declared examples contradict its pattern.

    This makes the vocabulary a *checked specification* rather than a comment. A
    regex that has drifted from what its author thought it matched fails here, at
    load, with the offending string quoted -- instead of silently over- or
    under-matching every file in the repository until someone notices a hook
    firing on ordinary prose.

    The over-broad direction is the one that matters most. ``\\breviewers?\\b``
    was a correct regex for a wrong rule: it matched the noun a contributing guide
    has to use, so the response to it firing was to rewrite prose around the word
    rather than to question the rule. Requiring every rule to ship strings it must
    *not* match is what makes that mistake load-bearing rather than cosmetic.

    Raises
    ------
    ValueError
        If a rule declares no examples at all, or an example on the wrong side.
    """
    if not rule.matches or not rule.not_matches:
        msg = (
            f"hygiene rule {rule.rule_id!r} must declare both `matches` and "
            "`not_matches` examples; a rule nobody has demonstrated firing is a "
            "rule nobody knows works, and a rule nobody has demonstrated staying "
            "quiet is a rule that will be worked around"
        )
        raise ValueError(msg)
    for text in rule.matches:
        if not rule.pattern.search(text):
            msg = (
                f"hygiene rule {rule.rule_id!r} does not match its own declared "
                f"example {text!r}"
            )
            raise ValueError(msg)
    for text in rule.not_matches:
        if rule.pattern.search(text):
            msg = (
                f"hygiene rule {rule.rule_id!r} wrongly matches its declared "
                f"non-example {text!r} -- the pattern is too wide"
            )
            raise ValueError(msg)


def load_vocabulary(path: Path = VOCABULARY_PATH) -> Vocabulary:
    """Read and compile the forbidden-term vocabulary.

    Parameters
    ----------
    path : Path
        Location of the vocabulary file.

    Returns
    -------
    Vocabulary
        Compiled rules plus the exempt-path set.

    Raises
    ------
    FileNotFoundError
        If the file is missing. Not defaulted away: without it the scanner has no
        rules and would pass everything.
    ValueError
        If the file is malformed, declares an unknown schema or an unknown flag,
        or contains a pattern that does not compile.
    """
    if not path.exists():
        msg = (
            f"hygiene vocabulary not found at {path}; the scanner cannot run "
            "without its term list and will not fall back to an empty one"
        )
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"hygiene vocabulary at {path} is not a mapping"
        raise ValueError(msg)
    # Checked before anything else: a file declaring a schema this scanner does
    # not understand may use fields that do not exist yet, so reporting a
    # per-rule complaint about it would be reporting a symptom of the version
    # mismatch rather than the mismatch.
    schema = str(raw.get("schema", ""))
    if schema != VOCABULARY_SCHEMA:
        msg = (
            f"hygiene vocabulary schema mismatch: file declares {schema!r}, "
            f"scanner expects {VOCABULARY_SCHEMA!r}"
        )
        raise ValueError(msg)

    rules: list[TermRule] = []
    for row in raw.get("terms") or ():
        flags = 0
        for name in row.get("flags") or ():
            if name not in _FLAG_MAP:
                msg = f"unknown flag {name!r} on term {row.get('id')!r}"
                raise ValueError(msg)
            flags |= _FLAG_MAP[name]
        try:
            compiled = re.compile(row["pattern"], flags)
        except re.error as exc:
            msg = f"term {row.get('id')!r} has an invalid pattern: {exc}"
            raise ValueError(msg) from exc
        rule = TermRule(
            rule_id=str(row["id"]),
            label=str(row.get("label", row["id"])),
            pattern=compiled,
            matches=tuple(str(x) for x in (row.get("matches") or ())),
            not_matches=tuple(str(x) for x in (row.get("not_matches") or ())),
        )
        _check_examples(rule)
        rules.append(rule)
    return Vocabulary(
        schema=schema,
        rules=tuple(rules),
        exempt_paths=frozenset(str(x) for x in (raw.get("exempt_paths") or ())),
    )


# "Gini" is a documented statistical eponym: flagged only in narrative docs.
# Path-conditional, so it stays here rather than in the flat vocabulary list.
# Written out in full, not assembled from fragments -- it is not in the term list,
# so there is nothing for it to collide with.
GINI_PATTERN = re.compile("gini", re.IGNORECASE)
NARRATIVE_GLOBS = (
    "README*.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "CONTRIBUTING*.md",
    "docs/*.md",
)

ARTIFACT_NAME = re.compile(
    r"(^|/)("
    r"PKG-INFO|SOURCES\.txt|.*\.egg-info.*|_version\.py|"
    r"\.baseline_perf(\..*)?|"
    r"structure\.txt|scm_version\.json|scm_file_list\.json|"
    r".*\.whl|.*\.log|test-report\.html|\.coverage(\..*)?|coverage\.xml|"
    r"dependency_links\.txt|entry_points\.txt|requires\.txt|top_level\.txt|"
    r".*\.py[co]"
    r")$"
)

# Byte-code caches must never be tracked: they are non-reproducible build
# artefacts (round-5 audit: 86 __pycache__ files had crept into the index).
PYCACHE_PATH = re.compile(r"(^|/)__pycache__(/|$)")

FORBIDDEN_PATH = re.compile(
    r"(^|/)(vault|STATE|uploads|PROJECT_DOCUMENTATION)(/|$)|"
    r"^docs/DECISIONS\.md$|\.pem$|(^|/)id_rsa|\.secrets$|credentials"
)

SIZE_LIMIT_BYTES = 500 * 1024
SIZE_ALLOWLIST: frozenset[str] = frozenset()

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

#: Loaded once per process; the scanner is invoked per staged file by pre-commit.
_VOCABULARY = load_vocabulary()


def _active_vocabulary() -> Vocabulary:
    """Return the process vocabulary, read at call time.

    ``scan_file`` must not bind ``_VOCABULARY`` as a default argument.  A default
    is evaluated once, when the ``def`` runs, so the module attribute and the
    parameter would be two different objects -- and a caller (or a test) that
    replaces the module attribute would silently keep scanning against the old
    rules.  That is the same late-binding trap the benchmark driver had, in a
    hook whose entire job is to notice things.
    """
    return _VOCABULARY


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


def scan_file(rel: str, vocabulary: Vocabulary | None = None) -> list[str]:
    """Return a list of violation messages for one repository-relative path."""
    active = _active_vocabulary() if vocabulary is None else vocabulary
    problems: list[str] = []
    if rel in active.exempt_paths:
        return problems

    if ARTIFACT_NAME.search(rel):
        problems.append(
            f"{rel}: build artifact must not be committed / "
            "آرتیفکت بیلد نباید کامیت شود"
        )
    if PYCACHE_PATH.search(rel):
        problems.append(
            f"{rel}: byte-code cache directory must not be committed / "
            "پوشه __pycache__ نباید کامیت شود"
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
        for rule in active.rules:
            if rule.pattern.search(line):
                problems.append(
                    f"{rel}:{lineno}: {rule.label} [{rule.rule_id}] / "
                    "واژگان داخلی پروژه در فایل عمومی"
                )
        if _is_narrative(rel) and GINI_PATTERN.search(line):
            problems.append(
                f"{rel}:{lineno}: eponym reserved for code identifiers "
                f"[{GINI_PATTERN.pattern}] / واژهٔ اختصاصی کد در سند روایی"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the scanner over staged, tracked or explicitly listed files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="explicit file list")
    parser.add_argument("--staged", action="store_true", help="scan git staged files")
    parser.add_argument("--all", action="store_true", help="scan all git tracked files")
    parser.add_argument(
        "--list-rules",
        action="store_true",
        help="print the loaded term ids and exit (the vocabulary is readable now)",
    )
    args = parser.parse_args(argv)

    if args.list_rules:
        for rule in _VOCABULARY.rules:
            print(f"{rule.rule_id}\t{rule.label}")
        return 0

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
    print(
        f"hygiene scan PASSED — {len(targets)} file(s) checked against "
        f"{len(_active_vocabulary().rules)} term rule(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
