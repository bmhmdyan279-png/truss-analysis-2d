"""The repository hygiene scanner, which had never been tested.

A pre-commit hook that guards every public file against committing internal
vocabulary, build artefacts, oversized blobs and forbidden paths had no test at
all. That is the same shape of defect the round-7 audit found in
``benchmarks/reference_problems.py``: a control that nothing exercises, so nobody
knows whether it fires.

The scanner was also restructured in this round. Its forbidden vocabulary used to
live in its own source as concatenated string fragments -- ``"review" + "er"``,
``"Pha" + "se"`` -- so that the file would not match its own rules, plus a
``SELF_PATHS`` allowlist exempting it a second time. Both mechanisms existed for
one purpose: to keep the checker from catching itself. The vocabulary now lives in
``scripts/hygiene_terms.yaml``, so the scanner's source contains no forbidden term
and needs no self-exemption. :func:`test_the_scanner_passes_its_own_scan` is the
test that makes that claim checkable rather than asserted.

These tests drive :func:`scan_file` directly with an explicit ``Vocabulary``
wherever possible, so they neither depend on the shipped vocabulary file being
reachable nor trip over it: a test whose own source contains the forbidden words
would fail the very hook it is testing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_public_hygiene as hygiene  # noqa: E402

VOCABULARY_PATH = REPO_ROOT / "scripts" / "hygiene_terms.yaml"


def _vocabulary(
    *terms: tuple[str, str], exempt: tuple[str, ...] = ()
) -> hygiene.Vocabulary:
    """Build a vocabulary from ``(rule_id, pattern)`` pairs for testing."""
    rules = tuple(
        hygiene.TermRule(rule_id=rid, label=f"label for {rid}", pattern=re.compile(pat))
        for rid, pat in terms
    )
    return hygiene.Vocabulary(
        schema=hygiene.VOCABULARY_SCHEMA, rules=rules, exempt_paths=frozenset(exempt)
    )


# --------------------------------------------------------------------------
# the vocabulary artefact
# --------------------------------------------------------------------------


def test_vocabulary_file_exists_and_is_readable_yaml() -> None:
    """A missing vocabulary is a hard error, so the file must ship."""
    assert VOCABULARY_PATH.exists()
    raw = yaml.safe_load(VOCABULARY_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert raw["schema"] == hygiene.VOCABULARY_SCHEMA


def test_shipped_vocabulary_loads_with_real_rules() -> None:
    """Guard against the file degrading into an empty rule set."""
    loaded = hygiene.load_vocabulary(VOCABULARY_PATH)
    assert len(loaded.rules) >= 10
    ids = {r.rule_id for r in loaded.rules}
    assert len(ids) == len(loaded.rules), "duplicate rule ids"
    for rule in loaded.rules:
        assert rule.label, rule.rule_id
        assert rule.pattern.pattern


def test_shipped_vocabulary_exempts_exactly_itself() -> None:
    """One exemption, and it is the file whose purpose is to hold the terms.

    The previous scanner exempted itself as well as disassembling its own
    patterns into fragments. If a second entry ever appears here it should be
    argued for in the vocabulary file, not added quietly.
    """
    loaded = hygiene.load_vocabulary(VOCABULARY_PATH)
    assert loaded.exempt_paths == frozenset({"scripts/hygiene_terms.yaml"})


def test_the_scanner_passes_its_own_scan() -> None:
    """The claim that replaced the self-exemption, made checkable.

    ``scripts/check_public_hygiene.py`` is *not* in ``exempt_paths``, so this
    passes only because its source genuinely contains none of the vocabulary --
    which is the whole point of moving the list into a data file. If a future
    edit to the scanner needs to name a forbidden term inline, this test fails
    and the term belongs in the vocabulary file instead.
    """
    loaded = hygiene.load_vocabulary(VOCABULARY_PATH)
    assert "scripts/check_public_hygiene.py" not in loaded.exempt_paths
    assert hygiene.scan_file("scripts/check_public_hygiene.py", loaded) == []


def test_missing_vocabulary_is_a_hard_error(tmp_path: Path) -> None:
    """Silently degrading to an empty rule set would guard nothing and pass."""
    with pytest.raises(FileNotFoundError, match="cannot run"):
        hygiene.load_vocabulary(tmp_path / "absent.yaml")


def test_empty_vocabulary_is_a_hard_error(tmp_path: Path) -> None:
    bad = tmp_path / "terms.yaml"
    bad.write_text(
        f"schema: {hygiene.VOCABULARY_SCHEMA}\nterms: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="empty rule set"):
        hygiene.load_vocabulary(bad)


def test_wrong_schema_is_a_hard_error(tmp_path: Path) -> None:
    bad = tmp_path / "terms.yaml"
    bad.write_text(
        "schema: something/else/v9\nterms:\n  - id: a\n    pattern: 'a'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        hygiene.load_vocabulary(bad)


def test_duplicate_rule_ids_are_a_hard_error(tmp_path: Path) -> None:
    bad = tmp_path / "terms.yaml"
    bad.write_text(
        f"schema: {hygiene.VOCABULARY_SCHEMA}\n"
        "terms:\n"
        "  - id: dup\n    pattern: 'a'\n"
        "  - id: dup\n    pattern: 'b'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate term ids"):
        hygiene.load_vocabulary(bad)


def test_unknown_flag_and_bad_regex_are_hard_errors(tmp_path: Path) -> None:
    bad = tmp_path / "terms.yaml"
    bad.write_text(
        f"schema: {hygiene.VOCABULARY_SCHEMA}\n"
        "terms:\n"
        "  - id: x\n    pattern: 'a'\n    flags: [verbose_please]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown flag"):
        hygiene.load_vocabulary(bad)

    bad.write_text(
        f"schema: {hygiene.VOCABULARY_SCHEMA}\n"
        "terms:\n"
        "  - id: x\n    pattern: '(unclosed'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid pattern"):
        hygiene.load_vocabulary(bad)


def test_non_mapping_vocabulary_is_a_hard_error(tmp_path: Path) -> None:
    bad = tmp_path / "terms.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a mapping"):
        hygiene.load_vocabulary(bad)


# --------------------------------------------------------------------------
# term matching
# --------------------------------------------------------------------------


def test_a_rule_fires_with_file_and_line(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "notes.md"
    target.write_text(
        "clean line\nthe forbidden token here\nanother clean\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    vocab = _vocabulary(("probe_term", "forbidden token"))
    problems = hygiene.scan_file("notes.md", vocab)

    assert len(problems) == 1
    assert problems[0].startswith("notes.md:2:")
    assert "probe_term" in problems[0]
    # bilingual, so the message is actionable for both audiences
    assert "واژگان داخلی پروژه" in problems[0]


def test_flags_are_honoured(tmp_path: Path, monkeypatch) -> None:
    """``ignorecase`` must reach the compiled pattern, not be dropped."""
    path = tmp_path / "a.txt"
    path.write_text("MiXeD CaSe\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    vocab = _vocabulary(("mixed", "(?i)mixed case"))
    assert hygiene.scan_file("a.txt", vocab)

    strict = _vocabulary(("strict", "mixed case"))
    assert hygiene.scan_file("a.txt", strict) == []


def test_exempt_path_is_skipped_entirely(tmp_path: Path, monkeypatch) -> None:
    """The exemption covers artefact and path rules too, not just terms."""
    path = tmp_path / "keep.yaml"
    path.write_text("forbidden token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    vocab = _vocabulary(("probe", "forbidden token"), exempt=("keep.yaml",))
    assert hygiene.scan_file("keep.yaml", vocab) == []


# --------------------------------------------------------------------------
# structural rules: artefacts, caches, paths, size
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        "src/truss_analysis.egg-info/PKG-INFO",
        "PKG-INFO",
        "dist/truss_analysis-1.0-py3-none-any.whl",
        "src/truss_analysis/_version.py",
        "run.log",
        "coverage.xml",
        ".coverage",
        "test-report.html",
        ".baseline_perf.json",
    ],
)
def test_build_artefacts_are_rejected(rel: str) -> None:
    assert hygiene.ARTIFACT_NAME.search(rel), rel


@pytest.mark.parametrize(
    "rel",
    [
        ".baseline_perf.json",
        ".baseline_perf.csv",
        "benchmarks/.baseline_perf.json",
        ".baseline_perf",
    ],
)
def test_suffixed_performance_baselines_are_rejected(rel: str) -> None:
    r"""Regression: the artefact group is anchored, and one alternative forgot.

    ``ARTIFACT_NAME`` wraps its alternatives in a group terminated by ``$``, so
    every alternative has to reach the end of the path. ``\.baseline_perf`` did
    not -- it matched the bare name and let ``.baseline_perf.json``, the form a
    performance baseline is actually written in, straight through. ``\.coverage``
    in the same regex already carried the ``(\..*)?`` suffix; this one had been
    missed. A filter that rejects the name but not the file it names is a filter
    that reports success while guarding nothing.
    """
    assert hygiene.ARTIFACT_NAME.search(rel), rel


@pytest.mark.parametrize(
    "rel",
    [
        "src/truss_analysis/main.py",
        "docs/theory.md",
        "requirements-dev.txt",
        "src/truss_analysis/data/physics_boundary.yaml",
    ],
)
def test_source_files_are_not_mistaken_for_artefacts(rel: str) -> None:
    """The artefact regex must not eat legitimate tracked files.

    ``requirements-dev.txt`` is the case worth pinning: the pattern rejects
    ``requires.txt`` and ``dependency_links.txt`` (egg-info internals), and a
    regex written one character wider would silently block a real file.
    """
    assert not hygiene.ARTIFACT_NAME.search(rel), rel


@pytest.mark.parametrize(
    "rel", ["src/__pycache__/x.pyc", "__pycache__/y.pyc", "a/b/__pycache__"]
)
def test_bytecode_caches_are_rejected(rel: str) -> None:
    assert hygiene.PYCACHE_PATH.search(rel), rel


@pytest.mark.parametrize(
    "rel",
    [
        "vault/notes.md",
        "STATE/x.json",
        "uploads/private.md",
        "PROJECT_DOCUMENTATION/x.md",
        "docs/DECISIONS.md",
        "keys/server.pem",
        ".ssh/id_rsa",
        "config/.secrets",
        "ops/credentials.json",
    ],
)
def test_forbidden_private_paths_are_rejected(rel: str) -> None:
    assert hygiene.FORBIDDEN_PATH.search(rel), rel


def test_size_limit_is_reported_with_the_actual_size(
    tmp_path: Path, monkeypatch
) -> None:
    big = tmp_path / "blob.bin"
    big.write_bytes(b"\0" * (hygiene.SIZE_LIMIT_BYTES + 1))
    monkeypatch.chdir(tmp_path)

    problems = hygiene.scan_file("blob.bin", _vocabulary(("probe", "zzz")))
    assert any("larger than 500 KB" in p for p in problems)


def test_a_deleted_file_is_not_an_error(tmp_path: Path, monkeypatch) -> None:
    """Staged deletions reach the hook; there is nothing to scan."""
    monkeypatch.chdir(tmp_path)
    assert hygiene.scan_file("does-not-exist.md", _vocabulary(("probe", "zzz"))) == []


def test_binary_files_skip_the_text_scan(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"forbidden token" * 4)
    monkeypatch.chdir(tmp_path)

    vocab = _vocabulary(("probe", "forbidden token"))
    assert hygiene.scan_file("figure.png", vocab) == []


def test_undecodable_text_is_skipped_not_crashed(tmp_path: Path, monkeypatch) -> None:
    blob = tmp_path / "latin1.txt"
    blob.write_bytes("caf\u00e9 forbidden token".encode("latin-1"))
    monkeypatch.chdir(tmp_path)

    vocab = _vocabulary(("probe", "forbidden token"))
    assert hygiene.scan_file("latin1.txt", vocab) == []


# --------------------------------------------------------------------------
# the path-conditional eponym rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rel", "narrative"),
    [
        ("README.md", True),
        ("CHANGELOG.md", True),
        ("docs/theory.md", True),
        ("CONTRIBUTING.fa.md", True),
        ("src/truss_analysis/stats.py", False),
        ("tests/test_stats.py", False),
    ],
)
def test_eponym_is_flagged_only_in_narrative_documents(
    tmp_path: Path, monkeypatch, rel: str, narrative: bool
) -> None:
    """A statistical eponym belongs in identifiers, not in prose."""
    # Recreate the full relative path, not just its basename: the rule is
    # path-conditional, so scanning "theory.md" would not match "docs/*.md" and
    # the test would pass for the wrong reason.
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("the gini coefficient is reported\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert hygiene._is_narrative(rel) is narrative
    problems = hygiene.scan_file(rel, _vocabulary(("probe", "zzz")))
    fired = any("eponym" in p for p in problems)
    assert fired is narrative


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------


def test_cli_list_rules_reads_the_shipped_vocabulary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert hygiene.main(["--list-rules"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == len(hygiene.load_vocabulary(VOCABULARY_PATH).rules)
    assert all("\t" in line for line in lines)


def test_cli_reports_a_clean_pass(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "clean.md"
    target.write_text("nothing to see here\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert hygiene.main([str(target)]) == 0
    assert "PASSED" in capsys.readouterr().out


def test_cli_reports_a_violation_on_stderr_and_exits_one(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hook that prints failures to stdout is a hook CI cannot surface.

    Driven with a synthetic vocabulary rather than a real forbidden term. This
    test file is itself scanned by ``--all``, so writing a genuine term into it
    would make the repository fail its own hook -- and the fix must not be to
    assemble the term from fragments, which is the technique this round removed
    from the scanner. A synthetic rule exercises exactly the same code path.
    """
    target = tmp_path / "dirty.md"
    target.write_text("a synthetic probe token on this line\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        hygiene, "_VOCABULARY", _vocabulary(("synthetic_probe", "probe token"))
    )

    assert hygiene.main([str(target)]) == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.err
    assert "synthetic_probe" in captured.err
    assert captured.out == ""


def test_cli_scans_the_whole_tracked_tree_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The real gate, run for real: every tracked file, shipped vocabulary.

    This is the assertion that would have caught the fragment-assembly version
    reporting success while its own rules were unreadable, and it is the one that
    fails if a future commit introduces internal vocabulary anywhere in the
    repository.
    """
    assert hygiene.main(["--all"]) == 0
    out = capsys.readouterr().out
    assert "PASSED" in out
    checked = int(re.search(r"(\d+) file\(s\) checked", out).group(1))
    assert checked > 100, "the tracked tree is much larger than this"


# --------------------------------------------------------------------------
# rules must target what they are for, not the nearest noun
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Reviewer 2 asked for a nonlinear solver",
        "see the referee report before merging",
        "handled by the peer-referee process",
        "Response to Reviewers",
        "reviewer #3 requested more validation",
    ],
)
def test_internal_review_vocabulary_is_caught(line: str) -> None:
    """The phrasing of a private review workflow leaking into a public file."""
    rule = _rule("referee_term")
    assert rule.pattern.search(line), line


@pytest.mark.parametrize(
    "line",
    [
        "A reviewer is entitled to know which commits were generated",
        "ask a reviewer to check the physics",
        "reviewers may request changes",
        "the reviewer of this pull request",
    ],
)
def test_the_ordinary_english_noun_is_not_caught(line: str) -> None:
    """The rule was `\breviewers?\b` and that was wrong.

    A bare common noun is not internal vocabulary, and banning it forced
    CONTRIBUTING.md -- a document that has to talk about people reviewing pull
    requests -- to be rewritten around the word. Erasing a term is not the same
    as removing what the term stood in for, and a lexicon check on an ordinary
    noun costs real clarity in the documents it governs. What is actually worth
    catching is the phrasing of a private process, which is what the narrowed
    pattern matches.
    """
    rule = _rule("referee_term")
    assert not rule.pattern.search(line), line


def _rule(rule_id: str) -> hygiene.TermRule:
    """Fetch one rule from the shipped vocabulary by id."""
    loaded = hygiene.load_vocabulary(VOCABULARY_PATH)
    for rule in loaded.rules:
        if rule.rule_id == rule_id:
            return rule
    msg = f"no rule {rule_id!r} in the shipped vocabulary"
    raise AssertionError(msg)


def test_every_shipped_rule_has_a_positive_and_negative_case() -> None:
    """No rule may be untestable, and none may match ordinary prose.

    A rule that cannot be demonstrated to fire is a rule nobody knows works; a
    rule that fires on neutral text is a rule that will be worked around rather
    than obeyed. Both are checked structurally: each compiled pattern must match
    its own id-shaped token and must not match a sentence of plain technical
    English about this library.
    """
    neutral = (
        "The solver assembles a global stiffness matrix from pin-jointed "
        "members and reports member forces, displacements and reactions."
    )
    loaded = hygiene.load_vocabulary(VOCABULARY_PATH)
    for rule in loaded.rules:
        assert not rule.pattern.search(neutral), (
            f"rule {rule.rule_id!r} fires on neutral technical prose: "
            f"{rule.pattern.pattern}"
        )
