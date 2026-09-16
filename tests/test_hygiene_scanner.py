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


def _minimal_yaml(*extra: str) -> str:
    """A syntactically valid vocabulary with one well-formed rule."""
    base = (
        f"schema: {hygiene.VOCABULARY_SCHEMA}\n"
        "terms:\n"
        "  - id: probe\n"
        "    label: probe rule\n"
        "    pattern: 'zzz'\n"
        "    matches: ['zzz here']\n"
        "    not_matches: ['clean line']\n"
    )
    return base + "".join(extra)


def test_empty_vocabulary_is_a_hard_error(tmp_path: Path) -> None:
    bad = tmp_path / "terms.yaml"
    bad.write_text(
        f"schema: {hygiene.VOCABULARY_SCHEMA}\nterms: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="empty rule set"):
        hygiene.load_vocabulary(bad)


def test_a_well_formed_vocabulary_loads(tmp_path: Path) -> None:
    """The positive case for the loader, so the error tests are not vacuous."""
    good = tmp_path / "terms.yaml"
    good.write_text(_minimal_yaml(), encoding="utf-8")
    loaded = hygiene.load_vocabulary(good)
    assert len(loaded.rules) == 1
    assert loaded.rules[0].rule_id == "probe"


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
        _minimal_yaml(
            "  - id: probe\n"
            "    pattern: 'yyy'\n"
            "    matches: ['yyy here']\n"
            "    not_matches: ['clean']\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate term ids"):
        hygiene.load_vocabulary(bad)


def test_unknown_flag_and_bad_regex_are_hard_errors(tmp_path: Path) -> None:
    bad = tmp_path / "terms.yaml"
    bad.write_text(
        f"schema: {hygiene.VOCABULARY_SCHEMA}\n"
        "terms:\n"
        "  - id: x\n"
        "    pattern: 'a'\n"
        "    flags: [verbose_please]\n"
        "    matches: ['a']\n"
        "    not_matches: ['b']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown flag"):
        hygiene.load_vocabulary(bad)

    bad.write_text(
        f"schema: {hygiene.VOCABULARY_SCHEMA}\n"
        "terms:\n"
        "  - id: x\n"
        "    pattern: '(unclosed'\n"
        "    matches: ['a']\n"
        "    not_matches: ['b']\n",
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
#
# The example strings live in scripts/hygiene_terms.yaml and are read from there,
# never written into this file. A test of a forbidden-term detector cannot
# contain the forbidden terms: this file is scanned by `--all`, so inlining a real
# positive case would make the repository fail its own hook. The fix for that is
# not to reassemble the terms from fragments -- which is the technique this round
# removed from the scanner -- but to put the examples in the one file whose entire
# purpose is to carry the vocabulary, and have the loader verify them. That is
# also why the rule's `not_matches` half is enforced: it is what catches a pattern
# too wide to obey.


def _rule(rule_id: str) -> hygiene.TermRule:
    """Fetch one rule from the shipped vocabulary by id."""
    loaded = hygiene.load_vocabulary(VOCABULARY_PATH)
    for rule in loaded.rules:
        if rule.rule_id == rule_id:
            return rule
    msg = f"no rule {rule_id!r} in the shipped vocabulary"
    raise AssertionError(msg)


def test_every_shipped_rule_declares_both_halves_of_its_examples() -> None:
    """The loader enforces this; the test pins that it does."""
    loaded = hygiene.load_vocabulary(VOCABULARY_PATH)
    for rule in loaded.rules:
        assert rule.matches, rule.rule_id
        assert rule.not_matches, rule.rule_id


def test_every_shipped_rule_matches_its_declared_examples() -> None:
    """A rule that cannot be demonstrated firing is a rule nobody knows works."""
    for rule in hygiene.load_vocabulary(VOCABULARY_PATH).rules:
        for text in rule.matches:
            assert rule.pattern.search(text), (rule.rule_id, text)


def test_no_shipped_rule_matches_its_declared_non_examples() -> None:
    """The half that catches a rule too wide to obey.

    ``referee_term`` used to be a bare common English noun, which banned the word
    from CONTRIBUTING.md -- a document that has to talk about people reviewing
    pull requests -- and the response was to rewrite prose around it rather than
    to question the rule. Every rule now ships strings it must not match, and
    ``load_vocabulary`` refuses one that violates them.
    """
    for rule in hygiene.load_vocabulary(VOCABULARY_PATH).rules:
        for text in rule.not_matches:
            assert not rule.pattern.search(text), (rule.rule_id, text)


def test_the_narrowed_referee_rule_separates_process_from_prose() -> None:
    """The specific rule the round-7 audit called theatre, checked both ways."""
    rule = _rule("referee_term")
    assert rule.matches, "the rule must ship positive examples"
    assert rule.not_matches, "and negative ones -- that is the whole point"
    # the negatives are ordinary contributing-guide prose; assert their shape
    # rather than their text, so this file never has to contain a positive case
    assert any("pull request" in t for t in rule.not_matches)


def test_a_rule_with_contradictory_examples_will_not_load(tmp_path: Path) -> None:
    """The self-check must be load-bearing, not decorative."""
    bad = tmp_path / "terms.yaml"
    bad.write_text(
        f"schema: {hygiene.VOCABULARY_SCHEMA}\n"
        "terms:\n"
        "  - id: broken\n"
        "    pattern: 'zzz'\n"
        "    matches: ['this does not contain the token']\n"
        "    not_matches: ['fine']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match its own declared example"):
        hygiene.load_vocabulary(bad)


def test_an_over_broad_rule_will_not_load(tmp_path: Path) -> None:
    bad = tmp_path / "terms.yaml"
    bad.write_text(
        f"schema: {hygiene.VOCABULARY_SCHEMA}\n"
        "terms:\n"
        "  - id: toowide\n"
        "    pattern: 'the'\n"
        "    matches: ['the thing']\n"
        "    not_matches: ['also the thing']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="too wide"):
        hygiene.load_vocabulary(bad)


def test_a_rule_without_examples_will_not_load(tmp_path: Path) -> None:
    bad = tmp_path / "terms.yaml"
    bad.write_text(
        f"schema: {hygiene.VOCABULARY_SCHEMA}\n"
        "terms:\n"
        "  - id: noexamples\n"
        "    pattern: 'zzz'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must declare both"):
        hygiene.load_vocabulary(bad)


def test_no_shipped_rule_fires_on_neutral_technical_prose() -> None:
    """A rule that matches ordinary prose gets worked around, not obeyed."""
    neutral = (
        "The solver assembles a global stiffness matrix from pin-jointed "
        "members and reports member forces, displacements and reactions. "
        "Steel properties degrade with temperature per EN 1993-1-2, and a "
        "reviewer may request changes to a pull request."
    )
    for rule in hygiene.load_vocabulary(VOCABULARY_PATH).rules:
        assert not rule.pattern.search(neutral), (
            f"rule {rule.rule_id!r} fires on neutral technical prose: "
            f"{rule.pattern.pattern}"
        )
