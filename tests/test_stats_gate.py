"""The README statistics gate must measure one canonical thing.

The round-7 audit found the gate was environment-dependent: ``measure()`` probed
whether ``openseespy`` imported and passed the three reference-solver
``--ignore`` flags only when it did not.  The intention was to measure the full
suite where the reference solver really loads; the effect was that HEAD
advertised 820 tests / 94.2% measured on a machine without ``openseespy`` while
earlier commits on the same tree advertised 838 / 94.19% measured on one with
it, so ``make stats-check`` was red on one machine and green on another for the
same commit.

These tests pin the fix: the canonical configuration is unconditional, it is the
same configuration CI runs, and the release version is patched alongside the
counts -- which it previously was not, leaving the CLI transcript at 2.5.0, the
BibTeX record at 2.5.0 and ``CITATION.cff`` at 2.8.0 in one tree.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import update_readme_stats as stats

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# the canonical configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("available", [True, False])
def test_canonical_ignores_do_not_depend_on_the_machine(
    monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    """The whole fix: one configuration, whatever this machine can import."""
    monkeypatch.setattr(stats, "_reference_solver_available", lambda: available)
    assert stats._canonical_ignores() == [
        f"--ignore={path}" for path in stats.REFERENCE_SOLVER_FILES
    ]
    assert len(stats.REFERENCE_SOLVER_FILES) == 3


def test_canonical_ignores_match_what_ci_runs() -> None:
    """Drift guard: parse the workflow rather than trusting a comment.

    The ``test`` and ``coverage`` jobs both pass three ``--ignore`` flags.  If
    those ever change and this script does not follow, the READMEs would quote a
    suite no CI runner executes -- which is a subtler version of the same defect.
    """
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    ci_ignores = sorted(set(re.findall(r"--ignore=([\w/.]+\.py)", workflow)))
    assert ci_ignores, "CI no longer passes --ignore flags; update this test"
    assert ci_ignores == sorted(stats.REFERENCE_SOLVER_FILES)


def test_the_ignored_files_exist() -> None:
    """Ignoring a path that does not exist would silently pass."""
    for rel in stats.REFERENCE_SOLVER_FILES:
        assert (REPO_ROOT / rel).exists(), rel


def test_the_ignored_files_are_run_by_a_mandatory_job() -> None:
    """Ignoring a file from the *statistics* must not mean ignoring it forever.

    The canonical configuration excludes the three reference-solver files so
    the published test count is the same number on every machine.  That is the
    right reason to exclude them from ``make stats`` and the wrong reason to
    exclude them from CI: ``physics_boundary`` ships a digest of the
    cross-validation evidence inside ``solver_metadata``, so a claim was
    travelling in the payload that no gate checked.  The ``reference-solver``
    job is that gate.

    Parsed from the workflow rather than trusted from a comment, for the same
    reason as the test above: a job that is quietly deleted, renamed, or given
    an ``if:`` condition must fail here instead of turning the verification
    column of the matrix back into an unenforced claim.
    """
    import yaml

    workflow_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    assert "reference-solver" in jobs, (
        "the mandatory reference-solver job is gone; the three files the stats "
        "gate ignores would then be run by nothing"
    )
    job = jobs["reference-solver"]
    # unconditional: an `if:` here would make the gate advisory again
    assert "if" not in job, "the reference-solver job must not be conditional"

    commands = " ".join(str(step.get("run", "")) for step in job["steps"])
    for rel in stats.REFERENCE_SOLVER_FILES:
        assert rel in commands, f"{rel} is not run by the reference-solver job"
    # and it must not re-introduce the ignores it exists to undo
    assert "--ignore" not in commands, (
        "the reference-solver job must run the bridge, not ignore it"
    )
    # the extra has to be installed, or every test importorskips to green
    assert "validation" in commands, "the job must install the validation extra"
    assert "import openseespy" in commands, (
        "the job must assert the reference solver imported before running the "
        "suite: an importorskip that skips is green"
    )


# --------------------------------------------------------------------------
# the release version
# --------------------------------------------------------------------------


def test_released_version_comes_from_the_latest_tag() -> None:
    version = stats._released_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version

    tags = subprocess.run(
        ["git", "tag", "--list", "v*"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tags, "the repository has no release tags"
    newest = max(tags, key=lambda t: [int(x) for x in t.removeprefix("v").split(".")])
    assert version == newest.removeprefix("v")


def test_released_version_falls_back_when_there_are_no_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source export with no git metadata must still produce a version."""

    def _no_tags(cmd, **kwargs):
        class _Result:
            returncode = 128
            stdout = ""

        return _Result()

    monkeypatch.setattr(stats.subprocess, "run", _no_tags)
    fallback = stats._released_version()
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'fallback_version\s*=\s*"([^"]+)"', pyproject)
    assert declared is not None
    assert fallback == declared.group(1)


@pytest.mark.parametrize("path", ["CITATION.cff"])
def test_citation_quotes_the_released_version(path: str) -> None:
    """The audit found CITATION.cff at 2.8.0 while the READMEs said 2.5.0."""
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    found = stats.CITATION_VERSION.search(text)
    assert found is not None, f"{path} has no machine-readable version field"
    assert found.group(0).split('"')[1] == stats._released_version()


@pytest.mark.parametrize("readme", ["README.md", "README.fa.md"])
def test_readmes_quote_the_released_version(readme: str) -> None:
    """Both the CLI transcript and the BibTeX record, in both languages."""
    version = stats._released_version()
    text = (REPO_ROOT / readme).read_text(encoding="utf-8")

    transcript = re.search(r"\$ truss-analysis version\n([\d.]+)", text)
    assert transcript is not None, "the CLI transcript block changed shape"
    assert transcript.group(1) == version

    bibtex = re.search(r"  version = \{([\d.]+)\},", text)
    assert bibtex is not None, "the BibTeX block changed shape"
    assert bibtex.group(1) == version


def test_version_patterns_actually_match_the_readmes() -> None:
    """A pattern that silently stops matching is a gate that silently stops gating."""
    for readme in ("README.md", "README.fa.md"):
        text = (REPO_ROOT / readme).read_text(encoding="utf-8")
        for pattern, _template in stats.VERSION_PATTERNS:
            assert pattern.search(text), (
                f"{pattern.pattern!r} no longer matches {readme}"
            )


# --------------------------------------------------------------------------
# rendering is pure, so --check can compare without writing
# --------------------------------------------------------------------------


def test_render_is_pure_and_reports_missing_patterns() -> None:
    text = "badge/coverage-91.0%25-brightgreen and 100 tests"
    fmt = {
        "n": "964",
        "cov1": "94.4",
        "cov2": "94.42",
        "modules": "49",
        "version": "2.8.0",
        "fa_n": "۹۶۴",
        "fa_cov1": "۹۴٫۴",
        "fa_cov2": "۹۴٫۴۲",
        "fa_version": "۲٫۸٫۰",
    }
    patterns = [
        (
            re.compile(r"badge/coverage-[\d.]+%25-brightgreen"),
            "badge/coverage-{cov1}%25-brightgreen",
        ),
        (re.compile(r"(\d+) tests"), "{n} tests"),
        (re.compile(r"a pattern that is not present"), "{n}"),
    ]
    rendered, hits, missing = stats.render(text, "t.md", patterns, fmt)

    assert "coverage-91.0" in text, "render must not mutate its input"
    assert "coverage-94.4%25" in rendered
    assert "964 tests" in rendered
    assert hits == 2
    assert len(missing) == 1


def test_patch_writes_what_render_computed(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("$ truss-analysis version\n0.0.1\n", encoding="utf-8")
    hits = stats.patch(
        target, stats.VERSION_PATTERNS, {"version": "2.8.0", "fa_version": "۲٫۸٫۰"}
    )
    assert hits == 1
    assert "2.8.0" in target.read_text(encoding="utf-8")


def test_check_stale_detects_a_wrong_number(tmp_path: Path, monkeypatch) -> None:
    """The gate must fail when the committed text disagrees with the measurement."""
    stale = tmp_path / "README.md"
    stale.write_text("$ truss-analysis version\n0.0.1\n", encoding="utf-8")
    fresh = tmp_path / "README.fa.md"
    fresh.write_text("$ truss-analysis version\n2.8.0\n", encoding="utf-8")

    monkeypatch.setattr(stats, "README_EN", stale)
    monkeypatch.setattr(stats, "README_FA", fresh)
    fmt = {"version": "2.8.0", "fa_version": "۲٫۸٫۰"}

    assert stats.check_stale(stats.VERSION_PATTERNS, stats.VERSION_PATTERNS, fmt) == 1


def test_the_committed_tree_passes_its_own_gate_without_remeasuring() -> None:
    """``--check`` with the measured values supplied must be a no-op.

    Running the real measurement here would take two minutes and duplicate the
    CI ``stats`` job.  Supplying the numbers the script would have measured
    exercises the same comparison path -- ``check_stale`` renders in memory and
    diffs -- which is the part that can silently stop working.
    """
    version = stats._released_version()
    fmt = {
        "n": re.search(
            r"test suite of (\d+) tests",
            stats.README_EN.read_text(encoding="utf-8"),
        ).group(1),
        "cov1": re.search(
            r"\(([\d.]+) % coverage", stats.README_EN.read_text(encoding="utf-8")
        ).group(1),
        "cov2": "0.0",
        "modules": "49",
        "version": version,
        "fa_n": "۰",
        "fa_cov1": "۰",
        "fa_cov2": "۰",
        "fa_version": version,
    }
    # the version patterns are language-independent, so check those exactly
    assert stats.check_stale(stats.VERSION_PATTERNS, stats.VERSION_PATTERNS, fmt) == 0
