"""Packaging hygiene tests: metadata completeness and dependency sync.

These tests guard the packaging decisions that keep the published artifact
lean and the repository consistent:

* ``pyproject.toml`` is the single source of truth for dependencies and
  ``requirements*.txt`` are generated mirrors (checked via
  ``scripts/sync_requirements.py --check``);
* project metadata (urls, classifiers, scripts, extras) is complete;
* ``CITATION.cff`` and ``MANIFEST.in`` exist and stay lean;
* the version fallback for git-less source exports is configured.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Python 3.10 compatibility: tomllib is available from 3.11 onward.
try:
    import tomllib
except ImportError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def test_requirements_files_in_sync() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "sync_requirements.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_project_urls_present() -> None:
    urls = _pyproject()["project"]["urls"]
    for key in ("Repository", "Documentation", "Issues", "Changelog"):
        assert key in urls, key
        assert urls[key].startswith("https://")


def test_classifiers_complete() -> None:
    classifiers = _pyproject()["project"]["classifiers"]
    joined = "\n".join(classifiers)
    assert "License :: OSI Approved :: MIT License" in joined
    for ver in ("3.10", "3.11", "3.12"):
        assert f"Programming Language :: Python :: {ver}" in joined
    assert "Topic :: Scientific/Engineering" in joined
    assert "Typing :: Typed" in joined


def test_extras_declared() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]
    assert set(extras) >= {"dev", "viz", "validation", "retrofit"}
    assert any("openseespy" in dep for dep in extras["validation"])
    assert any("matplotlib" in dep for dep in extras["viz"])


def test_console_script_maps_to_main() -> None:
    scripts = _pyproject()["project"]["scripts"]
    assert scripts["truss-analysis"] == "truss_analysis.main:main"


def test_setuptools_scm_fallback_configured() -> None:
    scm = _pyproject()["tool"]["setuptools_scm"]
    assert scm["write_to"] == "src/truss_analysis/_version.py"
    assert scm["fallback_version"]  # non-empty fallback for git-less exports
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "src/truss_analysis/_version.py" in gitignore


def test_citation_cff_is_lean() -> None:
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert citation.startswith("# Citation metadata")
    assert "cff-version: 1.2.0" in citation
    assert "truss_analysis" in citation
    # software citation only: no article title, no DOI placeholder noise
    for banned in ("doi:", "preferred-citation", "journal"):
        assert banned not in citation.lower()


def test_manifest_prunes_development_content() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for pruned in ("prune .github", "prune tests", "prune examples"):
        assert pruned in manifest
    assert "exclude Dockerfile" in manifest


def test_mypy_strict_configured() -> None:
    mypy_cfg = _pyproject()["tool"]["mypy"]
    assert mypy_cfg.get("strict") is True


def test_ruff_rules_extended() -> None:
    ruff_cfg = _pyproject()["tool"]["ruff"]["lint"]
    selected = set(ruff_cfg["select"])
    assert selected >= {"E", "F", "I", "W", "UP", "B", "SIM", "RUF", "PT", "N", "D"}


def test_no_bytecode_tracked_in_git() -> None:
    """Byte-code caches must never be in the git index (round-5 audit).

    86 ``__pycache__/*.pyc`` files had crept into the tracked tree; the
    hygiene scanner now rejects them, and this test pins the index itself
    so a re-introduction fails the suite rather than a manual review.
    """
    import shutil

    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        import pytest

        pytest.skip("git metadata not available")
    proc = subprocess.run(
        [git, "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = proc.stdout.splitlines()
    offenders = [
        path
        for path in tracked
        if "__pycache__" in path or path.endswith((".pyc", ".pyo"))
    ]
    assert not offenders, f"tracked byte-code artifacts: {offenders[:5]}"
