#!/usr/bin/env python3
"""Regenerate requirements*.txt from pyproject.toml.

pyproject.toml is the single source of truth for dependencies. The two
requirements files are convenience mirrors for users who prefer
``pip install -r requirements.txt``; running this script after any dependency
change keeps them byte-identical to what ``tests/test_packaging.py`` expects.

Usage::

    python scripts/sync_requirements.py [--check]

With ``--check`` the script exits non-zero when the files are out of date
(without writing), which is what the packaging test and CI rely on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Python 3.10 compatibility: fallback to tomli if tomllib is not available
try:
    import tomllib
except ImportError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent

HEADER = (
    "# AUTO-GENERATED from pyproject.toml - do not edit by hand.\n"
    "# Single source of truth for dependencies: pyproject.toml\n"
    "# Regenerate with: python scripts/sync_requirements.py\n"
)


def render() -> dict[str, str]:
    """Return {filename: contents} derived from pyproject.toml."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    project = pyproject["project"]
    core: list[str] = project["dependencies"]
    extras: dict[str, list[str]] = project["optional-dependencies"]

    runtime = HEADER + "\n".join(core) + "\n"
    dev_stack = extras["dev"] + extras["viz"] + extras["validation"]
    development = (
        HEADER
        + "-r requirements.txt\n\n"
        + "# dev + viz + validation extras "
        + "(mirrors: pip install -e '.[dev,viz,validation]')\n"
        + "\n".join(dev_stack)
        + "\n"
    )
    return {"requirements.txt": runtime, "requirements-dev.txt": development}


def main(argv: list[str] | None = None) -> int:
    """Entry point: write (default) or check (--check) the mirror files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the requirements files are out of date",
    )
    args = parser.parse_args(argv)

    expected = render()
    stale: list[str] = []
    for name, contents in expected.items():
        path = REPO_ROOT / name
        current = path.read_text("utf-8") if path.exists() else None
        if current != contents:
            stale.append(name)

    if args.check:
        if stale:
            print(
                "requirements files out of date vs pyproject.toml: "
                + ", ".join(stale)
                + " (run: python scripts/sync_requirements.py)",
                file=sys.stderr,
            )
            return 1
        print("requirements files in sync with pyproject.toml")
        return 0

    for name in stale:
        (REPO_ROOT / name).write_text(expected[name], encoding="utf-8")
    print(
        "regenerated from pyproject.toml: "
        + (", ".join(stale) if stale else "(already up to date)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
