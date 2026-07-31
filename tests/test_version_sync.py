import os
import re
import subprocess
import tomllib


def test_version_sync():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    pyproject_version = (
        data.get("project", {}).get("version")
        or data.get("version")
        or data.get("tool", {}).get("poetry", {}).get("version")
    )

    with open("CHANGELOG.md", encoding="utf-8") as f:
        changelog = f.read()
    match = re.search(r"##\s*\[?([\d.]+)\]?", changelog)
    assert match, "No version found in CHANGELOG.md"
    changelog_version = match.group(1)

    assert (
        pyproject_version == changelog_version
    ), f"Version mismatch: pyproject ({pyproject_version}) != CHANGELOG ({changelog_version})"

    if os.getenv("CI"):
        res = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            git_tag = res.stdout.strip().lstrip("v")
            assert (
                pyproject_version == git_tag
            ), f"Version mismatch: pyproject ({pyproject_version}) != Git Tag ({git_tag})"
