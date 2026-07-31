import os
import subprocess
import tomllib
from pathlib import Path


def test_version_sync():
    pyproject_path = Path("pyproject.toml")
    changelog_path = Path("CHANGELOG.md")

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    version = pyproject["project"]["version"]

    with open(changelog_path, encoding="utf-8") as f:
        changelog = f.read()

    assert (
        f"[{version}]" in changelog or f"Version {version}" in changelog
    ), f"Version {version} not in CHANGELOG.md"

    if os.getenv("CI"):
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            tag_version = result.stdout.strip().lstrip("v")
            assert (
                tag_version == version
            ), f"Git tag {tag_version} != pyproject {version}"
