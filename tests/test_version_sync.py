import re
from pathlib import Path


def test_version_sync():
    root = Path(__file__).parent.parent
    toml_content = (root / "pyproject.toml").read_text(encoding="utf-8")
    match_toml = re.search(
        r'^version\s*=\s*["\']([0-9]+\.[0-9]+\.[0-9]+)["\']', toml_content, re.MULTILINE
    )
    assert match_toml, "No semantic version found in pyproject.toml"

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    match_changelog = re.search(r"##\s+\[?v?([0-9]+\.[0-9]+\.[0-9]+)\]?", changelog)
    assert match_changelog, "No semantic version found in CHANGELOG.md"
    assert match_toml.group(1) == match_changelog.group(1), "Version mismatch"
