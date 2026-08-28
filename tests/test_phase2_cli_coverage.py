"""Phase 2 CLI coverage tests — updated for unified CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from truss_analysis.main import main


def test_cli_coverage_execution():
    """Run all example JSONs through the unified CLI to exercise main.py paths."""
    examples = list(Path("examples").glob("*.json"))
    if not examples:
        pytest.skip("No example JSON files found")
    for ex in examples:
        # Call main() directly so coverage tracks the code
        with patch.object(sys, "argv", ["truss-analysis", str(ex)]):
            result = main()
            assert result == 0, f"CLI failed on {ex}"


def test_cli_help_exits_cleanly():
    """Verify --help flag does not crash."""
    with patch.object(sys, "argv", ["truss-analysis", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


def test_cli_missing_input_errors():
    """Verify missing input produces a FileNotFoundError."""
    with patch.object(sys, "argv", ["truss-analysis", "nonexistent_file.json"]):
        with pytest.raises(FileNotFoundError):
            main()
