"""CLI coverage tests for the unified entry point and its subcommands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from truss_analysis.main import main


def test_cli_coverage_execution():
    """Run all example JSONs through the unified CLI to exercise main.py paths."""
    examples = sorted(Path("examples").glob("*.json"))
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


def test_cli_missing_input_returns_usage_error():
    """Verify a missing input file returns exit code 2 (usage error)."""
    with patch.object(sys, "argv", ["truss-analysis", "nonexistent_file.json"]):
        assert main() == 2


def test_cli_analyze_subcommand_explicit(tmp_path, capsys):
    """The explicit ``analyze`` subcommand matches the legacy invocation."""
    model = {
        "units": "SI",
        "nodes": [
            {
                "id": "1",
                "x": 0.0,
                "y": 0.0,
                "is_support": True,
                "support_dx": True,
                "support_dy": True,
            },
            {"id": "2", "x": 2.0, "y": 0.0, "is_support": True, "support_dy": True},
            {"id": "3", "x": 1.0, "y": 1.0, "is_support": False},
        ],
        "elements": [
            {"id": "e1", "node_i": "1", "node_j": "3", "E": 210e9, "A": 0.01},
            {"id": "e2", "node_i": "2", "node_j": "3", "E": 210e9, "A": 0.01},
            {"id": "e3", "node_i": "1", "node_j": "2", "E": 210e9, "A": 0.01},
        ],
        "loads": [{"node_id": "3", "Fx": 0.0, "Fy": -1000.0}],
    }
    p = tmp_path / "model.json"
    p.write_text(json.dumps(model), encoding="utf-8")
    out_json = tmp_path / "result.json"
    rc = main(["analyze", str(p), "--quiet", "-o", str(out_json)])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "SUCCESS"
    assert payload["equilibrium"]["is_valid"]


def test_cli_validate_subcommand(tmp_path, capsys):
    """``validate`` accepts a good model and rejects a broken one."""
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "units": "SI",
                "nodes": [
                    {
                        "id": "1",
                        "x": 0.0,
                        "y": 0.0,
                        "is_support": True,
                        "support_dx": True,
                        "support_dy": True,
                    },
                    {
                        "id": "2",
                        "x": 1.0,
                        "y": 0.0,
                        "is_support": True,
                        "support_dy": True,
                    },
                    {"id": "3", "x": 0.5, "y": 0.5, "is_support": False},
                ],
                "elements": [
                    {"id": "e1", "node_i": "1", "node_j": "3", "E": 210e9, "A": 0.01},
                    {"id": "e2", "node_i": "2", "node_j": "3", "E": 210e9, "A": 0.01},
                    {"id": "e3", "node_i": "1", "node_j": "2", "E": 210e9, "A": 0.01},
                ],
                "loads": [{"node_id": "3", "Fx": 0.0, "Fy": -1000.0}],
            }
        ),
        encoding="utf-8",
    )
    assert main(["validate", str(good)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["n_nodes"] == 3

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nodes": [], "elements": []}), encoding="utf-8")
    assert main(["validate", str(bad)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is False
    assert report["errors"]

    assert main(["validate", str(tmp_path / "missing.json")]) == 2


def test_cli_generate_subcommand(tmp_path, capsys):
    """``generate`` emits canonical JSON for a parametric truss."""
    out = tmp_path / "gen.json"
    rc = main(
        [
            "generate",
            "--family",
            "warren",
            "--panels",
            "4",
            "--span",
            "16",
            "--height",
            "2.4",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    model = json.loads(out.read_text(encoding="utf-8"))
    assert model["nodes"]
    assert model["elements"]

    # stdout path
    rc = main(
        [
            "generate",
            "--family",
            "pratt",
            "--panels",
            "2",
            "--span",
            "8",
            "--height",
            "1.5",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["nodes"]

    # invalid parameters -> usage error
    rc = main(
        [
            "generate",
            "--family",
            "warren",
            "--panels",
            "1",
            "--span",
            "8",
            "--height",
            "1.5",
        ]
    )
    assert rc == 2


def test_cli_version_subcommand(capsys):
    """``version`` prints the package version and exits 0."""
    from truss_analysis import __version__

    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_invalid_input_returns_failure_code(tmp_path, capsys):
    """An input that fails validation returns exit code 1, not a traceback."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"units": "SI", "nodes": [{"id": "1"}], "elements": []}),
        encoding="utf-8",
    )
    assert main(["analyze", str(bad)]) == 1
