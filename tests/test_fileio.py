from __future__ import annotations

import pytest
from truss_analysis.exceptions import InputValidationError
from truss_analysis.fileio import load_json


def test_load_json_valid(tmp_path):
    p = tmp_path / "test.json"
    p.write_text('{"nodes": {}}', encoding="utf-8")
    data = load_json(p)
    assert "nodes" in data


def test_load_json_missing():
    with pytest.raises(FileNotFoundError):
        load_json("nonexistent.json")


def test_load_json_invalid_format(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{invalid json}", encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_json(p)
