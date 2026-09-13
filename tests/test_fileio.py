from __future__ import annotations

import pytest

from truss_analysis.exceptions import InputValidationError
from truss_analysis.fileio import load_json


def test_load_json_valid(tmp_path):
    p = tmp_path / "test.json"
    p.write_text('{"nodes": [], "elements": []}', encoding="utf-8")
    data = load_json(p)
    assert "nodes" in data


def test_load_json_missing(tmp_path):
    p = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_json(str(p))


def test_load_json_invalid_format(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{invalid json}", encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_json(p)


def test_load_json_invalid_schema(tmp_path):
    p = tmp_path / "schema.json"
    p.write_text('{"nodes": {}}', encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_json(p)


def test_load_json_file_too_large(tmp_path):
    """Test that files exceeding MAX_INPUT_BYTES are rejected."""
    from truss_analysis.fileio import MAX_INPUT_BYTES

    p = tmp_path / "large.json"
    # Create a file larger than MAX_INPUT_BYTES (10MB)
    large_content = (
        '{"nodes": ['
        + ",".join([f'{{"id": "n{i}", "x": 0, "y": 0}}' for i in range(500000)])
        + "]}"
    )
    p.write_text(large_content, encoding="utf-8")

    # Verify file is actually large enough
    assert p.stat().st_size > MAX_INPUT_BYTES

    with pytest.raises(InputValidationError, match="File too large"):
        load_json(p)


def test_load_json_missing_elements_key(tmp_path):
    """Test that missing 'elements' key is rejected."""
    p = tmp_path / "missing_elements.json"
    p.write_text('{"nodes": []}', encoding="utf-8")
    with pytest.raises(InputValidationError, match="Missing key: 'elements'"):
        load_json(p)


def test_load_json_nodes_not_list(tmp_path):
    """Test that non-list 'nodes' is rejected."""
    p = tmp_path / "nodes_dict.json"
    p.write_text('{"nodes": {}, "elements": []}', encoding="utf-8")
    with pytest.raises(InputValidationError, match="'nodes' must be a list"):
        load_json(p)


def test_load_json_elements_not_list(tmp_path):
    """Test that non-list 'elements' is rejected."""
    p = tmp_path / "elements_dict.json"
    p.write_text('{"nodes": [], "elements": {}}', encoding="utf-8")
    with pytest.raises(InputValidationError, match="'elements' must be a list"):
        load_json(p)


def test_load_json_element_references_nonexistent_node(tmp_path):
    """Test that element referencing non-existent node is rejected."""
    p = tmp_path / "bad_ref.json"
    p.write_text(
        (
            '{"nodes": [{"id": "A", "x": 0, "y": 0}],'
            ' "elements": [{"id": "e1", "node_i": "A", "node_j": "B"}]}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(InputValidationError, match="references non-existent nodes"):
        load_json(p)


def test_load_json_loads_not_list(tmp_path):
    """Test that non-list 'loads' is rejected."""
    p = tmp_path / "loads_dict.json"
    p.write_text(
        '{"nodes": [], "elements": [], "loads": {}}',
        encoding="utf-8"
    )
    with pytest.raises(InputValidationError, match="'loads' must be a list"):
        load_json(p)


def test_load_json_load_item_not_dict(tmp_path):
    """Test that non-dict load item is rejected."""
    p = tmp_path / "loads_list.json"
    p.write_text(
        '{"nodes": [], "elements": [], "loads": ["not_a_dict"]}',
        encoding="utf-8"
    )
    with pytest.raises(InputValidationError, match="Each load must be a dictionary"):
        load_json(p)


def test_load_json_load_missing_node_id_and_id(tmp_path):
    """Test that load missing both node_id and id is rejected."""
    p = tmp_path / "load_no_id.json"
    p.write_text(
        '{"nodes": [], "elements": [], "loads": [{"Fx": 1, "Fy": 0}]}',
        encoding="utf-8"
    )
    with pytest.raises(InputValidationError, match="Load missing 'node_id' or 'id'"):
        load_json(p)


def test_load_json_load_missing_fx_and_fy(tmp_path):
    """Test that load missing both Fx and Fy is rejected."""
    p = tmp_path / "load_no_force.json"
    p.write_text(
        '{"nodes": [], "elements": [], "loads": [{"id": "L1"}]}',
        encoding="utf-8"
    )
    with pytest.raises(InputValidationError, match="Load missing 'Fx' or 'Fy'"):
        load_json(p)

