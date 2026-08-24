"""Golden tests with analytical solutions for validation."""

import json
import tempfile
from pathlib import Path

from truss_analysis.main import run


def test_golden_pratt_truss():
    """
    Pratt Truss with central load.
    Analytical solution:
    - AB = BC = 5 (tension)
    - AD = CF = 0
    - DE = EF = 0
    - BE = 0
    - AE = CE = -5√2 ≈ -7.07 (compression)
    - Reactions: Ry_A = Ry_C = 5, Rx_A = 0
    """
    data = {
        "nodes": [
            {
                "id": "A",
                "x": 0,
                "y": 0,
                "is_support": True,
                "support_dx": True,
                "support_dy": True,
            },
            {"id": "B", "x": 2, "y": 0},
            {"id": "C", "x": 4, "y": 0, "is_support": True, "support_dy": True},
            {"id": "D", "x": 0, "y": 2},
            {"id": "E", "x": 2, "y": 2},
            {"id": "F", "x": 4, "y": 2},
        ],
        "elements": [
            {"id": "AB", "node_i": "A", "node_j": "B", "E": 1, "A": 1},
            {"id": "BC", "node_i": "B", "node_j": "C", "E": 1, "A": 1},
            {"id": "AD", "node_i": "A", "node_j": "D", "E": 1, "A": 1},
            {"id": "CF", "node_i": "C", "node_j": "F", "E": 1, "A": 1},
            {"id": "DE", "node_i": "D", "node_j": "E", "E": 1, "A": 1},
            {"id": "EF", "node_i": "E", "node_j": "F", "E": 1, "A": 1},
            {"id": "BE", "node_i": "B", "node_j": "E", "E": 1, "A": 1},
            {"id": "AE", "node_i": "A", "node_j": "E", "E": 1, "A": 1},
            {"id": "CE", "node_i": "C", "node_j": "E", "E": 1, "A": 1},
        ],
        "loads": [{"node_id": "E", "Fx": 0, "Fy": -10}],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        result = run(temp_path, "SI")

        # Check reactions
        assert abs(result.reactions["A"]["Ry"] - 5.0) < 1e-6
        assert abs(result.reactions["C"]["Ry"] - 5.0) < 1e-6
        assert abs(result.reactions["A"]["Rx"]) < 1e-6

        # Check member forces
        force_dict = {f["element"]: f["force"] for f in result.forces}
        assert abs(force_dict["AB"] - 5.0) < 1e-6
        assert abs(force_dict["BC"] - 5.0) < 1e-6
        assert abs(force_dict["AD"]) < 1e-6
        assert abs(force_dict["CF"]) < 1e-6
        assert abs(force_dict["DE"]) < 1e-6
        assert abs(force_dict["EF"]) < 1e-6
        assert abs(force_dict["BE"]) < 1e-6
        assert abs(force_dict["AE"] - (-5 * 2**0.5)) < 1e-5
        assert abs(force_dict["CE"] - (-5 * 2**0.5)) < 1e-5

    finally:
        Path(temp_path).unlink()


def test_golden_warren_truss():
    """
    Warren Truss with central load.
    Analytical solution:
    - AB = P/4 = 2.5 (tension)
    - AC = BC = -P√5/4 ≈ -2.795 (compression)
    - Reactions: Ry_A = Ry_B = 5, Rx_A = 0
    """
    data = {
        "nodes": [
            {
                "id": "A",
                "x": 0,
                "y": 0,
                "is_support": True,
                "support_dx": True,
                "support_dy": True,
            },
            {"id": "B", "x": 2, "y": 0, "is_support": True, "support_dy": True},
            {"id": "C", "x": 1, "y": 2},
        ],
        "elements": [
            {"id": "AB", "node_i": "A", "node_j": "B", "E": 1, "A": 1},
            {"id": "AC", "node_i": "A", "node_j": "C", "E": 1, "A": 1},
            {"id": "BC", "node_i": "B", "node_j": "C", "E": 1, "A": 1},
        ],
        "loads": [{"node_id": "C", "Fx": 0, "Fy": -10}],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        result = run(temp_path, "SI")

        # Check reactions
        assert abs(result.reactions["A"]["Ry"] - 5.0) < 1e-6
        assert abs(result.reactions["B"]["Ry"] - 5.0) < 1e-6

        # Check member forces
        force_dict = {f["element"]: f["force"] for f in result.forces}
        assert abs(force_dict["AB"] - 2.5) < 1e-6
        expected_ac = -10 * 5**0.5 / 4
        assert abs(force_dict["AC"] - expected_ac) < 1e-5
        assert abs(force_dict["BC"] - expected_ac) < 1e-5

    finally:
        Path(temp_path).unlink()


def test_golden_cantilever_truss():
    """
    Cantilever Truss with load at free end.
    Analytical solution:
    - AB = -P = -10 (compression)
    - BC = P√2 ≈ 14.14 (tension)
    - AC = -P = -10 (compression)
    - Reactions: Rx_A = 10, Ry_A = 10, Rx_C = -10, Ry_C = 0
    """
    data = {
        "nodes": [
            {
                "id": "A",
                "x": 0,
                "y": 0,
                "is_support": True,
                "support_dx": True,
                "support_dy": True,
            },
            {"id": "B", "x": 2, "y": 0},
            {"id": "C", "x": 0, "y": 2, "is_support": True, "support_dx": True},
        ],
        "elements": [
            {"id": "AB", "node_i": "A", "node_j": "B", "E": 1, "A": 1},
            {"id": "BC", "node_i": "B", "node_j": "C", "E": 1, "A": 1},
            {"id": "AC", "node_i": "A", "node_j": "C", "E": 1, "A": 1},
        ],
        "loads": [{"node_id": "B", "Fx": 0, "Fy": -10}],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    try:
        result = run(temp_path, "SI")

        # Check reactions
        assert abs(result.reactions["A"]["Rx"] - 10.0) < 1e-6
        assert abs(result.reactions["A"]["Ry"] - 10.0) < 1e-6
        assert abs(result.reactions["C"]["Rx"] - (-10.0)) < 1e-6
        assert abs(result.reactions["C"]["Ry"]) < 1e-6

        # Check member forces
        force_dict = {f["element"]: f["force"] for f in result.forces}
        assert abs(force_dict["AB"] - (-10.0)) < 1e-6
        assert abs(force_dict["BC"] - 10 * 2**0.5) < 1e-5
        assert abs(force_dict["AC"] - (-10.0)) < 1e-6

    finally:
        Path(temp_path).unlink()
