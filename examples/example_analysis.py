"""Example analysis script (updated for the src-layout package)."""

import json
import tempfile
from pathlib import Path

from truss_analysis import run

data = {
    "units": "SI",
    "nodes": [
        {
            "id": 1,
            "x": 0,
            "y": 0,
            "is_support": True,
            "support_dx": True,
            "support_dy": True,
        },
        {"id": 2, "x": 2, "y": 0, "is_support": True, "support_dy": True},
        {"id": 3, "x": 1, "y": 1.5, "is_support": False},
    ],
    "elements": [
        {"id": 1, "node_i": 1, "node_j": 3, "A": 0.01, "E": 210e9},
        {"id": 2, "node_i": 2, "node_j": 3, "A": 0.01, "E": 210e9},
        {"id": 3, "node_i": 1, "node_j": 2, "A": 0.01, "E": 210e9},
    ],
    "loads": [{"node_id": 3, "Fx": 0, "Fy": -10000}],
}

with tempfile.NamedTemporaryFile(
    "w", suffix=".json", delete=False, encoding="utf-8"
) as f:
    json.dump(data, f)
    path = f.name

result = run(path, check_buckling=True)
print(result.summary())
Path(path).unlink()
