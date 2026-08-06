from __future__ import annotations

import json
import os

from .exceptions import InputValidationError


def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    size = os.path.getsize(filepath)
    if size > 10 * 1024 * 1024:
        raise InputValidationError(f"File too large: {size} bytes.")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise InputValidationError("Invalid JSON format.")

    for key in ["nodes", "elements"]:
        if key not in data:
            raise InputValidationError(f"Missing key: '{key}'")
        if not isinstance(data[key], list):
            raise InputValidationError(f"'{key}' must be a list.")

    node_ids = {n.get("id") for n in data["nodes"]}
    for elem in data["elements"]:
        if elem.get("node_i") not in node_ids or elem.get("node_j") not in node_ids:
            raise InputValidationError(
                f"Element {elem.get('id')} references non-existent nodes."
            )

    if "loads" in data:
        loads = data["loads"]
        if not isinstance(loads, list):
            raise InputValidationError("'loads' must be a list of force objects.")
        for lf in loads:
            if not isinstance(lf, dict):
                raise InputValidationError("Each load must be a dictionary.")
            if "node_id" not in lf and "id" not in lf:
                raise InputValidationError("Load missing 'node_id' or 'id'.")
            if "Fx" not in lf and "Fy" not in lf:
                raise InputValidationError("Load missing 'Fx' or 'Fy'.")
    return data
