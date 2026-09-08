"""JSON input loading with structural pre-validation.

:func:`load_json` is the single entry point for reading truss models from
disk. Besides parsing, it enforces the minimal contract every input file
must satisfy (top-level ``nodes``/``elements`` lists, element-to-node
referential integrity, well-formed ``loads``) so that downstream modules
can rely on the returned mapping's shape.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .exceptions import InputValidationError

#: Input files larger than this are rejected before parsing (bytes).
MAX_INPUT_BYTES = 10 * 1024 * 1024


def load_json(filepath: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and pre-validate a truss model from a JSON file.

    Parameters
    ----------
    filepath : str or os.PathLike[str]
        Path to the input JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed model with at least the keys ``nodes`` and ``elements``;
        ``loads`` and ``units`` are optional.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    InputValidationError
        If the file is too large, is not valid JSON, or violates the
        structural contract described in the module docstring.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    size = os.path.getsize(filepath)
    if size > MAX_INPUT_BYTES:
        raise InputValidationError(f"File too large: {size} bytes.")
    try:
        with open(filepath, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as exc:
        raise InputValidationError("Invalid JSON format.") from exc

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
