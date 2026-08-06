import json
import os


class FileValidationError(Exception):
    pass


def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    size = os.path.getsize(filepath)
    if size > 10 * 1024 * 1024:
        raise FileValidationError(f"File too large: {size} bytes.")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ["nodes", "elements"]:
        if key not in data:
            raise FileValidationError(f"Missing key: '{key}'")
        if not isinstance(data[key], list):
            raise FileValidationError(f"'{key}' must be a list.")

    node_ids = {n.get("id") for n in data["nodes"]}
    for elem in data["elements"]:
        if elem.get("node_i") not in node_ids or elem.get("node_j") not in node_ids:
            raise FileValidationError(
                f"Element {elem.get('id')} references non-existent nodes."
            )
    return data
