import csv
import json
from typing import Any


def enforce_locale(data: dict[str, Any]) -> dict[str, Any]:
    """Ensures the output dictionary contains the _locale meta field."""
    if isinstance(data, dict):
        data["_locale"] = "fa"
    return data


def write_json_safe(data: dict[str, Any], filepath: str) -> None:
    """Writes JSON with UTF-8-SIG (BOM) for correct Excel/Windows display."""
    safe_data = enforce_locale(data.copy())
    with open(filepath, "w", encoding="utf-8-sig") as f:
        json.dump(safe_data, f, indent=2, ensure_ascii=False)


def write_csv_safe(rows: list[dict[str, Any]], filepath: str) -> None:
    """Writes CSV with UTF-8-SIG and guaranteed English keys."""
    if not rows:
        return
    for row in rows:
        row["_locale"] = "fa"
    fieldnames = list(rows[0].keys())
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
