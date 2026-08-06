from __future__ import annotations

import json
from pathlib import Path

from .exceptions import InputValidationError

MAX_FILE_SIZE_MB = 10


def load_json(filepath: str | Path) -> dict:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise InputValidationError(
            f"File size ({size_mb:.1f}MB) exceeds limit ({MAX_FILE_SIZE_MB}MB)."
        )

    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise InputValidationError(f"Invalid JSON format: {e}") from None
