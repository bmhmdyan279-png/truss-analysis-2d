#!/usr/bin/env python3
"""Backward-compatibility shim: run truss_analysis from repo root.

Exit Code Taxonomy:
    0: Success (موفقیت)
    1: Input error (خطای ورودی)
    2: Solver error (خطای حل‌گر)
    3: Output/Visualization error (خطای خروجی)
    4: Internal error (خطای داخلی)
"""

from __future__ import annotations

import io
import sys
from typing import TextIO


def _configure_stdio(stream: TextIO) -> TextIO:
    """Force UTF-8 on stdout/stderr (Windows TTY mojibake guard)."""
    if not hasattr(stream, "reconfigure"):
        return io.TextIOWrapper(stream.buffer, encoding="utf-8")
    stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    return stream


def main() -> int:
    """Entry point with the documented exit-code taxonomy."""
    sys.stdout = _configure_stdio(sys.stdout)
    sys.stderr = _configure_stdio(sys.stderr)
    try:
        from truss_analysis.main import main as app_main
    except ImportError as exc:
        print(
            f"Internal error (4): Failed to import truss_analysis.main ({exc})",
            file=sys.stderr,
        )
        return 4
    try:
        return int(app_main())
    except Exception as exc:
        print(f"Internal error (4): {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
