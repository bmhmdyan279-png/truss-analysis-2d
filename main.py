#!/usr/bin/env python3
import sys
import io

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

try:
    from truss_analysis.main import main as app_main

    if __name__ == "__main__":
        sys.exit(app_main())
except ImportError as e:
    print(
        f"Internal error (4): Failed to import truss_analysis.main ({e})",
        file=sys.stderr,
    )
    sys.exit(4)
except Exception as e:
    print(f"Internal error (4): {e}", file=sys.stderr)
    sys.exit(4)
