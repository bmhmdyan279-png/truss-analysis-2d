#!/usr/bin/env python3
"""Entry point for truss-analysis-2d CLI."""

import sys


def _main():
    """Main entry point with UTF-8 wrapper for Windows compatibility."""
    # Set UTF-8 encoding ONLY when running as script (not on import)
    import io

    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except (AttributeError, Exception):
        pass

    try:
        from truss_analysis.main import main as app_main

        return app_main()
    except ImportError as e:
        print(
            f"Internal error (4): Failed to import truss_analysis.main ({e})",
            file=sys.stderr,
        )
        return 4
    except Exception as e:
        print(f"Internal error (4): {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(_main())
