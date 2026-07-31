#!/usr/bin/env python3
"""Backward compatibility shim for running truss_analysis from the root."""

import warnings

from truss_analysis.main import main

if __name__ == "__main__":
    warnings.warn(
        "Running root main.py is deprecated. Please use 'truss-analyze' command instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
