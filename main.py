#!/usr/bin/env python3
"""Backward compatibility shim for running truss_analysis from the root."""

from truss_analysis.main import main

if __name__ == "__main__":
    main()
