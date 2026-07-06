#!/usr/bin/env python3
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
from truss_analysis.main import main


def run_and_export():
    input_file = os.path.join(os.path.dirname(__file__), "../reference_problem.json")
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    print(f"Running analysis on {input_file}...")
    sys.argv = ["truss-analyze", input_file]
    try:
        main()
    except SystemExit:
        pass

    print("\nAnalysis complete. Check truss_analysis.log for results.")


if __name__ == "__main__":
    run_and_export()
