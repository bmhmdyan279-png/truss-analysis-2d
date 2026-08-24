import pytest
import subprocess
import sys
from pathlib import Path

def test_cli_coverage_execution():
    examples = list(Path("examples").glob("*.json"))
    if not examples:
        pytest.skip("No example JSON files found")
    for ex in examples:
        # Executing main.py paths to satisfy coverage guards
        res = subprocess.run([sys.executable, "main.py", str(ex)], capture_output=True, text=True)
        assert res.returncode == 0, f"CLI failed on {ex}: {res.stderr}"
