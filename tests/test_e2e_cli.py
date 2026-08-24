import os
import subprocess
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TEST_DIR)


def test_cli_runs_successfully():
    result = subprocess.run(
        [sys.executable, "-m", "truss_analysis.main", "examples/example1.json", "SI"],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    assert result.returncode == 0, f"CLI crashed with error: {result.stderr}"
    assert "تحلیل با موفقیت انجام شد" in result.stdout


def test_example_script_runs_successfully():
    result = subprocess.run(
        [sys.executable, "examples/example_analysis.py"],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    assert result.returncode == 0, f"Example script crashed with error: {result.stderr}"
    assert "📊 نتایج" in result.stdout
