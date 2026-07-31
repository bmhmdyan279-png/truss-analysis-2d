import os
import subprocess
import sys
import time
from pathlib import Path


def test_help_performance_regression():
    baseline_file = Path(__file__).parent.parent / ".baseline_perf"
    start = time.perf_counter()
    env = os.environ.copy()
    src_path = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).parent.parent,
        env=env,
    )
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, f"main.py --help failed: {result.stderr}"
    if baseline_file.exists():
        baseline = float(baseline_file.read_text().strip())
        assert (
            elapsed <= baseline * 1.20
        ), f"Regression: {elapsed:.4f}s > {baseline * 1.20:.4f}s"
    else:
        baseline_file.write_text(f"{elapsed:.6f}")
        assert elapsed < 2.0, f"Initial baseline too high: {elapsed:.4f}s"
