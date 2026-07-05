"""
truss_analysis — A professional 2D truss analysis engine.
"""

try:
    from importlib.metadata import version

    __version__ = version("truss-analysis-2d")
except Exception:
    __version__ = "1.4.0"

from .main import main

__all__ = ["main", "__version__"]
