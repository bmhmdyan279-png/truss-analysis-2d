from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("truss-analysis-2d")
    except PackageNotFoundError:
        __version__ = "2.0.0-dev"
except ImportError:
    __version__ = "2.0.0-dev"
