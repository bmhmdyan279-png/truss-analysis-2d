import importlib.resources as pkg_resources
import os
from contextlib import contextmanager

import matplotlib.font_manager as fm


@contextmanager
def safe_persian_font():
    """
    Context manager for safe Persian font loading using importlib.resources.as_file.
    Provides a real file path to Matplotlib to avoid BytesIO issues on Windows/Linux.
    """
    try:
        font_ref = pkg_resources.files("src.utils").joinpath(
            "fonts/Vazirmatn-Regular.ttf"
        )
        with pkg_resources.as_file(font_ref) as font_path:
            fm.fontManager.addfont(str(font_path))
            prop = fm.FontProperties(fname=str(font_path))
            yield prop
    except Exception:
        fallback_path = os.path.join(
            os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "tahoma.ttf"
        )
        fm.fontManager.addfont(fallback_path)
        prop = fm.FontProperties(fname=fallback_path)
        yield prop
