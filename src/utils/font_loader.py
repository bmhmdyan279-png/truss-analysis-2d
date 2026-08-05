import os
from contextlib import contextmanager

import matplotlib.font_manager as fm


@contextmanager
def safe_persian_font():
    """Return a FontProperties with a Persian-capable font, with cross-OS fallback."""
    font_path = os.path.join(
        os.path.dirname(__file__), "fonts", "Vazirmatn-Regular.ttf"
    )
    fallback_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:\\Windows\\Fonts\\tahoma.ttf",  # Windows
    ]
    try:
        fm.fontManager.addfont(str(font_path))
        yield fm.FontProperties(fname=str(font_path))
    except FileNotFoundError:
        chosen = None
        for path in fallback_paths:
            if os.path.exists(path):
                chosen = path
                break
        if chosen:
            fm.fontManager.addfont(chosen)
            yield fm.FontProperties(fname=chosen)
        else:
            print("Warning: No Persian fallback font found. Tofu may appear.")
            yield fm.FontProperties()
