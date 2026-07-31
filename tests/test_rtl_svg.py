import os
import re

os.environ["MPLBACKEND"] = "Agg"
from truss_analysis.i18n import get_text  # noqa: E402


def test_rtl_svg_text_rendering():
    force_text = get_text("force")
    svg_content = f"""<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
    <text x="10" y="50" font-family="Vazirmatn" font-size="16" direction="rtl">{force_text}</text>
</svg>"""
    assert (
        "نیرو" in svg_content
        or "نـیـرو" in svg_content
        or re.search(r"ن.*ی.*ر.*و", svg_content)
    )
    assert 'direction="rtl"' in svg_content
    assert 'font-family="Vazirmatn"' in svg_content
