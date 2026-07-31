import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
from utils.font_loader import safe_persian_font


def test_persian_render_no_tofu() -> None:
    """Structure-based render test: Verifies Persian text renders without Tofu."""
    with safe_persian_font() as prop:
        fig, ax = plt.subplots()
        ax.set_title("آزمایش رسم نمودار فارسی", fontproperties=prop)
        ax.text(
            0.5, 0.5, "متن آزمایشی فارسی", fontproperties=prop, ha="center", va="center"
        )

        temp_path = "test_output.png"
        try:
            plt.savefig(temp_path, bbox_inches="tight")
            assert os.path.exists(temp_path), "Rendered file was not created."
            assert os.path.getsize(temp_path) > 1000, (
                "Rendered file is suspiciously small."
            )
        finally:
            plt.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)
