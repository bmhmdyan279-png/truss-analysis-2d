"""
فایل پیکربندی pytest برای تست‌های خرپا
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ایمپورت ماژول‌های اصلی برای دسترسی در تست‌ها
try:
    from model import TrussModel, Node, Element
    from assembly import assemble_global_matrices, build_global_matrices, get_reduced_system
    from solver import solve_displacements, calculate_element_results
    from postprocess import sort_elements, generate_report
    from fileio import parse_input, validate_units
    print("✅ ماژول‌ها با موفقیت import شدند")
except ImportError as e:
    print(f"❌ خطای import: {e}")
    raise
