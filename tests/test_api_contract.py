# tests/test_api_contract.py
"""تست‌های قرارداد API برای جلوگیری از regression"""

import pytest

from truss_analysis.model import TrussModel


class TestAPIContract:
    """تست‌های تضمین سازگاری API"""

    def test_input_data_must_be_list_not_dict(self):
        """nodes و elements باید list باشند، نه dict"""
        # ❌ این باید خطا بدهد
        invalid_data = {
            "nodes": {"1": {"id": 1, "x": 0, "y": 0}},  # dict!
            "elements": [],
        }
        with pytest.raises((TypeError, ValueError)):
            TrussModel(invalid_data)

    def test_backward_compatibility_add_methods(self):
        """اگر add_node/add_element اضافه شد، باید کار کند"""
        # این تست برای آینده است
        pass
