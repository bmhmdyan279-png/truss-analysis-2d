# tests/test_edge_cases.py - نسخه اصلاح‌شده
import numpy as np

from truss_analysis.model import TrussModel


def test_singular_matrix_supports():
    """تست تشخیص ماتریس منفرد با تکیه‌گاه ناکافی"""
    input_data = {
        "units": "SI",
        "nodes": [
            {"id": 1, "x": 0, "y": 0, "is_support": True},
            {"id": 2, "x": 1, "y": 0, "is_support": False},
        ],
        "elements": [{"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "A": 0.01}],
        "loads": {"node_forces": [{"node_id": 2, "Fx": 1000, "Fy": 0}]},
    }
    model = TrussModel(input_data)

    # فقط یک تکیهگاه → ماتریس منفرد
    assert len(model.supported_nodes) == 1
    assert len(model.fixed_dofs) == 2  # فقط 2 DOF قفل

    # بررسی که سیستم می‌تواند تشخیص دهد ماتریس منفرد است
    from truss_analysis.assembly import build_global_matrices

    K, F = build_global_matrices(model)

    # rank باید کمتر از اندازه ماتریس باشد
    rank = np.linalg.matrix_rank(K)
    assert rank < K.shape[0], "ماتریس باید منفرد باشد"


def test_extreme_loads():
    """تست پایداری عددی با بارهای بسیار بزرگ"""
    input_data = {
        "units": "SI",
        "nodes": [
            {"id": 1, "x": 0, "y": 0, "is_support": True},
            {"id": 2, "x": 1, "y": 0, "is_support": True},
        ],
        "elements": [{"id": 1, "node_i": 1, "node_j": 2, "E": 200e9, "A": 0.01}],
        "loads": {"node_forces": [{"node_id": 2, "Fx": 1e12, "Fy": 0}]},
    }
    model = TrussModel(input_data)
    assert len(model.loads) == 1

    # بررسی overflow رخ نمی‌دهد
    from truss_analysis.assembly import build_global_matrices

    K, F = build_global_matrices(model)
    assert np.all(np.isfinite(K)), "ماتریس سختی نباید NaN/Inf داشته باشد"
    assert np.all(np.isfinite(F)), "بردار بار نباید NaN/Inf داشته باشد"
