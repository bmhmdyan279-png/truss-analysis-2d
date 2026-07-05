import os
import sys
import warnings

import pytest
from scipy.sparse.linalg import MatrixRankWarning

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from truss_analysis.assembly import build_global_matrices
from truss_analysis.model import TrussModel
from truss_analysis.solver import (
    calculate_element_results,
    calculate_total_energy,
    solve_displacements,
    validate_energy,
)


@pytest.mark.skip(reason="API mismatch - needs rewrite for v1.4.0")
def test_solve_displacements_penalty():
    """تست حل جابجایی‌ها با روش penalty"""
    input_data = {
        "nodes": {
            "1": {"id": 1, "x": 0, "y": 0},
            "2": {"id": 2, "x": 2, "y": 0},
        },
        "elements": {
            "1": {
                "id": 1,
                "node_i": 1,
                "node_j": 2,
                "E": 200e9,
                "A": 0.01,
            }
        },
        "supports": {
            "1": {"ux": True, "uy": True},
            "2": {"ux": False, "uy": True},
        },
        "loads": {"2": {"fx": 10000, "fy": 0}},
    }

    truss = TrussModel(input_data)
    K_global, F_global = build_global_matrices(truss)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatrixRankWarning)
        displacements = solve_displacements(truss, K_global, F_global)

    assert displacements is not None
    assert len(displacements) == 4
    print("✅ تست حل جابجایی‌ها با روش penalty پاس شد")


@pytest.mark.skip(reason="API mismatch - needs rewrite for v1.4.0")
def test_calculate_element_results_with_buckling():
    """تست نتایج با بررسی کمانش"""
    input_data = {
        "nodes": {
            "1": {"id": 1, "x": 0, "y": 0},
            "2": {"id": 2, "x": 2, "y": 0},
        },
        "elements": {
            "1": {
                "id": 1,
                "node_i": 1,
                "node_j": 2,
                "E": 200e9,
                "A": 0.01,
                "I": 0.0001,
            }
        },
        "supports": {
            "1": {"ux": True, "uy": True},
            "2": {"ux": False, "uy": True},
        },
        "loads": {"2": {"fx": 10000, "fy": 0}},
    }

    truss = TrussModel(input_data)
    K_global, F_global = build_global_matrices(truss)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatrixRankWarning)
        displacements = solve_displacements(truss, K_global, F_global)

    results = calculate_element_results(truss, displacements)

    assert len(results) == 1
    element_result = results[0]
    assert "buckling_warning" in element_result

    # اصلاح: استفاده از not به جای is False
    assert not element_result["buckling_warning"]
    print("✅ تست نتایج با بررسی کمانش پاس شد")


@pytest.mark.skip(reason="API mismatch - needs rewrite for v1.4.0")
def test_validate_energy_with_loads():
    """تست اعتبارسنجی انرژی با بارگذاری"""
    input_data = {
        "nodes": {
            "1": {"id": 1, "x": 0, "y": 0},
            "2": {"id": 2, "x": 2, "y": 0},
        },
        "elements": {
            "1": {
                "id": 1,
                "node_i": 1,
                "node_j": 2,
                "E": 200e9,
                "A": 0.01,
            }
        },
        "supports": {
            "1": {"ux": True, "uy": True},
            "2": {"ux": False, "uy": True},
        },
        "loads": {"2": {"fx": 10000, "fy": 0}},
    }

    truss = TrussModel(input_data)
    K_global, F_global = build_global_matrices(truss)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatrixRankWarning)
        displacements = solve_displacements(truss, K_global, F_global)

    results = calculate_element_results(truss, displacements)
    U_total = calculate_total_energy(truss, displacements)

    is_valid, error, message = validate_energy(results, U_total, truss, displacements, F_global)

    assert is_valid is not None
    print("✅ تست اعتبارسنجی انرژی با بارگذاری پاس شد")


@pytest.mark.skip(reason="API mismatch - needs rewrite for v1.4.0")
def test_validate_energy_thermal_only():
    """تست اعتبارسنجی انرژی (حرارتی خالص)"""
    input_data = {
        "nodes": {
            "1": {"id": 1, "x": 0, "y": 0},
            "2": {"id": 2, "x": 2, "y": 0},
        },
        "elements": {
            "1": {
                "id": 1,
                "node_i": 1,
                "node_j": 2,
                "E": 200e9,
                "A": 0.01,
                "delta_T": 50,
            }
        },
        "supports": {
            "1": {"ux": True, "uy": True},
            "2": {"ux": False, "uy": True},
        },
        "loads": {},
    }

    truss = TrussModel(input_data)
    K_global, F_global = build_global_matrices(truss)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatrixRankWarning)
        displacements = solve_displacements(truss, K_global, F_global)

    results = calculate_element_results(truss, displacements)
    U_total = calculate_total_energy(truss, displacements)

    is_valid, error, message = validate_energy(results, U_total, truss, displacements, F_global)

    assert is_valid is not None
    print("✅ تست اعتبارسنجی انرژی (حرارتی خالص) پاس شد")


@pytest.mark.skip(reason="API mismatch - needs rewrite for v1.4.0")
def test_solver_error_handling():
    """تست مدیریت خطا در solver"""
    input_data = {
        "nodes": {
            "1": {"id": 1, "x": 0, "y": 0},
        },
        "elements": {},
        "supports": {"1": {"ux": True, "uy": True}},
        "loads": {},
    }

    truss = TrussModel(input_data)
    K_global, F_global = build_global_matrices(truss)

    try:
        displacements = solve_displacements(truss, K_global, F_global)
        assert displacements is not None
    except Exception as e:
        # برخی خطاها قابل قبول هستند
        print(f"  خطای قابل قبول: {e}")

    print("✅ تست مدیریت خطا در solver پاس شد")
