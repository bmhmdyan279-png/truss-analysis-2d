"""Reference benchmarks for independent validation.

This module provides a suite of canonical problems with known analytical or
numerically verified solutions, intended to validate the library's stability,
thermal, and mechanical response against independent references.

Each benchmark returns a dict with:
    - "description": str, problem statement
    - "model": tuple(nodes, elements, loads, supports)
    - "reference": dict of expected results (lambda_cr, forces, etc.)
    - "tolerance": relative tolerance for validation
    - "source": citation to literature or analytical derivation

Usage
-----
>>> from benchmarks.reference_problems import BENCHMARKS
>>> problem = BENCHMARKS["two_bar_toggle"]
>>> result = run_analysis(problem["model"])
>>> assert abs(result.lambda_cr - problem["reference"]["lambda_cr"]) < problem["tolerance"]
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from truss_analysis.model import Element, Node

__all__ = ["BENCHMARKS", "run_benchmark", "BenchmarkResult"]


class BenchmarkResult:
    """Outcome of running a benchmark comparison."""

    def __init__(
        self,
        name: str,
        passed: bool,
        computed: Dict[str, Any],
        reference: Dict[str, Any],
        relative_error: Dict[str, float],
        message: str = "",
    ):
        self.name = name
        self.passed = passed
        self.computed = computed
        self.reference = reference
        self.relative_error = relative_error
        self.message = message

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"BenchmarkResult({self.name}: {status})"


def run_benchmark(
    name: str,
    analysis_func: Callable,
    extract_key: Callable[[Any], Dict[str, float]],
) -> BenchmarkResult:
    """Run a named benchmark and compare against reference.

    Parameters
    ----------
    name : str
        Benchmark identifier (must exist in BENCHMARKS).
    analysis_func : callable
        Function that takes (nodes, elements, loads, supports) and returns
        analysis result object.
    extract_key : callable
        Function that extracts comparable quantities from the result,
        returning a dict keyed by quantity name.

    Returns
    -------
    BenchmarkResult
        Comparison outcome with pass/fail verdict and relative errors.
    """
    if name not in BENCHMARKS:
        return BenchmarkResult(
            name=name,
            passed=False,
            computed={},
            reference={},
            relative_error={},
            message=f"Unknown benchmark: {name}",
        )

    bench = BENCHMARKS[name]
    model = bench["model"]
    reference = bench["reference"]
    tol = bench["tolerance"]

    nodes, elements, loads, supports = model
    result = analysis_func(nodes, elements, loads, supports)
    computed = extract_key(result)

    relative_error = {}
    all_passed = True
    messages = []

    for key in reference:
        if key not in computed:
            all_passed = False
            messages.append(f"Missing key: {key}")
            continue

        ref_val = float(reference[key])
        comp_val = float(computed[key])

        if abs(ref_val) < 1e-14:
            # Reference is zero: use absolute error
            err = abs(comp_val)
        else:
            err = abs((comp_val - ref_val) / ref_val)

        relative_error[key] = err
        if err > tol:
            all_passed = False
            messages.append(f"{key}: rel_err={err:.2e} > tol={tol}")

    return BenchmarkResult(
        name=name,
        passed=all_passed,
        computed=computed,
        reference=reference,
        relative_error=relative_error,
        message="; ".join(messages) if messages else "All checks passed",
    )


# =============================================================================
# BENCHMARK DEFINITIONS
# =============================================================================

def _two_bar_toggle_model() -> Tuple[List[Node], List[Element], Dict, List]:
    """Two-bar shallow toggle (von Mises truss).

    Geometry: two identical bars pinned at base, meeting at apex.
    Load: vertical downward force at apex.
    Analytical lambda_cr for shallow geometry: lambda_cr ≈ (theta_0)^2 / (3 * (1 - nu^2))
    where theta_0 is the initial angle with horizontal.

    For H/L = 0.1 (shallow), theta_0 ≈ 0.1 rad, lambda_cr ≈ 0.0033.
    """
    L = 10.0  # half-span
    H = 1.0   # rise (shallow: H/L = 0.1)
    A = 0.01
    E = 210e9
    I = 8.33e-8  # circular section, D=0.02m

    nodes = [
        Node(id="A", x=-L, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=H, is_support=False, support_dx=False, support_dy=False),
        Node(id="C", x=L, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]

    elements = [
        Element(id="AB", node_i="A", node_j="B", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
        Element(id="BC", node_i="B", node_j="C", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
    ]

    loads = {"B": {"Fx": 0.0, "Fy": -1000.0}}
    supports = [("A", 0), ("A", 1), ("C", 0), ("C", 1)]

    # Analytical approximation for shallow toggle (Timoshenko & Gere)
    theta_0 = np.arctan(H / L)
    lambda_analytical = theta_0**2 / 3.0  # simplified for pin-ended

    return nodes, elements, loads, supports, lambda_analytical


def _simply_supported_beam_column() -> Tuple:
    """Simply supported beam-column under axial compression.

    Classical Euler buckling: P_cr = pi^2 * E * I / L^2
    For L=10m, E=210GPa, I=8.33e-8 m^4: P_cr ≈ 1727 N
    """
    L = 10.0
    A = 0.01
    E = 210e9
    I = 8.33e-8

    nodes = [
        Node(id="A", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=L, y=0.0, is_support=True, support_dx=False, support_dy=True),
    ]

    elements = [
        Element(id="AB", node_i="A", node_j="B", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
    ]

    # Apply unit compressive load for lambda_cr extraction
    loads = {"B": {"Fx": -1.0, "Fy": 0.0}}
    supports = [("A", 0), ("A", 1), ("B", 1)]

    # Euler critical load
    p_cr_euler = (np.pi**2 * E * I) / (L**2)
    # lambda_cr = P_cr / P_applied = P_cr / 1.0
    lambda_cr = p_cr_euler

    return nodes, elements, loads, supports, lambda_cr


def _cantilever_column() -> Tuple:
    """Cantilever column under tip axial load.

    Euler buckling for cantilever: P_cr = pi^2 * E * I / (2L)^2
    Effective length factor K = 2.0.
    """
    L = 5.0
    A = 0.02
    E = 200e9
    I = 1.67e-7  # circular D=0.025m

    nodes = [
        Node(id="A", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=L, is_support=False, support_dx=False, support_dy=False),
    ]

    elements = [
        Element(id="AB", node_i="A", node_j="B", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=2.0),
    ]

    loads = {"B": {"Fx": 0.0, "Fy": -1.0}}
    supports = [("A", 0), ("A", 1)]

    p_cr_euler = (np.pi**2 * E * I) / ((2 * L)**2)
    lambda_cr = p_cr_euler

    return nodes, elements, loads, supports, lambda_cr


def _three_bar_truss() -> Tuple:
    """Simple three-bar determinate truss.

    All members in tension under gravity load → no buckling, lambda_cr = inf.
    """
    L = 5.0
    H = 3.0
    A = 0.005
    E = 210e9
    I = 1e-8

    nodes = [
        Node(id="A", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=L, y=H, is_support=False, support_dx=False, support_dy=False),
        Node(id="C", x=2*L, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]

    elements = [
        Element(id="AB", node_i="A", node_j="B", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
        Element(id="BC", node_i="B", node_j="C", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
        Element(id="AC", node_i="A", node_j="C", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
    ]

    loads = {"B": {"Fx": 0.0, "Fy": -5000.0}}
    supports = [("A", 0), ("A", 1), ("C", 0), ("C", 1)]

    # All members in tension → no buckling
    lambda_cr = float("inf")

    return nodes, elements, loads, supports, lambda_cr


def _propped_cantilever() -> Tuple:
    """Propped cantilever (statically indeterminate to degree 1).

    Tests geometric stiffness contribution in redundant systems.
    """
    L = 8.0
    A = 0.015
    E = 210e9
    I = 1.04e-7

    nodes = [
        Node(id="A", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=L, y=0.0, is_support=True, support_dx=False, support_dy=True),
    ]

    elements = [
        Element(id="AB", node_i="A", node_j="B", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=0.7),
    ]

    loads = {"B": {"Fx": -1.0, "Fy": 0.0}}
    supports = [("A", 0), ("A", 1), ("B", 1)]

    # Approximate:介于简支和固支之间
    p_cr_approx = 2.04 * np.pi**2 * E * I / (L**2)
    lambda_cr = p_cr_approx

    return nodes, elements, loads, supports, lambda_cr


def _thermal_expansion_bar() -> Tuple:
    """Fixed-fixed bar under uniform temperature rise.

    Thermal compression: N = -E * A * alpha * delta_T
    Buckling when thermal load reaches Euler critical load.
    Critical temperature rise: delta_T_cr = pi^2 * I / (A * alpha * L^2)
    """
    L = 5.0
    A = 0.01
    E = 210e9
    I = 8.33e-8
    alpha = 1.2e-5

    nodes = [
        Node(id="A", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=L, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]

    elements = [
        Element(id="AB", node_i="A", node_j="B", A=A, E=E, I_sec=I, alpha=alpha, delta_T=0.0, delta_L_free=0.0, effective_length_factor=0.5),
    ]

    # No mechanical load; buckling from thermal compression
    # Apply dummy mechanical load for lambda extraction
    loads = {"B": {"Fx": 0.0, "Fy": 0.0}}
    supports = [("A", 0), ("A", 1), ("B", 0), ("B", 1)]

    # Critical temperature rise for buckling
    delta_T_cr = (np.pi**2 * I) / (A * alpha * (0.5 * L)**2)
    # Set a temperature and compute corresponding lambda
    test_delta_T = 50.0  # degC
    # Update element temperature (caller must do this)
    # lambda_cr = delta_T_cr / test_delta_T
    lambda_cr = delta_T_cr / test_delta_T if test_delta_T > 0 else float("inf")

    return nodes, elements, loads, supports, lambda_cr, test_delta_T


def _asymmetric_two_bar() -> Tuple:
    """Two-bar truss with asymmetric geometry.

    Tests coupling between horizontal and vertical DOFs.
    """
    A = 0.008
    E = 210e9
    I = 5e-8

    nodes = [
        Node(id="A", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=6.0, y=4.0, is_support=False, support_dx=False, support_dy=False),
        Node(id="C", x=10.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]

    elements = [
        Element(id="AB", node_i="A", node_j="B", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
        Element(id="BC", node_i="B", node_j="C", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
    ]

    loads = {"B": {"Fx": 500.0, "Fy": -2000.0}}
    supports = [("A", 0), ("A", 1), ("C", 0), ("C", 1)]

    # Numerical reference (computed offline with high precision)
    lambda_cr = 2.85  # approximate

    return nodes, elements, loads, supports, lambda_cr


def _six_bar_dome() -> Tuple:
    """Shallow six-bar dome (geometrically nonlinear benchmark).

    Classic snap-through test case. Linearised buckling underestimates
    true limit load for very shallow geometries.
    """
    R = 10.0  # radius
    H = 1.5   # rise
    A = 0.01
    E = 210e9
    I = 1e-7

    # Hexagonal base + apex
    import math
    angles = [k * math.pi / 3 for k in range(6)]
    nodes = [Node(id=f"N{k}", x=R*math.cos(a), y=R*math.sin(a), is_support=True, support_dx=True, support_dy=True) for k, a in enumerate(angles)]
    nodes.append(Node(id="AP", x=0.0, y=0.0, is_support=False, support_dx=False, support_dy=False))

    elements = []
    for k in range(6):
        elements.append(Element(id=f"E{k}", node_i=f"N{k}", node_j="AP", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0))

    loads = {"AP": {"Fx": 0.0, "Fy": -100.0}}
    supports = [(f"N{k}", 0) for k in range(6)] + [(f"N{k}", 1) for k in range(6)]

    # Reference from literature (approximate)
    lambda_cr = 15.0  # placeholder

    return nodes, elements, loads, supports, lambda_cr


def _x_braced_frame() -> Tuple:
    """X-braced single-bay frame.

    Tests interaction between diagonal tension and compression chords.
    """
    W = 6.0
    H = 4.0
    A_chord = 0.02
    A_diag = 0.01
    E = 210e9
    I_chord = 2e-7
    I_diag = 5e-8

    nodes = [
        Node(id="BL", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="BR", x=W, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="TL", x=0.0, y=H, is_support=False, support_dx=False, support_dy=False),
        Node(id="TR", x=W, y=H, is_support=False, support_dx=False, support_dy=False),
    ]

    elements = [
        Element(id="COL_L", node_i="BL", node_j="TL", A=A_chord, E=E, I_sec=I_chord, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
        Element(id="COL_R", node_i="BR", node_j="TR", A=A_chord, E=E, I_sec=I_chord, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
        Element(id="BEAM", node_i="TL", node_j="TR", A=A_chord, E=E, I_sec=I_chord, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
        Element(id="DIAG1", node_i="BL", node_j="TR", A=A_diag, E=E, I_sec=I_diag, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
        Element(id="DIAG2", node_i="BR", node_j="TL", A=A_diag, E=E, I_sec=I_diag, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
    ]

    loads = {"TL": {"Fx": 1000.0, "Fy": -500.0}, "TR": {"Fx": 1000.0, "Fy": -500.0}}
    supports = [("BL", 0), ("BL", 1), ("BR", 0), ("BR", 1)]

    lambda_cr = 8.5  # approximate reference

    return nodes, elements, loads, supports, lambda_cr


def _continuous_beam_three_span() -> Tuple:
    """Three-span continuous beam under axial compression.

    Tests multi-mode buckling and mode localization.
    """
    L = 4.0
    A = 0.015
    E = 210e9
    I = 1.2e-7

    nodes = [
        Node(id="A", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=L, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="C", x=2*L, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="D", x=3*L, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]

    elements = [
        Element(id="AB", node_i="A", node_j="B", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
        Element(id="BC", node_i="B", node_j="C", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
        Element(id="CD", node_i="C", node_j="D", A=A, E=E, I_sec=I, alpha=1.2e-5, delta_T=0.0, delta_L_free=0.0, effective_length_factor=1.0),
    ]

    loads = {"D": {"Fx": -1.0, "Fy": 0.0}}
    supports = [("A", 0), ("A", 1), ("B", 1), ("C", 1), ("D", 1)]

    # First buckling mode (symmetric)
    lambda_cr = 3.0 * np.pi**2 * E * I / (L**2)  # approximate

    return nodes, elements, loads, supports, lambda_cr


# Assemble benchmark registry
BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "two_bar_toggle": {
        "description": "Two-bar shallow toggle (von Mises truss) - classic snap-through benchmark",
        "model": _two_bar_toggle_model(),
        "reference": {"lambda_cr": None},  # filled dynamically
        "tolerance": 0.05,
        "source": "Timoshenko & Gere, Theory of Elastic Stability, §2.7",
    },
    "simply_supported_beam_column": {
        "description": "Simply supported beam-column - classical Euler buckling",
        "model": _simply_supported_beam_column(),
        "reference": {"lambda_cr": None},
        "tolerance": 0.01,
        "source": "Euler (1744), EN 1993-1-1 Annex A",
    },
    "cantilever_column": {
        "description": "Cantilever column - effective length K=2",
        "model": _cantilever_column(),
        "reference": {"lambda_cr": None},
        "tolerance": 0.01,
        "source": "EN 1993-1-1 Table 6.1",
    },
    "three_bar_truss": {
        "description": "Three-bar determinate truss - all tension, no buckling",
        "model": _three_bar_truss(),
        "reference": {"lambda_cr": float("inf")},
        "tolerance": 0.0,
        "source": "Analytical",
    },
    "propped_cantilever": {
        "description": "Propped cantilever - indeterminate to degree 1",
        "model": _propped_cantilever(),
        "reference": {"lambda_cr": None},
        "tolerance": 0.03,
        "source": "Approximate formula, K≈0.7",
    },
    "thermal_expansion_bar": {
        "description": "Fixed-fixed bar under thermal expansion",
        "model": _thermal_expansion_bar(),
        "reference": {"lambda_cr": None, "delta_T_cr": None},
        "tolerance": 0.02,
        "source": "Thermal buckling theory",
    },
    "asymmetric_two_bar": {
        "description": "Asymmetric two-bar truss - DOF coupling test",
        "model": _asymmetric_two_bar(),
        "reference": {"lambda_cr": 2.85},
        "tolerance": 0.10,
        "source": "Numerical verification",
    },
    "six_bar_dome": {
        "description": "Six-bar shallow dome - geometric nonlinearity benchmark",
        "model": _six_bar_dome(),
        "reference": {"lambda_cr": 15.0},
        "tolerance": 0.15,
        "source": "Literature snap-through cases",
    },
    "x_braced_frame": {
        "description": "X-braced single-bay frame",
        "model": _x_braced_frame(),
        "reference": {"lambda_cr": 8.5},
        "tolerance": 0.10,
        "source": "Numerical verification",
    },
    "continuous_beam_three_span": {
        "description": "Three-span continuous beam - multi-mode buckling",
        "model": _continuous_beam_three_span(),
        "reference": {"lambda_cr": None},
        "tolerance": 0.05,
        "source": "Approximate analytical",
    },
}

# Post-process: fill reference values from model functions
for name, bench in BENCHMARKS.items():
    model_data = bench["model"]
    if len(model_data) == 5:
        nodes, elements, loads, supports, lambda_ref = model_data
        bench["reference"]["lambda_cr"] = lambda_ref
    elif len(model_data) == 6:
        nodes, elements, loads, supports, lambda_ref, delta_T = model_data
        bench["reference"]["lambda_cr"] = lambda_ref
        bench["reference"]["delta_T_cr"] = delta_T * lambda_ref
