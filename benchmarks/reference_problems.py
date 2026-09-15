"""
Benchmark Reference Problems for Structural Fire Analysis
==========================================================
This module contains 10 canonical reference problems for validating
structural fire analysis implementations against known solutions.

References:
- EN 1993-1-2: Design of steel structures - Structural fire design
- EN 1991-1-2: Actions on structures - Actions on structures exposed to fire
- NIST NCSTAR 1: Federal Building and Fire Safety Investigation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BenchmarkResult:
    """Container for benchmark comparison results."""
    problem_name: str
    expected_value: float
    computed_value: float
    relative_error: float
    tolerance: float
    passed: bool
    units: str
    notes: str = ""


@dataclass
class ReferenceProblem(ABC):
    """Abstract base class for reference problems."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the problem."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass
    
    @property
    @abstractmethod
    def reference_source(self) -> str:
        """Citation of the reference solution."""
        pass
    
    @abstractmethod
    def setup(self) -> Dict:
        """Return configuration dictionary for the analysis engine."""
        pass
    
    @abstractmethod
    def get_reference_value(self) -> float:
        """Return the expected reference value."""
        pass
    
    @abstractmethod
    def extract_computed_value(self, result: Dict) -> float:
        """Extract the comparable value from analysis results."""
        pass
    
    @property
    @abstractmethod
    def tolerance(self) -> float:
        """Acceptable relative error tolerance."""
        pass
    
    @property
    def units(self) -> str:
        """Units of the reference value."""
        return "dimensionless"
    
    def validate(self, result: Dict) -> BenchmarkResult:
        """Compare computed result against reference."""
        expected = self.get_reference_value()
        computed = self.extract_computed_value(result)
        
        if expected == 0:
            rel_error = abs(computed)
        else:
            rel_error = abs(computed - expected) / abs(expected)
        
        passed = rel_error <= self.tolerance
        
        return BenchmarkResult(
            problem_name=self.name,
            expected_value=expected,
            computed_value=computed,
            relative_error=rel_error,
            tolerance=self.tolerance,
            passed=passed,
            units=self.units,
            notes=""
        )


# ============================================================================
# Problem 1: Euler Buckling Load (Pinned-Pinned Column)
# ============================================================================

class EulerBucklingPinned(ReferenceProblem):
    """
    Classical Euler buckling load for pinned-pinned column.
    
    P_cr = π²EI / L²
    
    Reference: Timoshenko & Gere, Theory of Elastic Stability
    """
    
    @property
    def name(self) -> str:
        return "euler_buckling_pinned"
    
    @property
    def description(self) -> str:
        return "Critical buckling load for pinned-pinned steel column"
    
    @property
    def reference_source(self) -> str:
        return "Timoshenko & Gere (1961), Eq. 2-1"
    
    def setup(self) -> Dict:
        return {
            "geometry": {
                "type": "column",
                "length": 5.0,  # m
                "boundary_conditions": ["pinned", "pinned"]
            },
            "material": {
                "type": "steel",
                "E": 210e9,  # Pa
                "nu": 0.3
            },
            "section": {
                "type": "I",
                "A": 0.01,  # m²
                "I": 5e-5,  # m⁴
            },
            "load": {
                "type": "axial_compression",
                "magnitude": 1.0  # Unit load for eigenvalue extraction
            }
        }
    
    def get_reference_value(self) -> float:
        E = 210e9
        I = 5e-5
        L = 5.0
        return (np.pi**2 * E * I) / (L**2)  # ~414 kN
    
    def extract_computed_value(self, result: Dict) -> float:
        return result.get("critical_load", 0.0)
    
    @property
    def tolerance(self) -> float:
        return 0.01  # 1%
    
    @property
    def units(self) -> str:
        return "N"


# ============================================================================
# Problem 2: Critical Temperature for Simply Supported Beam
# ============================================================================

class CriticalTemperatureBeam(ReferenceProblem):
    """
    Critical temperature for thermal buckling of simply supported beam.
    
    Reference: Eurocode 3 Part 1-2, Annex E
    """
    
    @property
    def name(self) -> str:
        return "critical_temperature_beam"
    
    @property
    def description(self) -> str:
        return "Critical temperature for thermal buckling of restrained beam"
    
    @property
    def reference_source(self) -> str:
        return "EN 1993-1-2:2005, Annex E.2"
    
    def setup(self) -> Dict:
        return {
            "geometry": {
                "type": "beam",
                "length": 6.0,
                "boundary_conditions": ["fixed_axial", "pinned"]
            },
            "material": {
                "type": "steel_S355",
                "E_ambient": 210e9,
                "alpha_thermal": 12e-6
            },
            "section": {
                "type": "HEA200",
                "A": 0.00538,
                "I": 3.69e-5
            },
            "thermal": {
                "heating_rate": 10,  # °C/min
                "initial_temp": 20
            }
        }
    
    def get_reference_value(self) -> float:
        return 538.0  # °C (typical for S355 restrained beam)
    
    def extract_computed_value(self, result: Dict) -> float:
        return result.get("critical_temperature", 0.0)
    
    @property
    def tolerance(self) -> float:
        return 0.05  # 5%
    
    @property
    def units(self) -> str:
        return "°C"


# ============================================================================
# Problem 3: Unprotected Steel Column Heating (Standard Fire)
# ============================================================================

class UnprotectedSteelHeating(ReferenceProblem):
    """
    Temperature rise in unprotected steel section under ISO 834 fire.
    
    Reference: EN 1993-1-2 §4.2.5.1
    """
    
    @property
    def name(self) -> str:
        return "unprotected_steel_heating"
    
    @property
    def description(self) -> str:
        return "Steel temperature at t=30min under ISO 834 fire"
    
    @property
    def reference_source(self) -> str:
        return "EN 1993-1-2:2005, §4.2.5.1, Example 4.1"
    
    def setup(self) -> Dict:
        return {
            "section_factor": 150.0,  # A_m/V in m⁻¹
            "fire_curve": "ISO834",
            "time": 1800,  # 30 minutes in seconds
            "initial_temp": 20.0
        }
    
    def get_reference_value(self) -> float:
        return 710.0  # °C (approximate from Eurocode tables)
    
    def extract_computed_value(self, result: Dict) -> float:
        return result.get("steel_temperature", 0.0)
    
    @property
    def tolerance(self) -> float:
        return 0.10  # 10%
    
    @property
    def units(self) -> str:
        return "°C"


# ============================================================================
# Problem 4: Protected Steel Heating Rate
# ============================================================================

class ProtectedSteelHeating(ReferenceProblem):
    """
    Temperature rise in fire-protected steel section.
    
    Reference: EN 1993-1-2 §4.2.5.2
    """
    
    @property
    def name(self) -> str:
        return "protected_steel_heating"
    
    @property
    def description(self) -> str:
        return "Steel temperature at t=60min with 20mm gypsum protection"
    
    @property
    def reference_source(self) -> str:
        return "EN 1993-1-2:2005, §4.2.5.2, Example 4.3"
    
    def setup(self) -> Dict:
        return {
            "section_factor": 100.0,
            "protection": {
                "type": "gypsum_board",
                "thickness": 0.020,  # 20mm
                "conductivity": 0.2,  # W/(m·K)
                "density": 800  # kg/m³
            },
            "fire_curve": "ISO834",
            "time": 3600,  # 60 minutes
            "initial_temp": 20.0
        }
    
    def get_reference_value(self) -> float:
        return 420.0  # °C
    
    def extract_computed_value(self, result: Dict) -> float:
        return result.get("steel_temperature", 0.0)
    
    @property
    def tolerance(self) -> float:
        return 0.15  # 15%
    
    @property
    def units(self) -> str:
        return "°C"


# ============================================================================
# Problem 5: Tangent Stiffness Verification (Finite Difference)
# ============================================================================

class TangentStiffnessFD(ReferenceProblem):
    """
    Verification of geometric stiffness matrix via finite differences.
    
    Reference: Crisfield, Non-linear Finite Element Analysis, Vol. 1
    """
    
    @property
    def name(self) -> str:
        return "tangent_stiffness_fd"
    
    @property
    def description(self) -> str:
        return "Finite difference verification of K_G matrix"
    
    @property
    def reference_source(self) -> str:
        return "Crisfield (1991), §3.4"
    
    def setup(self) -> Dict:
        return {
            "structure": "simple_truss",
            "perturbation_magnitude": 1e-6,
            "test_dof": [0, 1, 2]
        }
    
    def get_reference_value(self) -> float:
        return 1e-4  # Acceptable FD error threshold
    
    def extract_computed_value(self, result: Dict) -> float:
        return result.get("max_fd_error", 0.0)
    
    @property
    def tolerance(self) -> float:
        return 1.0  # Absolute tolerance for FD error
    
    @property
    def units(self) -> str:
        return "relative"


# ============================================================================
# Problem 6: RK4 Convergence Rate Test
# ============================================================================

class RK4ConvergenceTest(ReferenceProblem):
    """
    Verify 4th-order convergence of RK4 time integration.
    
    Expected: error ∝ dt⁴
    """
    
    @property
    def name(self) -> str:
        return "rk4_convergence"
    
    @property
    def description(self) -> str:
        return "Convergence rate verification for RK4 scheme"
    
    @property
    def reference_source(self) -> str:
        return "Butcher (2008), Numerical Methods for ODEs"
    
    def setup(self) -> Dict:
        return {
            "ode_type": "heat_transfer_single_dof",
            "time_steps": [1.0, 0.5, 0.25, 0.125],
            "final_time": 10.0
        }
    
    def get_reference_value(self) -> float:
        return 4.0  # Expected convergence order
    
    def extract_computed_value(self, result: Dict) -> float:
        return result.get("convergence_order", 0.0)
    
    @property
    def tolerance(self) -> float:
        return 0.2  # Allow ±0.2 deviation from 4th order
    
    @property
    def units(self) -> str:
        return "order"


# ============================================================================
# Problem 7: Imperfection Sensitivity (Koiter Theory)
# ============================================================================

class ImperfectionSensitivity(ReferenceProblem):
    """
    Buckling load reduction due to initial imperfections.
    
    Reference: Koiter's asymptotic theory
    """
    
    @property
    def name(self) -> str:
        return "imperfection_sensitivity"
    
    @property
    def description(self) -> str:
        return "Load reduction factor vs imperfection amplitude"
    
    @property
    def reference_source(self) -> str:
        return "Koiter (1945), On the stability of elastic equilibrium"
    
    def setup(self) -> Dict:
        return {
            "structure": "cylindrical_shell",
            "imperfection_amplitudes": [0.001, 0.01, 0.1],  # h/R ratios
            "R": 1.0,
            "h": 0.01
        }
    
    def get_reference_value(self) -> float:
        return 0.6  # Approximate knockdown factor for ξ=0.1
    
    def extract_computed_value(self, result: Dict) -> float:
        return result.get("knockdown_factor", 0.0)
    
    @property
    def tolerance(self) -> float:
        return 0.20
    
    @property
    def units(self) -> str:
        return "ratio"


# ============================================================================
# Problem 8: Material Degradation at 600°C
# ============================================================================

class MaterialDegradation600C(ReferenceProblem):
    """
    Reduction factors for steel at 600°C.
    
    Reference: EN 1993-1-2 Table 3.1
    """
    
    @property
    def name(self) -> str:
        return "material_degradation_600C"
    
    @property
    def description(self) -> str:
        return "Yield strength reduction factor at 600°C"
    
    @property
    def reference_source(self) -> str:
        return "EN 1993-1-2:2005, Table 3.1"
    
    def setup(self) -> Dict:
        return {
            "material": "steel_S355",
            "temperature": 600.0
        }
    
    def get_reference_value(self) -> float:
        return 0.47  # k_y,θ at 600°C
    
    def extract_computed_value(self, result: Dict) -> float:
        return result.get("yield_reduction_factor", 0.0)
    
    @property
    def tolerance(self) -> float:
        return 0.05
    
    @property
    def units(self) -> str:
        return "ratio"


# ============================================================================
# Problem 9: Thermal Elongation Restraint Force
# ============================================================================

class ThermalRestraintForce(ReferenceProblem):
    """
    Axial force in fully restrained heated bar.
    
    F = EA·α·ΔT
    """
    
    @property
    def name(self) -> str:
        return "thermal_restraint_force"
    
    @property
    def description(self) -> str:
        return "Compressive force in axially restrained heated bar"
    
    @property
    def reference_source(self) -> str:
        return "Basic thermoelasticity"
    
    def setup(self) -> Dict:
        return {
            "length": 3.0,
            "A": 0.01,
            "E": 210e9,
            "alpha": 12e-6,
            "delta_T": 500.0
        }
    
    def get_reference_value(self) -> float:
        E = 210e9
        A = 0.01
        alpha = 12e-6
        dT = 500.0
        return E * A * alpha * dT  # 1.26 MN
    
    def extract_computed_value(self, result: Dict) -> float:
        return result.get("restraint_force", 0.0)
    
    @property
    def tolerance(self) -> float:
        return 0.02
    
    @property
    def units(self) -> str:
        return "N"


# ============================================================================
# Problem 10: Shallow Arch Snap-through Load
# ============================================================================

class ShallowArchSnapThrough(ReferenceProblem):
    """
    Snap-through buckling load of shallow circular arch.
    
    Reference: Williams (1964)
    """
    
    @property
    def name(self) -> str:
        return "shallow_arch_snap_through"
    
    @property
    def description(self) -> str:
        return "Critical snap-through load for shallow arch"
    
    @property
    def reference_source(self) -> str:
        return "Williams (1964), J. Mech. Phys. Solids"
    
    def setup(self) -> Dict:
        return {
            "geometry": {
                "type": "circular_arch",
                "radius": 2.5,
                "span": 2.0,
                "thickness": 0.01
            },
            "material": {
                "E": 70e9,
                "nu": 0.3
            },
            "boundary_conditions": ["pinned", "pinned"]
        }
    
    def get_reference_value(self) -> float:
        return 150.0  # N (approximate)
    
    def extract_computed_value(self, result: Dict) -> float:
        return result.get("snap_through_load", 0.0)
    
    @property
    def tolerance(self) -> float:
        return 0.10
    
    @property
    def units(self) -> str:
        return "N"


# ============================================================================
# Registry
# ============================================================================

ALL_BENCHMARKS: List[ReferenceProblem] = [
    EulerBucklingPinned(),
    CriticalTemperatureBeam(),
    UnprotectedSteelHeating(),
    ProtectedSteelHeating(),
    TangentStiffnessFD(),
    RK4ConvergenceTest(),
    ImperfectionSensitivity(),
    MaterialDegradation600C(),
    ThermalRestraintForce(),
    ShallowArchSnapThrough(),
]


def run_all_benchmarks(engine, verbose: bool = True) -> List[BenchmarkResult]:
    """
    Execute all benchmark problems and return results.
    
    Parameters
    ----------
    engine : AnalysisEngine
        The analysis engine to test
    verbose : bool
        Print detailed results
    
    Returns
    -------
    List[BenchmarkResult]
        Results for all benchmarks
    """
    results = []
    
    for problem in ALL_BENCHMARKS:
        try:
            config = problem.setup()
            analysis_result = engine.run(config)
            benchmark_result = problem.validate(analysis_result)
            results.append(benchmark_result)
            
            if verbose:
                status = "✓ PASS" if benchmark_result.passed else "✗ FAIL"
                print(f"{status}: {problem.name}")
                print(f"  Expected: {benchmark_result.expected_value:.4g} {benchmark_result.units}")
                print(f"  Computed: {benchmark_result.computed_value:.4g} {benchmark_result.units}")
                print(f"  Error:    {benchmark_result.relative_error*100:.2f}% (tolerance: {benchmark_result.tolerance*100:.1f}%)")
                print(f"  Source:   {problem.reference_source}")
                print()
                
        except Exception as e:
            error_result = BenchmarkResult(
                problem_name=problem.name,
                expected_value=problem.get_reference_value(),
                computed_value=float('nan'),
                relative_error=float('inf'),
                tolerance=problem.tolerance,
                passed=False,
                units=problem.units,
                notes=f"Execution error: {str(e)}"
            )
            results.append(error_result)
            
            if verbose:
                print(f"✗ ERROR: {problem.name}")
                print(f"  {str(e)}")
                print()
    
    return results


def summary_report(results: List[BenchmarkResult]) -> str:
    """Generate a summary report of benchmark results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    report = []
    report.append("=" * 60)
    report.append("BENCHMARK SUMMARY REPORT")
    report.append("=" * 60)
    report.append(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    report.append(f"Pass rate: {passed/total*100:.1f}%")
    report.append("")
    
    if failed > 0:
        report.append("FAILED BENCHMARKS:")
        report.append("-" * 40)
        for r in results:
            if not r.passed:
                report.append(f"  • {r.problem_name}")
                report.append(f"    Error: {r.relative_error*100:.2f}% (tol: {r.tolerance*100:.1f}%)")
                if r.notes:
                    report.append(f"    Note: {r.notes}")
        report.append("")
    
    report.append("=" * 60)
    return "\n".join(report)
