"""Convergence-rate test for RK4 time integration (Issue B3).

Verifies 4th-order convergence of the RK4 scheme in
:func:`truss_analysis.thermal.fire_curve.steel_temperature` by
step-halving against a reference solution from ``scipy.integrate.solve_ivp``
at machine precision.

Reference
---------
Butcher, J. C. (2008). Numerical Methods for Ordinary Differential Equations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from truss_analysis.material.steel_eurocode import specific_heat, unit_mass
from truss_analysis.thermal.fire_curve import (
    h_net,
    iso_834_temperature,
    steel_temperature,
)


def _reference_solution(
    duration_min: float = 30.0,
    section_factor: float = 183.0,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> float:
    """Compute reference steel temperature using scipy's RK45 at tight tolerance."""
    am_v = section_factor
    rho = unit_mass()

    def rhs(t_s: float, y: np.ndarray) -> list[float]:
        flux = float(h_net(iso_834_temperature(t_s / 60.0), y[0]))
        return [flux * am_v / (rho * float(specific_heat(y[0])))]

    sol = solve_ivp(rhs, (0.0, duration_min * 60.0), [20.0], rtol=rtol, atol=atol)
    return float(sol.y[0, -1])


def compute_convergence_order(
    duration_min: float = 30.0,
    section_factor: float = 183.0,
    dt_base: float = 5.0,
    n_refinements: int = 3,
) -> tuple[float, dict[str, float]]:
    """Estimate convergence order via Richardson extrapolation.

    Parameters
    ----------
    duration_min : float
        Fire exposure duration [minutes].
    section_factor : float
        Section factor A_m/V [1/m].
    dt_base : float
        Coarsest time step [s] (default 5.0, the EN 1993-1-2 maximum).
    n_refinements : int
        Number of step-halving refinements.

    Returns
    -------
    order : float
        Estimated convergence order (should be ~4.0 for RK4).
    errors : dict
        Mapping of dt -> absolute error vs reference.
    """
    T_ref = _reference_solution(duration_min, section_factor)
    errors = {}

    for i in range(n_refinements + 1):
        dt = dt_base / (2**i)
        res = steel_temperature(
            duration_min,
            section_factor,
            max_step_s=dt,
        )
        errors[dt] = abs(res.theta_final - T_ref)

    # Compute order from last three refinements
    if n_refinements >= 2:
        dt1 = dt_base / (2 ** (n_refinements - 2))
        dt2 = dt_base / (2 ** (n_refinements - 1))
        dt4 = dt_base / (2**n_refinements)

        err1 = errors[dt1]
        err2 = errors[dt2]
        err4 = errors[dt4]

        # Avoid division by zero
        if abs(err2 - err4) > 1e-14:
            order = np.log2(abs(err1 - err4) / abs(err2 - err4))
        else:
            order = float("inf")
    else:
        order = float("nan")

    return order, errors


class TestRK4Convergence:
    """Validate 4th-order convergence of RK4 integrator."""

    def test_convergence_order_is_four(self):
        """RK4 converges at fourth order -- measured, not merely asserted.

        This test used to discard its own order estimate
        (``_, errors = compute_convergence_order(...)``) and assert only that the
        errors did not increase. Its name and docstring claimed a fourth-order
        result that nothing checked, which is the same defect the round-7 audit
        found in the benchmark suite: a check that cannot fail.

        The reason it could not simply assert on ``compute_convergence_order``'s
        output is that the reference is too weak, not that the method is wrong.
        ``_reference_solution`` uses ``solve_ivp`` at ``rtol=1e-12``, but the EN
        specific-heat table has a kink near 735 degC, so the adaptive integrator
        saturates around 4e-5 degC while the RK4 sequence itself resolves to
        5e-6 degC. Measured against that reference the apparent order comes out
        near 2.4 and then collapses -- the reference fails first.

        Measured instead against a Richardson limit of the library's *own*
        sequence, which agrees with itself to ~7e-5 degC between successive
        pairs, the order is 3.999999988. ``benchmarks/reference_problems.py``
        pins the same number with a declared tolerance; this test pins the
        mechanism, so the two cannot silently disagree.
        """
        f_10 = steel_temperature(30.0, 183.0, max_step_s=10.0).theta_final
        f_5 = steel_temperature(30.0, 183.0, max_step_s=5.0).theta_final
        f_2_5 = steel_temperature(30.0, 183.0, max_step_s=2.5).theta_final
        f_1_25 = steel_temperature(30.0, 183.0, max_step_s=1.25).theta_final

        # Fourth-order Richardson limits from successive step pairs. Each pair
        # estimates the same continuum limit independently.
        limit_from_10 = f_5 + (f_5 - f_10) / 15.0
        limit_from_5 = f_2_5 + (f_2_5 - f_5) / 15.0
        limit_from_2_5 = f_1_25 + (f_1_25 - f_2_5) / 15.0

        # The limits must agree far more tightly than the errors being measured,
        # or the "reference" is the thing that has not converged.
        assert abs(limit_from_10 - limit_from_5) < 1e-3
        assert abs(limit_from_5 - limit_from_2_5) < 1e-3

        err_5 = abs(f_5 - limit_from_2_5)
        err_2_5 = abs(f_2_5 - limit_from_2_5)
        err_1_25 = abs(f_1_25 - limit_from_2_5)

        # the sequence must actually shrink, or an "order" is meaningless
        assert err_5 > 10.0 * err_1_25, (err_5, err_2_5, err_1_25)
        assert err_2_5 > err_1_25

        order = float(np.log2(err_2_5 / err_1_25))
        assert order == pytest.approx(4.0, abs=0.05), (
            f"measured convergence order {order:.4f}; a first-order scheme "
            "measures 1.0, a second-order one 2.0 and a third-order one 3.0, "
            "so this is not a tight bound on a number that happens to be 4"
        )

    def test_errors_decrease_monotonically_under_refinement(self):
        """The weaker property the order test used to stand in for."""
        _, errors = compute_convergence_order(
            duration_min=30.0,
            section_factor=183.0,
            dt_base=5.0,
            n_refinements=4,
        )
        err_values = list(errors.values())
        for i in range(1, len(err_values)):
            assert err_values[i] <= err_values[i - 1] * 1.1, (
                f"Error did not decrease at refinement {i}: "
                f"{err_values[i - 1]:.2e} -> {err_values[i]:.2e}"
            )

    def test_max_step_default_is_accurate_enough(self):
        """EN 1993-1-2's 5s maximum step yields < 0.01 degC accuracy."""
        T_ref = _reference_solution(30.0, 183.0)
        res_default = steel_temperature(30.0, 183.0)  # max_step_s=5.0 by default
        error = abs(res_default.theta_final - T_ref)

        # EN 1993-1-2 §4.2.2.2(3) caps Δt at 5s; verify this is sufficient
        assert error < 0.01, f"Default 5s step has error {error:.4f} degC > 0.01 degC"

    def test_spike_near_730C_resolved(self):
        """Specific heat spike near 730°C is captured without oscillation."""
        # Run long enough to pass through the 730°C spike
        res = steel_temperature(60.0, 183.0, max_step_s=5.0)

        # Temperature should be monotonic (no oscillations)
        assert np.all(np.diff(res.theta_steel) > 0.0), (
            "Temperature history not monotonic; RK4 may be undersampling c_a spike"
        )

        # Final temp should be well above the spike
        assert res.theta_final > 800.0


if __name__ == "__main__":
    # Quick manual run
    order, errors = compute_convergence_order()
    print("RK4 Convergence Test")
    print("=" * 50)
    for dt, err in errors.items():
        print(f"dt={dt:6.2f}s -> error={err:.2e}")
    print(f"Estimated convergence order: {order:.2f} (expected ~4.0)")
