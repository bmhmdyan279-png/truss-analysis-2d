from __future__ import annotations

import numpy as np
import pytest
from truss_analysis.exceptions import EnergyValidationError, SingularMatrixError
from truss_analysis.solver import check_energy, solve


def test_solve_simple():
    k = np.array([[2.0, -1.0], [-1.0, 2.0]])
    f = np.array([10.0, 0.0])
    fixed = [1]
    u = solve(k, f, fixed)
    assert abs(u[0] - 5.0) < 1e-6
    assert u[1] == 0.0


def test_singular_matrix_raises_truss_3001():
    k = np.zeros((4, 4))
    f = np.array([10.0, 0.0, 0.0, 0.0])
    fixed = [2, 3]
    with pytest.raises(SingularMatrixError):
        solve(k, f, fixed)


def test_check_energy_pass_no_thermal():
    u = np.array([1.0, 0.0])
    f_mech = np.array([2.0, 0.0])
    strain_energy = 1.0
    prestress_work = 0.0
    assert check_energy(u, f_mech, strain_energy, prestress_work) is True


def test_check_energy_pass_with_thermal():
    """Check energy balance with thermal loads.
    Formula: W_mech = strain_energy + 0.5 * prestress_work
    W_mech = 0.5 * u^T f_mech = 0.5 * 2 * 4 = 4.0
    With prestress=1.0: strain must be 3.5
    """
    u = np.array([2.0, 0.0])
    f_mech = np.array([4.0, 0.0])
    # W_mech = 0.5 * 2 * 4 = 4.0
    # Formula: 4.0 = strain + 0.5 * 1.0 => strain = 3.5
    assert check_energy(u, f_mech, 3.5, 1.0) is True


def test_check_energy_fail():
    u = np.array([1.0, 0.0])
    f_mech = np.array([2.0, 0.0])
    with pytest.raises(EnergyValidationError):
        check_energy(u, f_mech, 5.0, 0.0)
