"""Regression tests for the self-equilibrated (imposed-strain) energy balance.

The generalized Clapeyron identity implemented by
:func:`truss_analysis.solver.check_energy` is

    ``W_mech = U_strain + 0.5 * W_prestress``.

For a *self-equilibrated* problem (``F_mechanical = 0``, e.g. a fully
restrained bar heated by ``delta_T``) this reduces to
``U_strain = -0.5 * W_prestress``, which is generally **non-zero**. An
earlier revision short-circuited that case and demanded ``U_strain ~ 0``,
so the textbook thermal-stress problem raised ``EnergyValidationError``
out of :func:`truss_analysis.main.run`. These tests pin the corrected
behaviour with closed-form values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.exceptions import EnergyValidationError
from truss_analysis.main import run
from truss_analysis.model import Element, Node
from truss_analysis.postprocess import calculate_element_forces, imposed_strain_energy
from truss_analysis.solver import check_energy, solve

# Bar geometry / properties used throughout: L = 2 m, E = 210 GPa, A = 0.01 m^2
_L = 2.0
_E = 210e9
_A = 0.01
_ALPHA = 1.2e-5
_DELTA_T = 100.0
_K = _E * _A / _L


def _restrained_bar(delta_T: float = _DELTA_T) -> tuple[list[Node], list[Element]]:
    """Both ends fully fixed: the structure cannot relieve thermal strain."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=_L, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements = [
        Element(
            id="e1",
            node_i="1",
            node_j="2",
            E=_E,
            A=_A,
            alpha=_ALPHA,
            delta_T=delta_T,
        )
    ]
    return nodes, elements


def test_restrained_thermal_balance_closes_with_nonzero_strain_energy() -> None:
    """U_strain is large, W_mech is zero, and the balance still holds."""
    nodes, elements = _restrained_bar()
    K, F_ext, F_mechanical, fixed = assemble_global_matrices(nodes, elements)
    U = solve(K, F_ext, fixed)

    # Fully restrained => no displacement at all.
    assert np.allclose(U, 0.0, atol=1e-15)
    # No mechanical load was applied.
    assert np.allclose(F_mechanical, 0.0)

    _, strain_energy, prestress_work = calculate_element_forces(nodes, elements, U)

    # Closed forms: delta_L_thermal = alpha*dT*L, delta_L_mech = -delta_L_thermal
    delta_l = _ALPHA * _DELTA_T * _L
    assert strain_energy == pytest.approx(0.5 * _K * delta_l**2, rel=1e-12)
    assert prestress_work == pytest.approx(-_K * delta_l**2, rel=1e-12)

    # The two must cancel exactly: U_strain + 0.5*W_prestress == 0 == W_mech.
    assert strain_energy + 0.5 * prestress_work == pytest.approx(0.0, abs=1e-9)
    assert check_energy(U, F_mechanical, strain_energy, prestress_work) is True


def test_run_accepts_pure_thermal_load_on_indeterminate_model(tmp_path: Path) -> None:
    """End-to-end: the pipeline must not reject a restrained thermal state."""
    payload = {
        "units": "SI",
        "nodes": [
            {
                "id": "1",
                "x": 0.0,
                "y": 0.0,
                "is_support": True,
                "support_dx": True,
                "support_dy": True,
            },
            {
                "id": "2",
                "x": 1.0,
                "y": 0.0,
                "is_support": True,
                "support_dx": True,
                "support_dy": True,
            },
        ],
        "elements": [
            {
                "id": "e1",
                "node_i": "1",
                "node_j": "2",
                "E": 210e9,
                "A": 0.01,
                "alpha": 1.2e-5,
                "delta_T": 100.0,
            }
        ],
        "loads": [],
    }
    model_path = tmp_path / "restrained_thermal.json"
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run(str(model_path), quiet=True)
    assert result.status == "SUCCESS"

    # Thermal stress is real even though nothing moves: sigma = -E*alpha*dT
    force = result.element_forces[0]
    expected_force = -_E * _ALPHA * 100.0 * _A
    assert force["N"] == pytest.approx(expected_force, rel=1e-10)
    assert force["status"] == "Compression"


def test_genuinely_inconsistent_self_equilibrated_state_is_rejected() -> None:
    """The fix must not turn the guard into a no-op.

    With ``W_mech = 0`` a non-zero strain energy that is *not* balanced by
    the prestress term is a real inconsistency and must still raise.
    """
    U = np.zeros(2)
    F_mech = np.zeros(2)
    with pytest.raises(EnergyValidationError, match="Energy balance failed"):
        check_energy(U, F_mech, strain_energy=100.0, prestress_work=0.0)


def test_trivial_zero_energy_problem_passes() -> None:
    """No loads, no imposed strain: scale is zero and the balance is vacuous."""
    U = np.zeros(4)
    F_mech = np.zeros(4)
    assert check_energy(U, F_mech, 0.0, 0.0) is True


def test_energy_balance_is_scale_invariant() -> None:
    """With a supplied ``energy_scale`` the verdict does not depend on size.

    An absolute joule threshold is permissive for small models and
    over-strict for large ones. Supplying the characteristic imposed-strain
    energy makes the round-off floor track the problem instead of a constant,
    so a 1% imbalance is caught at every magnitude.
    """
    for factor in (1e-6, 1.0, 1e6):
        U = np.array([1e-3 * factor, 0.0])
        F_mech = np.array([2.0 * factor, 0.0])
        w_mech = 1e-3 * factor**2
        # consistent: U_strain == W_mech, no prestress
        assert check_energy(U, F_mech, w_mech, 0.0) is True
        assert check_energy(U, F_mech, w_mech, 0.0, energy_scale=w_mech) is True
        # inconsistent by 1%: caught at every magnitude once a physical
        # scale is supplied
        with pytest.raises(EnergyValidationError):
            check_energy(U, F_mech, 1.01 * w_mech, 0.0, energy_scale=w_mech)


def test_absolute_fallback_floor_is_documented_limitation() -> None:
    """Without ``energy_scale`` a sub-nanojoule imbalance is not resolvable.

    This pins the *known* limitation of the fallback path so it cannot widen
    silently: a balance in which every term vanishes analytically (free
    thermal expansion) has no meaningful relative error, so an absolute floor
    is unavoidable when the caller supplies no physical scale. ``run()``
    always supplies one; only direct callers hit this path.
    """
    U = np.array([1e-9, 0.0])
    F_mech = np.array([2e-6, 0.0])  # W_mech = 1e-15 J
    w_mech = 1e-15
    # 1% of 1e-15 J is 1e-17 J -- below the 1e-9 J fallback floor.
    assert check_energy(U, F_mech, 1.01 * w_mech, 0.0) is True
    # The same imbalance is caught as soon as a physical scale is supplied.
    with pytest.raises(EnergyValidationError):
        check_energy(U, F_mech, 1.01 * w_mech, 0.0, energy_scale=w_mech)


def test_energy_scale_floor_does_not_mask_real_imbalance() -> None:
    """The round-off floor must stay tighter than the relative tolerance.

    ``energy_roundoff_rel`` (1e-9) is an order of magnitude below
    ``energy_rel_tol`` (1e-8) by construction, so a floor derived from the
    characteristic imposed-strain energy can absorb round-off without ever
    accepting an error the relative test would reject.
    """
    U = np.array([1.0, 0.0])
    F_mech = np.array([2.0, 0.0])  # W_mech = 1.0
    energy_scale = 1.0  # floor = 1e-9, far below a 1% error of 1e-2
    with pytest.raises(EnergyValidationError):
        check_energy(U, F_mech, 1.01, 0.0, energy_scale=energy_scale)
    # but round-off at the 1e-12 level is absorbed
    assert check_energy(U, F_mech, 1.0 + 1e-12, 0.0, energy_scale=energy_scale) is True


def test_roundoff_floor_scales_with_supplied_energy_scale() -> None:
    """A physically derived floor replaces the arbitrary joule constant.

    Free thermal expansion makes every balance term vanish analytically, so
    the residual is pure round-off. With ``energy_scale`` supplied the floor
    tracks the characteristic imposed-strain energy instead of a fixed
    number of joules, which is what makes the verdict unit-agnostic.
    """
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=_L, y=0.0, is_support=True, support_dy=True),
    ]
    elements = [
        Element(
            id="e1",
            node_i="1",
            node_j="2",
            E=_E,
            A=_A,
            alpha=_ALPHA,
            delta_T=_DELTA_T,
        )
    ]
    scale = imposed_strain_energy(nodes, elements)
    # The imposed strain is real and large even though it stores no energy.
    assert scale == pytest.approx(0.5 * _K * (_ALPHA * _DELTA_T * _L) ** 2, rel=1e-12)

    K, F_ext, F_mechanical, fixed = assemble_global_matrices(nodes, elements)
    U = solve(K, F_ext, fixed)
    _, strain_energy, prestress_work = calculate_element_forces(nodes, elements, U)

    # Every balance term is round-off; the relative-only test would see ~100%.
    assert abs(strain_energy) < 1e-20
    assert check_energy(
        U, F_mechanical, strain_energy, prestress_work, energy_scale=scale
    )


def test_free_expansion_stores_no_strain_energy() -> None:
    """Companion case: unrestrained heating gives U_strain = 0, W_mech = 0."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=_L, y=0.0, is_support=True, support_dy=True),
    ]
    elements = [
        Element(
            id="e1",
            node_i="1",
            node_j="2",
            E=_E,
            A=_A,
            alpha=_ALPHA,
            delta_T=_DELTA_T,
        )
    ]
    K, F_ext, F_mechanical, fixed = assemble_global_matrices(nodes, elements)
    U = solve(K, F_ext, fixed)
    _, strain_energy, prestress_work = calculate_element_forces(nodes, elements, U)

    # Free expansion: no stress, no strain energy, no prestress work.
    assert strain_energy == pytest.approx(0.0, abs=1e-9)
    assert prestress_work == pytest.approx(0.0, abs=1e-9)
    assert U[2] == pytest.approx(_ALPHA * _DELTA_T * _L, rel=1e-10)
    assert (
        check_energy(
            U,
            F_mechanical,
            strain_energy,
            prestress_work,
            energy_scale=imposed_strain_energy(nodes, elements),
        )
        is True
    )
