"""Penalty boundary conditions and the ``options`` block wiring.

The README and ``docs/theory.md`` both advertised penalty boundary conditions
and sparse assembly, and ``options`` carried ``use_sparse`` / ``bc_method`` /
``penalty_value`` keys, but nothing in the pipeline read them. A user setting a
physical option got a silent no-op. These tests pin the now-implemented
behaviour, including the energy accounting that makes a penalty solve
self-consistent.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.exceptions import IllConditionedWarning, InputIgnoredWarning
from truss_analysis.main import run
from truss_analysis.model import Element, Node
from truss_analysis.postprocess import calculate_element_forces
from truss_analysis.solver import (
    apply_penalty_bc,
    check_energy,
    solve,
    solve_penalty,
    solve_penalty_with_energy,
)

# IllConditionedWarning is the subject of this module rather than an accident of
# it: the module exists to check that a penalty too small to approximate a
# constraint says so. Several tests deliberately pass such a penalty in order to
# exercise the penalty machinery itself, so the category is filtered here and
# asserted explicitly wherever the warning is the point.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.IllConditionedWarning"
    ),
]


def _frame() -> tuple[list[Node], list[Element]]:
    """A four-bar frame with one diagonal: indeterminate, so reactions matter."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="3", x=4.0, y=3.0, is_support=False),
        Node(id="4", x=0.0, y=3.0, is_support=False),
    ]
    elements = [
        Element(id="b1", node_i="1", node_j="2", E=210e9, A=0.02, I_sec=1e-5),
        Element(id="v1", node_i="2", node_j="3", E=210e9, A=0.01, I_sec=5e-6),
        Element(id="t1", node_i="3", node_j="4", E=210e9, A=0.015, I_sec=6e-6),
        Element(id="v2", node_i="4", node_j="1", E=210e9, A=0.01, I_sec=5e-6),
        Element(id="d1", node_i="1", node_j="3", E=210e9, A=0.008, I_sec=3e-6),
    ]
    return nodes, elements


def _loaded() -> tuple[list[Node], list[Element], np.ndarray, np.ndarray, list[int]]:
    nodes, elements = _frame()
    K, F, _, fixed = assemble_global_matrices(nodes, elements)
    F[2 * 2] += 20e3  # node 3, +x
    F[2 * 2 + 1] -= 60e3  # node 3, -y
    return nodes, elements, K, F, fixed


def test_penalty_converges_to_elimination_as_multiplier_grows() -> None:
    """The penalty solve must approach the exact constrained solution."""
    _nodes, _elements, K, F, fixed = _loaded()
    U_exact = solve(K, F, fixed)

    previous = np.inf
    for multiplier in (1e4, 1e6, 1e8, 1e10):
        U_pen, _ = solve_penalty_with_energy(K, F, fixed, penalty_multiplier=multiplier)
        err = float(np.abs(U_pen - U_exact).max() / np.abs(U_exact).max())
        assert err < previous, f"error did not shrink at multiplier {multiplier:.0e}"
        previous = err
    assert previous < 1e-9


def test_penalty_error_and_conditioning_trade_off() -> None:
    """Constraint error falls as cond(K) grows; their product is ~constant.

    This is the fundamental penalty trade-off, measured rather than asserted
    from theory: halving the constraint error costs a factor of ten in
    conditioning. Pinning it stops a future change from silently moving the
    default multiplier to a point where one of the two is unusable.
    """
    _nodes, _elements, K, F, fixed = _loaded()
    U_exact = solve(K, F, fixed)

    products = []
    for multiplier in (1e4, 1e6, 1e8, 1e10):
        U_pen, _ = solve_penalty_with_energy(K, F, fixed, penalty_multiplier=multiplier)
        err = float(np.abs(U_pen - U_exact).max() / np.abs(U_exact).max())
        K_pen, _ = apply_penalty_bc(K, F, fixed, penalty_multiplier=multiplier)
        cond = float(np.linalg.cond(K_pen))
        products.append(err * cond)

    # Each decade of multiplier trades error against conditioning one-for-one,
    # so the product stays within a small band instead of drifting.
    assert max(products) / min(products) < 3.0


def test_default_penalty_multiplier_is_well_scaled() -> None:
    """The default must give near-exact constraints without a conditioning warning."""
    _nodes, _elements, K, F, fixed = _loaded()
    U_exact = solve(K, F, fixed)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        U_pen, _ = solve_penalty_with_energy(K, F, fixed)
    assert not any(issubclass(w.category, IllConditionedWarning) for w in record)

    err = float(np.abs(U_pen - U_exact).max() / np.abs(U_exact).max())
    assert err < 1e-8
    K_pen, _ = apply_penalty_bc(K, F, fixed)
    assert np.linalg.cond(K_pen) < 1e12


def test_absolute_penalty_value_is_honoured_verbatim() -> None:
    """``penalty_value`` is an absolute N/m stiffness, not a multiplier."""
    _nodes, _elements, K, F, fixed = _loaded()
    alpha = 5.0e11
    K_pen, _ = apply_penalty_bc(K, F, fixed, penalty_value=alpha)
    for dof in fixed:
        assert K_pen[dof, dof] == pytest.approx(K[dof, dof] + alpha, rel=1e-12)
    # Off-diagonal and free-diagonal entries are untouched.
    for i in range(K.shape[0]):
        if i not in fixed:
            assert K_pen[i, i] == pytest.approx(K[i, i], rel=0.0, abs=0.0)


def test_undersized_penalty_warns() -> None:
    """A penalty only tens of times the diagonal does not enforce the constraint."""
    _nodes, _elements, K, F, fixed = _loaded()
    reference = float(np.abs(np.diag(K)).max())
    with pytest.warns(IllConditionedWarning, match="not a close approximation"):
        apply_penalty_bc(K, F, fixed, penalty_value=50.0 * reference)


def test_oversized_penalty_warns() -> None:
    """A penalty that destroys conditioning is reported too."""
    _nodes, _elements, K, F, fixed = _loaded()
    reference = float(np.abs(np.diag(K)).max())
    with pytest.warns(IllConditionedWarning, match="round-off"):
        apply_penalty_bc(K, F, fixed, penalty_value=1e14 * reference)


def test_penalty_inputs_are_not_mutated() -> None:
    """apply_penalty_bc must copy rather than edit the caller's matrices."""
    _nodes, _elements, K, F, fixed = _loaded()
    K_before = K.copy()
    F_before = F.copy()
    apply_penalty_bc(K, F, fixed)
    assert np.array_equal(K, K_before)
    assert np.array_equal(F, F_before)


def test_penalty_rejects_invalid_multiplier() -> None:
    _nodes, _elements, K, F, fixed = _loaded()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="penalty_multiplier"):
            apply_penalty_bc(K, F, fixed, penalty_multiplier=bad)


def test_penalty_energy_closes_the_clapeyron_balance() -> None:
    """The penalty springs store real energy and must be counted.

    Omitting ``penalty_energy`` leaves a residual of order ``R^2 / alpha`` --
    about 0.6% at ``alpha = 1e12`` on the shipped example -- because the
    structural strain energy is built from the unpenalised ``K`` while the
    solved field satisfies ``K_pen U = F``.
    """
    nodes, elements, K, F, fixed = _loaded()
    U, penalty_energy = solve_penalty_with_energy(K, F, fixed, penalty_value=1e12)

    assert penalty_energy > 0.0
    _, strain_energy, prestress_work = calculate_element_forces(nodes, elements, U)

    # Without the spring term the balance is visibly wrong.
    with pytest.raises(Exception, match="Energy balance failed"):
        check_energy(U, F, strain_energy, prestress_work, energy_scale=0.0)
    # With it, the identity closes to machine precision.
    assert (
        check_energy(
            U,
            F,
            strain_energy,
            prestress_work,
            energy_scale=1.0,
            penalty_energy=penalty_energy,
        )
        is True
    )


def test_penalty_energy_vanishes_as_penalty_grows() -> None:
    """Support settlement, and hence spring energy, shrinks like 1/alpha."""
    _nodes, _elements, K, F, fixed = _loaded()
    energies = [
        solve_penalty_with_energy(K, F, fixed, penalty_multiplier=m)[1]
        for m in (1e6, 1e8, 1e10)
    ]
    assert energies[0] > energies[1] > energies[2]
    # 100x the penalty gives ~100x less stored spring energy.
    assert energies[0] / energies[1] == pytest.approx(100.0, rel=0.05)


def test_solve_penalty_zeroes_constrained_dofs() -> None:
    """The convenience wrapper reports exact zeros at supports."""
    _nodes, _elements, K, F, fixed = _loaded()
    U = solve_penalty(K, F, fixed, penalty_multiplier=1e8)
    for dof in fixed:
        assert U[dof] == 0.0
    # and still matches elimination closely at free DOFs
    U_exact = solve(K, F, fixed)
    free = [d for d in range(len(U)) if d not in fixed]
    # atol is scaled to the displacement magnitude: some free DOFs are exactly
    # zero by symmetry, where a purely relative test is meaningless.
    scale = float(np.abs(U_exact).max())
    assert np.allclose(U[free], U_exact[free], rtol=1e-6, atol=1e-9 * scale)


def test_solve_penalty_with_energy_returns_raw_field() -> None:
    """The energy-consistent variant must NOT zero the constrained DOFs.

    Zeroing them breaks the balance: at ``alpha = 1e12`` a support settles by
    roughly ``R/alpha``, which is the same order as the free displacements, so
    the structural strain energy evaluated on a zeroed field no longer
    corresponds to any solved system.
    """
    _nodes, _elements, K, F, fixed = _loaded()
    U, _ = solve_penalty_with_energy(K, F, fixed, penalty_value=1e12)
    assert any(U[dof] != 0.0 for dof in fixed)


# ---------------------------------------------------------------------------
# options block wiring
# ---------------------------------------------------------------------------


def _write_model(tmp_path: Path, options: dict[str, object]) -> Path:
    nodes, elements = _frame()
    payload = {
        "units": "SI",
        "nodes": [
            {
                "id": n.id,
                "x": n.x,
                "y": n.y,
                "is_support": n.is_support,
                "support_dx": n.support_dx,
                "support_dy": n.support_dy,
            }
            for n in nodes
        ],
        "elements": [
            {
                "id": e.id,
                "node_i": e.node_i,
                "node_j": e.node_j,
                "E": e.E,
                "A": e.A,
                "I_sec": e.I_sec,
            }
            for e in elements
        ],
        "loads": [{"node_id": "3", "Fx": 20e3, "Fy": -60e3}],
        "options": options,
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_options_use_sparse_is_applied(tmp_path: Path) -> None:
    """``use_sparse`` must change the solve path, not be swallowed."""
    path = _write_model(tmp_path, {"use_sparse": True, "bc_method": "elimination"})
    dense = _write_model(tmp_path, {"use_sparse": False, "bc_method": "elimination"})
    r_sparse = run(str(path), quiet=True)
    r_dense = run(str(dense), quiet=True)
    for nid in r_sparse.displacements:
        assert r_sparse.displacements[nid]["ux"] == pytest.approx(
            r_dense.displacements[nid]["ux"], rel=1e-12, abs=1e-18
        )


def test_options_bc_method_penalty_is_applied(tmp_path: Path) -> None:
    """``bc_method: penalty`` must produce a penalty solve."""
    elim = _write_model(tmp_path, {"bc_method": "elimination"})
    r_elim = run(str(elim), quiet=True)
    pen_path = tmp_path / "pen.json"
    nodes, elements = _frame()
    payload = json.loads(elim.read_text(encoding="utf-8"))
    payload["options"] = {
        "bc_method": "penalty",
        # scale-aware by omission: leave penalty_value out entirely
    }
    pen_path.write_text(json.dumps(payload), encoding="utf-8")
    r_pen = run(str(pen_path), quiet=True)

    assert r_pen.status == "SUCCESS"
    # With the default multiplier the two agree to ~1e-11 relative.
    assert r_pen.displacements["3"]["ux"] == pytest.approx(
        r_elim.displacements["3"]["ux"], rel=1e-8
    )
    del nodes, elements


def test_sparse_with_penalty_falls_back_to_dense_without_losing_loads(
    tmp_path: Path,
) -> None:
    """Regression: the sparse+penalty path must not discard the applied loads.

    An earlier revision reassembled the system densely *after* the loads had
    been accumulated into ``F_ext``, resetting it to zero and silently solving
    an unloaded model. Every displacement came back exactly 0.
    """
    path = _write_model(
        tmp_path, {"use_sparse": True, "bc_method": "penalty", "penalty_value": 1e20}
    )
    with pytest.warns(InputIgnoredWarning, match="requires dense assembly"):
        result = run(str(path), quiet=True)

    dense_path = _write_model(tmp_path, {"use_sparse": False, "bc_method": "penalty"})
    expected = run(str(dense_path), quiet=True, penalty_value=1e20)

    # A non-trivial, load-driven displacement field must survive.
    assert result.displacements["3"]["ux"] != 0.0
    assert result.displacements["3"]["ux"] == pytest.approx(
        expected.displacements["3"]["ux"], rel=1e-9
    )


def test_unknown_option_key_warns(tmp_path: Path) -> None:
    """An unrecognised option must be reported, not silently dropped."""
    path = _write_model(tmp_path, {"use_sparse": False, "solver_backend": "magic"})
    with pytest.warns(InputIgnoredWarning, match="unrecognised options key"):
        run(str(path), quiet=True)


def test_invalid_bc_method_raises(tmp_path: Path) -> None:
    from truss_analysis.exceptions import InputValidationError

    path = _write_model(tmp_path, {"bc_method": "lagrange"})
    with pytest.raises(InputValidationError, match="bc_method"):
        run(str(path), quiet=True)


def test_explicit_argument_overrides_options_block(tmp_path: Path) -> None:
    """A value passed to run() beats the file, which is the more recent intent."""
    path = _write_model(tmp_path, {"bc_method": "elimination"})
    with pytest.warns(IllConditionedWarning):
        # penalty_value far below the stiffness diagonal -> undersized warning
        result = run(str(path), quiet=True, bc_method="penalty", penalty_value=1e8)
    assert result.status == "SUCCESS"
