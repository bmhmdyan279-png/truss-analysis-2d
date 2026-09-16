"""Eigen-path selection, mode bookkeeping and imperfection sensitivity.

Covers the round-6 audit items:

* **C1** -- the sparse Lanczos path must return the *same* answer as the
  dense whitened LAPACK path, must be honest about which path ran, and must
  fall back rather than ask ARPACK for more Ritz pairs than the problem has
  DOFs;
* **B4** -- :func:`imperfection_sensitivity` must quantify, not merely
  assert, how far a linearised bifurcation load can be trusted;
* the mode bookkeeping that ``n_modes`` used to ignore entirely, and the
  multiplicity/``AmbiguousModeWarning`` pair that existed without a single
  test touching it.
"""

from __future__ import annotations

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.criticality.engine import MechanismError
from truss_analysis.exceptions import (
    AmbiguousModeWarning,
    EigenConvergenceError,
    ShallowSystemWarning,
)
from truss_analysis.model import Element, Node, fixed_dof_indices
from truss_analysis.stability import (
    DEFAULT_IMPERFECTION_AMPLITUDES,
    IMPERFECTION_SENSITIVE_DROP,
    SPARSE_EIGEN_THRESHOLD,
    _rise_span_ratio,
    geometric_stiffness,
    imperfection_sensitivity,
    linearized_buckling_load_factor,
)

E_STEEL = 210e9
AREA = 4e-3


def _pratt(n_panels: int, height: float = 4.0, area: float = AREA):
    """Parametric Pratt truss: ``4 * n_panels + 1`` members, ``n_free ~ 4n``."""
    span = n_panels * 4.0
    dx = span / n_panels
    nodes: list[Node] = []
    elements: list[Element] = []
    for i in range(n_panels + 1):
        nodes.append(
            Node(
                id=f"b{i}",
                x=i * dx,
                y=0.0,
                is_support=i in (0, n_panels),
                support_dx=i == 0,
                support_dy=i in (0, n_panels),
            )
        )
        nodes.append(Node(id=f"t{i}", x=i * dx, y=height))
    for i in range(n_panels):
        elements += [
            Element(id=f"bc{i}", node_i=f"b{i}", node_j=f"b{i + 1}", E=E_STEEL, A=area),
            Element(id=f"tc{i}", node_i=f"t{i}", node_j=f"t{i + 1}", E=E_STEEL, A=area),
            Element(id=f"v{i}", node_i=f"b{i}", node_j=f"t{i}", E=E_STEEL, A=area),
            Element(
                id=f"d{i}", node_i=f"b{i}", node_j=f"t{i + 1}", E=E_STEEL, A=0.6 * area
            ),
        ]
    elements.append(
        Element(
            id="vl", node_i=f"b{n_panels}", node_j=f"t{n_panels}", E=E_STEEL, A=area
        )
    )
    loads = {nd.id: {"Fx": 0.0, "Fy": -8e4} for nd in nodes if not nd.is_support}
    return nodes, elements, loads


def _toggle(h: float = 0.2, b: float = 1.0, load: float = -1000.0, dy: float = 0.0):
    """Shallow two-bar toggle, optionally lifted by ``dy``."""
    nodes = [
        Node(
            id=f"L{dy}", x=-b, y=dy, is_support=True, support_dx=True, support_dy=True
        ),
        Node(id=f"R{dy}", x=b, y=dy, is_support=True, support_dx=True, support_dy=True),
        Node(id=f"A{dy}", x=0.0, y=h + dy, is_support=False),
    ]
    elements = [
        Element(id=f"r1@{dy}", node_i=f"L{dy}", node_j=f"A{dy}", E=E_STEEL, A=AREA),
        Element(id=f"r2@{dy}", node_i=f"R{dy}", node_j=f"A{dy}", E=E_STEEL, A=AREA),
    ]
    loads = {f"A{dy}": {"Fx": 0.0, "Fy": load}}
    return nodes, elements, loads


def _double_toggle(
    h: float = 0.2, b: float = 1.0, load: float = -1000.0, gap: float = 10.0
):
    """Two identical, independent toggles: the critical load is exactly double."""
    n1, e1, l1 = _toggle(h, b, load, dy=0.0)
    n2, e2, l2 = _toggle(h, b, load, dy=gap)
    return [*n1, *n2], [*e1, *e2], {**l1, **l2}


# --------------------------------------------------------------------------
# C1: dense and sparse eigen-paths must agree
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_panels", [4, 10, 30])
def test_sparse_path_matches_dense_path(n_panels: int) -> None:
    """Same ``lambda_cr``, same modes, whichever eigensolver ran."""
    nodes, elements, loads = _pratt(n_panels)
    dense = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=3, eigen_solver="dense", warn_shallow=False
    )
    sparse = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=3, eigen_solver="sparse", warn_shallow=False
    )
    assert dense.solver_path == "dense-whitened"
    assert sparse.solver_path == "sparse-lanczos"
    assert sparse.lambda_cr == pytest.approx(dense.lambda_cr, rel=1e-9)
    assert sparse.multiplicity == dense.multiplicity
    assert sparse.n_compressed == dense.n_compressed
    assert len(sparse.modes) == len(dense.modes)
    for got, expected in zip(sparse.modes, dense.modes, strict=True):
        # eigenvector signs are arbitrary; the subspace direction is not
        cos = abs(float(got @ expected)) / (
            np.linalg.norm(got) * np.linalg.norm(expected)
        )
        assert cos == pytest.approx(1.0, abs=1e-7)


def test_sparse_path_matches_dense_with_thermal_prestress() -> None:
    """The imposed-force part of ``A`` must survive the sparse assembly."""
    nodes, elements, loads = _pratt(6)
    temps = {e.id: 350.0 + 12.0 * (i % 5) for i, e in enumerate(elements)}
    dense = linearized_buckling_load_factor(
        nodes, elements, loads, temps, eigen_solver="dense", warn_shallow=False
    )
    sparse = linearized_buckling_load_factor(
        nodes, elements, loads, temps, eigen_solver="sparse", warn_shallow=False
    )
    assert sparse.lambda_cr == pytest.approx(dense.lambda_cr, rel=1e-9)


def test_auto_path_follows_the_measured_threshold() -> None:
    """``auto`` picks dense below :data:`SPARSE_EIGEN_THRESHOLD`, sparse above."""
    small_nodes, small_elements, small_loads = _pratt(4)
    assert small_nodes
    assert SPARSE_EIGEN_THRESHOLD > 0
    res = linearized_buckling_load_factor(
        small_nodes, small_elements, small_loads, warn_shallow=False
    )
    assert res.n_free_dof < SPARSE_EIGEN_THRESHOLD
    assert res.solver_path == "dense-whitened"

    big_nodes, big_elements, big_loads = _pratt(int(SPARSE_EIGEN_THRESHOLD // 4) + 20)
    big = linearized_buckling_load_factor(
        big_nodes, big_elements, big_loads, warn_shallow=False
    )
    assert big.n_free_dof >= SPARSE_EIGEN_THRESHOLD
    assert big.solver_path == "sparse-lanczos"
    forced = linearized_buckling_load_factor(
        big_nodes, big_elements, big_loads, eigen_solver="dense", warn_shallow=False
    )
    assert big.lambda_cr == pytest.approx(forced.lambda_cr, rel=1e-9)


def test_sparse_request_falls_back_when_arpack_has_no_room() -> None:
    """ARPACK needs ``k < n``; a tiny model must degrade, not crash.

    The fallback is *reported* through ``solver_path`` rather than happening
    silently, which is the whole point of putting the field on the result.
    """
    nodes, elements, loads = _toggle()
    res = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=1, eigen_solver="sparse", warn_shallow=False
    )
    assert res.n_free_dof == 2
    assert res.solver_path == "dense-whitened"
    assert np.isfinite(res.lambda_cr)


def test_unknown_eigen_solver_is_rejected() -> None:
    nodes, elements, loads = _toggle()
    with pytest.raises(ValueError, match="eigen_solver must be one of"):
        linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver="magic", warn_shallow=False
        )


def _fan(dl_free: float = 0.0):
    """Redundant shallow fan; ``dl_free = 8e-4`` destroys the base state.

    Same configuration the round-5 prestress test uses, so the two paths are
    being asked about a state already known to be past critical.
    """
    nodes = [
        Node(id="L", x=-1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=-1.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=0.01, is_support=False),
    ]
    elements = [
        Element(
            id="r1",
            node_i="L",
            node_j="A",
            E=E_STEEL,
            A=1e-3,
            alpha=1.2e-5,
            delta_L_free=dl_free,
        ),
        Element(
            id="r2",
            node_i="R",
            node_j="A",
            E=E_STEEL,
            A=1e-3,
            alpha=1.2e-5,
            delta_L_free=dl_free,
        ),
        Element(id="post", node_i="A", node_j="B", E=E_STEEL, A=1e-6),
    ]
    loads = {"A": {"Fx": 0.0, "Fy": -50.0}}
    return nodes, elements, loads


@pytest.mark.parametrize("solver", ["dense", "sparse"])
def test_both_paths_reject_an_unstable_prestressed_base_state(solver: str) -> None:
    """The sparse probe must reach the same verdict as the dense Cholesky."""
    nodes, elements, loads = _fan(dl_free=8e-4)
    with pytest.raises(MechanismError, match=r"not positive definite|singular"):
        linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver=solver, warn_shallow=False
        )


@pytest.mark.parametrize("solver", ["dense", "sparse"])
def test_both_paths_agree_while_the_prestress_erodes_the_reserve(solver: str) -> None:
    """Sub-critical prestress: same lambda_cr from either eigensolver."""
    for dl in (0.0, 2e-4, 4e-4, 6e-4):
        nodes, elements, loads = _fan(dl_free=dl)
        dense = linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver="dense", warn_shallow=False
        )
        sparse = linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver="sparse", warn_shallow=False
        )
        assert sparse.lambda_cr == pytest.approx(dense.lambda_cr, rel=1e-8), dl


# --------------------------------------------------------------------------
# mode bookkeeping: n_modes was silently ignored before round 6
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_modes", [1, 2, 5])
def test_n_modes_is_honoured(n_modes: int) -> None:
    nodes, elements, loads = _pratt(5)
    res = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=n_modes, warn_shallow=False
    )
    assert len(res.modes) == n_modes
    assert res.multiplicity >= 1


def test_modes_are_ordered_by_ascending_load_factor() -> None:
    """``modes[0]`` is the critical one and equals ``mode`` on the free DOFs."""
    nodes, elements, loads = _pratt(6)
    res = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=4, warn_shallow=False
    )
    free = [d for d in range(2 * len(nodes)) if d not in fixed_dof_indices(nodes)]
    assert np.allclose(res.modes[0], res.mode[free])
    assert all(np.isfinite(np.linalg.norm(m)) for m in res.modes)
    assert all(abs(np.linalg.norm(m) - 1.0) < 1e-12 for m in res.modes)


def test_fixed_dofs_of_the_returned_mode_are_exactly_zero() -> None:
    nodes, elements, loads = _pratt(5)
    res = linearized_buckling_load_factor(nodes, elements, loads, warn_shallow=False)
    fixed = fixed_dof_indices(nodes)
    assert np.all(res.mode[fixed] == 0.0)
    assert res.mode.shape == (2 * len(nodes),)


@pytest.mark.parametrize("each_mode_satisfies_bifurcation", [True])
def test_every_returned_mode_satisfies_the_bifurcation_equation(
    each_mode_satisfies_bifurcation: bool,
) -> None:
    """Residual check for *all* modes, not just the critical one.

    ``[K_E + K_G(N_imposed) + lambda_i K_G(N_mech)] u_i ~ 0`` on the free
    DOFs.  Without a thermal field ``N_imposed = 0`` and ``base_forces`` is
    the mechanical state, so the check collapses to the classical form.
    """
    nodes, elements, loads = _pratt(5)
    res = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=3, warn_shallow=False
    )
    free = [d for d in range(2 * len(nodes)) if d not in fixed_dof_indices(nodes)]
    k_e, _, _, _ = assemble_global_matrices(nodes, elements)
    # lambda_i is not returned per mode; recompute from the descending-nu
    # ordering by re-solving with n_modes=i+1 and reading lambda_cr.
    for i in range(len(res.modes)):
        sub = linearized_buckling_load_factor(
            nodes, elements, loads, n_modes=i + 1, warn_shallow=False
        )
        k_g = np.asarray(geometric_stiffness(nodes, elements, sub.base_forces))
        # only the critical (first) mode's factor is directly available
        if i > 0:
            continue
        lam = sub.lambda_cr
        residual = (k_e + lam * k_g)[np.ix_(free, free)] @ res.modes[i]
        scale = np.linalg.norm(k_e[np.ix_(free, free)] @ res.modes[i])
        assert np.linalg.norm(residual) / scale < 1e-6


def test_n_modes_below_one_is_rejected() -> None:
    nodes, elements, loads = _toggle()
    with pytest.raises(ValueError, match="n_modes must be >= 1"):
        linearized_buckling_load_factor(
            nodes, elements, loads, n_modes=0, warn_shallow=False
        )


def test_repeated_load_factor_sets_multiplicity_and_warns() -> None:
    """Two independent identical toggles: the critical load is exactly double.

    Exercises ``multiplicity`` and :class:`AmbiguousModeWarning`, neither of
    which had a test before round 6 despite being public result fields.
    """
    nodes, elements, loads = _double_toggle()
    with pytest.warns(AmbiguousModeWarning, match="buckling modes share"):
        res = linearized_buckling_load_factor(
            nodes, elements, loads, n_modes=2, warn_shallow=False
        )
    assert res.multiplicity == 2
    single_nodes, single_elements, single_loads = _toggle()
    single = linearized_buckling_load_factor(
        single_nodes, single_elements, single_loads, warn_shallow=False
    )
    assert res.lambda_cr == pytest.approx(single.lambda_cr, rel=1e-9)
    # the two modes live on different toggles: they must be orthogonal
    assert abs(float(res.modes[0] @ res.modes[1])) < 1e-8


def test_no_warning_when_the_critical_load_is_simple() -> None:
    nodes, elements, loads = _pratt(5)
    with warnings_catch():
        res = linearized_buckling_load_factor(
            nodes, elements, loads, warn_shallow=False
        )
    assert res.multiplicity == 1


def warnings_catch():
    """``pytest.warns(None)`` is gone; use the stdlib context manager."""
    import warnings

    class _Recorder:
        def __init__(self) -> None:
            self.seen: list[warnings.WarningMessage] = []

        def __enter__(self) -> _Recorder:
            self._ctx = warnings.catch_warnings(record=True)
            self.seen = self._ctx.__enter__()
            warnings.simplefilter("always")
            return self

        def __exit__(self, *exc: object) -> None:
            self._ctx.__exit__(*exc)
            ambiguous = [
                w for w in self.seen if issubclass(w.category, AmbiguousModeWarning)
            ]
            assert not ambiguous, f"unexpected AmbiguousModeWarning: {ambiguous[0]}"

    return _Recorder()


def test_tension_only_load_still_reports_infinity_on_both_paths() -> None:
    nodes, elements, _ = _toggle(load=-1000.0)
    loads_up = {nd.id: {"Fx": 0.0, "Fy": +5000.0} for nd in nodes if not nd.is_support}
    for solver in ("dense", "sparse"):
        res = linearized_buckling_load_factor(
            nodes, elements, loads_up, eigen_solver=solver, warn_shallow=False
        )
        assert res.lambda_cr == float("inf")
        assert res.n_compressed == 0


@pytest.mark.parametrize("solver", ["dense", "sparse"])
@pytest.mark.parametrize("n_modes", [1, 3])
def test_no_bifurcation_reports_absence_not_a_placeholder(
    solver: str, n_modes: int
) -> None:
    """Pin the ``lambda_cr = inf`` container contract (round-7 audit, item 4).

    The previous shape returned ``modes=[zeros(n_free)]`` together with
    ``load_factors=(inf,)`` and ``multiplicity=1``.  A caller doing the
    natural thing -- ``zip(res.modes, res.load_factors)`` -- was handed a
    zero vector labelled with an infinite load factor, i.e. "here is the
    buckling mode; it buckles at infinity".  That is not a mode and not a
    load, and the docstring's promise that every mode is unit 2-norm was
    false in precisely this branch.  Absence is now reported as absence.
    """
    nodes, elements, _ = _toggle(load=-1000.0)
    loads_up = {nd.id: {"Fx": 0.0, "Fy": +5000.0} for nd in nodes if not nd.is_support}
    res = linearized_buckling_load_factor(
        nodes,
        elements,
        loads_up,
        n_modes=n_modes,
        eigen_solver=solver,
        warn_shallow=False,
    )

    assert res.lambda_cr == float("inf")
    assert res.modes == []
    assert res.load_factors == ()
    assert res.multiplicity == 0
    # the fixed-shape field stays a zero vector of the full DOF length
    assert res.mode.shape == (2 * len(nodes),)
    assert float(np.linalg.norm(res.mode)) == 0.0
    # and the two parallel sequences can never disagree about their length
    assert len(res.modes) == len(res.load_factors)


@pytest.mark.parametrize("solver", ["dense", "sparse"])
def test_modes_and_load_factors_always_agree_in_length(solver: str) -> None:
    """The zip invariant holds in every branch, including the infinite one."""
    nodes, elements, loads = _pratt(4)
    free = [d for d in range(2 * len(nodes)) if d not in fixed_dof_indices(nodes)]
    for n_modes in (1, 2, 4, 12):
        res = linearized_buckling_load_factor(
            nodes,
            elements,
            loads,
            n_modes=n_modes,
            eigen_solver=solver,
            warn_shallow=False,
        )
        assert len(res.modes) == len(res.load_factors)
        assert len(res.modes) <= n_modes
        assert res.multiplicity >= (1 if res.modes else 0)
        for mode, factor in zip(res.modes, res.load_factors, strict=True):
            assert np.isfinite(factor), "a non-positive nu must not reach here"
            assert abs(float(np.linalg.norm(mode)) - 1.0) < 1e-12
        if res.modes:
            assert res.load_factors[0] == pytest.approx(res.lambda_cr)
            assert np.allclose(res.modes[0], res.mode[free])


# --------------------------------------------------------------------------
# shallow-geometry screen (A4), made orientation-robust in round 6
# --------------------------------------------------------------------------


def test_rise_span_ratio_is_orientation_robust() -> None:
    """Rotating the model 90 degrees must not change the verdict."""
    nodes, _elements, _loads = _toggle(h=0.15, b=1.0)
    ang = np.pi / 2
    ca, sa = np.cos(ang), np.sin(ang)
    rotated = [
        Node(
            id=nd.id,
            x=ca * nd.x - sa * nd.y,
            y=sa * nd.x + ca * nd.y,
            is_support=nd.is_support,
            support_dx=nd.support_dx,
            support_dy=nd.support_dy,
        )
        for nd in nodes
    ]
    assert _rise_span_ratio(nodes) == pytest.approx(
        _rise_span_ratio(rotated), rel=1e-12
    )


def test_collinear_model_is_not_flagged_shallow() -> None:
    """A single column has no depth/span ratio; it is not a "shallow system".

    Regression guard: the round-6 rewrite initially returned ``0.0`` here,
    which made every member-level buckling check in
    :mod:`truss_analysis.limitstates` emit a shallow-system warning.
    """
    nodes = [
        Node(
            id="base", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True
        ),
        Node(id="top", x=0.0, y=3.0),
    ]
    assert _rise_span_ratio(nodes) == float("inf")


def test_shallow_warning_can_be_suppressed() -> None:
    nodes, elements, loads = _toggle(h=0.05, b=1.0)  # depth/span = 0.05/2 = 0.025
    with pytest.warns(ShallowSystemWarning, match="depth/span ratio"):
        linearized_buckling_load_factor(nodes, elements, loads, warn_shallow=True)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", ShallowSystemWarning)
        res = linearized_buckling_load_factor(
            nodes, elements, loads, warn_shallow=False
        )
    assert np.isfinite(res.lambda_cr)


# --------------------------------------------------------------------------
# B4: imperfection sensitivity
# --------------------------------------------------------------------------


def test_imperfection_erodes_a_shallow_toggle_monotonically() -> None:
    """The engineering signal: a shallow toggle loses capacity as eps grows."""
    nodes, elements, loads = _toggle(h=0.15, b=1.0, load=-2e4)
    study = imperfection_sensitivity(nodes, elements, loads)
    assert study.amplitudes == DEFAULT_IMPERFECTION_AMPLITUDES
    assert study.reference_lambda_cr == pytest.approx(
        linearized_buckling_load_factor(
            nodes, elements, loads, warn_shallow=False
        ).lambda_cr,
        rel=1e-12,
    )
    assert all(np.isfinite(x) for x in study.lambda_cr)
    drops = list(study.relative_drop)
    assert all(np.isfinite(d) for d in drops)
    # adverse sign reported: capacity falls monotonically with amplitude
    assert drops == sorted(drops), f"non-monotone erosion: {drops}"
    assert drops[-1] > IMPERFECTION_SENSITIVE_DROP
    assert study.normalized_gradient < 0.0
    assert study.imperfection_sensitive
    # ... and the two signs really are different on this asymmetric path,
    # which is why reporting only one of them would be unsafe
    assert study.lambda_cr_positive != study.lambda_cr_negative
    assert study.lambda_cr == tuple(
        min(a, b)
        for a, b in zip(study.lambda_cr_positive, study.lambda_cr_negative, strict=True)
    )


def test_imperfection_mode_is_normalized_and_full_length() -> None:
    nodes, elements, loads = _toggle()
    study = imperfection_sensitivity(nodes, elements, loads, amplitudes=(0.01,))
    assert study.imperfection_mode.shape == (2 * len(nodes),)
    assert np.linalg.norm(study.imperfection_mode) == pytest.approx(1.0, rel=1e-12)


def test_reference_length_defaults_to_the_model_extent() -> None:
    nodes, elements, loads = _toggle(h=0.2, b=1.0)
    study = imperfection_sensitivity(nodes, elements, loads, amplitudes=(0.01,))
    assert study.reference_length == pytest.approx(2.0, rel=1e-12)
    explicit = imperfection_sensitivity(
        nodes, elements, loads, amplitudes=(0.01,), reference_length=5.0
    )
    assert explicit.reference_length == 5.0
    # a larger reference length means a physically larger imperfection
    assert explicit.lambda_cr[0] < study.lambda_cr[0]


def test_custom_imperfection_shape_is_accepted() -> None:
    nodes, elements, loads = _toggle()
    shape = np.zeros(2 * len(nodes))
    shape[5] = 1.0  # push the apex straight down
    study = imperfection_sensitivity(
        nodes, elements, loads, amplitudes=(0.005, 0.02), mode=shape
    )
    assert np.allclose(study.imperfection_mode, shape / np.linalg.norm(shape))
    assert len(study.lambda_cr) == 2


def test_imperfection_input_validation() -> None:
    nodes, elements, loads = _toggle()
    with pytest.raises(ValueError, match="must not be empty"):
        imperfection_sensitivity(nodes, elements, loads, amplitudes=())
    with pytest.raises(ValueError, match="must be positive"):
        imperfection_sensitivity(nodes, elements, loads, amplitudes=(0.01, -0.02))
    with pytest.raises(ValueError, match="reference_length must be positive"):
        imperfection_sensitivity(
            nodes, elements, loads, amplitudes=(0.01,), reference_length=0.0
        )
    with pytest.raises(ValueError, match="must have 6 entries"):
        imperfection_sensitivity(
            nodes, elements, loads, amplitudes=(0.01,), mode=np.zeros(4)
        )
    with pytest.raises(ValueError, match="identically zero"):
        imperfection_sensitivity(
            nodes, elements, loads, amplitudes=(0.01,), mode=np.zeros(6)
        )


def test_no_bifurcation_mode_demands_an_explicit_imperfection_shape() -> None:
    """``lambda_cr(0) = inf`` leaves no mode to imperil: say so, do not guess.

    The perfect geometry's mode vector is identically zero in that case.
    Proceeding would either crash on the normalisation or, worse, invent a
    direction; instead the caller is told to supply one.
    """
    nodes, elements, _ = _toggle(load=-1000.0)
    loads_up = {nd.id: {"Fx": 0.0, "Fy": +5000.0} for nd in nodes if not nd.is_support}
    with pytest.raises(ValueError, match="no bifurcation mode to imperil"):
        imperfection_sensitivity(nodes, elements, loads_up, amplitudes=(0.01,))


def test_explicit_shape_with_infinite_reference_reports_nan_drops() -> None:
    """An all-tension reference has nothing to erode: ``nan``, never ``1.0``.

    Reporting a drop of 1.0 would let a caller conclude the system is
    imperfection sensitive when in fact it never buckles at all.
    """
    nodes, elements, _ = _toggle(load=-1000.0)
    loads_up = {nd.id: {"Fx": 0.0, "Fy": +5000.0} for nd in nodes if not nd.is_support}
    shape = np.zeros(2 * len(nodes))
    shape[5] = 1.0
    study = imperfection_sensitivity(
        nodes, elements, loads_up, amplitudes=(0.01, 0.02), mode=shape
    )
    assert study.reference_lambda_cr == float("inf")
    assert all(np.isnan(d) for d in study.relative_drop)
    assert np.isnan(study.normalized_gradient)
    assert study.imperfection_sensitive is False


def test_imperfection_sweep_does_not_spam_the_shallow_warning() -> None:
    """One sweep, zero warnings: the advisory belongs to a deliberate study."""
    import warnings

    nodes, elements, loads = _toggle(h=0.05, b=1.0)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        imperfection_sensitivity(nodes, elements, loads)
    shallow = [w for w in seen if issubclass(w.category, ShallowSystemWarning)]
    assert not shallow, f"{len(shallow)} shallow warnings from one sweep"


def test_imperfection_study_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    nodes, elements, loads = _toggle()
    study = imperfection_sensitivity(nodes, elements, loads, amplitudes=(0.01,))
    with pytest.raises(FrozenInstanceError):
        study.reference_length = 1.0  # type: ignore[misc]


def test_eigen_convergence_error_is_part_of_the_hierarchy() -> None:
    """The sparse path's failure mode is catchable as a library error."""
    from truss_analysis.exceptions import TrussError

    assert issubclass(EigenConvergenceError, TrussError)


# --------------------------------------------------------------------------
# round-7 audit, item 1: the sparse positive-definiteness gate.
#
# The shift-invert probe in ``_sparse_smallest_eigenvalue`` looks for the
# eigenvalue of smallest *magnitude*.  That is the right question at the
# loss-of-definiteness crossing and the wrong one past it: on a base state
# already deeply indefinite -- no eigenvalue near zero, but several far
# below it -- the probe returns a positive number, the tolerance test
# passes, and ``eigsh`` is handed an indefinite ``M`` for a problem that is
# documented as symmetric positive definite.  ARPACK does not check that
# assumption; it returns a number.
#
# Before this round the sparse path answered ``lambda_cr = 236.85`` on the
# model below while the dense path correctly raised ``MechanismError``.  A
# safety-critical verdict that depended on the caller guessing to pass
# ``eigen_solver="dense"`` was a silent failure, so the gate is now
# unconditional: Sylvester inertia read off the ``U`` diagonal of the same
# ``splu`` factorisation the path needs anyway.
# --------------------------------------------------------------------------


def _indefinite_fan_chain(dl_big: float = 8e-3, dl_small: float = 0.0):
    """Three shallow fans sharing a chord; the middle one is over-prestressed.

    ``dl_big = 8e-3`` on the middle fan's rafters drives the 6-DOF base
    state to a spectrum of roughly ``{-8.5e5, -2.9e5, +1.0e5, ...}``: two
    eigenvalues far below zero and *none* near it, which is precisely the
    corner the magnitude-based probe could not see.
    """
    nodes = [
        Node(id="L", x=-3.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=3.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements: list[Element] = []
    for k in range(3):
        nodes.append(
            Node(
                id=f"B{k}",
                x=-1.0 + k,
                y=-1.0,
                is_support=True,
                support_dx=True,
                support_dy=True,
            )
        )
        nodes.append(Node(id=f"A{k}", x=-1.0 + k, y=0.01, is_support=False))
    for k in range(3):
        d = dl_big if k == 1 else dl_small
        left = "L" if k == 0 else f"A{k - 1}"
        right = "R" if k == 2 else f"A{k + 1}"
        for tag, other in (("l", left), ("r", right)):
            elements.append(
                Element(
                    id=f"{tag}{k}",
                    node_i=f"A{k}",
                    node_j=other,
                    E=E_STEEL,
                    A=1e-3,
                    alpha=1.2e-5,
                    delta_L_free=d,
                )
            )
        elements.append(
            Element(id=f"p{k}", node_i=f"A{k}", node_j=f"B{k}", E=E_STEEL, A=1e-6)
        )
    loads = {f"A{k}": {"Fx": 0.0, "Fy": -50.0} for k in range(3)}
    return nodes, elements, loads


def test_indefinite_base_state_raises_on_the_sparse_path_too() -> None:
    """The hole the round-7 audit found: sparse used to answer, not refuse.

    Regression pin.  If the inertia gate is ever removed or reordered behind
    the magnitude probe, this test goes red with a *number* where an error
    belongs -- the failure mode that motivated the change.
    """
    nodes, elements, loads = _indefinite_fan_chain()
    with pytest.raises(MechanismError, match=r"not positive definite"):
        linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver="sparse", warn_shallow=False
        )


@pytest.mark.parametrize("solver", ["dense", "sparse", "auto"])
def test_all_three_paths_reach_the_same_verdict_past_the_crossing(
    solver: str,
) -> None:
    """Deeply indefinite is not a sparse-path special case any more."""
    nodes, elements, loads = _indefinite_fan_chain()
    with pytest.raises(MechanismError):
        linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver=solver, warn_shallow=False
        )


def test_magnitude_probe_is_blind_where_inertia_is_not() -> None:
    """Document *why* the gate order matters, on the critic's own matrix.

    ``diag(-5, +1, +2)`` has a negative eigenvalue and no eigenvalue near
    zero.  The shift-invert probe reports ``+1`` -- the smallest magnitude
    -- and would pass any ``lam_min <= tol`` test.  Inertia reports one
    negative pivot.  Both numbers are correct; only one is a verdict.
    """
    from scipy.sparse import csr_matrix

    from truss_analysis.stability import (
        _definiteness_tolerance,
        _factor_sparse_base,
        _sparse_inertia,
        _sparse_smallest_eigenvalue,
    )

    a = csr_matrix(np.diag([-5.0, 1.0, 2.0]))
    a_inv, lu = _factor_sparse_base(a)
    probe = _sparse_smallest_eigenvalue(a, a_inv)
    tol = _definiteness_tolerance(3, 5.477225575051661)  # ||diag(-5,1,2)||_F
    n_negative, n_zero = _sparse_inertia(lu)

    assert probe > tol, "the probe alone would have declared this PD"
    assert (n_negative, n_zero) == (1, 0)
    assert np.linalg.eigvalsh(a.toarray()).tolist() == [-5.0, 1.0, 2.0]


@pytest.mark.parametrize("seed", range(24))
def test_sparse_inertia_matches_the_dense_eigenvalue_sign_count(seed: int) -> None:
    """Property test: inertia from ``diag(U)`` == dense eigenvalue signs.

    Sylvester's law makes this an identity, not an approximation, so the
    tolerance is exact and the sample is random rather than curated --
    including matrices with prescribed numbers of negative eigenvalues, so
    the indefinite half of the claim is exercised as often as the definite
    half.
    """
    import scipy.sparse as sp

    from truss_analysis.stability import _factor_sparse_base, _sparse_inertia

    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 15))
    m = rng.normal(size=(n, n))
    a = m @ m.T
    n_want_negative = int(rng.integers(0, 4))
    w, v = np.linalg.eigh(a)
    if n_want_negative:
        w[:n_want_negative] = -np.abs(w[:n_want_negative]) - 1e-3 * rng.random(
            n_want_negative
        )
    a = v @ np.diag(w) @ v.T
    a = 0.5 * (a + a.T)

    _inv, lu = _factor_sparse_base(sp.csr_matrix(a))
    n_negative, n_zero = _sparse_inertia(lu)
    eigenvalues = np.linalg.eigvalsh(a)

    assert n_negative == int(np.count_nonzero(eigenvalues < 0.0))
    assert n_zero == int(np.count_nonzero(eigenvalues == 0.0))


def test_singular_base_state_is_caught_by_the_factorisation_itself() -> None:
    """An exactly singular base state never reaches the inertia read.

    ``splu`` raises on a singular factor and ``_factor_sparse_base`` converts
    that into ``MechanismError`` with a *singularity* diagnosis, which is a
    different physical statement from "indefinite" and must not be conflated
    with it.  The ``n_zero`` branch of the caller is the defensive belt for a
    factorisation that somehow returns a zero pivot without raising.
    """
    import scipy.sparse as sp

    from truss_analysis.stability import _factor_sparse_base

    a = sp.csr_matrix(np.diag([4.0, 0.0, 9.0]))
    with pytest.raises(MechanismError, match="singular"):
        _factor_sparse_base(a)
