"""Cross-family transfer check for the criticality screening (level 5).

Why this module exists
----------------------
The screening framework itself is parameter-free mechanics: the criticality
index is exact linear algebra, not a fitted model, so there is nothing to
"train" in the usual supervised sense.  The transfer question that *is*
meaningful — and the one answered here — is:

    Can a cheap surrogate, learned on the member population of ONE truss
    family, reproduce the exact CI ranking of OTHER families?

A positive answer supports the family-independence of the screening signal:
the mapping "statics + geometry -> criticality rank" learned on Pratt trusses
carries over to Warren and Howe trusses.  A negative answer would mean the
ranking signal is family-specific and any shortcut model must be refitted.

Definition of "train" / "test" (fixed before measurement)
---------------------------------------------------------
* **Sample** = one (member, state) pair; a state is (topology, scenario,
  temperature).  Target = exact displacement CI of that member in that state
  (single-member stiffness perturbation, ``alpha``).
* **Features** = statics + geometry ONLY: one baseline solve per state
  (member force ratios) plus length/orientation/position/connectivity and
  the member's own temperature.  No information from the perturbation sweep
  may enter the features — that is the line which keeps the surrogate an
  independent predictor rather than a restatement of the engine.
* **Train** = fit :class:`RidgeSurrogate` on all samples of the training
  family (all its topologies, scenarios and temperatures).
* **Test** = per held-out state, Spearman rank correlation between
  predicted and exact CI over that state's members; aggregate = mean over
  states.  Gate: mean rho > :data:`CV_RHO_GATE` (0.85).
* **Baselines** reported alongside: (i) force-ratio-only ranking (single
  feature, no fitting), (ii) a permutation control (targets shuffled) which
  must land near zero — otherwise the metric itself is broken.

The ridge is implemented in closed form on standardised features (numpy
only): ``w = (Z^T Z + a I)^-1 Z^T (y - ybar)`` — deterministic, no optional
ML dependency, fully reproducible from ``(X, y, a)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..criticality.engine import compute_ci_for_topology
from ..criticality.scenarios import (
    T_AMBIENT,
    get_scenario_temperatures,
    member_centroids,
    span_bounds,
)
from ..limitstates import member_axial_forces
from ..model import Element, Node
from .metrics import rank_correlation

__all__ = [
    "CV_RHO_GATE",
    "FEATURES",
    "CrossValidationResult",
    "RidgeSurrogate",
    "StateRow",
    "cross_family_cv",
    "feature_matrix",
]

CV_RHO_GATE: float = 0.85

FEATURES: Tuple[str, ...] = (
    "force_ratio_abs",  # |N_i| / max_j |N_j| of the baseline state
    "force_sign",  # +1 tension / -1 compression / 0 zero
    "length_rel",  # L_i / mean member length of the topology
    "orientation_abs_cos",  # |cos(theta_i)|: 0 vertical, 1 horizontal
    "heated_flag",  # 1 if the member temperature exceeds ambient
    "temperature_scaled",  # T_i / 1000 [degC / 1000]
    "centroid_position",  # (x_c - min_x) / span
    "node_degree_mean",  # (deg(node_i) + deg(node_j)) / 2
)


def _member_lengths(
    nodes: Sequence[Node], elements: Sequence[Element]
) -> Dict[str, float]:
    node_map = {n.id: n for n in nodes}
    out: Dict[str, float] = {}
    for e in elements:
        ni, nj = node_map[e.node_i], node_map[e.node_j]
        out[e.id] = float(np.hypot(nj.x - ni.x, nj.y - ni.y))
    return out


def feature_matrix(
    nodes: Sequence[Node],
    elements: Sequence[Element],
    loads: Mapping[str, Mapping[str, float]],
    scenario: str,
    t_target: float,
    alpha: float = 0.7,
) -> Tuple[NDArray[np.float64], List[str], NDArray[np.float64]]:
    """Rows of :data:`FEATURES`, member ids, and the exact CI targets.

    Performs exactly one baseline solve (for the force features) plus the
    exact CI sweep that provides the *targets only*.
    """
    temps = get_scenario_temperatures(list(nodes), list(elements), scenario, t_target)
    forces = member_axial_forces(nodes, elements, loads, temps)
    result = compute_ci_for_topology(
        list(nodes), list(elements), dict(loads), {}, scenario, t_target, alpha
    )
    lengths = _member_lengths(nodes, elements)
    mean_len = float(np.mean(list(lengths.values())))
    max_force = max((abs(v) for v in forces.values()), default=0.0)
    min_x, span = span_bounds(list(nodes))
    centroids = member_centroids(list(nodes), list(elements))
    degree: Dict[str, int] = {n.id: 0 for n in nodes}
    for e in elements:
        degree[e.node_i] += 1
        degree[e.node_j] += 1
    node_map = {n.id: n for n in nodes}
    rows: List[List[float]] = []
    ids: List[str] = []
    targets: List[float] = []
    for e in elements:
        ni, nj = node_map[e.node_i], node_map[e.node_j]
        length = lengths[e.id]
        cos_t = abs((nj.x - ni.x) / length)
        n_force = forces[e.id]
        rows.append(
            [
                abs(n_force) / max_force if max_force > 0.0 else 0.0,
                float(np.sign(n_force)),
                length / mean_len if mean_len > 0.0 else 0.0,
                cos_t,
                1.0 if temps[e.id] > T_AMBIENT else 0.0,
                temps[e.id] / 1000.0,
                (centroids[e.id] - min_x) / span,
                (degree[e.node_i] + degree[e.node_j]) / 2.0,
            ]
        )
        ids.append(e.id)
        targets.append(result.ci_values[e.id])
    return np.asarray(rows, dtype=float), ids, np.asarray(targets, dtype=float)


@dataclass
class RidgeSurrogate:
    """Closed-form ridge regression on standardised features."""

    alpha: float = 1.0
    mean_: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    scale_: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    coef_: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    bias_: float = 0.0

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> "RidgeSurrogate":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            msg = f"shape mismatch: x{x.shape}, y{y.shape}"
            raise ValueError(msg)
        if x.shape[0] < 2:
            msg = "need at least two training samples"
            raise ValueError(msg)
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale_ = np.where(scale > 0.0, scale, 1.0)
        z = (x - self.mean_) / self.scale_
        ybar = float(y.mean())
        gram = z.T @ z + float(self.alpha) * np.eye(z.shape[1])
        self.coef_ = np.linalg.solve(gram, z.T @ (y - ybar))
        self.bias_ = ybar
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coef_ is None or self.mean_ is None or self.scale_ is None:
            msg = "RidgeSurrogate.predict called before fit"
            raise RuntimeError(msg)
        z = (np.asarray(x, dtype=float) - self.mean_) / self.scale_
        return z @ self.coef_ + self.bias_


@dataclass(frozen=True)
class StateRow:
    """Per-state transfer result on the held-out families."""

    topology: str
    scenario: str
    temperature: float
    n_members: int
    rho: float
    rho_force_baseline: float


@dataclass(frozen=True)
class CrossValidationResult:
    """Aggregated cross-family transfer evidence."""

    train_family: str
    test_families: Tuple[str, ...]
    n_train_samples: int
    features: Tuple[str, ...]
    ridge_alpha: float
    rows: Tuple[StateRow, ...]
    rho_mean: float
    rho_median: float
    rho_min: float
    rho_mean_force_baseline: float
    rho_by_family: Dict[str, float]
    gate: float = CV_RHO_GATE

    @property
    def passed(self) -> bool:
        return bool(np.isfinite(self.rho_mean) and self.rho_mean > self.gate)


def _family_of(topology_name: str) -> str:
    return topology_name.split("_")[0].lower()


def cross_family_cv(
    topologies: Sequence[Tuple[str, str, Sequence[Node], Sequence[Element], Mapping]],
    scenarios: Sequence[str] = ("local_left", "local_mid", "local_right"),
    temperatures: Sequence[float] = (200.0, 400.0, 600.0, 800.0),
    alpha: float = 0.7,
    train_family: str = "pratt",
    test_families: Sequence[str] = ("warren", "howe"),
    ridge_alpha: float = 1.0,
    shuffle_targets_seed: Optional[int] = None,
) -> CrossValidationResult:
    """Run the whole transfer experiment.

    ``topologies`` is a sequence of ``(name, family, nodes, elements,
    loads)``.  Local scenarios are used deliberately: under a uniform field
    the CI ranking is temperature-invariant (Lemma 1), so uniform states
    would add duplicate targets with conflicting feature values instead of
    information.
    """
    train_x: List[NDArray[np.float64]] = []
    train_y: List[NDArray[np.float64]] = []
    test_states: List[
        Tuple[str, str, float, NDArray[np.float64], NDArray[np.float64]]
    ] = []
    for name, family, nodes, elements, loads in topologies:
        fam = family.lower()
        if fam != train_family and fam not in {f.lower() for f in test_families}:
            continue
        for scenario in scenarios:
            for t in temperatures:
                x, _ids, y = feature_matrix(
                    nodes, elements, loads, scenario, float(t), alpha
                )
                if fam == train_family:
                    train_x.append(x)
                    train_y.append(y)
                else:
                    test_states.append((name, scenario, float(t), x, y))
    if not train_x:
        msg = f"no training topologies for family '{train_family}'"
        raise ValueError(msg)
    if not test_states:
        msg = f"no test topologies for families {tuple(test_families)}"
        raise ValueError(msg)
    x_train = np.vstack(train_x)
    y_train = np.concatenate(train_y)
    if shuffle_targets_seed is not None:
        # permutation control: detach targets from their members; the mean
        # test rho must collapse to ~0, proving the metric is not inflated
        # by construction (e.g. by member ordering or state duplication).
        rng = np.random.default_rng(shuffle_targets_seed)
        y_train = rng.permutation(y_train)
    model = RidgeSurrogate(alpha=ridge_alpha).fit(x_train, y_train)

    rows: List[StateRow] = []
    for name, scenario, t, x, y in test_states:
        pred = model.predict(x)
        rho = rank_correlation(
            {str(i): float(p) for i, p in enumerate(pred)},
            {str(i): float(v) for i, v in enumerate(y)},
        )
        rho_base = rank_correlation(
            {str(i): float(v) for i, v in enumerate(x[:, 0])},
            {str(i): float(v) for i, v in enumerate(y)},
        )
        rows.append(
            StateRow(
                topology=name,
                scenario=scenario,
                temperature=t,
                n_members=int(x.shape[0]),
                rho=rho,
                rho_force_baseline=rho_base,
            )
        )
    rhos = np.asarray([r.rho for r in rows], dtype=float)
    base = np.asarray([r.rho_force_baseline for r in rows], dtype=float)
    by_family: Dict[str, float] = {}
    for fam in test_families:
        sel = [r.rho for r in rows if _family_of(r.topology) == fam.lower()]
        if sel:
            by_family[str(fam)] = float(np.mean(sel))
    return CrossValidationResult(
        train_family=train_family,
        test_families=tuple(test_families),
        n_train_samples=int(x_train.shape[0]),
        features=FEATURES,
        ridge_alpha=float(ridge_alpha),
        rows=tuple(rows),
        rho_mean=float(np.mean(rhos)),
        rho_median=float(np.median(rhos)),
        rho_min=float(np.min(rhos)),
        rho_mean_force_baseline=float(np.mean(base)),
        rho_by_family=by_family,
    )
