"""Parametric topology generator for 2D pin-jointed planar trusses.

Generates Warren, Pratt, and Howe truss families plus three statically
determinate controls as JSON-compatible dictionaries conforming to the
truss-analysis-2d input schema.  Geometry and connectivity only: no thermal
loading is applied at this stage (``delta_T = 0``).

Load model (prompt-05, task T1)
-------------------------------
The pre-prompt-5 generator hard-coded ``Fy = -10 kN`` per node on nodes with
``not is_support and y > 0``.  That pattern is **family-dependent**: in Warren
the unloaded non-support bottom-chord nodes carry no load while in Pratt/Howe
the load set differs, so total demand changed between families and between
``n_panels`` — corrupting any cross-topology comparison (H2 in particular).

The model now is:

* **Loaded set = all non-support nodes**, independent of ``y`` and family.
* The vertical demand is a single configuration parameter ``total_load``
  [N] (default 100 kN), shared **equally** by the loaded nodes:
  ``Fy_i = -total_load / n_loaded``.  Keeping the *total* constant makes the
  demand comparable across ``n_panels`` and families (screening-level
  idealisation; a tributary-area deck model is a possible refinement, not a
  correctness requirement, because CI is a ratio and therefore invariant to
  the overall load scale — what matters is the *pattern*, and this pattern is
  family-independent by construction).
* Horizontal nodal loads are zero (gravity-type screening load).

Lemma 1 (correct statement, prompt-05 task T4)
----------------------------------------------
Lemma 1 (uniform-temperature invariance of the CI ranking) holds because a
uniform temperature field scales the **whole stiffness matrix** by
``k_E(theta)`` when sections are uniform: ``K(theta) = k_E(theta) K_0``, so
every perturbed/base displacement ratio — and therefore every CI and every
rank comparison — is temperature-invariant.  It does **not** follow from
"uniform sections imply tau = 1" by any eigenstructure argument; the legacy
wording was wrong and was removed.  The measured form of the lemma lives in
``tests/test_lemma1_uniform_invariance.py`` (prompt-04).

Determinism (prompt-05, task T6)
--------------------------------
:func:`model_to_json` emits canonical JSON (sorted keys, fixed separators,
no NaN) and :func:`content_hash` returns its SHA-256; generating the same
configuration twice yields byte-identical JSON and an identical hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from truss_analysis.sections import SquareHSS, idealised_square_hss

# ----------------------------------------------------------------------
#  Public API: both the object-oriented generator and a simple function
# ----------------------------------------------------------------------

__all__ = [
    "TrussFamily",
    "TrussConfig",
    "TopologyGenerator",
    "content_hash",
    "generate_topology",
    "model_to_json",
]

DEFAULT_TOTAL_LOAD: Final[float] = 100.0e3  # N, screening-level total demand


class TrussFamily(Enum):
    """Supported truss topology families."""

    WARREN = "warren"
    PRATT = "pratt"
    HOWE = "howe"


@dataclass(frozen=True)
class TrussConfig:
    """Immutable configuration for a parametric truss.

    Parameters
    ----------
    family:
        Truss family to generate.
    n_panels:
        Number of panels (must be >= 2).
    span:
        Total span length [m].
    height:
        Truss height [m].
    area:
        Uniform cross-sectional area [m²].
    youngs_modulus:
        Young's modulus [Pa].
    thermal_expansion:
        Coefficient of thermal expansion [1/°C].
    total_load:
        Total vertical demand [N] shared equally by all non-support nodes
        (see the module-level *Load model* section).
    section_thickness_ratio:
        Width-to-thickness ratio ``b/t`` of the idealised square HSS used to
        derive ``I_sec`` from ``area`` (see :mod:`truss_analysis.sections`).
    moment_of_inertia:
        Optional explicit override of ``I_sec`` [m⁴]; when ``None`` the
        idealised square HSS value is used.

    Raises
    ------
    ValueError
        If geometric parameters are non-physical.
    """

    family: TrussFamily
    n_panels: int
    span: float
    height: float
    area: float = 0.01
    youngs_modulus: float = 210.0e9
    thermal_expansion: float = 1.2e-5
    total_load: float = DEFAULT_TOTAL_LOAD
    section_thickness_ratio: float = 25.0
    moment_of_inertia: float | None = None

    def __post_init__(self) -> None:
        if self.n_panels < 2:
            raise ValueError(f"n_panels must be >= 2, got {self.n_panels}")
        if self.span <= 0.0:
            raise ValueError(f"span must be > 0, got {self.span}")
        if self.height <= 0.0:
            raise ValueError(f"height must be > 0, got {self.height}")
        if self.total_load <= 0.0:
            raise ValueError(f"total_load must be > 0, got {self.total_load}")

    @property
    def section(self) -> SquareHSS:
        """Idealised square HSS matching ``area`` at the configured b/t."""
        return idealised_square_hss(self.area, self.section_thickness_ratio)

    @property
    def i_sec(self) -> float:
        """Second moment of area [m⁴]: explicit override or idealised HSS."""
        if self.moment_of_inertia is not None:
            return self.moment_of_inertia
        return self.section.i_sec


class TopologyGenerator:
    """Generates a parametric truss model dictionary.

    Output schema mirrors ``reference_problem.json``:
    ``units``, ``temperature_change``, ``nodes``, ``elements``,
    ``loads``, ``options``.

    Parameters
    ----------
    config:
        Truss configuration.

    Examples
    --------
    >>> cfg = TrussConfig(TrussFamily.WARREN, n_panels=4, span=16.0, height=3.0)
    >>> model = TopologyGenerator(cfg).generate()
    >>> isinstance(model, dict) and "nodes" in model
    True
    """

    _OPTIONS: Final[dict[str, Any]] = {
        "use_sparse": True,
        "bc_method": "elimination",
        "penalty_value": 1.0e12,
        "plot_results": True,
        "displacement_scale": "auto",
    }

    def __init__(self, config: TrussConfig) -> None:
        self._cfg = config
        self._panel_len: float = config.span / config.n_panels

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def generate(self) -> dict[str, Any]:
        """Generate the complete truss model dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-serializable model dictionary.
        """
        nodes = self._build_nodes()
        elements = self._build_elements(nodes)
        loads = self._build_loads(nodes)
        return {
            "units": "SI",
            "temperature_change": 0.0,
            "nodes": nodes,
            "elements": elements,
            "loads": loads,
            "options": dict(self._OPTIONS),
        }

    # ──────────────────────────────────────────────────────────────
    # Determinate controls (negative controls for H1) — prompt-05 T3
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def generate_determinate_control(
        index: int = 1,
        span: float = 10.0,
        height: float = 2.0,
        area: float = 0.01,
        youngs_modulus: float = 210.0e9,
        total_load: float = DEFAULT_TOTAL_LOAD,
    ) -> dict[str, Any]:
        """Generate one of three **geometrically distinct** determinate trusses.

        ``index=1``: single triangle (3 nodes / 3 members).
        ``index=2``: two-panel Warren-like chain (5 nodes / 7 members).
        ``index=3``: the chain plus an apex node (6 nodes / 9 members).

        All three satisfy ``m + r = 2j`` exactly (statically determinate) and
        become mechanisms when any single member is removed — asserted in
        ``tests/test_topology_generator.py``.  The legacy single-triangle
        variant with three heights (DR-003) is replaced: height variety alone
        was not a geometric variety.

        Parameters
        ----------
        index:
            Control geometry selector (1, 2 or 3).
        span, height, area, youngs_modulus, total_load:
            As in :class:`TrussConfig`; the total load is shared equally by
            all non-support nodes.
        """
        if index not in (1, 2, 3):
            msg = f"index must be 1, 2 or 3, got {index}"
            raise ValueError(msg)
        i_sec = idealised_square_hss(area).i_sec
        template: dict[str, Any] = {
            "A": area,
            "E": youngs_modulus,
            "alpha": 1.2e-5,
            "delta_T": 0.0,
            "delta_L0": 0.0,
            "effective_length_factor": 1.0,
            "section_type": "idealised_square_hss",
            "I_sec": i_sec,
        }

        def node(nid: int, x: float, y: float, support: str = "") -> dict[str, Any]:
            out: dict[str, Any] = {
                "id": nid,
                "x": x,
                "y": y,
                "is_support": bool(support),
            }
            if support == "pin":
                out["support_dx"] = True
                out["support_dy"] = True
            elif support == "roller":
                out["support_dx"] = False
                out["support_dy"] = True
            return out

        if index == 1:
            nodes = [
                node(1, 0.0, 0.0, "pin"),
                node(2, span, 0.0, "roller"),
                node(3, span / 2.0, height),
            ]
            members = [(1, 2), (1, 3), (2, 3)]
        elif index == 2:
            nodes = [
                node(1, 0.0, 0.0, "pin"),
                node(2, span / 2.0, 0.0),
                node(3, span, 0.0, "roller"),
                node(4, span / 4.0, height),
                node(5, 3.0 * span / 4.0, height),
            ]
            members = [(1, 2), (2, 3), (4, 5), (1, 4), (4, 2), (2, 5), (5, 3)]
        else:
            nodes = [
                node(1, 0.0, 0.0, "pin"),
                node(2, span / 2.0, 0.0),
                node(3, span, 0.0, "roller"),
                node(4, span / 4.0, height),
                node(5, 3.0 * span / 4.0, height),
                node(6, span / 2.0, 2.0 * height),
            ]
            members = [
                (1, 2),
                (2, 3),
                (4, 5),
                (1, 4),
                (4, 2),
                (2, 5),
                (5, 3),
                (4, 6),
                (6, 5),
            ]
        loaded = [n["id"] for n in nodes if not n["is_support"]]
        share = -total_load / len(loaded)
        return {
            "units": "SI",
            "temperature_change": 0.0,
            "nodes": nodes,
            "elements": [
                {"id": i, "node_i": ni, "node_j": nj, **template}
                for i, (ni, nj) in enumerate(members, start=1)
            ],
            "loads": [{"node_id": nid, "Fx": 0.0, "Fy": share} for nid in loaded],
            "options": dict(TopologyGenerator._OPTIONS),
        }

    # ──────────────────────────────────────────────────────────────
    # Node generation
    # ──────────────────────────────────────────────────────────────

    def _build_nodes(self) -> list[dict[str, Any]]:
        cfg = self._cfg
        n = cfg.n_panels
        nodes: list[dict[str, Any]] = []
        nid = 1

        # Bottom chord: B_0 .. B_n
        for i in range(n + 1):
            is_left = i == 0
            is_right = i == n
            node: dict[str, Any] = {
                "id": nid,
                "x": round(i * self._panel_len, 10),
                "y": 0.0,
                "is_support": is_left or is_right,
            }
            if is_left:
                node["support_dx"] = True
                node["support_dy"] = True
            elif is_right:
                node["support_dx"] = False
                node["support_dy"] = True
            nodes.append(node)
            nid += 1

        # Top chord
        if cfg.family is TrussFamily.WARREN:
            # Top nodes at mid-panel positions: T_0 .. T_{n-1}
            for i in range(n):
                nodes.append(
                    {
                        "id": nid,
                        "x": round((i + 0.5) * self._panel_len, 10),
                        "y": cfg.height,
                        "is_support": False,
                    }
                )
                nid += 1
        else:
            # Pratt / Howe: top nodes at interior panel points T_1 .. T_{n-1}
            for i in range(1, n):
                nodes.append(
                    {
                        "id": nid,
                        "x": round(i * self._panel_len, 10),
                        "y": cfg.height,
                        "is_support": False,
                    }
                )
                nid += 1

        return nodes

    # ──────────────────────────────────────────────────────────────
    # Element generation
    # ──────────────────────────────────────────────────────────────

    def _build_elements(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cfg = self._cfg
        n = cfg.n_panels
        node_map = self._node_lookup(nodes)
        elements: list[dict[str, Any]] = []
        eid = 1

        def add(ni: int, nj: int) -> None:
            nonlocal eid
            elements.append(self._element(eid, ni, nj))
            eid += 1

        def b(i: int) -> int:
            """Bottom-chord node id at panel index *i*."""
            return self._find(node_map, i * self._panel_len, 0.0)

        # ── Bottom chord (common to all families) ──
        for i in range(n):
            add(b(i), b(i + 1))

        if cfg.family is TrussFamily.WARREN:
            self._build_warren(add, b, node_map, n)
        elif cfg.family is TrussFamily.PRATT:
            self._build_pratt(add, b, node_map, n)
        elif cfg.family is TrussFamily.HOWE:
            self._build_howe(add, b, node_map, n)

        return elements

    def _build_warren(
        self,
        add: Any,
        b: Any,
        node_map: dict[tuple[int, int], int],
        n: int,
    ) -> None:
        h = self._cfg.height
        pl = self._panel_len

        def t(i: int) -> int:
            return self._find(node_map, (i + 0.5) * pl, h)

        # Top chord
        for i in range(n - 1):
            add(t(i), t(i + 1))

        # Diagonals: B_i → T_i → B_{i+1}
        for i in range(n):
            add(b(i), t(i))
            add(t(i), b(i + 1))

    def _build_pratt(
        self,
        add: Any,
        b: Any,
        node_map: dict[tuple[int, int], int],
        n: int,
    ) -> None:
        h = self._cfg.height
        pl = self._panel_len

        def t(i: int) -> int:
            return self._find(node_map, i * pl, h)

        # Top chord
        for i in range(1, n - 1):
            add(t(i), t(i + 1))

        # Verticals
        for i in range(1, n):
            add(b(i), t(i))

        # Diagonals — Pratt: slope *down* toward mid-span.  For an odd panel
        # count the centre panel needs its own diagonal, otherwise the panel
        # is a shear mechanism (DR-020, measured cond(K) ~ 1e16).
        mid = n // 2
        for i in range(n):
            if i < mid:
                if i + 1 < n:
                    add(b(i), t(i + 1))
            elif i == mid and n % 2 == 1:
                add(b(i), t(i + 1))
            elif (n % 2 == 0) or (i > mid):
                if 1 <= i < n:
                    add(t(i), b(i + 1))

    def _build_howe(
        self,
        add: Any,
        b: Any,
        node_map: dict[tuple[int, int], int],
        n: int,
    ) -> None:
        h = self._cfg.height
        pl = self._panel_len

        def t(i: int) -> int:
            return self._find(node_map, i * pl, h)

        # Top chord
        for i in range(1, n - 1):
            add(t(i), t(i + 1))

        # Verticals
        for i in range(1, n):
            add(b(i), t(i))

        # Diagonals — Howe: slope *up* toward mid-span (mirror of Pratt),
        # including the centre-panel diagonal for odd panel counts (DR-020).
        mid = n // 2
        for i in range(n):
            if i < mid:
                if i + 1 < n:
                    add(t(i + 1), b(i))
            elif i == mid and n % 2 == 1:
                add(t(i + 1), b(i))
            elif (n % 2 == 0) or (i > mid):
                if 1 <= i < n:
                    add(b(i + 1), t(i))

    # ──────────────────────────────────────────────────────────────
    # Loads — family-independent total-load model (prompt-05 T1)
    # ──────────────────────────────────────────────────────────────

    def _build_loads(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Share ``total_load`` equally over **all** non-support nodes.

        No ``y``-threshold, no hard-coded magnitude: see the module-level
        *Load model* section for why this pattern is family-independent and
        size-normalised.
        """
        loaded = [node for node in nodes if not node["is_support"]]
        if not loaded:
            return []
        share = -self._cfg.total_load / len(loaded)
        return [{"node_id": node["id"], "Fx": 0.0, "Fy": share} for node in loaded]

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _node_lookup(
        nodes: list[dict[str, Any]],
    ) -> dict[tuple[int, int], int]:
        """Build ``(x_mm, y_mm) → node_id`` lookup with 0.1 mm precision."""
        lookup: dict[tuple[int, int], int] = {}
        for node in nodes:
            key = (round(node["x"] * 1000), round(node["y"] * 1000))
            lookup[key] = node["id"]
        return lookup

    @staticmethod
    def _find(node_map: dict[tuple[int, int], int], x: float, y: float) -> int:
        key = (round(x * 1000), round(y * 1000))
        try:
            return node_map[key]
        except KeyError as exc:
            raise KeyError(f"No node found at ({x:.6f}, {y:.6f})") from exc

    def _element(self, eid: int, ni: int, nj: int) -> dict[str, Any]:
        cfg = self._cfg
        return {
            "id": eid,
            "node_i": ni,
            "node_j": nj,
            "A": cfg.area,
            "E": cfg.youngs_modulus,
            "alpha": cfg.thermal_expansion,
            "delta_T": 0.0,
            "delta_L0": 0.0,
            "effective_length_factor": 1.0,
            "section_type": "idealised_square_hss",
            "I_sec": cfg.i_sec,
        }


# ======================================================================
#  Deterministic serialisation (prompt-05 T6)
# ======================================================================


def model_to_json(model: dict[str, Any]) -> str:
    """Canonical JSON: sorted keys, fixed separators, NaN rejected."""
    return json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(model: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON — stable across regenerations."""
    return hashlib.sha256(model_to_json(model).encode("utf-8")).hexdigest()


# ======================================================================
#  Simple function‑based interface (for scripts/compute_phase2_*.py)
# ======================================================================


def generate_topology(
    family: str,
    n_panels: int,
    span: float,
    height: float,
    area: float = 0.01,
    youngs_modulus: float = 210.0e9,
    thermal_expansion: float = 1.2e-5,
    total_load: float = DEFAULT_TOTAL_LOAD,
) -> dict[str, Any]:
    """Generate a complete truss model dictionary using the object‑oriented generator.

    This is a convenience wrapper around ``TopologyGenerator``, intended for
    scripts that expect a simple function with this signature.

    Parameters
    ----------
    family : str
        One of ``'warren'``, ``'pratt'``, or ``'howe'`` (case‑insensitive).
    n_panels : int
        Number of panels (must be >= 2).
    span : float
        Total span [m].
    height : float
        Truss height [m].
    area : float, optional
        Uniform cross‑sectional area [m²] (default 0.01).
    youngs_modulus : float, optional
        Young's modulus [Pa] (default 210.0e9).
    thermal_expansion : float, optional
        Coefficient of thermal expansion [1/°C] (default 1.2e-5).
    total_load : float, optional
        Total vertical demand [N] shared equally by all non-support nodes
        (default 100 kN).

    Returns
    -------
    dict[str, Any]
        Model dictionary conforming to the truss-analysis-2d input schema.

    Raises
    ------
    ValueError
        If ``family`` is unknown or geometric parameters are invalid.
    """
    try:
        fam = TrussFamily(family.lower())
    except ValueError as exc:
        raise ValueError(
            f"Unknown family '{family}'. Allowed: 'warren', 'pratt', 'howe'."
        ) from exc

    cfg = TrussConfig(
        family=fam,
        n_panels=n_panels,
        span=span,
        height=height,
        area=area,
        youngs_modulus=youngs_modulus,
        thermal_expansion=thermal_expansion,
        total_load=total_load,
    )
    return TopologyGenerator(cfg).generate()
