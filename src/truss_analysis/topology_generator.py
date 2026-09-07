"""Parametric topology generator for 2D pin-jointed planar trusses.

Generates Warren, Pratt, and Howe truss families as JSON-compatible
dictionaries conforming to the truss-analysis-2d input schema.

Phase 1 deliverable — geometry and connectivity only.
No thermal loading is applied at this stage (``delta_T = 0``).

Lemma 1 readiness
-----------------
The generator produces *uniform* section properties across all members.
This guarantees that a uniform temperature field (which scales every
member's E by the same factor) cannot alter the stiffness matrix
*eigenstructure*, preserving CI ranking (Kendall τ = 1).  Any future
thermal module must therefore apply degradation through this generator's
output rather than mutating individual member properties ad-hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

# ----------------------------------------------------------------------
#  Public API: both the object-oriented generator and a simple function
# ----------------------------------------------------------------------

__all__ = [
    "TrussFamily",
    "TrussConfig",
    "TopologyGenerator",
    "generate_topology",  # <-- added for scripts/compute_phase2_deterministic.py
]


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
    moment_of_inertia:
        Second moment of area [m⁴]. If ``None``, computed as ``A²/12``
        (square hollow-section approximation per Phase 1 scope-lock).

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
    moment_of_inertia: float | None = None

    def __post_init__(self) -> None:
        if self.n_panels < 2:
            raise ValueError(f"n_panels must be >= 2, got {self.n_panels}")
        if self.span <= 0.0:
            raise ValueError(f"span must be > 0, got {self.span}")
        if self.height <= 0.0:
            raise ValueError(f"height must be > 0, got {self.height}")

    @property
    def i_sec(self) -> float:
        """Effective second moment of area [m⁴]."""
        if self.moment_of_inertia is not None:
            return self.moment_of_inertia
        return self.area**2 / 12.0


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
    # Determinate control (negative control for H1)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def generate_determinate_control(
        span: float = 10.0,
        height: float = 2.0,
        area: float = 0.01,
        youngs_modulus: float = 210.0e9,
    ) -> dict[str, Any]:
        """Generate a simple 3-member statically determinate truss.

        This serves as the *negative control* for the H1 hypothesis
        test (Phase 8 scope): a determinate truss has no redundant
        load paths, so all members are equally critical.

        Parameters
        ----------
        span:
            Span length [m].
        height:
            Apex height [m].
        area:
            Cross-sectional area [m²].
        youngs_modulus:
            Young's modulus [Pa].

        Returns
        -------
        dict[str, Any]
            Model dictionary for a statically determinate triangle.
        """
        i_sec = area**2 / 12.0
        elem_template: dict[str, Any] = {
            "A": area,
            "E": youngs_modulus,
            "alpha": 1.2e-5,
            "delta_T": 0.0,
            "delta_L0": 0.0,
            "effective_length_factor": 1.0,
            "section_type": "rectangular",
            "I_sec": i_sec,
        }
        return {
            "units": "SI",
            "temperature_change": 0.0,
            "nodes": [
                {
                    "id": 1,
                    "x": 0.0,
                    "y": 0.0,
                    "is_support": True,
                    "support_dx": True,
                    "support_dy": True,
                },
                {
                    "id": 2,
                    "x": span,
                    "y": 0.0,
                    "is_support": True,
                    "support_dx": False,
                    "support_dy": True,
                },
                {"id": 3, "x": span / 2.0, "y": height, "is_support": False},
            ],
            "elements": [
                {"id": 1, "node_i": 1, "node_j": 2, **elem_template},
                {"id": 2, "node_i": 1, "node_j": 3, **elem_template},
                {"id": 3, "node_i": 2, "node_j": 3, **elem_template},
            ],
            "loads": [{"node_id": 3, "Fx": 0.0, "Fy": -10000.0}],
            "options": {
                "use_sparse": True,
                "bc_method": "elimination",
                "penalty_value": 1.0e12,
                "plot_results": True,
                "displacement_scale": "auto",
            },
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

        # Diagonals — Pratt: slope *down* toward mid-span
        mid = n // 2
        for i in range(n):
            if i < mid:
                if i + 1 < n:
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

        # Diagonals — Howe: slope *up* toward mid-span (mirror of Pratt)
        mid = n // 2
        for i in range(n):
            if i < mid:
                if i + 1 < n:
                    add(t(i + 1), b(i))
            elif (n % 2 == 0) or (i > mid):
                if 1 <= i < n:
                    add(b(i + 1), t(i))

    # ──────────────────────────────────────────────────────────────
    # Loads
    # ──────────────────────────────────────────────────────────────

    def _build_loads(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply default vertical loads to all top-chord (non-support) nodes."""
        loads: list[dict[str, Any]] = []
        for node in nodes:
            if not node["is_support"] and node["y"] > 0.0:
                loads.append(
                    {
                        "node_id": node["id"],
                        "Fx": 0.0,
                        "Fy": -10000.0,
                    }
                )
        return loads

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
            "section_type": "rectangular",
            "I_sec": cfg.i_sec,
        }


# ======================================================================
#  Simple function‑based interface (for scripts/compute_phase2_deterministic.py)
# ======================================================================


def generate_topology(
    family: str,
    n_panels: int,
    span: float,
    height: float,
    area: float = 0.01,
    youngs_modulus: float = 210.0e9,
    thermal_expansion: float = 1.2e-5,
) -> dict[str, Any]:
    """Generate a complete truss model dictionary using the object‑oriented generator.

    This is a convenience wrapper around ``TopologyGenerator``, intended for
    scripts that expect a simple function with this exact signature.

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

    Returns
    -------
    dict[str, Any]
        Model dictionary conforming to the truss‑analysis‑2d input schema.

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
    )
    return TopologyGenerator(cfg).generate()
