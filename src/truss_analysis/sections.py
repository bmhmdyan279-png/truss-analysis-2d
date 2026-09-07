"""Cross-section models for truss members (prompt-05, task T2).

History and honesty
-------------------
The pre-prompt-5 code computed ``I_sec = A**2 / 12`` and documented it as a
"square hollow-section approximation".  ``A**2/12`` is the exact second moment
of area of a *solid* square of side ``sqrt(A)``; for a hollow section it is
simply wrong (critic 8 D-3, CONTEXT_LOCK T2/DR-004).  This module replaces it
with an exact, documented model and never calls a solid-section formula
"hollow":

* :class:`SquareHSS` — an **idealised** square hollow section (uniform wall
  thickness, sharp corners): ``A = b^2 - (b - 2t)^2 = 4 t (b - t)`` and
  ``I = (b^4 - (b - 2t)^4) / 12`` (exact for the idealised shape; equal about
  both axes by symmetry).  These are NOT commercial product properties.
* :func:`idealised_square_hss` — solves ``(b, t)`` from a target area and a
  documented width-to-thickness ratio ``r = b/t``:
  ``A = 4 b^2 (r - 1) / r^2``  =>  ``b = (r / 2) * sqrt(A / (r - 1))``.
* :class:`SectionCatalog` — immutable named-section table with nearest-area
  lookup: the hook through which a real commercial table (e.g. EN 10210) can
  be plugged in later without code changes (prompt-05 option A; DL-025
  records why the parametric idealisation is the library default today).
* :func:`euler_buckling_load` — ``P_cr = pi^2 E I / (k L)^2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "SquareHSS",
    "SectionCatalog",
    "euler_buckling_load",
    "idealised_square_hss",
]

_MIN_THICKNESS_RATIO = 2.0  # r = b/t must keep (b - 2t) > 0


@dataclass(frozen=True)
class SquareHSS:
    """Idealised square hollow section, uniform wall thickness.

    Parameters
    ----------
    name:
        Identifier (e.g. ``"idealised-r25-A0.01"``).
    b:
        Outer width [m].
    t:
        Wall thickness [m]; must satisfy ``0 < 2t < b``.
    """

    name: str
    b: float
    t: float

    def __post_init__(self) -> None:
        if self.b <= 0.0:
            msg = f"outer width must be > 0, got {self.b}"
            raise ValueError(msg)
        if self.t <= 0.0:
            msg = f"wall thickness must be > 0, got {self.t}"
            raise ValueError(msg)
        if 2.0 * self.t >= self.b:
            msg = f"wall thickness leaves no void: 2t={2 * self.t} >= b={self.b}"
            raise ValueError(msg)

    @property
    def width_to_thickness(self) -> float:
        """r = b / t [-]."""
        return self.b / self.t

    @property
    def area(self) -> float:
        """Exact cross-sectional area ``4 t (b - t)`` [m²]."""
        return 4.0 * self.t * (self.b - self.t)

    @property
    def i_sec(self) -> float:
        """Exact second moment of area about either axis [m⁴]."""
        bi = self.b - 2.0 * self.t
        return (self.b**4 - bi**4) / 12.0

    @property
    def radius_of_gyration(self) -> float:
        """i = sqrt(I / A) [m]."""
        return math.sqrt(self.i_sec / self.area)


def idealised_square_hss(area: float, thickness_ratio: float = 25.0) -> SquareHSS:
    """Solve the idealised square HSS with a given area and b/t ratio.

    ``A = 4 b^2 (r - 1) / r^2`` inverted for ``b``, then ``t = b / r``.
    ``thickness_ratio`` must exceed 2 (otherwise the section is solid or
    negative-void).  The result reproduces ``area`` exactly (to fp round-off).
    """
    if area <= 0.0:
        msg = f"area must be > 0, got {area}"
        raise ValueError(msg)
    if thickness_ratio < _MIN_THICKNESS_RATIO:
        msg = (
            f"thickness_ratio must be >= {_MIN_THICKNESS_RATIO}, got {thickness_ratio}"
        )
        raise ValueError(msg)
    r = thickness_ratio
    b = (r / 2.0) * math.sqrt(area / (r - 1.0))
    t = b / r
    return SquareHSS(name=f"idealised-r{r:g}-A{area:g}", b=b, t=t)


@dataclass(frozen=True)
class SectionCatalog:
    """Immutable table of named sections with nearest-area selection.

    The library default catalogue is empty on purpose: commercial tables
    (EN 10210 etc.) are licensed data and must be supplied by the user via
    :meth:`from_rows`; until then :func:`idealised_square_hss` is the
    documented parametric model (DL-025).
    """

    name: str
    rows: tuple[SquareHSS, ...]

    @classmethod
    def from_rows(cls, name: str, rows: Sequence[SquareHSS]) -> SectionCatalog:
        return cls(name=name, rows=tuple(rows))

    def nearest_to_area(self, area: float) -> SquareHSS:
        """Row whose area is closest (relative distance) to ``area``."""
        if not self.rows:
            msg = f"catalogue '{self.name}' is empty"
            raise ValueError(msg)
        return min(self.rows, key=lambda row: abs(row.area - area) / row.area)


def euler_buckling_load(
    i_sec: float,
    length: float,
    youngs_modulus: float,
    effective_length_factor: float = 1.0,
) -> float:
    """Euler elastic buckling load ``P_cr = pi^2 E I / (k L)^2`` [N]."""
    if length <= 0.0:
        msg = f"length must be > 0, got {length}"
        raise ValueError(msg)
    kl = effective_length_factor * length
    return math.pi**2 * youngs_modulus * i_sec / kl**2
