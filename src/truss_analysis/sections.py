"""Cross-section models for truss members.

The library ships an *idealised* square hollow-section (HSS) model and an
immutable catalogue type through which real commercial tables can be
plugged in later without code changes.

Models
------
* :class:`SquareHSS` - an **idealised** square hollow section (uniform wall
  thickness, sharp corners): ``A = b^2 - (b - 2t)^2 = 4 t (b - t)`` and
  ``I = (b^4 - (b - 2t)^4) / 12`` (exact for the idealised shape; equal
  about both axes by symmetry).  These are NOT commercial product
  properties.
* :func:`idealised_square_hss` - solves ``(b, t)`` from a target area and a
  documented width-to-thickness ratio ``r = b/t``:
  ``A = 4 b^2 (r - 1) / r^2``  =>  ``b = (r / 2) * sqrt(A / (r - 1))``.
* :class:`SectionCatalog` - immutable named-section table with nearest-area
  lookup.
* :func:`euler_buckling_load` - ``P_cr = pi^2 E I / (k L)^2``.

Notes
-----
The solid-square formula ``I = A^2 / 12`` is the exact second moment of
area of a *solid* square of side ``sqrt(A)``; it is NOT a valid hollow
section approximation and is deliberately not used anywhere here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "SectionCatalog",
    "SquareHSS",
    "euler_buckling_load",
    "idealised_square_hss",
]

_MIN_THICKNESS_RATIO = 2.0  # r = b/t must keep (b - 2t) > 0


@dataclass(frozen=True)
class SquareHSS:
    """Idealised square hollow section with uniform wall thickness.

    Parameters
    ----------
    name : str
        Identifier (e.g. ``"idealised-r25-A0.01"``).
    b : float
        Outer width [m].
    t : float
        Wall thickness [m]; must satisfy ``0 < 2t < b``.

    Raises
    ------
    ValueError
        If ``b`` or ``t`` is non-positive, or the wall leaves no void.
    """

    name: str
    b: float
    t: float

    def __post_init__(self) -> None:
        """Validate that the section geometry leaves a positive void."""
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
        """Return the width-to-thickness ratio ``r = b / t`` [-]."""
        return self.b / self.t

    @property
    def area(self) -> float:
        """Return the exact cross-sectional area ``4 t (b - t)`` [m^2]."""
        return 4.0 * self.t * (self.b - self.t)

    @property
    def i_sec(self) -> float:
        """Return the exact second moment of area about either axis [m^4]."""
        bi = self.b - 2.0 * self.t
        return (self.b**4 - bi**4) / 12.0

    @property
    def radius_of_gyration(self) -> float:
        """Return the radius of gyration ``sqrt(I / A)`` [m]."""
        return math.sqrt(self.i_sec / self.area)


def idealised_square_hss(area: float, thickness_ratio: float = 25.0) -> SquareHSS:
    """Construct the idealised square HSS with a given area and ``b/t`` ratio.

    Inverts ``A = 4 b^2 (r - 1) / r^2`` for ``b``, then sets ``t = b / r``.

    Parameters
    ----------
    area : float
        Target cross-sectional area [m^2]; must be positive.
    thickness_ratio : float, default 25.0
        Width-to-thickness ratio ``r = b/t``; must be at least 2 (otherwise
        the section is solid or has a negative void).

    Returns
    -------
    SquareHSS
        Section whose :attr:`~SquareHSS.area` reproduces ``area`` exactly
        (to floating-point round-off).

    Raises
    ------
    ValueError
        If ``area`` is non-positive or ``thickness_ratio`` is below 2.
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
    (e.g. EN 10210) are licensed data and must be supplied by the user via
    :meth:`from_rows`. Until then :func:`idealised_square_hss` is the
    documented parametric model.

    Attributes
    ----------
    name : str
        Catalogue identifier.
    rows : tuple[SquareHSS, ...]
        Immutable sequence of available sections.
    """

    name: str
    rows: tuple[SquareHSS, ...]

    @classmethod
    def from_rows(cls, name: str, rows: Sequence[SquareHSS]) -> SectionCatalog:
        """Build a catalogue from a sequence of sections (copied to a tuple)."""
        return cls(name=name, rows=tuple(rows))

    def nearest_to_area(self, area: float) -> SquareHSS:
        """Return the row whose area is closest (relative distance) to ``area``.

        Raises
        ------
        ValueError
            If the catalogue is empty.
        """
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
    """Return the Euler elastic buckling load ``P_cr = pi^2 E I / (k L)^2`` [N].

    Parameters
    ----------
    i_sec : float
        Second moment of area [m^4].
    length : float
        Member length ``L`` [m]; must be positive.
    youngs_modulus : float
        Young's modulus ``E`` [Pa].
    effective_length_factor : float, default 1.0
        Effective-length factor ``k`` (pin-ended column: 1.0).

    Returns
    -------
    float
        Elastic critical (buckling) load [N].

    Raises
    ------
    ValueError
        If ``length`` is non-positive.
    """
    if length <= 0.0:
        msg = f"length must be > 0, got {length}"
        raise ValueError(msg)
    kl = effective_length_factor * length
    return math.pi**2 * youngs_modulus * i_sec / kl**2
