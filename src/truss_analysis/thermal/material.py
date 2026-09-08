"""Thermal material degradation models - compatibility layer.

The historical API (``get_eurocode_k_E`` / ``get_eurocode_k_y``) is preserved
for backwards compatibility, but every value now comes from the EN 1993-1-2:2005
single source of truth (:mod:`truss_analysis.material.steel_eurocode`, fixture
``data/en1993_1_2_table3_1.json``).

Behavioural change versus the legacy implementation (deliberate):
* the legacy tables truncated at 1000 degC (returning 1.0 below 20 and 0.0
  above 1000); the material model clamps into the standard's validity range
  [20, 1200] and linearly interpolates Table 3.1, so temperatures in
  (1000, 1200] now follow the standard instead of jumping to 0.
"""

from __future__ import annotations

import warnings

from ..material.steel_eurocode import k_E as _material_k_E
from ..material.steel_eurocode import k_y as _material_k_y

__all__ = ["get_eurocode_k_E", "get_eurocode_k_y"]

_DEPRECATION_K_E = (
    "truss_analysis.thermal.material.get_eurocode_k_E is deprecated; use "
    "truss_analysis.material.steel_eurocode.k_E (EN 1993-1-2:2005 model)."
)
_DEPRECATION_K_Y = (
    "truss_analysis.thermal.material.get_eurocode_k_y is deprecated; use "
    "truss_analysis.material.steel_eurocode.k_y (EN 1993-1-2:2005 model)."
)


def get_eurocode_k_E(temp_c: float) -> float:
    """Reduction factor for Young's modulus (deprecated wrapper).

    Delegate to the material model; emits :class:`DeprecationWarning`.
    """
    warnings.warn(_DEPRECATION_K_E, DeprecationWarning, stacklevel=2)
    return float(_material_k_E(temp_c))


def get_eurocode_k_y(temp_c: float) -> float:
    """Reduction factor for yield strength (deprecated wrapper).

    Delegate to the material model; emits :class:`DeprecationWarning`.
    """
    warnings.warn(_DEPRECATION_K_Y, DeprecationWarning, stacklevel=2)
    return float(_material_k_y(temp_c))
