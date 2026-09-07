"""Thermal Material Degradation Models — COMPATIBILITY LAYER.

The historical API (``get_eurocode_k_E`` / ``get_eurocode_k_y``) is preserved
for backwards compatibility, but every value now comes from the EN 1993-1-2:2005
single source of truth (:mod:`truss_analysis.material.steel_eurocode`, fixture
``data/en1993_1_2_table3_1.json``).  The pre-SSOT tables that lived in this
file were fabricated (k_E(600)=0.10 versus 0.310 in Table 3.1, k_E(700)=0.05
versus 0.130, truncation at 1000 degC with a silent 0.0 beyond) and have been
removed; see ``vault/reports/03_material_deviation.md``.

Behavioural change versus the legacy implementation (deliberate):
* legacy returned 1.0 for T < 20 and 0.0 for T > 1000; the SSOT clamps into
  the standard's validity range [20, 1200] and linearly interpolates, so
  temperatures in (1000, 1200] now follow Table 3.1 instead of jumping to 0.
"""

from __future__ import annotations

import warnings

from truss_analysis.material.steel_eurocode import k_E as _ssot_k_E
from truss_analysis.material.steel_eurocode import k_y as _ssot_k_y

__all__ = ["get_eurocode_k_E", "get_eurocode_k_y"]

_DEPRECATION_K_E = (
    "truss_analysis.thermal.material.get_eurocode_k_E is deprecated; use "
    "truss_analysis.material.steel_eurocode.k_E (EN 1993-1-2:2005 SSOT)."
)
_DEPRECATION_K_Y = (
    "truss_analysis.thermal.material.get_eurocode_k_y is deprecated; use "
    "truss_analysis.material.steel_eurocode.k_y (EN 1993-1-2:2005 SSOT)."
)


def get_eurocode_k_E(temp_c: float) -> float:
    """Reduction factor for Young's modulus (deprecated wrapper).

    Delegates to the SSOT; emits :class:`DeprecationWarning`.
    """
    warnings.warn(_DEPRECATION_K_E, DeprecationWarning, stacklevel=2)
    return float(_ssot_k_E(temp_c))


def get_eurocode_k_y(temp_c: float) -> float:
    """Reduction factor for yield strength (deprecated wrapper).

    Delegates to the SSOT; emits :class:`DeprecationWarning`.
    """
    warnings.warn(_DEPRECATION_K_Y, DeprecationWarning, stacklevel=2)
    return float(_ssot_k_y(temp_c))
