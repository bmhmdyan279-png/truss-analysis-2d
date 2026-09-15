"""Thermal layer: fire exposure curves, member heating, degradation shims.

* :mod:`truss_analysis.thermal.fire_curve` -- ISO 834 exposure curve and the
  EN 1993-1-2 §4.2.2.2 lumped-capacitance steel-heating solver (round-5
  audit: the bridge from a fire exposure to the prescribed-temperature
  structural chain).
* :mod:`truss_analysis.thermal.material` -- deprecated compatibility shims
  over :mod:`truss_analysis.material.steel_eurocode`.
"""

from .fire_curve import (
    ALPHA_C,
    EPSILON_RES,
    STEFAN_BOLTZMANN,
    SteelHeatingResult,
    h_net,
    iso_834_temperature,
    steel_temperature,
)
from .material import get_eurocode_k_E, get_eurocode_k_y

__all__ = [
    "ALPHA_C",
    "EPSILON_RES",
    "STEFAN_BOLTZMANN",
    "SteelHeatingResult",
    "get_eurocode_k_E",
    "get_eurocode_k_y",
    "h_net",
    "iso_834_temperature",
    "steel_temperature",
]
