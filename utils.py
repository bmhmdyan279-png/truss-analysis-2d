"""
توابع کمکی عمومی - جدا شده برای جلوگیری از Circular Import
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

UNIT_CONVERSION = {
    'SI': {
        'length': {'factor': 1.0, 'base': 'm', 'label': 'm'},
        'force': {'factor': 1.0, 'base': 'N', 'label': 'N'},
        'stress': {'factor': 1.0, 'base': 'Pa', 'label': 'Pa'},
        'energy': {'factor': 1.0, 'base': 'J', 'label': 'J'},
        'temperature': {'factor': 1.0, 'base': '°C', 'label': '°C'}
    },
    'SI-mm': {
        'length': {'factor': 0.001, 'base': 'm', 'label': 'mm'},
        'force': {'factor': 1.0, 'base': 'N', 'label': 'N'},
        'stress': {'factor': 1.0, 'base': 'Pa', 'label': 'Pa'},
        'energy': {'factor': 1.0, 'base': 'J', 'label': 'J'},
        'temperature': {'factor': 1.0, 'base': '°C', 'label': '°C'}
    },
    'SI-cm': {
        'length': {'factor': 0.01, 'base': 'm', 'label': 'cm'},
        'force': {'factor': 1.0, 'base': 'N', 'label': 'N'},
        'stress': {'factor': 1.0, 'base': 'Pa', 'label': 'Pa'},
        'energy': {'factor': 1.0, 'base': 'J', 'label': 'J'},
        'temperature': {'factor': 1.0, 'base': '°C', 'label': '°C'}
    },
    'Imperial': {
        'length': {'factor': 0.3048, 'base': 'm', 'label': 'ft'},
        'force': {'factor': 4.44822, 'base': 'N', 'label': 'lbf'},
        'stress': {'factor': 6894.76, 'base': 'Pa', 'label': 'psi'},
        'energy': {'factor': 1.35582, 'base': 'J', 'label': 'ft-lbf'},
        'temperature': {'factor': 1.0, 'base': '°C', 'label': '°F'}
    }
}


def validate_units(units: str) -> Dict:
    if units not in UNIT_CONVERSION:
        raise ValueError(f"واحد نامعتبر: '{units}'. واحدهای مجاز: {', '.join(UNIT_CONVERSION.keys())}")

    return UNIT_CONVERSION[units]


def convert_to_si(value: float, from_units: str, quantity_type: str) -> float:
    if from_units not in UNIT_CONVERSION:
        raise ValueError(f"واحد مبدا نامعتبر: {from_units}")

    if quantity_type not in UNIT_CONVERSION[from_units]:
        raise ValueError(f"نوع کمیت نامعتبر برای واحد {from_units}: {quantity_type}")

    conversion = UNIT_CONVERSION[from_units][quantity_type]
    return value * conversion['factor']


def convert_from_si(value: float, to_units: str, quantity_type: str) -> float:
    if to_units not in UNIT_CONVERSION:
        raise ValueError(f"واحد مقصد نامعتبر: {to_units}")

    if quantity_type not in UNIT_CONVERSION[to_units]:
        raise ValueError(f"نوع کمیت نامعتبر برای واحد {to_units}: {quantity_type}")

    conversion = UNIT_CONVERSION[to_units][quantity_type]
    return value / conversion['factor']


def format_with_units(value: float, units: str, quantity_type: str) -> str:
    if units not in UNIT_CONVERSION:
        units = 'SI'

    value_in_target = convert_from_si(convert_to_si(value, 'SI', quantity_type), units, quantity_type)
    unit_label = UNIT_CONVERSION[units][quantity_type]['label']

    abs_val = abs(value_in_target)

    if quantity_type == 'length':
        if units == 'SI':
            if abs_val < 0.001:
                return f"{value_in_target * 1000:.3f} mm"
            elif abs_val < 1:
                return f"{value_in_target * 100:.1f} cm"
            else:
                return f"{value_in_target:.3f} m"
        else:
            return f"{value_in_target:.3f} {unit_label}"

    elif quantity_type == 'force':
        if abs_val < 0.001:
            return f"{value_in_target * 1e6:.2f} μ{unit_label}"
        elif abs_val < 1:
            return f"{value_in_target * 1e3:.2f} m{unit_label}"
        elif abs_val < 1000:
            return f"{value_in_target:.2f} {unit_label}"
        elif abs_val < 1e6:
            return f"{value_in_target / 1e3:.2f} k{unit_label}"
        else:
            return f"{value_in_target / 1e6:.2f} M{unit_label}"

    elif quantity_type == 'stress':
        if abs_val < 1000:
            return f"{value_in_target:.1f} {unit_label}"
        elif abs_val < 1e6:
            return f"{value_in_target / 1e3:.1f} k{unit_label}"
        elif abs_val < 1e9:
            return f"{value_in_target / 1e6:.1f} M{unit_label}"
        else:
            return f"{value_in_target / 1e9:.2f} G{unit_label}"

    else:
        return f"{value_in_target:.4e} {unit_label}"