"""Thermal degradation models for steel structures."""

from .material import get_eurocode_k_E, get_eurocode_k_y

__all__ = ["get_eurocode_k_E", "get_eurocode_k_y"]
