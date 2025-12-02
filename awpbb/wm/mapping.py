"""Mapping between floating-point DCT coefficients and Paillier-friendly integers."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def compute_scale_and_offset(coeffs: np.ndarray, safety_margin: float = 10.0) -> Tuple[float, float]:
    """Determine scale S and offset so mapped ints stay positive."""
    max_abs = float(np.max(np.abs(coeffs))) if coeffs.size else 1.0
    offset = max_abs + safety_margin
    scale = 16.0  # fixed scale factor; adjust if you want more precision
    return scale, offset


def coeffs_to_ints(coeffs: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return np.rint((coeffs + offset) * scale).astype(np.int64)


def ints_to_coeffs(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return (values.astype(np.float64) / scale) - offset


def bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i // 8] |= 1 << (7 - (i % 8))
    return bytes(out)


def int_to_bits(value: int, width: int) -> list[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits
