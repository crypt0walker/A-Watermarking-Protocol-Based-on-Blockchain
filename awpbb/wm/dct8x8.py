"""8x8 DCT/IDCT helpers using numpy and Pillow."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_grayscale(path: str | Path, size: Tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float64)


def save_grayscale(arr: np.ndarray, path: str | Path) -> None:
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    img.save(path)


def load_color(path: str | Path, size: Tuple[int, int] | None = None) -> np.ndarray:
    """Load RGB image as float64 array of shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float64)


def save_color(arr: np.ndarray, path: str | Path) -> None:
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    img.save(path)


def to_blocks(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    assert h % 8 == 0 and w % 8 == 0, "image size must be multiple of 8"
    blocks = img.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2)
    return blocks.reshape(-1, 8, 8)


def from_blocks(blocks: np.ndarray, h: int, w: int) -> np.ndarray:
    blocks = blocks.reshape(h // 8, w // 8, 8, 8).swapaxes(1, 2)
    return blocks.reshape(h, w)


def _dct_matrix() -> np.ndarray:
    T = np.zeros((8, 8), dtype=np.float64)
    for u in range(8):
        for x in range(8):
            if u == 0:
                alpha = math.sqrt(1 / 8)
            else:
                alpha = math.sqrt(2 / 8)
            T[u, x] = alpha * math.cos(((2 * x + 1) * u * math.pi) / 16)
    return T


_T = _dct_matrix()
_Tt = _T.T


def dct_block(block: np.ndarray) -> np.ndarray:
    return _T @ block @ _Tt


def idct_block(block: np.ndarray) -> np.ndarray:
    return _Tt @ block @ _T
