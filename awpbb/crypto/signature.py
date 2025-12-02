"""Minimal RSA signatures (textbook RSA with SHA-256 hash)."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Tuple

from .paillier import _gcd, _is_probable_prime, _modinv


def _rand_prime(bits: int) -> int:
    assert bits >= 16
    while True:
        candidate = secrets.randbits(bits) | 1 | (1 << (bits - 1))
        if _is_probable_prime(candidate):
            return candidate


@dataclass
class RSAPublic:
    n: int
    e: int


@dataclass
class RSAPrivate:
    n: int
    d: int


def keygen(bits: int = 2048, e: int = 65537) -> Tuple[RSAPublic, RSAPrivate]:
    if bits < 512:
        raise ValueError("RSA key size too small")
    p = _rand_prime(bits // 2)
    q = _rand_prime(bits // 2)
    while q == p:
        q = _rand_prime(bits // 2)
    n = p * q
    phi = (p - 1) * (q - 1)
    if _gcd(e, phi) != 1:
        return keygen(bits, e)
    d = _modinv(e, phi)
    return RSAPublic(n=n, e=e), RSAPrivate(n=n, d=d)


def _hash(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest(), "big")


def sign(priv: RSAPrivate, data: bytes) -> int:
    h = _hash(data)
    if h >= priv.n:
        h = h % priv.n
    return pow(h, priv.d, priv.n)


def verify(pub: RSAPublic, data: bytes, sig: int) -> bool:
    h = _hash(data)
    if h >= pub.n:
        h = h % pub.n
    return pow(sig, pub.e, pub.n) == h % pub.n
