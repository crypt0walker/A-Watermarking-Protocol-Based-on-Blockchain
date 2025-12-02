"""Paillier cryptosystem utilities used for homomorphic embedding benchmarks.

The implementation is self contained (no external crypto deps) and uses
probabilistic Miller–Rabin for prime testing. It is designed for correctness
and reasonable performance at 1024/2048 bits, not for production security.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Tuple


# Miller–Rabin bases for deterministic checks up to 2^128; for larger key sizes
# we still get strong probabilistic assurance with additional random bases.
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


def _modinv(a: int, n: int) -> int:
    """Modular inverse using extended Euclid."""
    t, new_t = 0, 1
    r, new_r = n, a
    while new_r:
        q = r // new_r
        t, new_t = new_t, t - q * new_t
        r, new_r = new_r, r - q * new_r
    if r != 1:
        raise ValueError("a is not invertible")
    return t % n


def _is_probable_prime(n: int, rounds: int = 10) -> bool:
    """Probabilistic Miller–Rabin primality test."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # write n-1 as 2^s * d
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def check(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return True
        return False

    bases = list(_MR_BASES)
    # add random bases for larger numbers
    while len(bases) < rounds:
        candidate = secrets.randbelow(n - 3) + 2
        if candidate not in bases:
            bases.append(candidate)
    for a in bases:
        if not check(a % n):
            return False
    return True


def _rand_prime(bits: int) -> int:
    """Generate a random probable prime with the requested bit-length."""
    assert bits >= 16, "bits too small"
    while True:
        candidate = secrets.randbits(bits) | 1 | (1 << (bits - 1))
        if _is_probable_prime(candidate):
            return candidate


def _lcm(a: int, b: int) -> int:
    return a // _gcd(a, b) * b


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _L(u: int, n: int) -> int:
    return (u - 1) // n


@dataclass
class PaillierPublic:
    n: int
    g: int

    @property
    def n2(self) -> int:
        return self.n * self.n


@dataclass
class PaillierPrivate:
    lam: int
    mu: int


def keygen(bits: int = 1024) -> Tuple[PaillierPublic, PaillierPrivate]:
    """Generate Paillier keypair."""
    if bits < 512:
        raise ValueError("Paillier key size too small for realistic use")
    p = _rand_prime(bits // 2)
    q = _rand_prime(bits // 2)
    while q == p:
        q = _rand_prime(bits // 2)
    n = p * q
    lam = _lcm(p - 1, q - 1)
    g = n + 1
    n2 = n * n
    mu = _modinv(_L(pow(g, lam, n2), n), n)
    return PaillierPublic(n=n, g=g), PaillierPrivate(lam=lam, mu=mu)


def enc(pub: PaillierPublic, m: int, r: int | None = None) -> int:
    """Encrypt message m. 0 <= m < n."""
    if not (0 <= m < pub.n):
        raise ValueError("message out of range")
    if r is None:
        # choose random r in Zn*
        while True:
            r = secrets.randbelow(pub.n - 1) + 1
            if _gcd(r, pub.n) == 1:
                break
    n2 = pub.n2
    return (pow(pub.g, m, n2) * pow(r, pub.n, n2)) % n2


def dec(pub: PaillierPublic, priv: PaillierPrivate, c: int) -> int:
    """Decrypt ciphertext c."""
    n2 = pub.n2
    if not (0 <= c < n2):
        raise ValueError("ciphertext out of range")
    u = pow(c, priv.lam, n2)
    return (_L(u, pub.n) * priv.mu) % pub.n


def hom_add(pub: PaillierPublic, c1: int, c2: int) -> int:
    return (c1 * c2) % pub.n2


def hom_add_const(pub: PaillierPublic, c: int, k: int) -> int:
    # k can be negative; handle by using modular inverse
    if k >= 0:
        return (c * pow(pub.g, k, pub.n2)) % pub.n2
    # for negative k, use g^{-k} = (g^{-1})^{|k|}
    g_inv = _modinv(pub.g % pub.n2, pub.n2)
    return (c * pow(g_inv, -k, pub.n2)) % pub.n2


def hom_mul_const(pub: PaillierPublic, c: int, k: int) -> int:
    if k < 0:
        raise ValueError("negative scalar not supported for hom_mul_const")
    return pow(c, k, pub.n2)
