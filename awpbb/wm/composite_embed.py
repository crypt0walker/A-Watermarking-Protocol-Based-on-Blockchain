"""Composite embedding with Paillier-based QIM adjustments."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from awpbb.crypto.paillier import PaillierPublic, hom_add_const


def qim_embed_int(x: int, bit: int, delta: int) -> int:
    # 标准标量 QIM：两个余类相差 delta，量化步长为 2*delta，使用“就近”量化以最小化失真
    step = 2 * delta
    # x 为非负整数时可用整数舍入：round(x/step) = floor((x+step/2)/step)
    q = (x + (step // 2)) // step
    base = q * step
    return base if bit == 0 else base + delta


def qim_extract_int(x_bar: int, delta: int) -> int:
    # 判决：对 delta 格点就近取整后看奇偶
    q = (x_bar + (delta // 2)) // delta
    return int(q % 2)


def distribute_delta(alpha: List[int], dy: int) -> List[int]:
    """Split dy across components so that sum(alpha_i * dx_i) = dy."""
    denom = sum(a * a for a in alpha)
    deltas = [int(round(dy * a / denom)) for a in alpha]
    residual = dy - sum(a * d for a, d in zip(alpha, deltas))
    # Fix residual exactly by adjusting the largest-weight component
    if residual != 0:
        idx = max(range(len(alpha)), key=lambda i: abs(alpha[i]))
        a = alpha[idx]
        if a == 0:
            deltas[idx] += residual
        else:
            # add enough steps to cover residual with correct sign
            step = residual // a
            if residual % a != 0:
                step += 1 if residual * a > 0 else -1
            deltas[idx] += step
    # final sanity check; if still off, adjust by direct difference
    final = sum(a * d for a, d in zip(alpha, deltas))
    if final != dy:
        deltas[0] += (dy - final) // (alpha[0] if alpha[0] != 0 else 1)
    return deltas


def embed_group(
    pub: PaillierPublic,
    xs: List[int],
    cs: List[int],
    bit: int,
    alpha: List[int],
    delta: int,
) -> List[int]:
    y = sum(a * x for a, x in zip(alpha, xs))
    y_star = qim_embed_int(y, bit, delta)
    dy = y_star - y
    dxs = distribute_delta(alpha, dy)
    updated = list(cs)
    for i, dx in enumerate(dxs):
        updated[i] = hom_add_const(pub, cs[i], dx)
    return updated


def embed_bits(
    pub: PaillierPublic,
    xs: List[int],
    cs: List[int],
    bits: Iterable[int],
    alpha: List[int],
    delta: int,
    k: int = 4,
) -> List[int]:
    """Embed bits into encrypted samples using composite embedding.

    xs: plaintext host samples (ints)
    cs: encrypted host samples E(xs)
    bits: iterator of bits to embed
    alpha: weight vector (len>=k)
    delta: QIM step
    k: group size
    """
    cs_out = list(cs)
    alpha = alpha[:k]
    bit_list = list(bits)
    bidx = 0
    for i in range(0, len(xs), k):
        if bidx >= len(bit_list):
            break
        group_x = xs[i : i + k]
        group_c = cs[i : i + k]
        updated = embed_group(pub, group_x, group_c, bit_list[bidx], alpha[: len(group_x)], delta)
        cs_out[i : i + len(updated)] = updated
        bidx += 1
    return cs_out


def extract_bits(
    xs: List[int],
    alpha: List[int],
    delta: int,
    k: int = 4,
    max_bits: int | None = None,
) -> List[int]:
    alpha = alpha[:k]
    bits: List[int] = []
    for i in range(0, len(xs), k):
        if max_bits is not None and len(bits) >= max_bits:
            break
        group_x = xs[i : i + k]
        y = sum(a * x for a, x in zip(alpha[: len(group_x)], group_x))
        bits.append(qim_extract_int(y, delta))
    return bits
