#!/usr/bin/env python3
"""Benchmark a single RA/CP/B/J transaction (mock BC) with interactive input."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awpbb.crypto import paillier
from awpbb.crypto import signature as rsa
from awpbb.protocol.mock_bc import MockBC, TxRecord
from awpbb.wm import composite_embed
from awpbb.wm import dct8x8
from awpbb.wm import mapping


def _b64_int(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, "big")


def _hash_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def prompt_inputs() -> tuple[Path, int, int]:
    img_in = input("输入图像路径（默认 lenna.jpg）: ").strip()
    img_path = Path(img_in) if img_in else Path("lenna.jpg")

    size_in = input("输入图像尺寸 N（默认 256，表示 N×N）: ").strip()
    size = 256 if size_in == "" else int(size_in)

    wm_in = input("输入水印长度（bit，默认 64）: ").strip()
    wm_bits = 64 if wm_in == "" else int(wm_in)

    return img_path, size, wm_bits


def _prepare_blocks_color(img: np.ndarray, positions: List[Tuple[int, int]]):
    """Split RGB image into DCT blocks per channel; return blocks and mapping."""
    h, w, _ = img.shape
    blocks_by_c = []
    for c in range(3):
        chan_blocks = dct8x8.to_blocks(img[:, :, c])
        dct_blocks = np.array([dct8x8.dct_block(b) for b in chan_blocks])
        blocks_by_c.append(dct_blocks)
    # host index mapping: list of (channel, block_idx, (r,c))
    host_indices: List[Tuple[int, int, Tuple[int, int]]] = []
    num_blocks = blocks_by_c[0].shape[0]
    for blk_idx in range(num_blocks):
        for c in range(3):  # distribute均匀：块内遍历三个通道
            for pos in positions:
                host_indices.append((c, blk_idx, pos))
    return blocks_by_c, host_indices, h, w


def _extract_host_coeffs_color(blocks_by_c, host_indices: List[Tuple[int, int, Tuple[int, int]]]) -> List[float]:
    coefs = []
    for c, blk_idx, (r, col) in host_indices:
        coefs.append(float(blocks_by_c[c][blk_idx, r, col]))
    return coefs


def _apply_coeffs_color(blocks_by_c, host_indices: List[Tuple[int, int, Tuple[int, int]]], new_values: List[float]):
    out = [np.array(bc, copy=True) for bc in blocks_by_c]
    for val, (c, blk_idx, (r, col)) in zip(new_values, host_indices):
        out[c][blk_idx, r, col] = val
    return out


def embed_bits_from_offset(
    pub,
    xs: List[int],
    cs: List[int],
    bits: List[int],
    alpha: List[int],
    delta: int,
    k: int,
    start_group: int = 0,
) -> List[int]:
    """Embed bits starting from group offset (group size k)."""
    cs_out = list(cs)
    bidx = 0
    start = start_group * k
    for i in range(start, len(xs), k):
        if bidx >= len(bits):
            break
        group_x = xs[i : i + k]
        group_c = cs_out[i : i + k]
        updated = composite_embed.embed_group(pub, group_x, group_c, bits[bidx], alpha[: len(group_x)], delta)
        cs_out[i : i + len(updated)] = updated
        bidx += 1
    return cs_out


def main() -> None:
    img_path, size, wm_len = prompt_inputs()
    alpha = [1, 2, 4, 8]
    k = len(alpha)

    t0 = time.perf_counter()
    if not img_path.is_file():
        synth_size = (size, size)
        synth = np.fromfunction(lambda y, x: (x + y) % 256, synth_size, dtype=np.float64)
        img_path = Path("synthetic_input.png")
        dct8x8.save_grayscale(synth, img_path)
        print(f"[warn] 输入图像不存在，生成了合成图 {img_path}，请使用 --image 指向真实 lenna.jpg")
    img = dct8x8.load_color(img_path, size=(size, size))
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    pub_p, priv_p = paillier.keygen(1024)  # Paillier 密钥生成
    t_keygen = time.perf_counter() - t1

    t1 = time.perf_counter()
    ra_pub, ra_priv = rsa.keygen()  # RA 的 RSA 密钥
    t_ra_rsa_keygen = time.perf_counter() - t1

    t1 = time.perf_counter()
    cp_pub, cp_priv = rsa.keygen()  # CP 的 RSA 密钥
    t_cp_rsa_keygen = time.perf_counter() - t1

    nonce = int.from_bytes(_hash_bytes(b"nonce" + _b64_int(int(time.time() * 1e6))), "big") & ((1 << 64) - 1)
    t1 = time.perf_counter()
    encN = paillier.enc(pub_p, nonce)  # 同态加密 nonce
    token_sig = rsa.sign(ra_priv, _b64_int(pub_p.n) + _b64_int(encN))
    t_ra_issue = time.perf_counter() - t1

    positions = [(3, 3), (4, 2), (2, 4), (3, 4)]  # 中频位置
    blocks_by_c, host_indices, H, W = _prepare_blocks_color(img, positions)  # 彩色 DCT 分块
    host_coeffs = _extract_host_coeffs_color(blocks_by_c, host_indices)
    scale, offset = mapping.compute_scale_and_offset(np.array(host_coeffs))
    host_ints = mapping.coeffs_to_ints(np.array(host_coeffs), scale, offset).tolist()

    available_bits = len(host_ints) // k
    total_bits = wm_len + 64
    if total_bits > available_bits:
        raise ValueError(f"容量不足: 需要 {total_bits} bit, 宿主仅 {available_bits} bit")

    rng_bytes = _hash_bytes(b"WCP" + _b64_int(int(time.time() * 1e6)))
    wcp_bits = [(rng_bytes[i // 8] >> (7 - (i % 8))) & 1 for i in range(wm_len)]
    nonce_bits = [(nonce >> (63 - i)) & 1 for i in range(64)]
    wm_bits = wcp_bits + nonce_bits

    t1 = time.perf_counter()
    enc_host = [paillier.enc(pub_p, int(x)) for x in host_ints]
    t_enc = time.perf_counter() - t1

    # 分别计时嵌入 W_CP 与 nonce
    t1 = time.perf_counter()
    enc_after_wcp = embed_bits_from_offset(
        pub=pub_p,
        xs=host_ints,
        cs=enc_host,
        bits=wcp_bits,
        alpha=alpha,
        delta=32,
        k=k,
        start_group=0,
    )
    t_embed_wcp = time.perf_counter() - t1

    t1 = time.perf_counter()
    enc_embedded = embed_bits_from_offset(
        pub=pub_p,
        xs=host_ints,
        cs=enc_after_wcp,
        bits=nonce_bits,
        alpha=alpha,
        delta=32,
        k=k,
        start_group=len(wcp_bits),  # 继续在后续分组嵌入 nonce
    )
    t_embed_nonce = time.perf_counter() - t1
    print(f"CP: 已嵌入 W_CP（{wm_len}bit）耗时 {t_embed_wcp:.3f}s，嵌入 nonce（64bit）耗时 {t_embed_nonce:.3f}s")

    Xd = _hash_bytes(b"Xd" + _b64_int(enc_embedded[0]))
    t1 = time.perf_counter()
    sig_cp = rsa.sign(cp_priv, Xd + _b64_int(encN))
    t_cp_sign = time.perf_counter() - t1

    bc = MockBC()
    record = TxRecord(
        Xd=Xd,
        TX=int(time.time()),
        pkX_RA=_b64_int(pub_p.n),
        encN=_b64_int(encN),
        sigRA=token_sig,
        sigCP=sig_cp,
        buyer="buyer-1",
    )
    t1 = time.perf_counter()
    token_id = bc.register_by_cp(record)
    t_bc_reg = time.perf_counter() - t1

    t1 = time.perf_counter()
    bc.confirm_by_b(token_id, record)
    t_bc_conf = time.perf_counter() - t1

    t1 = time.perf_counter()
    decrypted = [paillier.dec(pub_p, priv_p, c) for c in enc_embedded]
    t_dec = time.perf_counter() - t1

    new_coeffs = mapping.ints_to_coeffs(np.array(decrypted), scale, offset).tolist()
    updated_blocks_by_c = _apply_coeffs_color(blocks_by_c, host_indices, new_coeffs)
    t1 = time.perf_counter()
    # 逐通道 IDCT 并重组彩色图
    recon_channels = []
    for c in range(3):
        spatial_blocks = np.array([dct8x8.idct_block(b) for b in updated_blocks_by_c[c]])
        recon_chan = dct8x8.from_blocks(spatial_blocks, H, W)
        recon_channels.append(recon_chan)
    recon_img = np.stack(recon_channels, axis=2)
    t_recon = time.perf_counter() - t1
    out_img_path = Path("watermarked.png")
    dct8x8.save_color(recon_img, out_img_path)

    t1 = time.perf_counter()
    bits_out = composite_embed.extract_bits(
        xs=decrypted,
        alpha=alpha,
        delta=32,
        k=k,
        max_bits=total_bits,
    )
    t_extract = time.perf_counter() - t1
    nonce_out = 0
    for b in bits_out[wm_len : wm_len + 64]:
        nonce_out = (nonce_out << 1) | b

    t1 = time.perf_counter()
    nonce_dec = paillier.dec(pub_p, priv_p, encN)
    t_ra_decrypt = time.perf_counter() - t1

    result = {
        "image": str(img_path),
        "size": img.shape,
        "paillier_bits": 1024,
        "delta": 32,
        "alpha": alpha,
        "watermark_bits": wm_len,
        "nonce_bits": 64,
        "success": nonce_out == nonce and nonce_dec == nonce,
        "timings_sec": {
            "RA_keygen_paillier": t_keygen,
            "RA_keygen_rsa": t_ra_rsa_keygen,
            "RA_issue": t_ra_issue,
            "CP_keygen_rsa": t_cp_rsa_keygen,
            "CP_encrypt": t_enc,
            "CP_embed_wcp": t_embed_wcp,
            "CP_embed_nonce": t_embed_nonce,
            "CP_sign": t_cp_sign,
            "CP_bc_register": t_bc_reg,
            "B_bc_confirm": t_bc_conf,
            "B_decrypt": t_dec,
            "B_reconstruct": t_recon,
            "J_extract": t_extract,
            "RA_decrypt_nonce": t_ra_decrypt,
            "load_image": t_load,
        },
        "token_id_hex": token_id.hex(),
        "watermarked_image": str(out_img_path),
    }
    print(json.dumps(result, indent=2))

    # 中文耗时打印（按耗时降序）
    stages = [
        ("CP Paillier 加密", t_enc, "CP"),
        ("B 解密", t_dec, "B"),
        ("CP 嵌入 W_CP", t_embed_wcp, "CP"),
        ("CP 嵌入 nonce", t_embed_nonce, "CP"),
        ("RA RSA 密钥生成", t_ra_rsa_keygen, "RA"),
        ("CP RSA 密钥生成", t_cp_rsa_keygen, "CP"),
        ("RA Paillier 密钥生成", t_keygen, "RA"),
        ("RA 发放 token（加密 nonce+签名）", t_ra_issue, "RA"),
        ("CP 签名交易", t_cp_sign, "CP"),
        ("J 提取水印", t_extract, "J"),
        ("B 重建图像", t_recon, "B"),
        ("RA 解密 nonce", t_ra_decrypt, "RA"),
        ("加载/缩放图像", t_load, "I/O"),
        ("CP 注册 mock BC", t_bc_reg, "CP"),
        ("B 确认 mock BC", t_bc_conf, "B"),
    ]
    stages_sorted = sorted(stages, key=lambda x: x[1], reverse=True)

    print("\n=== 各端耗时（秒，降序） ===")
    for name, tval, role in stages_sorted:
        print(f"{name}: {tval:.6f} [{role}]")

    # 1->n / 链式复用提示
    print("\n[复用提示] 1->n 或链式场景可复用/可省的步骤：")
    print(" - 密钥生成（Paillier/RSA）可多笔复用，不必每笔生成。")
    print(" - 若宿主内容不变，图像加载+DCT+宿主系数 Paillier 加密可缓存；每个买家仅重新嵌入自己的 W_CP 与 nonce。")
    print(" - BC 注册/确认、嵌入、解密/提取是每笔必做，无法省略。")

    if result["success"]:
        print(f"\n[成功] 水印与 nonce 提取验证通过，水印图像已保存至 {out_img_path}")
    else:
        print("\n[失败] 提取水印/nonce 不一致，请检查参数与实现。")


if __name__ == "__main__":
    main()
    
