#!/usr/bin/env python3
"""Off-chain (mock BC) single-transaction benchmark with quality self-check.

对齐 Frattolillo 关键假设（每笔交易一次性 Paillier 公私钥 + nonce token）：
- RA: 生成一次性 (pk_RA^X, sk_RA^X) 与随机 nonce N，并提供 E_pk(N)+签名；
- CP: 用 pk_RA^X 加密宿主 DCT 系数，并在密文域同态嵌入 W_CP || N（严格 composite embedding + QIM）；
- B: 在收到 sk_RA^X 后解密得到水印内容；
- J: 在明文水印图像上提取 W'，验证 nonce N' 与 RA 解密 N 一致。

实现细节（便于稳定提取与可视质量）：
- 输入为彩色 RGB 图像；嵌入在单一通道（默认 G 通道）的 8×8 DCT 中频系数；
- 输出为彩色 watermarked.png；
- 质量自校验：从保存后的 watermarked.png 重新提取水印，统计 BER/NC/PSNR(Y)。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import secrets
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awpbb.crypto import paillier
from awpbb.crypto import signature as rsa
from awpbb.protocol.mock_bc import MockBC, TxRecord
from awpbb.wm import composite_embed
from awpbb.wm import dct8x8


def _ms(sec: float) -> float:
    return sec * 1000.0


def _b(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, "big")


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _prompt_int(msg: str, default: int) -> int:
    s = input(msg).strip()
    return default if s == "" else int(s)


def _psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def _rgb_to_y_u8(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def _nc_bits(bits_a: List[int], bits_b: List[int]) -> float:
    if len(bits_a) == 0 or len(bits_a) != len(bits_b):
        return 0.0
    a = np.array([1 if x else -1 for x in bits_a], dtype=np.float64)
    b = np.array([1 if x else -1 for x in bits_b], dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _ber(bits_a: List[int], bits_b: List[int]) -> int:
    return sum((x ^ y) for x, y in zip(bits_a, bits_b))


def prompt_inputs() -> tuple[Path, int, int, int, int]:
    img_in = input("请输入图像文件路径（默认 Lenna.jpg）: ").strip()
    img_path = Path(img_in) if img_in else Path("Lenna.jpg")
    if not img_path.is_file():
        img_path = Path("lenna.jpg")
    size = _prompt_int("请输入图像边长(像素)（默认 256）：", 256)
    wm_bits = _prompt_int("请输入水印长度 (bit)（默认 128）：", 128)
    s = _prompt_int("请输入 QIM 步长 s（默认 256）：", 256)
    texture_pct = _prompt_int("请输入纹理保留分位数(%)（默认 70）：", 70)
    return img_path, size, wm_bits, s, texture_pct


def _load_rgb_u8(img_path: Path, size: int) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _block_variances(channel_u8: np.ndarray) -> np.ndarray:
    blocks = dct8x8.to_blocks(channel_u8.astype(np.float64))
    return np.var(blocks.reshape(blocks.shape[0], -1), axis=1)


def _make_slot_order(variances: np.ndarray, texture_pct: int, seed: int) -> Tuple[List[int], float, int]:
    tau = float(np.quantile(variances, texture_pct / 100.0))
    textured = [i for i, v in enumerate(variances) if v >= tau]
    non_textured = [i for i, v in enumerate(variances) if v < tau]
    rng = random.Random(seed)
    rng.shuffle(textured)
    rng.shuffle(non_textured)
    supplemented = len(non_textured)
    order = textured + non_textured
    return order, tau, supplemented


def _extract_coeffs(dct_blocks: np.ndarray, positions: List[Tuple[int, int]], slot_order: List[int]) -> np.ndarray:
    coeffs = np.zeros((len(slot_order), len(positions)), dtype=np.float64)
    for si, blk_idx in enumerate(slot_order):
        blk = dct_blocks[blk_idx]
        for pi, (r, c) in enumerate(positions):
            coeffs[si, pi] = blk[r, c]
    return coeffs.reshape(-1)


def _apply_coeffs(dct_blocks: np.ndarray, positions: List[Tuple[int, int]], slot_order: List[int], new_coeffs_flat: np.ndarray) -> np.ndarray:
    out = np.array(dct_blocks, copy=True)
    new_coeffs = new_coeffs_flat.reshape(len(slot_order), len(positions))
    for si, blk_idx in enumerate(slot_order):
        for pi, (r, c) in enumerate(positions):
            out[blk_idx, r, c] = new_coeffs[si, pi]
    return out


def main() -> None:
    if not getattr(paillier, "_G", False):
        print("[warn] 未检测到 gmpy2，Paillier 加/解密会非常慢。建议: python3 -m pip install gmpy2")

    img_path, size, wm_len, s, texture_pct = prompt_inputs()
    if not img_path.is_file():
        raise FileNotFoundError(f"找不到图像文件: {img_path}")
    if size % 8 != 0:
        raise ValueError("图像边长必须是 8 的倍数（便于 8×8 DCT 分块）")

    host_channel = 1  # 0=R,1=G,2=B；只在一个通道嵌入，输出仍为彩色
    positions = [(3, 3), (4, 2), (2, 4), (3, 4)]  # 每块 4 个系数 => composite group size k=4
    k = 4
    scale = 256.0  # 将 DCT 系数放大到整数域，降低 round-trip 取整误差
    safety_margin = 10.0

    # ===== 版权方：载入与缩放 =====
    t0 = time.perf_counter()
    rgb = _load_rgb_u8(img_path, size)
    t_load = time.perf_counter() - t0
    print(f"【版权方】载入并缩放图像：{_ms(t_load):.2f} ms | 尺寸 {size}×{size}")

    # ===== 槽位数（按块，一块一比特）=====
    blocks_per_side = size // 8
    num_blocks = blocks_per_side * blocks_per_side
    K_slots = num_blocks
    max_wm = max(1, K_slots - 64)
    print(f"可输入的水印长度范围：1 ~ {max_wm} bit")
    if wm_len < 1 or wm_len > max_wm:
        raise ValueError(f"水印长度非法: {wm_len}，必须在 1~{max_wm} 之间")
    print(f"水印槽位数 K={K_slots}，实际水印长度={wm_len} bit")

    # ===== 纹理统计（用于提示与均匀嵌入顺序）=====
    t0 = time.perf_counter()
    ch = rgb[:, :, host_channel]
    variances = _block_variances(ch)
    seed = int.from_bytes(_sha256(b"slot-order" + img_path.name.encode() + _b(size)), "big") & 0xFFFFFFFF
    slot_order, tau, supplemented = _make_slot_order(variances, texture_pct, seed)
    t_texture = time.perf_counter() - t0
    if supplemented > 0:
        print(f"【提示】纹理块不足，已自动补足 {supplemented} 个高能量块")
    print(f"【提示】纹理阈值 tau={tau:.6f}，可用块 {len(slot_order)}/{K_slots} | 计算耗时 {_ms(t_texture):.2f} ms")

    # ===== 版权方：DCT + 量化准备 =====
    t0 = time.perf_counter()
    blocks = dct8x8.to_blocks(ch.astype(np.float64))
    dct_blocks = np.array([dct8x8.dct_block(b) for b in blocks])
    host_coeffs = _extract_coeffs(dct_blocks, positions, slot_order)
    t_dct = time.perf_counter() - t0

    max_abs = float(np.max(np.abs(host_coeffs))) if host_coeffs.size else 1.0
    offset = max_abs + safety_margin
    delta_int = int(round(s * scale))
    print(
        f"【版权方】分块 DCT 与量化：{_ms(t_dct):.2f} ms | 块 {blocks_per_side}×{blocks_per_side}, |B|={len(positions)}, s={s}, scale={int(scale)}"
    )

    # 宿主整数表示：x = round((c + offset) * scale)
    host_ints_signed = np.rint((host_coeffs + offset) * scale).astype(np.int64).tolist()

    # 嵌入比特总数：W_CP + nonce
    total_bits = wm_len + 64
    if total_bits > K_slots:
        raise ValueError(f"容量不足: 需要 {total_bits} bit, K={K_slots}")

    # ===== RA：一次性 Paillier + RSA =====
    t0 = time.perf_counter()
    pub_p, priv_p = paillier.keygen(1024)
    t_ra_keygen_p = time.perf_counter() - t0

    t0 = time.perf_counter()
    ra_pub, ra_priv = rsa.keygen()
    t_ra_keygen_rsa = time.perf_counter() - t0

    t0 = time.perf_counter()
    cp_pub, cp_priv = rsa.keygen()
    t_cp_keygen_rsa = time.perf_counter() - t0

    # RA token: N, E(N), Sig_RA(pk||E(N))
    nonce = secrets.randbits(64)
    t0 = time.perf_counter()
    encN = paillier.enc(pub_p, nonce)
    sig_ra = rsa.sign(ra_priv, _b(pub_p.n) + _b(encN))
    t_ra_issue = time.perf_counter() - t0
    print(f"【RA】生成随机 nonce N（64bit）并用 pk_RA^X 加密：{_ms(t_ra_issue):.2f} ms")

    # ===== 版权方：生成 W_CP =====
    t0 = time.perf_counter()
    wcp_bytes = secrets.token_bytes((wm_len + 7) // 8)
    wcp_bits = [(wcp_bytes[i // 8] >> (7 - (i % 8))) & 1 for i in range(wm_len)]
    t_wcp = time.perf_counter() - t0
    print(f"【版权方】生成版权方水印比特：{_ms(t_wcp):.2f} ms | {wm_len} bit")

    nonce_bits = [(nonce >> (63 - i)) & 1 for i in range(64)]

    # ===== 版权方：加密载体槽位 =====
    t0 = time.perf_counter()
    enc_host = [paillier.enc(pub_p, int(x)) for x in host_ints_signed]
    t_cp_encrypt = time.perf_counter() - t0
    print(f"【版权方】加密载体槽位：{_ms(t_cp_encrypt):.2f} ms | 槽位 {K_slots} (系数 {len(enc_host)})")

    # ===== 版权方：同态复合嵌入（分开统计 W_CP 与 nonce）=====
    t0 = time.perf_counter()
    enc_after_wcp = enc_host[:]
    for bi in range(wm_len):
        start = bi * k
        enc_after_wcp[start : start + k] = composite_embed.embed_group(
            pub_p,
            host_ints_signed[start : start + k],
            enc_after_wcp[start : start + k],
            wcp_bits[bi],
            [1, 2, 4, 8],
            delta_int,
        )
    t_embed_wcp = time.perf_counter() - t0

    t0 = time.perf_counter()
    enc_embedded = enc_after_wcp
    for bi in range(64):
        slot = wm_len + bi
        start = slot * k
        enc_embedded[start : start + k] = composite_embed.embed_group(
            pub_p,
            host_ints_signed[start : start + k],
            enc_embedded[start : start + k],
            nonce_bits[bi],
            [1, 2, 4, 8],
            delta_int,
        )
    t_embed_nonce = time.perf_counter() - t0
    print(
        f"【云服务器】云端水印叠加（密文域）：{_ms(t_embed_wcp+t_embed_nonce):.2f} ms | 槽位 {total_bits}\n"
        f"  ├─ 同态嵌入 W_CP：{_ms(t_embed_wcp):.2f} ms\n"
        f"  └─ 同态嵌入 nonce：{_ms(t_embed_nonce):.2f} ms"
    )

    # ===== mock BC register/confirm =====
    Xd = _sha256(b"Xd" + _b(enc_embedded[0]))
    t0 = time.perf_counter()
    sig_cp = rsa.sign(cp_priv, Xd + _b(encN))
    t_cp_sign = time.perf_counter() - t0

    bc = MockBC()
    record = TxRecord(
        Xd=Xd,
        TX=int(time.time()),
        pkX_RA=_b(pub_p.n),
        encN=_b(encN),
        sigRA=sig_ra,
        sigCP=sig_cp,
        buyer="buyer-1",
    )
    t0 = time.perf_counter()
    token_id = bc.register_by_cp(record)
    t_bc_reg = time.perf_counter() - t0

    t0 = time.perf_counter()
    bc.confirm_by_b(token_id, record)
    t_bc_conf = time.perf_counter() - t0

    # ===== 买家：解密 =====
    t0 = time.perf_counter()
    dec_ints = [paillier.dec(pub_p, priv_p, c) for c in enc_embedded]
    t_b_dec = time.perf_counter() - t0

    # ===== 回填系数并 IDCT 重建（仅 host 通道）=====
    new_coeffs = (np.array(dec_ints, dtype=np.float64) / scale) - offset
    updated_dct = _apply_coeffs(dct_blocks, positions, slot_order, new_coeffs)
    t0 = time.perf_counter()
    spatial_blocks = np.array([dct8x8.idct_block(b) for b in updated_dct])
    ch_wm = dct8x8.from_blocks(spatial_blocks, size, size)
    ch_wm_u8 = np.clip(np.rint(ch_wm), 0, 255).astype(np.uint8)
    t_idct = time.perf_counter() - t0

    rgb_wm = rgb.copy()
    rgb_wm[:, :, host_channel] = ch_wm_u8

    t0 = time.perf_counter()
    out_path = Path("watermarked.png")
    Image.fromarray(rgb_wm).save(out_path)
    t_save = time.perf_counter() - t0
    print(f"【云服务器】IDCT 回填图像：{_ms(t_idct):.2f} ms | 块 {blocks_per_side}×{blocks_per_side}")
    print(f"【云服务器】保存水印图像：{_ms(t_save):.2f} ms | {out_path.resolve()}")

    # ===== 自校验：从保存后的图像重新提取水印 =====
    t0 = time.perf_counter()
    rgb_loaded = np.asarray(Image.open(out_path).convert("RGB"), dtype=np.uint8)
    ch_loaded = rgb_loaded[:, :, host_channel]
    blocks_loaded = dct8x8.to_blocks(ch_loaded.astype(np.float64))
    dct_loaded = np.array([dct8x8.dct_block(b) for b in blocks_loaded])
    coeffs_loaded = _extract_coeffs(dct_loaded, positions, slot_order)
    ints_loaded = np.rint((coeffs_loaded + offset) * scale).astype(np.int64).tolist()
    extracted_bits = composite_embed.extract_bits(
        xs=ints_loaded,
        alpha=[1, 2, 4, 8],
        delta=delta_int,
        k=4,
        max_bits=total_bits,
    )
    t_extract = time.perf_counter() - t0

    wcp_out = extracted_bits[:wm_len]
    nonce_out_bits = extracted_bits[wm_len : wm_len + 64]
    nonce_out = 0
    for b in nonce_out_bits:
        nonce_out = (nonce_out << 1) | int(b)

    wm_err = _ber(wcp_bits + nonce_bits, extracted_bits)
    non_wm_changed = 0
    for slot in range(total_bits, K_slots):
        start = slot * k
        orig = host_ints_signed[start : start + k]
        now = ints_loaded[start : start + k]
        if any(abs(int(a) - int(b)) > 1 for a, b in zip(orig, now)):
            non_wm_changed += 1

    print(f"【校验】水印槽位错误数：{wm_err}")
    print(f"【校验】非水印槽位错误数：{non_wm_changed}")
    print(f"【仲裁者】提取耗时：{_ms(t_extract):.2f} ms | nonce 匹配：{nonce_out == nonce}")

    psnr_y = _psnr_u8(_rgb_to_y_u8(rgb), _rgb_to_y_u8(rgb_loaded))
    print(f"【指标】PSNR(Y)：{psnr_y:.2f} dB")
    print(f"【指标】NC(版权水印 W_CP)：{_nc_bits(wcp_bits, wcp_out):.4f}")
    print(f"【指标】NC(nonce)：{_nc_bits(nonce_bits, nonce_out_bits):.4f}")

    # RA 解密 nonce（仲裁用）
    t0 = time.perf_counter()
    nonce_dec = paillier.dec(pub_p, priv_p, encN)
    t_ra_dec = time.perf_counter() - t0

    success = (wm_err == 0) and (non_wm_changed == 0) and (nonce_out == nonce) and (nonce_dec == nonce)

    timings = {
        "load_image": t_load,
        "texture_stats": t_texture,
        "dct_prepare": t_dct,
        "RA_keygen_paillier": t_ra_keygen_p,
        "RA_keygen_rsa": t_ra_keygen_rsa,
        "RA_issue_encN_sig": t_ra_issue,
        "CP_keygen_rsa": t_cp_keygen_rsa,
        "CP_gen_wcp_bits": t_wcp,
        "CP_encrypt_host": t_cp_encrypt,
        "Cloud_embed_wcp": t_embed_wcp,
        "Cloud_embed_nonce": t_embed_nonce,
        "CP_sign": t_cp_sign,
        "BC_register_mock": t_bc_reg,
        "BC_confirm_mock": t_bc_conf,
        "Buyer_decrypt": t_b_dec,
        "IDCT_reconstruct": t_idct,
        "save_image": t_save,
        "Judge_extract_from_image": t_extract,
        "RA_decrypt_nonce": t_ra_dec,
    }

    cp_total = t_load + t_texture + t_dct + t_cp_keygen_rsa + t_wcp + t_cp_encrypt + t_cp_sign + t_bc_reg
    cloud_total = t_embed_wcp + t_embed_nonce + t_idct + t_save
    user_total = t_bc_conf + t_b_dec
    ra_total = t_ra_keygen_p + t_ra_keygen_rsa + t_ra_issue + t_ra_dec
    judge_total = t_extract

    print("\n【耗时汇总】")
    print(f"  RA：{_ms(ra_total):.2f} ms")
    print(f"  版权方(CP)：{_ms(cp_total):.2f} ms")
    print(f"  云服务器：{_ms(cloud_total):.2f} ms")
    print(f"  买家：{_ms(user_total):.2f} ms")
    print(f"  仲裁者：{_ms(judge_total):.2f} ms")
    print(f"【输出】水印图已保存：{out_path.resolve()}")

    result = {
        "image": str(img_path),
        "size": [size, size, 3],
        "host_channel": host_channel,
        "K_slots": K_slots,
        "watermark_bits": wm_len,
        "nonce_bits": 64,
        "s": s,
        "scale": scale,
        "delta_int": delta_int,
        "texture_percentile": texture_pct,
        "tau": tau,
        "supplemented_blocks": supplemented,
        "success": success,
        "wm_errors": wm_err,
        "non_wm_slot_changed": non_wm_changed,
        "psnr_y_db": psnr_y,
        "nc_wcp": _nc_bits(wcp_bits, wcp_out),
        "nc_nonce": _nc_bits(nonce_bits, nonce_out_bits),
        "timings_sec": timings,
        "role_totals_sec": {
            "RA": ra_total,
            "CP": cp_total,
            "Cloud": cloud_total,
            "User": user_total,
            "Judge": judge_total,
            "Total": ra_total + cp_total + cloud_total + user_total + judge_total,
        },
        "token_id_hex": token_id.hex(),
        "watermarked_image": str(out_path),
    }

    print("\n--- JSON 结果（可用于后续汇总/作图）---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
