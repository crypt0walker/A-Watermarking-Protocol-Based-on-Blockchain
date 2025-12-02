# A-Watermarking-Protocol-Based-on-Blockchain

Python 实验脚本，复现 Frattolillo 协议核心（Paillier + 8×8 DCT + 严格 composite embedding + 真 RSA 签名），支持单次交易链下耗时测量，链上合约稿在 `awpbb/onchain/WatermarkProtocol.sol`。

## 快速使用

```bash
# 安装依赖（需要 numpy + pillow）
pip install numpy pillow

# 单次交易基准，默认读取 ./lenna.jpg，缩放 256x256，水印 64 bit（另加 64 bit nonce）
python scripts/benchmark_offchain.py \
  --image lenna.jpg \
  --size 256 \
  --wm-bits 64 \
  --paillier-bits 1024 \
  --delta 32 \
  --save-watermarked  # 可选，输出 watermarked.png
```

输出 JSON，包含各端耗时（秒）：

- `RA_keygen_paillier` / `RA_keygen_rsa` / `RA_issue`
- `CP_keygen_rsa` / `CP_encrypt` / `CP_embed` / `CP_sign` / `CP_bc_register`
- `B_bc_confirm` / `B_decrypt` / `B_reconstruct`
- `J_extract` / `RA_decrypt_nonce`
- 以及 `success` 标记（提取 nonce 与 RA 解密一致）。

CLI 参数可动态调整图像尺寸与水印长度：
- `--size N`：将输入图缩放为 N×N，默认 256。
- `--wm-bits K`：W_CP 长度（bit），最终嵌入长度为 `K + 64`（附加 nonce）。
- `--delta` / `--alpha`：QIM 步长、复合权重。

## 1-to-n 与链式场景的复用说明
- 可复用部分：Paillier 密钥生成/加解密、DCT 处理、严格 composite 嵌入/提取、签名/验签、mock BC/合约接口。
- 随 n 线性增长的部分：Paillier 加密/解密次数、复合嵌入次数、签名/验签次数、BC 注册/确认调用。
- 链式扩展：按链长度重复 CP→B→RA→BC 的相同流水，各阶段计时可在 `benchmark_offchain.py` 内循环 n 次累积或分摊；合约 `WatermarkProtocol.sol` 可直接用于 anvil 部署，对应调用替换 mock BC 部分。
