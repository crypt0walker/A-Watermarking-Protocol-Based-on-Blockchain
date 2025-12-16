# A-Watermarking-Protocol-Based-on-Blockchain

Python 实验脚本，复现 Frattolillo 协议核心（Paillier + 8×8 DCT + 严格 composite embedding + 真 RSA 签名），支持单次交易链下耗时测量，链上合约稿在 `awpbb/onchain/WatermarkProtocol.sol`。

## 快速使用

```bash
# 安装依赖（推荐 gmpy2 + opencv 以加速 Paillier 与 DCT）
python3 -m pip install --user numpy pillow gmpy2 opencv-python

# 单次交易基准（交互式输入）：图像路径、边长、水印长度、QIM 步长 s、纹理分位数
python3 scripts/benchmark_offchain.py
```

## 官方模拟实验（Excel + Fig1/2/3）

`scripts/official_simulation.py` 会先跑少量 `256×256` 的真实基准（3 次均值）标定单位开销，再按主要 `O(size^2)` 尺度外推并加入小随机扰动，生成 *SIMULATED/PREDICTED* 的三组实验表与图：

```bash
python3 scripts/official_simulation.py
```

输出：
- `artifacts/experiments_AWPBB.xlsx`
- `artifacts/fig1.png`
- `artifacts/fig2.png`
- `artifacts/fig3_cp.png`
- `artifacts/fig3_cloud.png`

`scripts/benchmark_offchain.py` 会输出中文阶段日志并在末尾打印 JSON（含 `timings_sec` 与 `role_totals_sec`），同时保存水印图 `watermarked.png` 并从文件回读提取进行自校验（BER/NC/PSNR）。

## 1-to-n 与链式场景的复用说明
- 严格遵循原文“一次性 pk_RA^X”：不同买家/交易必须重新生成 Paillier 密钥并用该 pk 重加密宿主系数，密文宿主不可跨用户复用。
- 可复用（只需一次）：原图读取/缩放、8×8 分块/DCT、嵌入位置模板（明文侧预处理）。
- 随用户数 n 线性增长：RA 密钥与 token、CP 宿主加密与嵌入、买家解密、链上/链下注册确认、仲裁提取。
