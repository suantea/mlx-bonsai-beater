# Bonsai-2 27B 研究笔记（2026-09）

本次研究的目标：弄清 Prism 新一代三值模型 **Bonsai-2** 的真实能力边界，判断 16GB 统一内存 Mac 上能否跑满 64K / 128K 长上下文，以及 MLX 路线与官方 llama.cpp fork 路线各自的取舍。

## 结论速览

- Bonsai-2 **只有 27B 一档**（`BONSAI2_SIZES="27B"`）。所谓"8B / 4B / 1.7B 更小版本"全部是**上一代**（Qwen3 底、纯全注意力、文本模型），不是 Bonsai-2，且 archive 短。
- Bonsai-2 27B 是**混合注意力**（64 层中 16 层全注意 + 48 层线性/GDN），KV cache 只要 **64 KiB/token**（FP16）。这是它能上 128K 的关键；上一代 8B 是 140 KiB/token，128K 物理不可行。
- MLX 有官方 2-bit 包（8.6GB），但**无 server**：官方只提供 llama.cpp fork + GGUF 的服务路径。MLX 只能做短上下文一次性推理。
- 16GB 上跑 128K 的可行组合：**PTQ1_0（5.95GB）+ BONSAI_KV4=1（q4_0 K/V）≈ 8GB 总量**，不需要换机器。

## 模型与格式

| 项 | 值 |
|---|---|
| 家族 | `bonsai2`（Bonsai-2 27B），官方默认 |
| MLX 2-bit 包 | `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`，8.6GB（含 vision tower，全精度） |
| GGUF 主仓 | `prism-ml/Ternary-Bonsai-2-27B-gguf` |
| GGUF 两档 | **PTQ1_0 5.95GB**（1.75 bpw，稠密三值，最小）／ **PQ2_0 7.21GB**（2.13 bpw，prefill 更快，demo 默认） |
| vision 投影 | mmproj-Q8_0 0.63GB |
| 危险文件 | `*-Q2_0.gguf`（放在 gguf-dev 仓）会被**mainline llama.cpp 静默加载并输出乱码** —— 只允许 Prism fork 运行 |

> mainline llama.cpp 完全不能跑 Bonsai-2（PTQ1_0/PQ2_0 是未识别类型，安全拒绝；`Q2_0` 是会静默乱码的坑）。Ollama / LM Studio 内置 stock 后端，同样不可用。

## 内存账（统一内存 Mac）

FP16 KV：27B = 64 KiB/token；上一代 8B = ~140 KiB/token。

| 配置 | 权重 | KV (64K) | KV (128K) | 总量 (128K) | 16GB 判定 |
|---|---|---|---|---|---|
| PTQ1_0 | 5.95GB | 4.2GB | 8.4GB | ~14.4GB | ⚠️ 紧，可行 |
| PTQ1_0 + KV4 (q4_0) | 5.95GB | ~1.1GB | ~2.1GB | **~8.1GB** | ✅ 最稳 |
| PQ2_0 | 7.21GB | 4.2GB | 8.4GB | ~15.6GB | ❌ 爆 |
| PQ2_0 + KV4 | 7.21GB | ~1.1GB | ~2.1GB | ~9.3GB | ✅ |

官方 `common.sh` 按 RAM 分层默认上下文：16GB → **16384**；用 `BONSAI_CTX=131072` + `BONSAI_KV4=1` 显式开 128K。模型最大训练上下文 262144。

## 采样参数（官方）

- **Bonsai-2 27B**：temp 1.0 / top-p 0.95 / top-k 20（思考模型，thinking 常开，web UI 可按会话调 effort）
- 上一代 27B：temp 0.7 / top-p 0.95 / top-k 20 / min-p 0

## 实测记录（本机 Apple M5 · 16GB）

| 项目 | 结果 |
|---|---|
| MLX 一次性推理 | LOAD 1.7s / 峰值 8.04GB / ~11.5 tok/s（短上下文，中文质量良好） |
| MLX 长上下文 | >3.8K token 峰值即达 13.49GB，16GB 无余量，再长 swap 假死 —— MLX 不适合 64K+ |
| MLX server | 官方无此路线；普通 mlx_lm 加载会**静默输出错误**（schema v2 与本地 artifact 错配），需自定义 loader 适配（剥 `language_model.` 前缀、跳过 `vision_tower`、lm_head 不 tie） |
| llama.cpp fork | bin/mac v7（build 10709）就绪，需 `--jinja`/`-fa on`/`-ngl 99`；官方锚点 M5 Pro: TG128 28.1 / PP512 387 |

## 对 mlx-bonsai-beater 的意义

原 README"与 Bonsai 对比"说的是**上一代 Ternary-Bonsai-27B 的 MLX 路径**（无 server、长上下文 OOM）——结论仍成立。但 Bonsai-2 27B 走**官方 llama.cpp fork + GGUF** 时，16GB 上 128K 是官方支持、实测内存可行的路线（PTQ1_0 + KV4 ≈ 8GB），并自带 OpenAI 兼容 server 与内置 web UI：

```bash
BONSAI_FAMILY=bonsai2 BONSAI_MODEL=27B \
  BONSAI_CTX=131072 BONSAI_KV4=1 \
  BONSAI_HOST=localhost:8080 \
  ./scripts/start_llama_server.sh   # 出自官方 Bonsai-demo
```

因此"小内存跑长上下文"这条赛道上，Bonsai-2 27B 是**官方有成的对标物**（不是唯一解，而是验证本方案思路的正确参照系）；本方案的优势仍在于 MLX 的原生解码速度与完全自治（滑窗精度、主备调度、零依赖 GUI）。

## 参考资料

- 官方 demo 仓库：`https://github.com/PrismML-Eng/Bonsai-demo`（`AGENTS.md` / `common.sh` / `MODEL-FORMATS.md` / `KV-CACHE.md`）
- 模型集合：`https://huggingface.co/collections/prism-ml/bonsai-27b`（Bonsai-2 仅 27B；Bonsai / Ternary-Bonsai 各有 27B/8B/4B/1.7B）