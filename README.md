# mlx-bonsai-beater

让一个 **16GB 统一内存的 Mac** 全场景跑满 35B 规模 MoE 模型：**50+ t/s 解码、最高 128K 上下文、双机主备调度**，并配一个开箱即用的本地聊天窗口。

名字来源：与我们实测的 **Ternary-Bonsai-27B**（同样定位"小内存跑大模型"）正面 PK——见 [与 Bonsai 的实测对比](#与-bonsai-27b-实测对比)。

## 为什么

| 约束 | 你说服自己 | 我们实测 |
|---|---|---|
| 16GB 内存 + MLX | 35B MoE 量化后能跑 | ✅ 10.49GB 全驻留 |
| 长上下文（64K+） | KV cache 塞得下 | ✅ 滑窗 full-attention，峰值 11.5GB |
| 速度 | 50+ t/s | ✅ 短上下文 52.6 t/s；64K 仍 50.5 t/s |
| 单机宕机怎么办 | 重启 | ✅ 第二台 Mac 经 SSH 隧道组成主备，路由器 2 秒内自动切兜底 |

## 架构

```
┌───────────── 客户端（永远只连 8080，无感） ─────────────┐
│  浏览器 GUI / curl / 任何 OpenAI SDK                     │
└──────────────────────┬────────────────────────────────  ┘
                       │ 127.0.0.1:8080
              ┌────────▼────────┐  每 2s 探活
              │   mlx-router   │◄──────────────┐
              └───────┬────────┘  主通→主 / 主断→备
        ┌─────────────┴─────────────┐
        │ 主: 对端(经 SSH 隧道 8090) │  备: 本机 8081
        ▼                           ▼
  serve_window(对端)            serve_window(本机)
  qwen36-35b-a3b-rq            qwen36-35b-a3b-rq
```

- **`scripts/serve_window.py`**：在 mlx_lm server 加载模型前注入滑窗 patch（真实建层处 `qwen3_5.TextModel`），并修复 server 端 batch 缓存的 trim 判定，让多轮对话的 prompt-cache 复用不断链。
- **`infra/mlx-router.py`**：OpenAI 兼容反向代理，主备探活 2 秒切换，透传 `X-MLX-Route` 头供客户端识别当前走主还是兜底。
- **`infra/mlx-server`**：`status / start / stop / tunnel 8090 / router 8080 / sync` 一键管理双机（部署时设置 `MLX_REMOTE`）。
- **`tools/requant_stream.py`**：流式 requantizer，把 udq2 存量量化版进一步压缩（如 2-bit MoE expert + 4-bit attention 投影），逐 tensor 处理、低峰内存。
- **`gui/`**：零依赖本地聊天窗口（纯 stdlib HTTP 反向代理 + 单页前端），oMLX / llama.cpp webui 风格，带模型选择、温度/token 参数、流式输出、主备路由徽标。

## 快速开始

```bash
# 1) 准备模型（mlx_lm 兼容的量化 MoE 目录，例如 udq2 的 Qwen3.x-35B-A3B）
#    已有成品可用 tools/requant_stream.py 压缩：
python3 tools/requant_stream.py --src $BASE --dst $OUT \
    --out-proj-bits 4 --expert-down-bits 2 --apply

# 2) 起本机服务（滑窗 2048，兜底端口 8081）
FA_WINDOW=2048 python3 scripts/serve_window.py \
    --model $OUT --port 8081 --trust-remote-code \
    --chat-template-args '{"enable_thinking":false}'

# 3) 起调度中心（统一入口 8080）
export MLX_BACKUP=http://127.0.0.1:8081
python3 infra/mlx-router.py
#    多机：先让对端跑同一 serve_window(8080)，
#         再 export MLX_REMOTE=user@peer && infra/mlx-server tunnel start

# 4) 打开聊天窗口
python3 gui/server.py --port 7860 --upstream http://127.0.0.1:8080
#   → http://127.0.0.1:7860
```

直接用 OpenAI SDK 也行：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -d '{"model":"<model>","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## 实测基准（本机：16GB 统一内存 Apple M5）

- **短上下文**：decode **52.6 t/s**（threshold 50+ 达标）
- **64K（65,443 tok）+ 滑窗 2048**：decode **50.5 t/s**、prefill **834 t/s**、峰值 **11.51GB** ✅
- **128K（131,706 tok）+ 滑窗 2048 + prefill_step 256**：decode **37.7 t/s**、峰值 **11.51GB**（无 OOM、无抖动）
- 质量（5 题含 SQL JOIN 陷阱判别）：**无退化**，评分与 35B 原版一致

## 与 Bonsai-27B 实测对比

同一台 M5 Air 16GB、同一套 5 题评测：

| 项目 | 本方案 (35B-A3B MoE) | Ternary-Bonsai-27B (dense) |
|---|---|---|
| 解码速度 | **52.6 t/s** | 6.5 t/s（MLX 实测） |
| prefill | **834 t/s** | 65 t/s |
| 32K 上下文 | ✅ 48.8 t/s | ❌ **OOM** |
| 64K / 128K | ✅ 50.5 / 37.7 t/s | ❌ OOM |
| 服务端 | ✅ OpenAI 兼容 server | ❌ 官方明示 *No MLX server yet* |
| 质量（5 题） | 持平 | 持平（重构类略优） |

> Bonsai-27B 官方宣称的 26–44 t/s / 262K 上下文是 **llama.cpp GGUF + 私有 fork** 路径的数字；在 Apple MLX 上它既无 server、长上下文也 OOM。它的成就是**智能密度**（5.9GB 塞进 27B 且质量保留），适合 llama.cpp/CUDA 生态。

## 项目结构

```
scripts/serve_window.py    滑窗 server 包装（FA_WINDOW/PRE_STEP 环境变量）
tools/requant_stream.py    流式量化压缩（udq2 → 更小）
tools/rq_window.py         滑窗加载 + 长上下文生成基准
bench/rq_quality.py        5 题质量评测（可对照任意 mlx 模型）
bench/bonsai_quality.py    Bonsai 专用评测（剥 language_model. 前缀等）
infra/mlx-router.py        主备调度中心（8080 统一入口）
infra/mlx-server           双机管理 CLI（status/tunnel/router/sync）
gui/server.py + static/    本地聊天窗口（零依赖）
```

## 已知限制

- 滑窗是全注意力的近似：模型只能看到最近 W 个 token（`FA_WINDOW`），超长程精确召回按需调大窗口。
- prefill 大 chunk 会瞬态占用大量内存 → server 侧强制 `--prefill-step-size 256`。
- 多机依赖 SSH 隧道连通对端；对端离线时自动降级本机（单机模式依然完整可用）。