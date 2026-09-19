#!/usr/bin/env python3
"""mlx_lm server 启动包装：注入滑窗 patch + server 端 trim 修复。

用法:
  FA_WINDOW=2048 ./serve_window.py --model $P --port 8080
环境变量:
  FA_WINDOW: 滑窗大小，0=关闭滑窗（全注意力）
  PRE_STEP:  prefill 步长（默认 256）
注意:
  - 必须在 import mlx_lm 前设置好 patch（server 会异步加载模型）
  - keep 参数与 server batch 路径不兼容，这里不设置
"""
import os
import sys

FA_WINDOW = int(os.environ.get("FA_WINDOW", "2048"))
PRE_STEP = int(os.environ.get("PRE_STEP", "256"))

import mlx.core as mx

# --- 在 server 加载模型前注入滑窗 patch（真实建层处 qwen3_5.TextModel） ---
if FA_WINDOW > 0:
    import mlx_lm.models.qwen3_5 as ql
    from mlx_lm.models.cache import ArraysCache, RotatingKVCache

    class TrimSafeRotatingKVCache(RotatingKVCache):
        def is_trimmable(self):
            return True

    def windowed_make_cache(self):
        return [
            ArraysCache(size=2)
            if getattr(l, "is_linear", False)
            else TrimSafeRotatingKVCache(max_size=FA_WINDOW)
            for l in self.layers
        ]

    ql.TextModel.make_cache = windowed_make_cache

    # --- server 端 batch 缓存修复 ---
    # BatchRotatingKVCache.is_trimmable 仍是窗口满返回 False（原版坑）；
    # ArraysCache 继承基类也是 False（qwen3_next 线性层 delta state 固定大小）。
    # trim 只回退 _idx/_offset，对旋转缓存与线性层都是安全的 → 统一放开。
    from mlx_lm.models.cache import BatchRotatingKVCache

    BatchRotatingKVCache.is_trimmable = lambda self: True
    # ArraysCache（线性层 delta state，固定大小压缩态）无 trim 方法，
    # server 复用历史时 trim_prompt_cache 会对每个元素调 .trim() → 必须补 no-op
    ArraysCache.is_trimmable = lambda self: True
    ArraysCache.trim = lambda self, n: n

# --- 接管标准入口 ---
from mlx_lm.server import main

if __name__ == "__main__":
    # server 默认 --prefill-step-size 2048，窗口化 + 大 chunk 会瞬态超大内存 → 注入 256
    if "--prefill-step-size" not in sys.argv:
        sys.argv += ["--prefill-step-size", str(PRE_STEP)]
    print(f"[serve_window] window={FA_WINDOW} prefill_step={PRE_STEP}", flush=True)
    main()