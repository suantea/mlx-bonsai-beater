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
import socket
import sys
import threading
import time

FA_WINDOW = int(os.environ.get("FA_WINDOW", "2048"))
PRE_STEP = int(os.environ.get("PRE_STEP", "256"))
# 单个请求最长时长（秒）。超时强杀该连接释放 GPU 队列，防止客户端断开后
# 生成线程永远占着推理队列拖死整台服务。64K 全程 ~2min、128K ~4min 远低于默认。
REQUEST_TIMEOUT = int(os.environ.get("MLX_REQUEST_TIMEOUT", "900"))

import mlx.core as mx

# --- 在 server 加载模型前注入滑窗 patch（真实建层处 qwen3_5.TextModel） ---
if FA_WINDOW > 0:
    import mlx_lm.models.qwen3_5 as ql
    from mlx_lm.models.cache import ArraysCache

    class WindowCache:
        """真滑窗 KV 缓存：物理序 = 逻辑序，只保留最近 max_size 个 token。

        旋转缓存（RotatingKVCache）在超出窗口后会把新 key 覆盖到物理最前，
        而 SDPA 按"物理位置即逻辑位置"做因果注意力，导致长 prompt 下 mask
        错位、生成退化。本缓存改为「追加 + 超窗裁头」，物理序永远等于时间序，
        与 mx.fast.scaled_dot_product_attention 天然兼容，无需旋转 mask。
        """
        def __init__(self, max_size):
            self.max_size = max_size
            self.keys = None
            self.values = None
            self.offset = 0
            self.start_position = 0
            self._idx = 0

        def update_and_fetch(self, keys, values):
            self.offset += keys.shape[2]
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                self.keys = mx.concatenate([self.keys, keys], axis=2)
                self.values = mx.concatenate([self.values, values], axis=2)
            if self.keys.shape[2] > self.max_size:
                drop = self.keys.shape[2] - self.max_size
                self.start_position += drop
                self.keys = self.keys[..., drop:, :]
                self.values = self.values[..., drop:, :]
            return self.keys, self.values

        @property
        def state(self):
            return self.keys, self.values

        @state.setter
        def state(self, v):
            self.keys, self.values = v
            self.offset = self.keys.shape[2] if self.keys is not None else 0
            self.start_position = 0

        def make_mask(self, N, window_size=None, return_array=False):
            # 物理序=逻辑序且窗口外的 key 已被裁掉，只需标准 causal mask
            if N <= 1:
                return None
            if return_array:
                rows = mx.arange(N)[:, None]
                cols = mx.arange(N)[None, :]
                return cols <= rows
            return "causal"

        def is_trimmable(self):
            # 未满窗口的 cache 才允许被 trim/复用（对齐官方 RotatingKVCache），
            # 防止短请求误复用长请求已滑窗的 KV（offset 与 keys 物理内容不匹配）。
            return self.offset < self.max_size

        def trim(self, n):
            if self.keys is None:
                return n
            n = min(self.offset, n)
            self.offset -= n
            return n

        @property
        def meta_state(self):
            return tuple(map(str, (self.max_size, self.start_position, self.offset)))

        @meta_state.setter
        def meta_state(self, v):
            self.max_size, self.start_position, self.offset = map(
                int, v
            )

        def empty(self):
            return self.keys is None

        @property
        def nbytes(self):
            return (
                self.keys.nbytes + self.values.nbytes
                if self.keys is not None
                else 0
            )

    def windowed_make_cache(self):
        return [
            ArraysCache(size=2)
            if getattr(l, "is_linear", False)
            else WindowCache(max_size=FA_WINDOW)
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

# --- 请求级看门狗：防止客户端断开后生成线程占死 GPU 队列 ---
# mlx_lm.server 的 APIHandler 继承 BaseHTTPRequestHandler，
# 客户端 BrokenPipe 后生成循环仍会跑完，让后续请求全部排队假死。
# 这里在每请求开始时起一个 daemon 定时器，超时就 shutdown 该连接，
# 让该请求的读写全部抛异常、线程退出、GPU 队列释放。
def _install_watchdog():
    from mlx_lm.server import APIHandler

    _orig_handle = APIHandler.handle
    _orig_setup = APIHandler.setup

    def _kill(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def watchdog_setup(self):
        _orig_setup(self)
        self._watchdog = threading.Timer(REQUEST_TIMEOUT, _kill, args=(self.connection,))
        self._watchdog.daemon = True
        self._watchdog.start()

    def watchdog_handle(self):
        try:
            _orig_handle(self)
        finally:
            wd = getattr(self, "_watchdog", None)
            if wd is not None:
                wd.cancel()

    APIHandler.setup = watchdog_setup
    APIHandler.handle = watchdog_handle
    print(f"[serve_window] watchdog: request timeout {REQUEST_TIMEOUT}s", flush=True)


if __name__ == "__main__":
    _install_watchdog()
    # server 默认 --prefill-step-size 2048，窗口化 + 大 chunk 会瞬态超大内存 → 注入 256
    if "--prefill-step-size" not in sys.argv:
        sys.argv += ["--prefill-step-size", str(PRE_STEP)]
    # 滑窗 KV 缓存跨请求复用（--prompt-cache-size 默认 10）会让短请求复用长请求
    # 已滑窗的缓存，offset/keys 不匹配导致生成退化 → 默认关闭，每请求独立构建。
    if "--prompt-cache-size" not in sys.argv:
        sys.argv += ["--prompt-cache-size", "0"]
    print(
        f"[serve_window] window={FA_WINDOW} prefill_step={PRE_STEP} prompt_cache=off",
        flush=True,
    )
    main()