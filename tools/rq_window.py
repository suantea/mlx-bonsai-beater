#!/usr/bin/env python3
import sys, os, time
import mlx.core as mx

WINDOW = int(os.environ.get("FA_WINDOW", "0"))

import mlx_lm.models.qwen3_5 as ql
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

if WINDOW > 0:
    class TrimSafeRotatingKVCache(RotatingKVCache):
        """RotatingKVCache 窗口满后 is_trimmable()=False，会断掉多轮 prompt-cache
        复用（can_trim_prompt_cache 要求所有层可 trim）。trim 本身安全，这里强制可 trim。"""
        def is_trimmable(self):
            return True

    def windowed_make_cache(self):
        return [
            ArraysCache(size=2)
            if l.is_linear
            else TrimSafeRotatingKVCache(max_size=WINDOW, keep=int(os.environ.get("FA_KEEP", "0")))
            for l in self.layers
        ]
    # 真实建层处是 qwen3_5.TextModel.make_cache（qwen3_5_moe.Model 只是转发），
    # patch 打在 qwen3_next 上不会生效
    ql.TextModel.make_cache = windowed_make_cache

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

path = os.path.expanduser(sys.argv[1])
target = int(sys.argv[2]) if len(sys.argv) > 2 else 64000
mt = int(sys.argv[3]) if len(sys.argv) > 3 else 64

mx.reset_peak_memory()
t0 = time.time()
model, tok = load(path)
print(f"[load] {time.time()-t0:.2f}s peak={mx.get_peak_memory()/1e9:.2f}GB  window={WINDOW or 'off'}")

para = ("The quick brown fox jumps over the lazy dog near the river bank. "
        "Machine learning models process tokens sequentially and attend to context. "
        "Memory bandwidth limits decoding speed in local inference setups. ")
filler = para * (target * 5 // len(tok.encode(para)) + 2)
needle = "\n\nNow, in about 150 words, summarize what this document is about.\n"
body = filler[: target * 5] + needle

msgs = [{"role": "user", "content": body}]
p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                            enable_thinking=False)
ids = tok.encode(p)
print(f"[prompt] {len(ids)} tokens")

mx.reset_peak_memory()
mx.clear_cache()
sampler = make_sampler(temp=0.0)
t2 = time.time()
out = generate(model, tok, prompt=ids, max_tokens=mt, sampler=sampler,
               prefill_step_size=int(os.environ.get("PRE_STEP","512")), verbose=True)
print(f"[wall] {time.time()-t2:.2f}s peak={mx.get_peak_memory()/1e9:.2f}GB")
print("=== tail output ==="); print(out[-400:])