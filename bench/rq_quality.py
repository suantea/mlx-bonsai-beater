#!/usr/bin/env python3
import sys, os, time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_tasks import TASKS

path = os.path.expanduser(sys.argv[1])
model, tok = load(path)
sampler = make_sampler(temp=0.0)
for name, q in TASKS.items():
    p = tok.apply_chat_template([{"role": "user", "content": q}],
                               add_generation_prompt=True, tokenize=False,
                               enable_thinking=False)
    t = time.time()
    out = generate(model, tok, prompt=p, max_tokens=700, sampler=sampler, verbose=False)
    print(f"\n{'='*70}\n### {name}  ({time.time()-t:.1f}s)\n{'='*70}")
    print(out.strip())
