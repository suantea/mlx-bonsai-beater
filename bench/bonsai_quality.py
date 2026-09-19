#!/usr/bin/env python3
import sys, os, time, json
import mlx.core as mx
import mlx.nn as nn

pack_root = os.path.expanduser(sys.argv[1])
sys.path.insert(0, os.path.join(pack_root, "runtime"))
from runtime import Packed
import artifact
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_tokenizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench_tasks import TASKS


def load_text_model(directory):
    config = json.loads(open(os.path.join(directory, "config.json")).read())
    if config.get("model_type") != "prism_hadamard_qwen35":
        raise ValueError("Unsupported packed model schema")
    model = TextModel(TextModelArgs.from_dict(config["text_config"]))
    weights = mx.load(os.path.join(directory, "model.safetensors"))
    prefix = "language_model."
    seen = set()
    for record in config["modules"]:
        if record["path"] in seen:
            raise ValueError("Duplicate packed module")
        seen.add(record["path"])
        parts = record["path"].split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        key = prefix + record["path"]
        arrays = [weights[key + "." + s] for s in ("weight", "scales", "biases")]
        if record["dtype"] != "float16":
            raise ValueError("Unsupported activation dtype")
        block = record["block"]
        signs = weights.get(key + ".signs")
        if block and signs is None:
            raise ValueError("Missing sign vector")
        original = getattr(parent, parts[-1])
        artifact.validate_record(original, record, arrays, signs)
        setattr(parent, parts[-1], Packed(arrays, block, signs, record["embedding"], mx.float16))
    weights = {
        k.removeprefix(prefix): v
        for k, v in weights.items()
        if not k.startswith("vision_tower.") and k.startswith(prefix)
    }
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, config


model, config = load_text_model(pack_root)
tok = load_tokenizer(pack_root)
sampler = make_sampler(temp=0.0)
for name, q in TASKS.items():
    p = tok.apply_chat_template([{"role": "user", "content": q}],
                               add_generation_prompt=True, tokenize=False,
                               enable_thinking=False)
    t = time.time()
    out = generate(model, tok, prompt=p, max_tokens=700, sampler=sampler, verbose=False)
    print(f"\n{'='*70}\n### {name}  ({time.time()-t:.1f}s)\n{'='*70}")
    print(out.strip())