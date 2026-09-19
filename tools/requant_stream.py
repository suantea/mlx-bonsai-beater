#!/usr/bin/env python3
"""Streaming requantizer for MLX safetensors checkpoints.

Reads udq2 (mixed 2/3/4/5/8-bit) and writes a smaller checkpoint:
  - drop vision_tower.* entirely (mlx_lm.sanitize skips it)
  - quantize BF16 out_proj/o_proj (linear_attn + self_attn) to N bits
  - optionally lower switch_mlp.down_proj bits
  - optionally change expert group_size (64 -> 128)
Everything is processed tensor-by-tensor so peak RAM stays low.
"""
import argparse, json, math, os, shutil, sys, struct

import numpy as np
import mlx.core as mx
from safetensors import safe_open
import safetensors.mlx as stmlx

BASE_BITS = 2
BASE_GROUP = 64
MODE = "affine"

DROP_PREFIXES = ("vision_tower.",)


def load_headers(dirpath, index):
    shards = sorted(set(index.values()))
    heads = {}
    for sh in shards:
        with open(os.path.join(dirpath, sh), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            heads[sh] = json.loads(f.read(n))
    return heads


def nbytes_quant(shape, bits, group, src_bits=None):
    packed = shape[-1]
    in_dim = packed if src_bits is None else packed * 32 // src_bits
    outer = 1
    for d in shape[:-1]:
        outer *= d
    new_packed = in_dim * bits // 32
    wbytes = outer * new_packed * 4
    groups = (in_dim + group - 1) // group
    return wbytes + outer * groups * 2 * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--out-proj-bits", type=int, default=4)
    ap.add_argument("--expert-down-bits", type=int, default=2)
    ap.add_argument("--expert-group", type=int, default=64)
    ap.add_argument("--shard-gb", type=float, default=2.0)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    dst = os.path.expanduser(args.dst)

    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    weight_map = idx["weight_map"]
    heads = load_headers(src, weight_map)
    src_cfg = json.load(open(os.path.join(src, "config.json")))
    src_q = src_cfg.get("quantization", {})
    name_set = set(weight_map)

    def src_bits_group(module):
        e = src_q.get(module)
        if e is None:
            return BASE_BITS, BASE_GROUP
        return e.get("bits", BASE_BITS), e.get("group_size", BASE_GROUP)

    def is_quantized(module):
        return f"{module}.scales" in name_set

    def plan(module):
        """return ('drop'|'keep'|'q', payload)"""
        for p in DROP_PREFIXES:
            if module.startswith(p):
                return ("drop", None)
        if not is_quantized(module):
            # BF16 tensor: optionally quantize
            if module.endswith("linear_attn.out_proj") or module.endswith("self_attn.o_proj"):
                return ("q", (args.out_proj_bits, BASE_GROUP))
            return ("keep", None)
        # quantized module
        if module.endswith("switch_mlp.down_proj"):
            return ("q", (args.expert_down_bits, args.expert_group))
        if module.endswith("switch_mlp.gate_proj") or module.endswith("switch_mlp.up_proj"):
            b, g = src_bits_group(module)
            if args.expert_group != g:
                return ("q", (b, args.expert_group))
            return ("keep", None)
        return ("keep", None)

    # ---------- dry run: project sizes ----------
    proj = {}
    handled = set()
    for name, sh in weight_map.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        if name.endswith(".weight"):
            module = name[: -len(".weight")]
        else:
            module = name  # bare bf16 param
        if name in handled:
            continue
        handled.add(name)
        act, payload = plan(module)
        if act == "drop":
            continue
        shape = heads[sh][name]["shape"]
        if act == "q":
            b, g = payload
            sb = src_bits_group(module)[0] if is_quantized(module) else None
            proj[name] = nbytes_quant(shape, b, g, src_bits=sb)
        elif name.endswith(".weight") and is_quantized(module):
            proj[name] = sum(
                heads[weight_map[f"{module}.{s}"]][f"{module}.{s}"]["data_offsets"][1]
                - heads[weight_map[f"{module}.{s}"]][f"{module}.{s}"]["data_offsets"][0]
                for s in ("weight", "scales", "biases")
            )
        else:
            proj[name] = int(np.prod(shape)) * 2

    total = sum(proj.values())
    src_total = sum(
        f["data_offsets"][1] - f["data_offsets"][0]
        for h in heads.values()
        for f in h.values()
        if isinstance(f, dict) and "data_offsets" in f
    )
    print(f"source weights : {src_total/1e9:.2f} GB")
    print(f"projected out  : {total/1e9:.2f} GB   (saved {(src_total-total)/1e9:.2f} GB)")
    if not args.apply:
        print("dry run (pass --apply to write)")
        return

    # ---------- apply ----------
    os.makedirs(dst, exist_ok=True)
    out_q = {"group_size": BASE_GROUP, "bits": BASE_BITS, "mode": MODE}
    out_tensors = {}
    out_bytes = 0
    out_files = []
    index_out = {}
    SHARD_LIMIT = args.shard_gb * 1e9

    def flush():
        nonlocal out_tensors, out_bytes, out_files
        if not out_tensors:
            return
        fn = f"model-{len(out_files)+1:05d}.safetensors"
        mx.save_safetensors(os.path.join(dst, fn), out_tensors)
        for k in out_tensors:
            index_out[k] = fn
        out_files.append(fn)
        print(f"  wrote {fn}: {len(out_tensors)} tensors, {out_bytes/1e9:.2f} GB")
        out_tensors = {}
        out_bytes = 0

    def add(name, t):
        nonlocal out_bytes
        out_tensors[name] = t
        out_bytes += t.nbytes

    # Build "units": a quantized module (weight+scales+biases) or a single bf16 tensor.
    units = {}          # primary shard -> list of (kind, module_or_name)
    for name in weight_map:
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        module = name[: -len(".weight")] if name.endswith(".weight") else name
        act, _ = plan(module)
        if act == "drop":
            continue
        if is_quantized(module):
            units.setdefault(weight_map[f"{module}.weight"], []).append(("q3", module))
        else:
            units.setdefault(weight_map[name], []).append(("one", name))

    def load_temp(n):
        tmp = mx.load(os.path.join(src, weight_map[n]))
        v = tmp.pop(n, None)
        del tmp
        return v

    for sh, ulist in units.items():
        data = mx.load(os.path.join(src, sh))
        for kind, key in ulist:
            if kind == "one":
                add(key, data.pop(key))
            else:
                module = key
                wname = f"{module}.weight"
                act, payload = plan(module)
                w = data.pop(wname, None)
                if w is None:
                    w = load_temp(wname)
                s = data.pop(f"{module}.scales", None)
                if s is None:
                    s = load_temp(f"{module}.scales")
                bi = data.pop(f"{module}.biases", None)
                if bi is None:
                    bi = load_temp(f"{module}.biases")
                if act == "q":
                    b, g = payload
                    if is_quantized(module):
                        sb, sg = src_bits_group(module)
                        wf = mx.dequantize(w, s, bi, group_size=sg, bits=sb)
                    else:
                        wf = w
                    q, qs, qb = mx.quantize(wf, group_size=g, bits=b)
                    add(wname, q)
                    add(f"{module}.scales", qs)
                    add(f"{module}.biases", qb)
                    out_q[module] = {"group_size": g, "bits": b, "mode": MODE}
                else:
                    add(wname, w)
                    add(f"{module}.scales", s)
                    add(f"{module}.biases", bi)
                    b, g = src_bits_group(module)
                    out_q[module] = {"group_size": g, "bits": b, "mode": MODE}
            if out_bytes >= SHARD_LIMIT:
                flush()
        del data

    flush()

    # ---------- config + metadata ----------
    new_cfg = dict(src_cfg)
    new_cfg["quantization"] = out_q
    json.dump(new_cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    json.dump(
        {"metadata": {"total_size": sum(proj.values())}, "weight_map": index_out},
        open(os.path.join(dst, "model.safetensors.index.json"), "w"),
        indent=2,
    )
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json", "preprocessor_config.json", "vocab.json",
              "merges.txt", "special_tokens_map.json"]:
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, f))
    print("done ->", dst)


if __name__ == "__main__":
    main()
