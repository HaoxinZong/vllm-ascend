#!/usr/bin/env python3
#
# Replay one dense Ascend FIA op from MiniMax dense_attn.fia.* dumps.
#

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch_npu


def load_dump(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "tensor" not in obj:
        raise ValueError(f"{path} is not a MiniMax tensor dump")
    return obj


def find_dump(
    root: Path,
    rank: str,
    phase: str,
    layer: int,
    count: int,
    name: str,
) -> Path | None:
    rank_dir = root / rank
    search_root = rank_dir if rank_dir.is_dir() else root
    matches: list[Path] = []
    for path in search_root.rglob("*.pt"):
        try:
            obj = load_dump(path)
        except Exception:
            continue
        if str(obj.get("phase")) != phase:
            continue
        if int(obj.get("count", -1)) != count:
            continue
        if obj.get("layer_id") is not None and int(obj["layer_id"]) != layer:
            continue
        if str(obj.get("name")) != name:
            continue
        matches.append(path)
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"multiple dumps matched {name}: {matches}")
    return matches[0]


def load_tensor(
    root: Path,
    rank: str,
    phase: str,
    layer: int,
    count: int,
    name: str,
    required: bool = True,
) -> torch.Tensor | None:
    path = find_dump(root, rank, phase, layer, count, name)
    if path is None:
        if required:
            raise FileNotFoundError(f"cannot find dump point {name}")
        return None
    return load_dump(path)["tensor"]


def to_actual_seq_lengths(x: torch.Tensor | None) -> list[int] | None:
    if x is None:
        return None
    return [int(v) for v in x.reshape(-1).tolist()]


def compare_tensors(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict[str, float | bool]:
    if a.shape != b.shape:
        return {
            "shape_match": False,
            "allclose": False,
            "max_abs": math.inf,
            "mean_abs": math.inf,
            "bad_pct": 100.0,
            "cos": float("nan"),
        }

    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    diff = (af - bf).abs()
    denom = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-12)
    bad = diff > (atol + rtol * bf.abs())
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item() if af.numel() else 1.0
    return {
        "shape_match": True,
        "allclose": bool(not bad.any().item()),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "bad_pct": float(bad.float().mean().item() * 100.0) if bad.numel() else 0.0,
        "cos": cos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay dense npu_fused_infer_attention_score from dumps.")
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--rank", default="pp0_tp0")
    parser.add_argument("--phase", default="prefill")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--input-layout", default="TND")
    parser.add_argument("--sparse-mode", type=int, default=3)
    parser.add_argument("--pre-tokens", type=int, default=None)
    parser.add_argument("--next-tokens", type=int, default=None)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    query = load_tensor(args.dump_dir, args.rank, args.phase, args.layer, args.count, "dense_attn.fia.query")
    key = load_tensor(args.dump_dir, args.rank, args.phase, args.layer, args.count, "dense_attn.fia.key")
    value = load_tensor(args.dump_dir, args.rank, args.phase, args.layer, args.count, "dense_attn.fia.value")
    block_table = load_tensor(args.dump_dir, args.rank, args.phase, args.layer, args.count, "dense_attn.fia.block_table")
    actual_q = load_tensor(
        args.dump_dir, args.rank, args.phase, args.layer, args.count, "dense_attn.fia.actual_seq_lengths_q"
    )
    actual_kv = load_tensor(
        args.dump_dir, args.rank, args.phase, args.layer, args.count, "dense_attn.fia.actual_seq_lengths_kv"
    )
    attn_mask = load_tensor(
        args.dump_dir, args.rank, args.phase, args.layer, args.count, "dense_attn.fia.attn_mask", required=False
    )
    dumped_out = load_tensor(
        args.dump_dir, args.rank, args.phase, args.layer, args.count, "dense_attn.fia.out", required=False
    )

    assert query is not None
    assert key is not None
    assert value is not None
    assert block_table is not None

    scale = args.scale
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    query_npu = query.npu()
    key_npu = key.npu()
    value_npu = value.npu()
    block_table_npu = block_table.to(torch.int32).npu()
    attn_mask_npu = attn_mask.npu() if attn_mask is not None else None

    kwargs: dict[str, Any] = {
        "query": query_npu,
        "key": key_npu,
        "value": value_npu,
        "atten_mask": attn_mask_npu,
        "block_table": block_table_npu,
        "input_layout": args.input_layout,
        "block_size": args.block_size,
        "actual_seq_lengths": to_actual_seq_lengths(actual_q),
        "actual_seq_lengths_kv": to_actual_seq_lengths(actual_kv),
        "num_key_value_heads": int(key.shape[-2]),
        "num_heads": int(query.shape[-2]),
        "scale": float(scale),
        "sparse_mode": args.sparse_mode,
    }
    if args.pre_tokens is not None:
        kwargs["pre_tokens"] = args.pre_tokens
    if args.next_tokens is not None:
        kwargs["next_tokens"] = args.next_tokens

    print("Replay FIA params:")
    for key_name in (
        "input_layout",
        "block_size",
        "num_heads",
        "num_key_value_heads",
        "scale",
        "sparse_mode",
        "pre_tokens",
        "next_tokens",
    ):
        if key_name in kwargs:
            print(f"  {key_name}: {kwargs[key_name]}")
    print(f"  query: {tuple(query.shape)} {query.dtype}")
    print(f"  key: {tuple(key.shape)} {key.dtype}")
    print(f"  value: {tuple(value.shape)} {value.dtype}")
    print(f"  block_table: {tuple(block_table.shape)} {block_table.dtype}")
    print(f"  attn_mask: {None if attn_mask is None else (tuple(attn_mask.shape), attn_mask.dtype)}")
    print(f"  actual_seq_lengths: {kwargs['actual_seq_lengths']}")
    print(f"  actual_seq_lengths_kv: {kwargs['actual_seq_lengths_kv']}")

    out, _ = torch_npu.npu_fused_infer_attention_score(**kwargs)
    out_cpu = out.view(query.shape[0], query.shape[-2], query.shape[-1]).detach().cpu()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tensor": out_cpu, "params": kwargs}, args.save)
        print(f"saved replay output to {args.save}")

    if dumped_out is not None:
        stats = compare_tensors(out_cpu, dumped_out, args.atol, args.rtol)
        print("Compare replay output vs dumped dense_attn.fia.out:")
        print(f"  shape_match: {stats['shape_match']}")
        print(f"  allclose: {stats['allclose']}  atol={args.atol} rtol={args.rtol}")
        print(f"  max_abs: {stats['max_abs']:.8e}")
        print(f"  mean_abs: {stats['mean_abs']:.8e}")
        print(f"  bad_pct: {stats['bad_pct']:.4f}")
        print(f"  cos: {stats['cos']:.8e}")


if __name__ == "__main__":
    main()
