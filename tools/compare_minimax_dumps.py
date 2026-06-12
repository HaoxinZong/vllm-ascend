#!/usr/bin/env python3
#
# Compare MiniMax debug tensor dumps from two runs.
#

from __future__ import annotations

import argparse
import fnmatch
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PHASE_ORDER = {
    "prefill": 0,
    "decode": 1,
    "mixed": 2,
    "unknown": 3,
}

POINT_ORDER = {
    "attn.hidden_in": 0,
    "attn.qkv": 1,
    "attn.cos": 2,
    "attn.sin": 3,
    "attn.q_after_qknorm_rope": 4,
    "attn.k_after_qknorm_rope": 5,
    "attn.v": 6,
    "attn.out_before_o_proj": 7,
    "attn.out": 8,
    "moe.input": 20,
    "moe.router_logits": 21,
    "moe.out_before_extra_reduce": 22,
    "moe.out_after_extra_reduce": 23,
}


@dataclass(frozen=True)
class DumpKey:
    pp_rank: int
    tp_rank: int
    phase: str
    module: str
    name: str
    count: int


@dataclass
class DumpRecord:
    key: DumpKey
    path: Path
    layer_id: int | None
    shape: tuple[int, ...] | None
    dtype: str | None


@dataclass
class CompareResult:
    key: DumpKey
    good: DumpRecord
    bad: DumpRecord
    status: str
    max_abs: float
    mean_abs: float
    max_rel: float
    bad_pct: float
    cosine: float
    good_shape: tuple[int, ...]
    bad_shape: tuple[int, ...]
    positions_match: bool | None


def load_pt(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "tensor" not in obj:
        raise ValueError(f"{path} is not a MiniMax dump dict")
    return obj


def parse_rank_from_dir(path: Path) -> tuple[int, int] | None:
    for part in reversed(path.parts):
        match = re.fullmatch(r"pp(\d+)_tp(\d+)", part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def layer_id_from_module(module: str) -> int | None:
    parts = module.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def parse_phase_from_filename(path: Path) -> str:
    parts = path.stem.split("_", 2)
    if len(parts) >= 2 and parts[1] in PHASE_ORDER:
        return parts[1]
    return "unknown"


def parse_count_from_filename(path: Path) -> int:
    first = path.stem.split("_", 1)[0]
    try:
        return int(first)
    except ValueError:
        return 0


def build_record(path: Path) -> DumpRecord:
    obj = load_pt(path)
    tensor = obj["tensor"]
    rank = parse_rank_from_dir(path)
    pp_rank = int(obj.get("pp_rank", rank[0] if rank else 0))
    tp_rank = int(obj.get("tp_rank", rank[1] if rank else 0))
    phase = str(obj.get("phase") or parse_phase_from_filename(path))
    module = str(obj.get("module") or "unknown")
    name = str(obj.get("name") or path.stem)
    count = int(obj.get("count", parse_count_from_filename(path)))
    layer_id = obj.get("layer_id")
    if layer_id is None:
        layer_id = layer_id_from_module(module)
    else:
        layer_id = int(layer_id)

    shape = obj.get("shape")
    if shape is None and isinstance(tensor, torch.Tensor):
        shape = tuple(tensor.shape)
    elif shape is not None:
        shape = tuple(shape)

    dtype = obj.get("dtype")
    if dtype is None and isinstance(tensor, torch.Tensor):
        dtype = str(tensor.dtype)

    return DumpRecord(
        key=DumpKey(
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            phase=phase,
            module=module,
            name=name,
            count=count,
        ),
        path=path,
        layer_id=layer_id,
        shape=shape,
        dtype=str(dtype) if dtype is not None else None,
    )


def parse_int_ranges(value: str | None) -> set[int] | None:
    if not value:
        return None
    items: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                start, end = end, start
            items.update(range(start, end + 1))
        else:
            items.add(int(part))
    return items


def parse_rank(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"pp(\d+)_tp(\d+)", value)
    if match:
        return int(match.group(1)), int(match.group(2))
    if "," in value:
        pp_str, tp_str = value.split(",", 1)
        return int(pp_str), int(tp_str)
    raise argparse.ArgumentTypeError(
        f"rank must look like pp0_tp0 or 0,0, got {value!r}"
    )


def matches_filters(
    record: DumpRecord,
    ranks: set[tuple[int, int]] | None,
    phases: set[str] | None,
    layers: set[int] | None,
    counts: set[int] | None,
    name_globs: list[str],
) -> bool:
    key = record.key
    if ranks is not None and (key.pp_rank, key.tp_rank) not in ranks:
        return False
    if phases is not None and key.phase not in phases:
        return False
    if layers is not None and record.layer_id not in layers:
        return False
    if counts is not None and key.count not in counts:
        return False
    if name_globs:
        full_name = f"{key.module}.{key.name}"
        if not any(
            fnmatch.fnmatch(key.name, pattern)
            or fnmatch.fnmatch(full_name, pattern)
            for pattern in name_globs
        ):
            return False
    return True


def build_index(
    root: Path,
    ranks: set[tuple[int, int]] | None,
    phases: set[str] | None,
    layers: set[int] | None,
    counts: set[int] | None,
    name_globs: list[str],
) -> tuple[dict[DumpKey, DumpRecord], list[str]]:
    index: dict[DumpKey, DumpRecord] = {}
    warnings: list[str] = []
    for path in sorted(root.rglob("*.pt")):
        if not path.is_file():
            continue
        try:
            record = build_record(path)
        except Exception as exc:
            warnings.append(f"skip {path}: {exc}")
            continue
        if not matches_filters(record, ranks, phases, layers, counts, name_globs):
            continue
        if record.key in index:
            warnings.append(
                "duplicate key, keeping first: "
                f"{record.key} old={index[record.key].path} new={path}"
            )
            continue
        index[record.key] = record
    return index, warnings


def sort_key(key: DumpKey, record: DumpRecord) -> tuple[Any, ...]:
    layer_id = record.layer_id if record.layer_id is not None else 10**9
    return (
        key.pp_rank,
        key.tp_rank,
        PHASE_ORDER.get(key.phase, PHASE_ORDER["unknown"]),
        key.count,
        layer_id,
        POINT_ORDER.get(key.name, 10_000),
        key.module,
        key.name,
    )


def to_compare_float(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.is_complex():
        return tensor.abs().to(torch.float32)
    if tensor.dtype == torch.bool:
        return tensor.to(torch.float32)
    return tensor.to(torch.float32)


def same_positions(good_obj: dict[str, Any], bad_obj: dict[str, Any]) -> bool | None:
    good_pos = good_obj.get("positions")
    bad_pos = bad_obj.get("positions")
    if good_pos is None and bad_pos is None:
        return None
    if good_pos is None or bad_pos is None:
        return False
    if not isinstance(good_pos, torch.Tensor) or not isinstance(bad_pos, torch.Tensor):
        return good_pos == bad_pos
    return good_pos.shape == bad_pos.shape and torch.equal(good_pos.cpu(), bad_pos.cpu())


def compare_pair(
    key: DumpKey,
    good: DumpRecord,
    bad: DumpRecord,
    atol: float,
    rtol: float,
) -> CompareResult:
    good_obj = load_pt(good.path)
    bad_obj = load_pt(bad.path)
    good_tensor = good_obj["tensor"]
    bad_tensor = bad_obj["tensor"]

    good_shape = tuple(good_tensor.shape)
    bad_shape = tuple(bad_tensor.shape)
    positions_match = same_positions(good_obj, bad_obj)

    if good_shape != bad_shape:
        return CompareResult(
            key=key,
            good=good,
            bad=bad,
            status="SHAPE",
            max_abs=math.inf,
            mean_abs=math.inf,
            max_rel=math.inf,
            bad_pct=100.0,
            cosine=math.nan,
            good_shape=good_shape,
            bad_shape=bad_shape,
            positions_match=positions_match,
        )

    good_f = to_compare_float(good_tensor)
    bad_f = to_compare_float(bad_tensor)
    if good_f.numel() == 0:
        status = "OK" if positions_match is not False else "POS"
        return CompareResult(
            key=key,
            good=good,
            bad=bad,
            status=status,
            max_abs=0.0,
            mean_abs=0.0,
            max_rel=0.0,
            bad_pct=0.0,
            cosine=math.nan,
            good_shape=good_shape,
            bad_shape=bad_shape,
            positions_match=positions_match,
        )

    diff = (good_f - bad_f).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = torch.maximum(good_f.abs(), bad_f.abs()).clamp_min(1e-12)
    max_rel = float((diff / denom).max().item())
    close_mask = torch.isclose(good_f, bad_f, rtol=rtol, atol=atol, equal_nan=True)
    bad_pct = 100.0 * (1.0 - float(close_mask.to(torch.float32).mean().item()))

    good_vec = good_f.flatten()
    bad_vec = bad_f.flatten()
    norm = torch.linalg.vector_norm(good_vec) * torch.linalg.vector_norm(bad_vec)
    cosine = math.nan
    if float(norm.item()) > 0.0:
        cosine = float(torch.dot(good_vec, bad_vec).div(norm).item())

    status = "OK" if bool(close_mask.all().item()) else "BAD"
    if positions_match is False:
        status = "POS" if status == "OK" else f"{status}+POS"

    return CompareResult(
        key=key,
        good=good,
        bad=bad,
        status=status,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_rel=max_rel,
        bad_pct=bad_pct,
        cosine=cosine,
        good_shape=good_shape,
        bad_shape=bad_shape,
        positions_match=positions_match,
    )


def unravel_index(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    out = []
    for size in reversed(shape):
        out.append(flat_idx % size)
        flat_idx //= size
    return tuple(reversed(out))


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.4e}"


def print_rows(results: list[CompareResult], max_rows: int, only_bad: bool) -> None:
    visible = [
        result
        for result in results
        if not only_bad or result.status not in ("OK",)
    ][:max_rows]
    if not visible:
        return

    headers = [
        "status",
        "rank",
        "phase",
        "cnt",
        "layer",
        "point",
        "max_abs",
        "mean_abs",
        "max_rel",
        "bad%",
        "cos",
        "shape",
    ]
    rows: list[list[str]] = []
    for result in visible:
        key = result.key
        layer = result.good.layer_id
        rows.append(
            [
                result.status,
                f"pp{key.pp_rank}_tp{key.tp_rank}",
                key.phase,
                str(key.count),
                "-" if layer is None else str(layer),
                key.name,
                format_float(result.max_abs),
                format_float(result.mean_abs),
                format_float(result.max_rel),
                f"{result.bad_pct:.2f}",
                format_float(result.cosine),
                "x".join(str(x) for x in result.good_shape),
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    if len(results) > len(visible):
        print(f"... omitted {len(results) - len(visible)} rows; use --max-rows to show more")


def print_top_diffs(result: CompareResult, topk: int) -> None:
    if topk <= 0 or result.status == "SHAPE":
        return
    good_obj = load_pt(result.good.path)
    bad_obj = load_pt(result.bad.path)
    good = to_compare_float(good_obj["tensor"])
    bad = to_compare_float(bad_obj["tensor"])
    diff = (good - bad).abs().flatten()
    if diff.numel() == 0:
        return
    k = min(topk, diff.numel())
    values, indices = torch.topk(diff, k=k)

    print()
    print("Top differing elements for first mismatch:")
    print(f"  good: {result.good.path}")
    print(f"  bad : {result.bad.path}")
    print("  idx  good  bad  abs_diff  rel_diff")
    shape = tuple(good.shape)
    good_flat = good.flatten()
    bad_flat = bad.flatten()
    for value, flat_idx_tensor in zip(values.tolist(), indices.tolist()):
        flat_idx = int(flat_idx_tensor)
        idx = unravel_index(flat_idx, shape)
        good_value = float(good_flat[flat_idx].item())
        bad_value = float(bad_flat[flat_idx].item())
        denom = max(abs(good_value), abs(bad_value), 1e-12)
        rel = value / denom
        print(
            "  "
            f"{idx}  "
            f"{good_value:.8e}  "
            f"{bad_value:.8e}  "
            f"{value:.8e}  "
            f"{rel:.8e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MiniMax tensor dumps and locate the first divergence."
    )
    parser.add_argument("good_dir", type=Path, help="Dump directory from the good run")
    parser.add_argument("bad_dir", type=Path, help="Dump directory from the bad run")
    parser.add_argument(
        "--rank",
        action="append",
        type=parse_rank,
        help="Rank to compare, e.g. pp0_tp0 or 0,0. Can be repeated.",
    )
    parser.add_argument(
        "--phase",
        help="Comma-separated phases to compare: prefill,decode,mixed,unknown",
    )
    parser.add_argument("--layers", help="Layer filter, e.g. 0-7 or 0,1,8-11")
    parser.add_argument("--counts", help="Dump count filter, e.g. 0 or 0-2")
    parser.add_argument(
        "--names",
        help="Comma-separated glob filters for dump points, e.g. attn.* or *router*",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-rows", type=int, default=80)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--only-bad",
        action="store_true",
        help="Only print mismatched rows in the table.",
    )
    parser.add_argument(
        "--stop-on-first",
        action="store_true",
        help="Stop comparison after the first mismatch.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Exit non-zero if either directory is missing matched dump keys.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ranks = set(args.rank) if args.rank else None
    phases = {item.strip() for item in args.phase.split(",")} if args.phase else None
    layers = parse_int_ranges(args.layers)
    counts = parse_int_ranges(args.counts)
    name_globs = [item.strip() for item in args.names.split(",")] if args.names else []
    name_globs = [item for item in name_globs if item]

    good_index, good_warnings = build_index(
        args.good_dir, ranks, phases, layers, counts, name_globs
    )
    bad_index, bad_warnings = build_index(
        args.bad_dir, ranks, phases, layers, counts, name_globs
    )

    for warning in good_warnings + bad_warnings:
        print(f"warning: {warning}")

    good_keys = set(good_index)
    bad_keys = set(bad_index)
    common_keys = good_keys & bad_keys
    missing_in_good = sorted(
        bad_keys - good_keys,
        key=lambda key: sort_key(key, bad_index[key]),
    )
    missing_in_bad = sorted(
        good_keys - bad_keys,
        key=lambda key: sort_key(key, good_index[key]),
    )
    sorted_common = sorted(
        common_keys,
        key=lambda key: sort_key(key, good_index[key]),
    )

    print(
        f"Indexed good={len(good_index)} bad={len(bad_index)} "
        f"common={len(common_keys)} "
        f"missing_in_good={len(missing_in_good)} missing_in_bad={len(missing_in_bad)}"
    )
    print(f"Tolerance: atol={args.atol:g} rtol={args.rtol:g}")

    results: list[CompareResult] = []
    first_bad: CompareResult | None = None
    for key in sorted_common:
        result = compare_pair(
            key,
            good_index[key],
            bad_index[key],
            atol=args.atol,
            rtol=args.rtol,
        )
        results.append(result)
        if result.status != "OK" and first_bad is None:
            first_bad = result
            if args.stop_on_first:
                break

    print_rows(results, max_rows=args.max_rows, only_bad=args.only_bad)

    ok_count = sum(1 for result in results if result.status == "OK")
    bad_count = len(results) - ok_count
    print()
    print(f"Compared {len(results)} common dumps: OK={ok_count} mismatch={bad_count}")

    if first_bad is None:
        print("First mismatch: none")
    else:
        key = first_bad.key
        layer = first_bad.good.layer_id
        print("First mismatch:")
        print(
            f"  rank=pp{key.pp_rank}_tp{key.tp_rank} "
            f"phase={key.phase} count={key.count} "
            f"layer={layer} point={key.name}"
        )
        print(f"  module={key.module}")
        print(
            "  "
            f"status={first_bad.status} "
            f"max_abs={format_float(first_bad.max_abs)} "
            f"mean_abs={format_float(first_bad.mean_abs)} "
            f"max_rel={format_float(first_bad.max_rel)} "
            f"bad_pct={first_bad.bad_pct:.2f} "
            f"cos={format_float(first_bad.cosine)}"
        )
        print(f"  good={first_bad.good.path}")
        print(f"  bad ={first_bad.bad.path}")
        if first_bad.positions_match is False:
            print("  note=positions differ; confirm both runs used the same prompt/order")
        print_top_diffs(first_bad, args.topk)

    if missing_in_good:
        print()
        print("Keys missing in good:")
        for key in missing_in_good[:20]:
            print(f"  {key}")
        if len(missing_in_good) > 20:
            print(f"  ... {len(missing_in_good) - 20} more")

    if missing_in_bad:
        print()
        print("Keys missing in bad:")
        for key in missing_in_bad[:20]:
            print(f"  {key}")
        if len(missing_in_bad) > 20:
            print(f"  ... {len(missing_in_bad) - 20} more")

    if args.strict_missing and (missing_in_good or missing_in_bad):
        return 2
    return 1 if first_bad is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
