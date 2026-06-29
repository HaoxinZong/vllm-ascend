#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare MiniMax M3 tensor dumps from vllm-ascend and upstream vLLM."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _is_topk_path(path: str) -> bool:
    name = path.lower()
    return "topk" in name or "topk_idx" in name or "topk_indices" in name


def _valid_topk_sets(tensor: torch.Tensor) -> list[tuple[int, ...]]:
    if tensor.numel() == 0:
        return []
    if tensor.ndim == 0:
        return [(int(tensor.item()),)]
    rows = tensor.reshape(-1, tensor.shape[-1]).to(torch.int64)
    result: list[tuple[int, ...]] = []
    for row in rows:
        values = [int(item) for item in row.tolist() if int(item) >= 0]
        result.append(tuple(sorted(set(values))))
    return result


def _compare_topk(path: str, left: torch.Tensor, right: torch.Tensor) -> list[str]:
    if left.shape[:-1] != right.shape[:-1]:
        return [
            f"{path}: topk prefix shape mismatch "
            f"{tuple(left.shape)} vs {tuple(right.shape)}"
        ]
    left_sets = _valid_topk_sets(left)
    right_sets = _valid_topk_sets(right)
    if left_sets == right_sets:
        return []
    for idx, (left_set, right_set) in enumerate(zip(left_sets, right_sets)):
        if left_set != right_set:
            return [f"{path}: topk set mismatch at row {idx}: {left_set} vs {right_set}"]
    return [f"{path}: topk set mismatch"]


def _compare_tensor(
    path: str,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> list[str]:
    if _is_topk_path(path) and left.dtype in (torch.int16, torch.int32, torch.int64):
        return _compare_topk(path, left, right)
    if tuple(left.shape) != tuple(right.shape):
        return [f"{path}: shape mismatch {tuple(left.shape)} vs {tuple(right.shape)}"]
    if left.dtype != right.dtype:
        return [f"{path}: dtype mismatch {left.dtype} vs {right.dtype}"]
    if left.dtype == torch.bool:
        if torch.equal(left, right):
            return []
        diff = int((left != right).sum().item())
        return [f"{path}: bool mismatch, diff_count={diff}"]
    if not torch.is_floating_point(left) and not torch.is_complex(left):
        if torch.equal(left, right):
            return []
        diff = int((left != right).sum().item())
        return [f"{path}: integer mismatch, diff_count={diff}"]

    left_f = left.float()
    right_f = right.float()
    equal = torch.allclose(left_f, right_f, atol=atol, rtol=rtol, equal_nan=True)
    if equal:
        return []
    delta = (left_f - right_f).abs()
    return [
        f"{path}: allclose failed max_abs={delta.max().item():.6g} "
        f"mean_abs={delta.mean().item():.6g} atol={atol} rtol={rtol}"
    ]


def _compare_value(
    path: str,
    left: Any,
    right: Any,
    *,
    atol: float,
    rtol: float,
    strict_payload_keys: bool,
) -> list[str]:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return _compare_tensor(path, left, right, atol=atol, rtol=rtol)
    if left is None or right is None:
        return [] if left is right else [f"{path}: None mismatch"]
    if isinstance(left, dict) and isinstance(right, dict):
        messages: list[str] = []
        left_keys = set(left)
        right_keys = set(right)
        if strict_payload_keys and left_keys != right_keys:
            messages.append(
                f"{path}: key mismatch only_left={sorted(left_keys - right_keys)} "
                f"only_right={sorted(right_keys - left_keys)}"
            )
        for key in sorted(left_keys & right_keys):
            messages.extend(
                _compare_value(
                    f"{path}.{key}",
                    left[key],
                    right[key],
                    atol=atol,
                    rtol=rtol,
                    strict_payload_keys=strict_payload_keys,
                )
            )
        return messages
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        messages = []
        if len(left) != len(right):
            messages.append(f"{path}: length mismatch {len(left)} vs {len(right)}")
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            messages.extend(
                _compare_value(
                    f"{path}[{idx}]",
                    left_item,
                    right_item,
                    atol=atol,
                    rtol=rtol,
                    strict_payload_keys=strict_payload_keys,
                )
            )
        return messages
    return [] if left == right else [f"{path}: value mismatch {left!r} vs {right!r}"]


def _load_dump(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "payload" not in data:
        raise ValueError(f"{path} is not a MiniMax M3 tensor dump")
    return data


def compare_dirs(args: argparse.Namespace) -> int:
    left_dir = Path(args.left)
    right_dir = Path(args.right)
    left_files = {path.name: path for path in left_dir.glob("*.pt")}
    right_files = {path.name: path for path in right_dir.glob("*.pt")}

    failures: list[str] = []
    missing_left = sorted(set(right_files) - set(left_files))
    missing_right = sorted(set(left_files) - set(right_files))
    if missing_left:
        failures.append(f"missing in left: {missing_left}")
    if missing_right:
        failures.append(f"missing in right: {missing_right}")

    compared = 0
    for name in sorted(set(left_files) & set(right_files)):
        left = _load_dump(left_files[name])
        right = _load_dump(right_files[name])
        compared += 1
        if left.get("tag") != right.get("tag"):
            failures.append(f"{name}: tag mismatch {left.get('tag')} vs {right.get('tag')}")
        if left.get("layer_idx") != right.get("layer_idx"):
            failures.append(
                f"{name}: layer mismatch {left.get('layer_idx')} vs {right.get('layer_idx')}"
            )
        failures.extend(
            f"{name}: {message}"
            for message in _compare_value(
                "payload",
                left["payload"],
                right["payload"],
                atol=args.atol,
                rtol=args.rtol,
                strict_payload_keys=args.strict_payload_keys,
            )
        )

    if failures:
        print(f"Compared {compared} files, found {len(failures)} issue(s).")
        for failure in failures[: args.max_failures]:
            print(f"FAIL {failure}")
        if len(failures) > args.max_failures:
            print(f"... {len(failures) - args.max_failures} more failure(s)")
        return 1

    print(f"Compared {compared} files: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", help="Ascend dump directory")
    parser.add_argument("right", help="vLLM dump directory")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--max-failures", type=int, default=50)
    parser.add_argument(
        "--strict-payload-keys",
        action="store_true",
        help="Fail when a payload key exists on only one side.",
    )
    args = parser.parse_args()
    return compare_dirs(args)


if __name__ == "__main__":
    raise SystemExit(main())
