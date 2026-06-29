# SPDX-License-Identifier: Apache-2.0
"""Small opt-in tensor dump helpers for MiniMax M3 debugging."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

import torch

_DUMP_COUNTS: defaultdict[tuple[str, str, int | None], int] = defaultdict(int)
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def layer_idx_from_name(name: str | None) -> int | None:
    if not name:
        return None
    match = _LAYER_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _clone_for_dump(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(_clone_for_dump(item) for item in value)
    if isinstance(value, list):
        return [_clone_for_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_for_dump(item) for key, item in value.items()}
    return repr(value)


def _sanitize_tag(tag: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", tag).strip("_")


def dump_tensor_point(
    tag: str,
    *,
    layer_idx: int | None = None,
    layer_name: str | None = None,
    dump_dir: str | None = None,
    dump_dir_env: str = "MINIMAX_M3_DUMP_TENSOR_DIR",
    first_n_layers_env: str = "MINIMAX_M3_DUMP_FIRST_N_LAYERS",
    first_n_layers: int | None = None,
    limit_env: str = "MINIMAX_M3_DUMP_LIMIT_PER_POINT",
    source: str = "ascend",
    **payload: Any,
) -> None:
    dump_dir = dump_dir or os.getenv(dump_dir_env)
    if not dump_dir:
        return

    if layer_idx is None:
        layer_idx = layer_idx_from_name(layer_name)
    if layer_idx is not None:
        if first_n_layers is None:
            first_n_layers = int(os.getenv(first_n_layers_env, "3"))
        if layer_idx >= first_n_layers:
            return

    limit = int(os.getenv(limit_env, "1"))
    key = (os.path.abspath(dump_dir), tag, layer_idx)
    count = _DUMP_COUNTS[key]
    layer_part = "model" if layer_idx is None else f"layer{layer_idx:02d}"
    sanitized_tag = _sanitize_tag(tag)
    if count >= limit:
        first_file = os.path.join(dump_dir, f"0000_{layer_part}_{sanitized_tag}.pt")
        if os.path.exists(first_file):
            return
        count = 0
        _DUMP_COUNTS[key] = 0

    file_name = f"{count:04d}_{layer_part}_{sanitized_tag}.pt"
    os.makedirs(dump_dir, exist_ok=True)
    torch.save(
        {
            "source": source,
            "tag": tag,
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "payload": _clone_for_dump(payload),
        },
        os.path.join(dump_dir, file_name),
    )
    _DUMP_COUNTS[key] += 1
