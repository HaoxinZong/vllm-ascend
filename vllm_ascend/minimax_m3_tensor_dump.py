# SPDX-License-Identifier: Apache-2.0
"""Tensor dump entry point for Ascend MiniMax M3 debugging."""

from __future__ import annotations

from typing import Any

from vllm_ascend.debug_tensor_dump import dump_tensor_point


def dump_minimax_m3_tensor_point(
    tag: str,
    *,
    layer_idx: int | None = None,
    layer_name: str | None = None,
    **payload: Any,
) -> None:
    dump_tensor_point(
        tag,
        layer_idx=layer_idx,
        layer_name=layer_name,
        dump_dir_env="MINIMAX_M3_DUMP_TENSOR_DIR",
        first_n_layers_env="MINIMAX_M3_DUMP_FIRST_N_LAYERS",
        limit_env="MINIMAX_M3_DUMP_LIMIT_PER_POINT",
        source="ascend_minimax_m3",
        **payload,
    )
