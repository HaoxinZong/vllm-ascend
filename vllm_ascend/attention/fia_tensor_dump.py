# SPDX-License-Identifier: Apache-2.0
"""Tensor dump entry point for Ascend fused infer attention debugging."""

from __future__ import annotations

import os
from typing import Any

from vllm_ascend.debug_tensor_dump import dump_tensor_point


def dump_fia_inputs(tag: str, *, layer_idx: int | None, **payload: Any) -> None:
    first_n_layers = os.getenv(
        "VLLM_ASCEND_DUMP_FIA_FIRST_N_LAYERS",
        os.getenv("VLLM_ASCEND_DUMP_FIRST_N_LAYERS"),
    )
    dump_tensor_point(
        f"attention_v1_{tag}",
        layer_idx=layer_idx,
        dump_dir_env="VLLM_ASCEND_DUMP_FIA_INPUTS",
        first_n_layers_env="VLLM_ASCEND_DUMP_FIA_FIRST_N_LAYERS",
        first_n_layers=int(first_n_layers) if first_n_layers is not None else None,
        limit_env="VLLM_ASCEND_DUMP_FIA_LIMIT",
        source="ascend_fia",
        **payload,
    )
