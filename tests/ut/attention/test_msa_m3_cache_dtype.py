# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_ascend.attention.msa_m3 import (
    AscendMiniMaxM3SparseImpl,
    _prepare_main_kv_cache_update,
    _resolve_main_kv_cache_dtype,
    minimax_m3_sparse_attn_ascendc_prefill,
)


def test_all_topologies_use_bfloat16_main_kv_cache() -> None:
    for pp_size in (1, 8):
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)
        )
        cache_config = SimpleNamespace(cache_dtype="fp8")

        assert _resolve_main_kv_cache_dtype(vllm_config, cache_config) == "bfloat16"


def test_main_kv_cache_update_uses_actual_cache_dtype() -> None:
    tensor = torch.tensor([-500.0, 2.0, 500.0], dtype=torch.float32)
    bf16_cache = torch.empty(3, dtype=torch.bfloat16)

    update = _prepare_main_kv_cache_update(tensor, bf16_cache)

    assert update.dtype == torch.bfloat16


def test_all_topologies_use_sparse_attention_score_prefill() -> None:
    impl = AscendMiniMaxM3SparseImpl(
        num_heads=8,
        head_size=128,
        scale=0.125,
        num_kv_heads=1,
        kv_cache_dtype="bfloat16",
        topk_blocks=2048,
        sparse_block_size=128,
    )

    assert (
        impl.minimax_m3_sparse_attn_ascendc
        is minimax_m3_sparse_attn_ascendc_prefill
    )
