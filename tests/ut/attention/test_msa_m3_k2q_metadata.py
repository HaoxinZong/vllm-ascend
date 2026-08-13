# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch

from vllm_ascend.attention.msa_m3 import _build_k2q_block_metadata
from vllm_ascend.attention.msa_m3_npu_new import minimax_m3_sparse_attn


def test_build_k2q_block_metadata_on_cpu() -> None:
    cu_block_lens, total_rows, max_kv = _build_k2q_block_metadata(
        torch.tensor([1, 128, 129, 257], dtype=torch.int32),
        block_size=128,
    )

    assert cu_block_lens.device.type == "cpu"
    assert cu_block_lens.tolist() == [0, 1, 2, 4, 7]
    assert total_rows == 7
    assert max_kv == 3


@patch(
    "torch.ops._C_ascend.npu_sparse_attention_score_prefill",
    create=True,
)
@patch("vllm_ascend.attention.msa_m3_npu_new.npu_k2q_csr")
def test_prefill_passes_precomputed_k2q_stats(
    mock_k2q_csr,
    mock_sparse_attention,
) -> None:
    q = torch.zeros(2, 1, 4)
    key = torch.zeros(1, 128, 1, 4)
    value = torch.zeros_like(key)
    output = torch.empty_like(q)
    mock_k2q_csr.return_value = (
        torch.tensor([[0, 2]], dtype=torch.int32),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([[0, 0]], dtype=torch.int32),
    )
    mock_sparse_attention.return_value = q

    minimax_m3_sparse_attn(
        q,
        (key, value),
        torch.tensor([[[0], [0]]], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        2,
        1,
        0.125,
        output,
        block_size=128,
        cu_block_lens=torch.tensor([0, 1], dtype=torch.int32),
        k2q_total_rows=1,
        k2q_max_kv=1,
    )

    assert mock_k2q_csr.call_args.kwargs["total_rows"] == 1
    assert mock_k2q_csr.call_args.kwargs["max_kv"] == 1
