# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 block-sparse attention wiring for Ascend.

This mirrors the upstream MiniMax M3 attention contract: sparse layers own a
main paged K/V cache plus a side cache for the index-key branch.  The current
Ascend implementation uses PyTorch tensor ops for the M3-specific top-k and
selected-block attention path, so the model exercises the real sparse dataflow
without falling back to dense attention.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    split_decodes_and_prefills,
)

SPARSE_BLOCK_SIZE = 128
IndexerKVDType = str


class MiniMaxM3SparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "float16",
    ]

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE_ASCEND"

    @staticmethod
    def get_impl_cls() -> type["MiniMaxM3SparseImpl"]:
        return MiniMaxM3SparseImpl

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseMetadataBuilder"]:
        return MiniMaxM3SparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [SPARSE_BLOCK_SIZE]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class MiniMaxM3IndexerBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "float16",
    ]

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE_INDEXER_ASCEND"

    @staticmethod
    def get_impl_cls() -> type["MiniMaxM3IndexerImpl"]:
        return MiniMaxM3IndexerImpl

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMetadataBuilder"]:
        return MiniMaxM3IndexerMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [SPARSE_BLOCK_SIZE]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Plane 0 stores index_k. Plane 1 is deliberately unused; keeping the
        # standard FullAttentionSpec layout avoids special-casing cache zeroing.
        return (2, num_blocks, block_size, num_kv_heads, head_size)


@dataclass
class MiniMaxM3SparseMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_seq_len: int
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int


class MiniMaxM3SparseMetadataBuilder(
    AttentionMetadataBuilder[MiniMaxM3SparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.decode_threshold = 1
        if vllm_config.speculative_config is not None:
            self.decode_threshold += (
                vllm_config.speculative_config.num_speculative_tokens
            )
        self.reorder_batch_threshold = self.decode_threshold

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        assert isinstance(common_attn_metadata, AscendCommonAttentionMetadata)
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
            )
        )
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].to("cpu")
        return MiniMaxM3SparseMetadata(
            seq_lens=common_attn_metadata.seq_lens[:num_reqs],
            seq_lens_cpu=seq_lens_cpu,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu[
                : num_reqs + 1
            ],
            block_table=common_attn_metadata.block_table_tensor[:num_reqs],
            slot_mapping=common_attn_metadata.slot_mapping[:num_actual_tokens],
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=num_actual_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
        )


MiniMaxM3IndexerMetadata = MiniMaxM3SparseMetadata


class MiniMaxM3IndexerMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    pass


class MiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        block_size: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType = "bf16",
    ) -> None:
        super().__init__()
        if indexer_kv_dtype not in ("bf16", "auto"):
            raise NotImplementedError(
                f"MiniMax M3 indexer_kv_dtype={indexer_kv_dtype!r} is not "
                "supported on Ascend yet."
            )
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.block_size = block_size
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return MiniMaxM3IndexerBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=torch.bfloat16,
        )

    def forward(self) -> None:
        return None


class MiniMaxM3IndexerImpl(nn.Module):
    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        score_type: str = "max",
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType = "bf16",
    ) -> None:
        super().__init__()
        if score_type != "max":
            raise NotImplementedError(
                f"MiniMax M3 sparse_score_type={score_type!r} is not supported."
            )
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.num_index_heads = num_index_heads
        self.index_head_dim = index_head_dim
        self.index_cache = MiniMaxM3IndexerCache(
            head_dim=index_head_dim,
            block_size=sparse_block_size,
            prefix=f"{prefix}.index_cache",
            cache_config=cache_config,
            indexer_kv_dtype=indexer_kv_dtype,
        )

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, MiniMaxM3SparseMetadata)
        iq = index_query[: index_md.num_actual_tokens].view(
            -1,
            self.num_index_heads,
            self.index_head_dim,
        )
        index_cache = self.index_cache.kv_cache[0, :, :, 0, :]

        decode_topk = None
        prefill_topk = None
        if index_md.num_decode_tokens > 0:
            decode_topk = _select_topk_blocks(
                iq[: index_md.num_decode_tokens],
                index_cache,
                index_md,
                self.topk_blocks,
                self.block_size,
                self.scale,
                self.init_blocks,
                self.local_blocks,
                token_offset=0,
                num_tokens=index_md.num_decode_tokens,
            )
        if index_md.num_prefill_tokens > 0:
            prefill_topk = _select_topk_blocks(
                iq[index_md.num_decode_tokens :],
                index_cache,
                index_md,
                self.topk_blocks,
                self.block_size,
                self.scale,
                self.init_blocks,
                self.local_blocks,
                token_offset=index_md.num_decode_tokens,
                num_tokens=index_md.num_prefill_tokens,
            )
        return decode_topk, prefill_topk


class MiniMaxM3Indexer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.impl = MiniMaxM3IndexerImpl(**kwargs)

    @property
    def index_cache(self) -> MiniMaxM3IndexerCache:
        return self.impl.index_cache

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.impl(index_query)


class MiniMaxM3SparseImpl(AttentionImplBase[MiniMaxM3SparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: tuple[torch.Tensor | None, torch.Tensor | None],
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output.fill_(0)
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)
        q = query[: main_md.num_actual_tokens].view(
            -1,
            self.num_heads,
            self.head_size,
        )
        out = output[: main_md.num_actual_tokens].view(
            -1,
            self.num_heads,
            self.head_size,
        )
        out.zero_()
        decode_topk, prefill_topk = topk_idx
        _sparse_attention_selected_blocks(
            q,
            kv_cache,
            decode_topk,
            prefill_topk,
            main_md,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.scale,
            out,
        )
        return output


def _request_for_token(md: MiniMaxM3SparseMetadata, token_idx: int) -> tuple[int, int]:
    starts = md.query_start_loc_cpu
    req = int(torch.searchsorted(starts, token_idx, right=True).item()) - 1
    req = max(0, min(req, md.seq_lens_cpu.numel() - 1))
    local_q = token_idx - int(starts[req].item())
    return req, local_q


def _allowed_len(md: MiniMaxM3SparseMetadata, token_idx: int) -> tuple[int, int, int]:
    req, local_q = _request_for_token(md, token_idx)
    q_len = int(md.query_start_loc_cpu[req + 1].item() - md.query_start_loc_cpu[req].item())
    seq_len = int(md.seq_lens_cpu[req].item())
    context_len = seq_len - q_len
    return req, local_q, context_len + local_q + 1


def _force_init_local_blocks(
    scores: torch.Tensor,
    allowed_blocks: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    if allowed_blocks == 0:
        return scores
    force = torch.zeros_like(scores, dtype=torch.bool)
    if init_blocks > 0:
        force[: min(init_blocks, allowed_blocks)] = True
    if local_blocks > 0:
        start = max(0, allowed_blocks - local_blocks)
        force[start:allowed_blocks] = True
    if torch.any(force):
        scores = scores.clone()
        scores[force] = scores.max() + 1.0
    return scores


def _select_topk_blocks(
    index_query: torch.Tensor,
    index_cache: torch.Tensor,
    md: MiniMaxM3SparseMetadata,
    topk: int,
    block_size: int,
    scale: float,
    init_blocks: int,
    local_blocks: int,
    *,
    token_offset: int,
    num_tokens: int,
) -> torch.Tensor:
    topk_idx = torch.full(
        (index_query.shape[1], num_tokens, topk),
        -1,
        dtype=torch.int32,
        device=index_query.device,
    )
    for local_token in range(num_tokens):
        token_idx = token_offset + local_token
        req, _, allowed = _allowed_len(md, token_idx)
        allowed_blocks = (allowed + block_size - 1) // block_size
        if allowed_blocks <= 0:
            continue
        logical_blocks = torch.arange(
            allowed_blocks,
            dtype=torch.long,
            device=index_query.device,
        )
        physical_blocks = md.block_table[req, :allowed_blocks].to(
            device=index_query.device,
            dtype=torch.long,
        )
        block_scores = []
        for logical_block, physical_block in zip(logical_blocks, physical_blocks):
            valid = min(block_size, allowed - int(logical_block.item()) * block_size)
            keys = index_cache[physical_block, :valid].to(index_query.dtype)
            scores = torch.matmul(
                index_query[local_token],
                keys.transpose(0, 1),
            ) * scale
            block_scores.append(scores.max(dim=-1).values)
        scores = torch.stack(block_scores, dim=-1)
        k = min(topk, allowed_blocks)
        for head in range(index_query.shape[1]):
            head_scores = _force_init_local_blocks(
                scores[head],
                allowed_blocks,
                init_blocks,
                local_blocks,
            )
            _, selected = torch.topk(head_scores, k=k, dim=-1)
            topk_idx[head, local_token, :k] = selected.to(torch.int32)
    return topk_idx


def _sparse_attention_selected_blocks(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    decode_topk: torch.Tensor | None,
    prefill_topk: torch.Tensor | None,
    md: MiniMaxM3SparseMetadata,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
    output: torch.Tensor,
) -> None:
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]
    heads_per_kv = num_heads // num_kv_heads
    for token_idx in range(md.num_actual_tokens):
        req, _, allowed = _allowed_len(md, token_idx)
        if token_idx < md.num_decode_tokens:
            selected = decode_topk
            local_token = token_idx
        else:
            selected = prefill_topk
            local_token = token_idx - md.num_decode_tokens
        if selected is None:
            continue
        for kv_head in range(num_kv_heads):
            logical_blocks = selected[kv_head, local_token]
            keys = []
            values = []
            for logical_block_tensor in logical_blocks:
                logical_block = int(logical_block_tensor.item())
                if logical_block < 0:
                    continue
                block_start = logical_block * SPARSE_BLOCK_SIZE
                valid = min(SPARSE_BLOCK_SIZE, allowed - block_start)
                if valid <= 0:
                    continue
                physical_block = int(md.block_table[req, logical_block].item())
                keys.append(key_cache[physical_block, :valid, kv_head])
                values.append(value_cache[physical_block, :valid, kv_head])
            if not keys:
                continue
            k = torch.cat(keys, dim=0).to(query.dtype)
            v = torch.cat(values, dim=0).to(query.dtype)
            head_start = kv_head * heads_per_kv
            head_end = head_start + heads_per_kv
            q = query[token_idx, head_start:head_end]
            scores = torch.matmul(q, k.transpose(0, 1)) * scale
            probs = torch.softmax(scores, dim=-1)
            output[token_idx, head_start:head_end] = torch.matmul(probs, v)
