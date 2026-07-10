# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 Triton index score and top-k kernels for Ascend.

Public API
----------
``minimax_m3_index_score`` computes paged block scores for prefill.
``minimax_m3_index_topk`` selects sparse logical blocks for prefill.
``minimax_m3_index_decode`` computes decode block scores and selects top-k.

Data flow
---------
Prefill: score -> chunk-local top-k -> pairwise merge -> finalize.
Decode: chunk-local fused score/top-k -> pairwise merge -> finalize.
For a single decode chunk, the fused kernel writes final indices directly.

Internal indices are 1-based so zero represents padding. Public APIs return
0-based logical block ids and ``-1`` padding.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

__all__ = [
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_index_decode",
]

SPARSE_BLOCK_SIZE = 128
TOPK_SELECTION_TILE = 128
TOPK_COMPUTE_MIN_TILE = 16
TOPK_NUM_WARPS = 4
TOPK_NUM_STAGES = 2


@triton.jit
def _select_topk_pairs(
    scores,
    indices,
    valid_mask,
    topk_size: tl.constexpr,
    block_size: tl.constexpr,
):
    off_k = tl.arange(0, block_size)
    off_t = tl.arange(0, topk_size)

    valid_i32 = valid_mask.to(tl.int32)
    work_valid_i32 = valid_i32
    work_scores = tl.where(work_valid_i32 != 0, scores, float("-inf"))
    selected_scores = tl.full((topk_size,), -1e30, dtype=tl.float32)
    selected_indices = tl.full((topk_size,), 0, dtype=tl.int32)

    for rank in tl.static_range(0, topk_size):
        best_score = tl.max(work_scores, axis=0)
        best_offset = tl.argmax(work_scores, axis=0).to(tl.int32)

        selected_i32 = (
            (off_k == best_offset).to(tl.int32) * work_valid_i32
        )
        best_index = tl.sum(
            tl.where(selected_i32 != 0, indices, 0),
            axis=0,
        ).to(tl.int32)
        has_best_i32 = (best_index > 0).to(tl.int32)

        selected_scores = tl.where(
            off_t == rank,
            tl.where(has_best_i32 != 0, best_score, -1e30),
            selected_scores,
        )
        selected_indices = tl.where(
            off_t == rank,
            tl.where(has_best_i32 != 0, best_index, 0),
            selected_indices,
        )

        selected_lane_i32 = (off_k == best_offset).to(tl.int32)
        work_valid_i32 = work_valid_i32 * (1 - selected_lane_i32)
        work_scores = tl.where(
            selected_lane_i32 != 0,
            float("-inf"),
            work_scores,
        )

    return selected_scores, selected_indices


@triton.jit
def _merge_topk_pairs(
    left_scores,
    left_indices,
    right_scores,
    right_indices,
    topk_size: tl.constexpr,
):
    """Exact stable merge of two descending top-k runs.

    `left_*` and `right_*` are already sorted in descending score order, with
    valid 1-based indices packed before zero padding.  The function therefore
    performs a standard two-pointer merge instead of re-running top-k on each
    side.  Dynamic register-vector indexing is expressed as a one-hot gather.
    """
    off_t = tl.arange(0, topk_size)

    # Keep cursors as Triton scalars throughout the static merge loop.
    zero_i32 = tl.sum(off_t.to(tl.int32) * 0, axis=0)
    left_pos = zero_i32
    right_pos = zero_i32

    out_scores = tl.full((topk_size,), -1e30, dtype=tl.float32)
    out_indices = tl.full((topk_size,), 0, dtype=tl.int32)

    for rank in tl.static_range(0, topk_size):
        # One-hot gathers emulate runtime indexing into register vectors.
        left_head_i32 = (off_t == left_pos).to(tl.int32)
        right_head_i32 = (off_t == right_pos).to(tl.int32)

        left_score = tl.sum(
            tl.where(left_head_i32 != 0, left_scores, 0.0),
            axis=0,
        )
        right_score = tl.sum(
            tl.where(right_head_i32 != 0, right_scores, 0.0),
            axis=0,
        )
        left_index = tl.sum(
            tl.where(left_head_i32 != 0, left_indices, 0),
            axis=0,
        ).to(tl.int32)
        right_index = tl.sum(
            tl.where(right_head_i32 != 0, right_indices, 0),
            axis=0,
        ).to(tl.int32)

        left_has_i32 = (left_index > 0).to(tl.int32)
        right_has_i32 = (right_index > 0).to(tl.int32)
        left_ge_right_i32 = (left_score >= right_score).to(tl.int32)

        # Equal scores preserve the left run order.
        take_left_i32 = left_has_i32 * (
            (1 - right_has_i32) + right_has_i32 * left_ge_right_i32
        )
        take_right_i32 = right_has_i32 * (1 - take_left_i32)
        has_best_i32 = take_left_i32 + take_right_i32

        best_score = tl.where(
            take_left_i32 != 0,
            left_score,
            right_score,
        )
        best_index = tl.where(
            take_left_i32 != 0,
            left_index,
            right_index,
        )

        out_scores = tl.where(
            off_t == rank,
            tl.where(has_best_i32 != 0, best_score, -1e30),
            out_scores,
        )
        out_indices = tl.where(
            off_t == rank,
            tl.where(has_best_i32 != 0, best_index, 0),
            out_indices,
        )

        # Padded runs leave both cursors unchanged.
        left_pos = left_pos + take_left_i32
        right_pos = right_pos + take_right_i32

    return out_scores, out_indices


# ---------------------------------------------------------------------------
# Index block-score kernel (paged). score[h, token, block] = max over the
# 128-token block of (idx_q . index_k), causal-masked. BLOCK_SIZE_K == 128 so
# each K-tile is exactly one page (BLOCKS_PER_K_BLOCK == 1).
# ---------------------------------------------------------------------------
# Metadata pointers may be sliced from a mixed batch. Scalar loads avoid
# alignment specialization and recompiles for equivalent shapes.

@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads,
    head_dim: tl.constexpr,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    # Block table row for this request.
    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Causal window: only blocks up to the last query token's position.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        # index-K for this page: [BLOCK_SIZE_D, BLOCK_SIZE_K] (transposed)
        # we don't need masked load for K, because KV cache ensures
        # allocation is multiple of BLOCK_SIZE_K.
        # for tokens beyond seqlen, they will be masked in qk later.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        qk = tl.dot(q, k) * sm_scale_log2e
        # apply causal mask as needed
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(off_q[:, None] >= pos[None, :], qk, float("-inf"))
        # one sparse block per K-tile -> max over the 128 positions
        score = tl.max(qk, axis=1)  # [BLOCK_SIZE_Q]
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            * stride_s_n
            + blk * stride_s_k
        )
        q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
        tl.store(s_ptrs, score, mask=q_store_mask)


# ---------------------------------------------------------------------------
# Prefill and decode top-k kernels.
# ---------------------------------------------------------------------------
@triton.jit(
    do_not_specialize=["query_start", "batch_start"],
    do_not_specialize_on_alignment=["prefix_lens"],
)
def _prefill_topk_partial_kernel(
    s_ptr,
    scores_partial_ptr,
    indices_partial_ptr,
    cu_seqlens,
    prefix_lens,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    chunk_blocks,
    num_chunks,
    query_start,
    batch_start,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ps_c,
    stride_ps_h,
    stride_ps_n,
    stride_ps_t,
    stride_pi_c,
    stride_pi_h,
    stride_pi_n,
    stride_pi_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_BLOCK: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    """Select sorted local top-k runs for a tiled prefill-query range.

    ``query_start`` and ``batch_start`` make a large logical launch a sequence
    of bounded physical launches.  Every program visits ``QUERY_TILE`` adjacent
    query rows for one request/head/chunk tuple.
    """
    pid_q_tile = tl.program_id(0)
    pid_b = tl.program_id(1) + batch_start
    pid_hc = tl.program_id(2)
    pid_h = pid_hc // num_chunks
    pid_chunk = pid_hc - pid_h * num_chunks

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    prefix_len = tl.load(prefix_lens + pid_b)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    chunk_start = pid_chunk * chunk_blocks
    block_ids = chunk_start + off_k
    init_i32 = (block_ids < init_blocks).to(tl.int32)
    query_base = query_start + pid_q_tile * QUERY_TILE

    for q_offset in tl.static_range(0, QUERY_TILE):
        pid_q = query_base + q_offset
        valid_q_i32 = (pid_q < q_len).to(tl.int32)
        query_idx = seq_start + pid_q
        valid_blocks = (
            prefix_len + pid_q + BLOCK_SIZE_BLOCK
        ) // BLOCK_SIZE_BLOCK
        valid_i32 = (
            (block_ids < valid_blocks)
            & (off_k < chunk_blocks)
            & (valid_q_i32 != 0)
        ).to(tl.int32)

        score = tl.load(
            s_ptr
            + pid_h * stride_s_h
            + query_idx * stride_s_n
            + block_ids * stride_s_k,
            mask=valid_i32 != 0,
            other=-1e30,
        ).to(tl.float32)
        score = tl.where(score != score, -1e30, score)

        local_i32 = (
            block_ids >= tl.maximum(0, valid_blocks - local_blocks)
        ).to(tl.int32)
        if MASK_INIT:
            score = tl.where(
                (valid_i32 * init_i32) != 0,
                score - 1e29,
                score,
            )
        else:
            score = tl.where(
                (valid_i32 * init_i32) != 0,
                1e30,
                score,
            )
        if MASK_LOCAL:
            score = tl.where(
                (valid_i32 * local_i32) != 0,
                score - 1e28,
                score,
            )
        else:
            score = tl.where(
                (valid_i32 * local_i32) != 0,
                1e29,
                score,
            )

        indices = tl.where(valid_i32 != 0, block_ids + 1, 0).to(tl.int32)
        selected_scores, selected_indices = _select_topk_pairs(
            score,
            indices,
            valid_i32 != 0,
            BLOCK_SIZE_T,
            BLOCK_SIZE_K,
        )

        tl.store(
            scores_partial_ptr
            + pid_chunk * stride_ps_c
            + pid_h * stride_ps_h
            + query_idx * stride_ps_n
            + off_t * stride_ps_t,
            selected_scores,
            mask=valid_q_i32 != 0,
        )
        tl.store(
            indices_partial_ptr
            + pid_chunk * stride_pi_c
            + pid_h * stride_pi_h
            + query_idx * stride_pi_n
            + off_t * stride_pi_t,
            selected_indices,
            mask=valid_q_i32 != 0,
        )

@triton.jit(
    do_not_specialize=["num_input_chunks", "total_q", "query_start"]
)
def _topk_pair_merge_kernel(
    in_scores_ptr,
    in_indices_ptr,
    out_scores_ptr,
    out_indices_ptr,
    num_input_chunks,
    total_q,
    query_start,
    stride_is_c,
    stride_is_h,
    stride_is_n,
    stride_is_t,
    stride_ii_c,
    stride_ii_h,
    stride_ii_n,
    stride_ii_t,
    stride_os_c,
    stride_os_h,
    stride_os_n,
    stride_os_t,
    stride_oi_c,
    stride_oi_h,
    stride_oi_n,
    stride_oi_t,
    BLOCK_SIZE_T: tl.constexpr,
    QUERY_TILE: tl.constexpr,
):
    """Merge a bounded range of query rows from adjacent top-k chunks."""
    pid_n_tile = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_out_chunk = tl.program_id(2)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    left_chunk = 2 * pid_out_chunk
    right_chunk = left_chunk + 1
    right_exists_i32 = (right_chunk < num_input_chunks).to(tl.int32)
    query_base = query_start + pid_n_tile * QUERY_TILE

    for q_offset in tl.static_range(0, QUERY_TILE):
        pid_n = query_base + q_offset
        valid_n_i32 = (pid_n < total_q).to(tl.int32)
        load_mask = valid_n_i32 != 0

        left_scores = tl.load(
            in_scores_ptr
            + left_chunk * stride_is_c
            + pid_h * stride_is_h
            + pid_n * stride_is_n
            + off_t * stride_is_t,
            mask=load_mask,
            other=-1e30,
        ).to(tl.float32)
        left_indices = tl.load(
            in_indices_ptr
            + left_chunk * stride_ii_c
            + pid_h * stride_ii_h
            + pid_n * stride_ii_n
            + off_t * stride_ii_t,
            mask=load_mask,
            other=0,
        ).to(tl.int32)
        right_mask = load_mask & (right_exists_i32 != 0)
        right_scores = tl.load(
            in_scores_ptr
            + right_chunk * stride_is_c
            + pid_h * stride_is_h
            + pid_n * stride_is_n
            + off_t * stride_is_t,
            mask=right_mask,
            other=-1e30,
        ).to(tl.float32)
        right_indices = tl.load(
            in_indices_ptr
            + right_chunk * stride_ii_c
            + pid_h * stride_ii_h
            + pid_n * stride_ii_n
            + off_t * stride_ii_t,
            mask=right_mask,
            other=0,
        ).to(tl.int32)

        merged_scores, merged_indices = _merge_topk_pairs(
            left_scores,
            left_indices,
            right_scores,
            right_indices,
            BLOCK_SIZE_T,
        )

        tl.store(
            out_scores_ptr
            + pid_out_chunk * stride_os_c
            + pid_h * stride_os_h
            + pid_n * stride_os_n
            + off_t * stride_os_t,
            merged_scores,
            mask=load_mask,
        )
        tl.store(
            out_indices_ptr
            + pid_out_chunk * stride_oi_c
            + pid_h * stride_oi_h
            + pid_n * stride_oi_n
            + off_t * stride_oi_t,
            merged_indices,
            mask=load_mask,
        )

@triton.jit(do_not_specialize=["total_q", "query_start"])
def _topk_finalize_kernel(
    indices_partial_ptr,
    indices_final_ptr,
    topk,
    total_q,
    query_start,
    stride_pi_c,
    stride_pi_h,
    stride_pi_n,
    stride_pi_t,
    stride_f_h,
    stride_f_n,
    stride_f_t,
    BLOCK_SIZE_T: tl.constexpr,
    QUERY_TILE: tl.constexpr,
):
    """Convert one bounded range of 1-based partial indices to API output."""
    pid_n_tile = tl.program_id(0)
    pid_h = tl.program_id(1)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    query_base = query_start + pid_n_tile * QUERY_TILE

    for q_offset in tl.static_range(0, QUERY_TILE):
        pid_n = query_base + q_offset
        valid_n_i32 = (pid_n < total_q).to(tl.int32)
        indices = tl.load(
            indices_partial_ptr
            + pid_h * stride_pi_h
            + pid_n * stride_pi_n
            + off_t * stride_pi_t,
            mask=valid_n_i32 != 0,
            other=0,
        ).to(tl.int32)
        output = tl.where(
            (off_t < topk) & (indices > 0),
            indices - 1,
            -1,
        )
        tl.store(
            indices_final_ptr
            + pid_h * stride_f_h
            + pid_n * stride_f_n
            + off_t * stride_f_t,
            output.to(indices_final_ptr.dtype.element_ty),
            mask=(valid_n_i32 != 0) & (off_t < topk),
        )


# Decode uses the same stable pairwise merge/finalize path for multi-chunk
# contexts. Launch tiling is only needed by prefill top-k.
@triton.jit(do_not_specialize=["num_input_chunks"])
def _decode_topk_pair_merge_kernel(
    in_scores_ptr,
    in_indices_ptr,
    out_scores_ptr,
    out_indices_ptr,
    num_input_chunks,
    stride_is_c,
    stride_is_h,
    stride_is_n,
    stride_is_t,
    stride_ii_c,
    stride_ii_h,
    stride_ii_n,
    stride_ii_t,
    stride_os_c,
    stride_os_h,
    stride_os_n,
    stride_os_t,
    stride_oi_c,
    stride_oi_h,
    stride_oi_n,
    stride_oi_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_out_chunk = tl.program_id(2)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    left_chunk = 2 * pid_out_chunk
    right_chunk = left_chunk + 1
    right_exists_i32 = (right_chunk < num_input_chunks).to(tl.int32)

    left_scores = tl.load(
        in_scores_ptr + left_chunk * stride_is_c + pid_h * stride_is_h
        + pid_n * stride_is_n + off_t * stride_is_t,
    ).to(tl.float32)
    left_indices = tl.load(
        in_indices_ptr + left_chunk * stride_ii_c + pid_h * stride_ii_h
        + pid_n * stride_ii_n + off_t * stride_ii_t,
    ).to(tl.int32)
    right_scores = tl.load(
        in_scores_ptr + right_chunk * stride_is_c + pid_h * stride_is_h
        + pid_n * stride_is_n + off_t * stride_is_t,
        mask=right_exists_i32 != 0,
        other=-1e30,
    ).to(tl.float32)
    right_indices = tl.load(
        in_indices_ptr + right_chunk * stride_ii_c + pid_h * stride_ii_h
        + pid_n * stride_ii_n + off_t * stride_ii_t,
        mask=right_exists_i32 != 0,
        other=0,
    ).to(tl.int32)

    merged_scores, merged_indices = _merge_topk_pairs(
        left_scores, left_indices, right_scores, right_indices, BLOCK_SIZE_T
    )
    tl.store(
        out_scores_ptr + pid_out_chunk * stride_os_c + pid_h * stride_os_h
        + pid_n * stride_os_n + off_t * stride_os_t,
        merged_scores,
    )
    tl.store(
        out_indices_ptr + pid_out_chunk * stride_oi_c + pid_h * stride_oi_h
        + pid_n * stride_oi_n + off_t * stride_oi_t,
        merged_indices,
    )


@triton.jit
def _decode_topk_finalize_kernel(
    indices_partial_ptr,
    indices_final_ptr,
    topk,
    stride_pi_c,
    stride_pi_h,
    stride_pi_n,
    stride_pi_t,
    stride_f_h,
    stride_f_n,
    stride_f_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    indices = tl.load(
        indices_partial_ptr + pid_h * stride_pi_h + pid_n * stride_pi_n
        + off_t * stride_pi_t,
    ).to(tl.int32)
    output = tl.where((off_t < topk) & (indices > 0), indices - 1, -1)
    tl.store(
        indices_final_ptr + pid_h * stride_f_h + pid_n * stride_f_n
        + off_t * stride_f_t,
        output.to(indices_final_ptr.dtype.element_ty),
        mask=off_t < topk,
    )


# ---------------------------------------------------------------------------
# Decode: fused block scoring and chunk-local top-k.
#
# Each program handles one query token and one logical-block chunk, retaining
# only that chunk's top-k pairs. Decode never materializes a full score tensor.
# ---------------------------------------------------------------------------
@triton.jit
def _decode_fused_local_topk(
    q_ptr,
    ik_cache_ptr,
    block_table_ptr,
    seq_lens,
    pid_n,
    pid_chunk,
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    init_blocks,
    local_blocks,
    sm_scale,
    decode_query_len,
    chunk_blocks,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_BLOCK: tl.constexpr,
):
    """Compute one decode chunk's sorted local top-k run.

    Both decode write paths use this helper so direct output and partial output
    share the same score, priority, and local-selection logic.
    """
    req_id = pid_n // decode_query_len
    q_offset = pid_n - req_id * decode_query_len
    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (
        kv_len + BLOCK_SIZE_BLOCK - 1
    ) // BLOCK_SIZE_BLOCK

    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(
        chunk_start + chunk_blocks,
        num_blocks,
    )
    local_start = tl.maximum(0, num_blocks - local_blocks)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    off_h = tl.arange(0, num_idx_heads)
    bt_row = block_table_ptr + req_id * stride_bt_b

    q = tl.load(
        q_ptr
        + pid_n * stride_q_n
        + off_h[None, :] * stride_q_h
        + off_d[:, None] * stride_q_d,
    )  # [D, H]

    topk_scores = tl.full(
        (num_idx_heads, TOPK_WIDTH),
        float("-inf"),
        dtype=tl.float32,
    )
    topk_indices = tl.full(
        (num_idx_heads, TOPK_WIDTH),
        0,
        dtype=tl.int32,
    )
    sm_scale_log2e = sm_scale * 1.4426950409

    for blk in tl.range(chunk_start, chunk_end):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[:, None] * stride_ik_pos
            + off_d * stride_ik_d,
        )  # [K, D]

        qk = tl.dot(k, q) * sm_scale_log2e  # [K, H]
        qk = tl.where(
            pos[:, None] < kv_len,
            qk,
            float("-inf"),
        )
        scores = tl.max(qk, axis=0)
        is_init = blk < init_blocks
        is_local = (
            (blk >= local_start) & (blk < num_blocks)
        )
        scores = tl.where(
            is_local,
            1e29,
            tl.where(is_init, 1e30, scores),
        )
        scores = tl.where(scores != scores, -1e30, scores)

        # Keep one unordered local set while scanning this chunk.
        # Strict replacement preserves the earlier block on score ties.
        off_t = tl.arange(0, TOPK_WIDTH)
        worst_score = tl.min(topk_scores, axis=1)
        worst_pos = tl.argmin(topk_scores, axis=1)
        replace = scores > worst_score
        replace_mask = (
            (off_t[None, :] == worst_pos[:, None])
            & replace[:, None]
        )
        topk_scores = tl.where(
            replace_mask,
            scores[:, None],
            topk_scores,
        )
        topk_indices = tl.where(
            replace_mask,
            (blk + 1).to(tl.int32),
            topk_indices,
        )

    # Canonicalize the local set before it enters the existing merge tree.
    # Output order is descending score, then ascending logical block id.
    off_t = tl.arange(0, TOPK_WIDTH)
    work_scores = topk_scores
    work_indices = topk_indices
    sorted_scores = tl.full(
        (num_idx_heads, TOPK_WIDTH),
        -1e30,
        dtype=tl.float32,
    )
    sorted_indices = tl.full(
        (num_idx_heads, TOPK_WIDTH),
        0,
        dtype=tl.int32,
    )

    for rank in tl.static_range(0, TOPK_WIDTH):
        valid = work_indices > 0
        ranked_scores = tl.where(valid, work_scores, float("-inf"))
        best_score = tl.max(ranked_scores, axis=1)
        at_best_score = valid & (
            ranked_scores == best_score[:, None]
        )
        tie_index = tl.where(
            at_best_score,
            work_indices,
            2147483647,
        )
        best_index = tl.min(tie_index, axis=1)
        selected = at_best_score & (
            work_indices == best_index[:, None]
        )

        has_best = best_index != 2147483647
        selected_score = tl.where(has_best, best_score, -1e30)
        selected_index = tl.where(has_best, best_index, 0).to(tl.int32)
        slot = off_t[None, :] == rank
        sorted_scores = tl.where(
            slot,
            selected_score[:, None],
            sorted_scores,
        )
        sorted_indices = tl.where(
            slot,
            selected_index[:, None],
            sorted_indices,
        )

        work_scores = tl.where(
            selected,
            float("-inf"),
            work_scores,
        )
        work_indices = tl.where(selected, 0, work_indices)

    return sorted_scores, sorted_indices


@triton.jit(
    do_not_specialize=["chunk_blocks", "decode_query_len"]
)
def _decode_fused_topk_partial_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    scores_partial_ptr,  # [num_chunks, num_idx_heads, total_q, topk_width]
    indices_partial_ptr,  # [num_chunks, num_idx_heads, total_q, topk_width]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    init_blocks,
    local_blocks,
    sm_scale,
    decode_query_len,
    chunk_blocks,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_ps_c,
    stride_ps_h,
    stride_ps_n,
    stride_ps_t,
    stride_pi_c,
    stride_pi_h,
    stride_pi_n,
    stride_pi_t,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_BLOCK: tl.constexpr,
):
    """Score one decode block chunk and write its local top-k run."""
    pid_n = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    off_h = tl.arange(0, num_idx_heads)
    off_t = tl.arange(0, TOPK_WIDTH)

    topk_scores, topk_indices = _decode_fused_local_topk(
        q_ptr,
        ik_cache_ptr,
        block_table_ptr,
        seq_lens,
        pid_n,
        pid_chunk,
        num_idx_heads,
        head_dim,
        TOPK_WIDTH,
        init_blocks,
        local_blocks,
        sm_scale,
        decode_query_len,
        chunk_blocks,
        stride_q_n,
        stride_q_h,
        stride_q_d,
        stride_ik_blk,
        stride_ik_pos,
        stride_ik_d,
        stride_bt_b,
        BLOCK_SIZE_K,
        BLOCK_SIZE_BLOCK,
    )

    tl.store(
        scores_partial_ptr
        + pid_chunk * stride_ps_c
        + off_h[:, None] * stride_ps_h
        + pid_n * stride_ps_n
        + off_t[None, :] * stride_ps_t,
        topk_scores,
    )
    tl.store(
        indices_partial_ptr
        + pid_chunk * stride_pi_c
        + off_h[:, None] * stride_pi_h
        + pid_n * stride_pi_n
        + off_t[None, :] * stride_pi_t,
        topk_indices,
    )


@triton.jit(
    do_not_specialize=["chunk_blocks", "decode_query_len"]
)
def _decode_fused_topk_direct_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    indices_final_ptr,  # [num_idx_heads, total_q, topk]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    topk,
    init_blocks,
    local_blocks,
    sm_scale,
    decode_query_len,
    chunk_blocks,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_f_h,
    stride_f_n,
    stride_f_t,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_BLOCK: tl.constexpr,
):
    """Single-chunk decode: score, select, sort, and write API output."""
    pid_n = tl.program_id(0)
    off_h = tl.arange(0, num_idx_heads)
    off_t = tl.arange(0, TOPK_WIDTH)

    _, topk_indices = _decode_fused_local_topk(
        q_ptr,
        ik_cache_ptr,
        block_table_ptr,
        seq_lens,
        pid_n,
        0,
        num_idx_heads,
        head_dim,
        TOPK_WIDTH,
        init_blocks,
        local_blocks,
        sm_scale,
        decode_query_len,
        chunk_blocks,
        stride_q_n,
        stride_q_h,
        stride_q_d,
        stride_ik_blk,
        stride_ik_pos,
        stride_ik_d,
        stride_bt_b,
        BLOCK_SIZE_K,
        BLOCK_SIZE_BLOCK,
    )

    output = tl.where(
        (off_t[None, :] < topk) & (topk_indices > 0),
        topk_indices - 1,
        -1,
    )
    tl.store(
        indices_final_ptr
        + off_h[:, None] * stride_f_h
        + pid_n * stride_f_n
        + off_t[None, :] * stride_f_t,
        output.to(indices_final_ptr.dtype.element_ty),
        mask=off_t[None, :] < topk,
    )


# ---------------------------------------------------------------------------
# Python launch planning and public wrappers.
# ---------------------------------------------------------------------------
PREFILL_TOPK_PROGRAM_BUDGET = 32768
PREFILL_TOPK_MAX_QUERY_TILE = 8


def _topk_compute_width(topk: int) -> int:
    return max(TOPK_COMPUTE_MIN_TILE, triton.next_power_of_2(topk))


def _topk_select_width(topk: int) -> int:
    return max(TOPK_SELECTION_TILE, _topk_compute_width(topk))


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _prefill_topk_query_tile(
    max_query_len: int,
    batch: int,
    num_heads: int,
    num_chunks: int,
) -> int:
    """Choose a small static Q tile for launch-safe prefill top-k.

    The chosen tile is one for normal shapes.  It grows only after the
    flattened select launch would exceed ``PREFILL_TOPK_PROGRAM_BUDGET`` and is
    capped at eight to avoid turning the compile-time query loop into a large
    register footprint.  Larger inputs are covered by multiple launches.
    """
    launch_width = max(1, batch * num_heads * num_chunks)
    required_tile = _ceil_div(
        max_query_len * launch_width,
        PREFILL_TOPK_PROGRAM_BUDGET,
    )
    return min(
        PREFILL_TOPK_MAX_QUERY_TILE,
        _next_power_of_two(required_tile),
    )


def _prefill_topk_launch_ranges(
    max_query_len: int,
    batch: int,
    head_chunk_width: int,
    query_tile: int,
):
    """Yield bounded (batch, query) regions for the prefill partial kernel.

    The physical grid for every yielded region is:
    ``ceil(query_count / query_tile) * batch_count * head_chunk_width``.
    This helper keeps it at or below the configured launch budget.
    """
    if head_chunk_width > PREFILL_TOPK_PROGRAM_BUDGET:
        raise RuntimeError(
            "prefill top-k head/chunk width exceeds the launch budget; "
            "split the score chunks before calling this operator"
        )

    max_query_tiles = max(
        1,
        PREFILL_TOPK_PROGRAM_BUDGET // head_chunk_width,
    )
    total_query_tiles = _ceil_div(max_query_len, query_tile)
    query_tiles_per_launch = min(total_query_tiles, max_query_tiles)
    batch_per_launch = max(
        1,
        PREFILL_TOPK_PROGRAM_BUDGET
        // (query_tiles_per_launch * head_chunk_width),
    )
    query_span = query_tiles_per_launch * query_tile

    for batch_start in range(0, batch, batch_per_launch):
        batch_count = min(batch_per_launch, batch - batch_start)
        for query_start in range(0, max_query_len, query_span):
            query_count = min(query_span, max_query_len - query_start)
            program_count = (
                _ceil_div(query_count, query_tile)
                * batch_count
                * head_chunk_width
            )
            if program_count > PREFILL_TOPK_PROGRAM_BUDGET:
                raise AssertionError("invalid prefill top-k launch plan")
            yield batch_start, batch_count, query_start, query_count


def _linear_topk_launch_ranges(
    total_q: int,
    launch_width: int,
    query_tile: int,
):
    """Yield bounded flattened-token regions for merge/finalize kernels."""
    if launch_width > PREFILL_TOPK_PROGRAM_BUDGET:
        raise RuntimeError(
            "top-k merge/finalize width exceeds the launch budget; "
            "split the input chunks before calling this operator"
        )
    query_tiles_per_launch = max(
        1,
        PREFILL_TOPK_PROGRAM_BUDGET // launch_width,
    )
    query_span = query_tiles_per_launch * query_tile
    for query_start in range(0, total_q, query_span):
        query_count = min(query_span, total_q - query_start)
        program_count = _ceil_div(query_count, query_tile) * launch_width
        if program_count > PREFILL_TOPK_PROGRAM_BUDGET:
            raise AssertionError("invalid linear top-k launch plan")
        yield query_start, query_count


def _merge_topk_levels(
    scores: torch.Tensor,
    indices: torch.Tensor,
    num_heads: int,
    total_q: int,
    topk_width: int,
    query_tile: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hierarchically merge chunk-local runs with the same query tiling."""
    current_scores = scores
    current_indices = indices
    current_chunks = scores.shape[0]
    while current_chunks > 1:
        next_chunks = triton.cdiv(current_chunks, 2)
        next_scores = torch.empty(
            (next_chunks, num_heads, total_q, topk_width),
            dtype=torch.float32,
            device=scores.device,
        )
        next_indices = torch.empty(
            (next_chunks, num_heads, total_q, topk_width),
            dtype=torch.int32,
            device=indices.device,
        )
        for query_start, query_count in _linear_topk_launch_ranges(
            total_q,
            num_heads * next_chunks,
            query_tile,
        ):
            _topk_pair_merge_kernel[
                (triton.cdiv(query_count, query_tile), num_heads, next_chunks)
            ](
                current_scores,
                current_indices,
                next_scores,
                next_indices,
                current_chunks,
                total_q,
                query_start,
                current_scores.stride(0),
                current_scores.stride(1),
                current_scores.stride(2),
                current_scores.stride(3),
                current_indices.stride(0),
                current_indices.stride(1),
                current_indices.stride(2),
                current_indices.stride(3),
                next_scores.stride(0),
                next_scores.stride(1),
                next_scores.stride(2),
                next_scores.stride(3),
                next_indices.stride(0),
                next_indices.stride(1),
                next_indices.stride(2),
                next_indices.stride(3),
                BLOCK_SIZE_T=topk_width,
                QUERY_TILE=query_tile,
                num_warps=TOPK_NUM_WARPS,
                num_stages=TOPK_NUM_STAGES,
            )
        current_scores = next_scores
        current_indices = next_indices
        current_chunks = next_chunks
    return current_scores, current_indices


def _merge_topk_levels_decode(
    scores: torch.Tensor,
    indices: torch.Tensor,
    num_heads: int,
    total_q: int,
    topk_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original decode merge path, independent from prefill query tiling."""
    current_scores = scores
    current_indices = indices
    current_chunks = scores.shape[0]
    while current_chunks > 1:
        next_chunks = triton.cdiv(current_chunks, 2)
        next_scores = torch.empty(
            (next_chunks, num_heads, total_q, topk_width),
            dtype=torch.float32,
            device=scores.device,
        )
        next_indices = torch.empty(
            (next_chunks, num_heads, total_q, topk_width),
            dtype=torch.int32,
            device=indices.device,
        )
        _decode_topk_pair_merge_kernel[(total_q, num_heads, next_chunks)](
            current_scores,
            current_indices,
            next_scores,
            next_indices,
            current_chunks,
            current_scores.stride(0),
            current_scores.stride(1),
            current_scores.stride(2),
            current_scores.stride(3),
            current_indices.stride(0),
            current_indices.stride(1),
            current_indices.stride(2),
            current_indices.stride(3),
            next_scores.stride(0),
            next_scores.stride(1),
            next_scores.stride(2),
            next_scores.stride(3),
            next_indices.stride(0),
            next_indices.stride(1),
            next_indices.stride(2),
            next_indices.stride(3),
            BLOCK_SIZE_T=topk_width,
            num_warps=TOPK_NUM_WARPS,
            num_stages=TOPK_NUM_STAGES,
        )
        current_scores = next_scores
        current_indices = next_indices
        current_chunks = next_chunks
    return current_scores, current_indices


def _prefill_score_q_tile(max_query_len: int) -> int:
    """Choose the smallest score Q tile that covers the request.

    The score kernel is unchanged.  This only avoids running a 64-row QK tile
    for short continuation chunks such as q_len=16.
    """
    if max_query_len <= 16:
        return 16
    if max_query_len <= 32:
        return 32
    return 64


@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale: float,
) -> torch.Tensor:
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads

    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )

    block_size_q = _prefill_score_q_tile(max_query_len)
    _index_block_score_kernel[
        (triton.cdiv(max_query_len, block_size_q), batch * num_idx_heads)
    ](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_idx_heads,
        head_dim,
        sm_scale,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=block_size_q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )
    return score


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Select sparse logical-block top-k with launch-safe adaptive Q tiling.

    The public tensor layout and priority semantics are unchanged.  Only the
    physical launch plan changes when the original one-query-per-program grid
    would be too large for Ascend.
    """
    assert topk > 0

    num_heads, total_q, score_block_stride = score.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk_width = _topk_compute_width(topk)
    select_width = _topk_select_width(topk)
    num_chunks = max(1, triton.cdiv(score_block_stride, select_width))
    chunk_blocks = triton.cdiv(score_block_stride, num_chunks)
    query_tile = _prefill_topk_query_tile(
        max_query_len,
        batch,
        num_heads,
        num_chunks,
    )

    partial_scores = torch.empty(
        (num_chunks, num_heads, total_q, topk_width),
        dtype=torch.float32,
        device=score.device,
    )
    partial_indices = torch.empty(
        (num_chunks, num_heads, total_q, topk_width),
        dtype=torch.int32,
        device=score.device,
    )

    head_chunk_width = num_heads * num_chunks
    for batch_start, batch_count, query_start, query_count in _prefill_topk_launch_ranges(
        max_query_len,
        batch,
        head_chunk_width,
        query_tile,
    ):
        _prefill_topk_partial_kernel[
            (
                triton.cdiv(query_count, query_tile),
                batch_count,
                head_chunk_width,
            )
        ](
            score,
            partial_scores,
            partial_indices,
            cu_seqlens_q,
            prefix_lens,
            init_blocks,
            local_blocks,
            chunk_blocks,
            num_chunks,
            query_start,
            batch_start,
            score.stride(0),
            score.stride(1),
            score.stride(2),
            partial_scores.stride(0),
            partial_scores.stride(1),
            partial_scores.stride(2),
            partial_scores.stride(3),
            partial_indices.stride(0),
            partial_indices.stride(1),
            partial_indices.stride(2),
            partial_indices.stride(3),
            BLOCK_SIZE_K=select_width,
            BLOCK_SIZE_T=topk_width,
            BLOCK_SIZE_BLOCK=SPARSE_BLOCK_SIZE,
            QUERY_TILE=query_tile,
            MASK_INIT=False,
            MASK_LOCAL=False,
            num_warps=TOPK_NUM_WARPS,
            num_stages=TOPK_NUM_STAGES,
        )

    _, final_indices = _merge_topk_levels(
        partial_scores,
        partial_indices,
        num_heads,
        total_q,
        topk_width,
        query_tile,
    )
    topk_idx = torch.empty(
        (num_heads, total_q, topk),
        dtype=torch.int32,
        device=score.device,
    )
    for query_start, query_count in _linear_topk_launch_ranges(
        total_q,
        num_heads,
        query_tile,
    ):
        _topk_finalize_kernel[
            (triton.cdiv(query_count, query_tile), num_heads)
        ](
            final_indices,
            topk_idx,
            topk,
            total_q,
            query_start,
            final_indices.stride(0),
            final_indices.stride(1),
            final_indices.stride(2),
            final_indices.stride(3),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            BLOCK_SIZE_T=topk_width,
            QUERY_TILE=query_tile,
            num_warps=TOPK_NUM_WARPS,
            num_stages=TOPK_NUM_STAGES,
        )
    return topk_idx

@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    sm_scale: float,
    decode_query_len: int,
) -> torch.Tensor:
    """Decode block scoring and local top-k without a full score tensor.

    Each fused program scores one logical-block chunk for one query token and
    writes only its local top-k pairs.  The existing pairwise merge tree then
    combines those sorted runs.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads
    assert total_q == seq_lens.shape[0] * decode_query_len
    assert topk > 0

    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score_block_stride = round_up(max_block, 16)
    topk_width = _topk_compute_width(topk)
    select_width = _topk_select_width(topk)
    num_chunks = max(
        1,
        triton.cdiv(score_block_stride, select_width),
    )
    chunk_blocks = triton.cdiv(score_block_stride, num_chunks)

    topk_idx = torch.empty(
        (num_idx_heads, total_q, topk),
        dtype=torch.int32,
        device=idx_q.device,
    )

    # A single chunk can emit public indices directly and avoid intermediates.
    if num_chunks == 1:
        _decode_fused_topk_direct_kernel[(total_q,)](
            idx_q,
            index_kv_cache,
            topk_idx,
            block_table,
            seq_lens,
            num_idx_heads,
            head_dim,
            TOPK_WIDTH=topk_width,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sm_scale=sm_scale,
            decode_query_len=decode_query_len,
            chunk_blocks=chunk_blocks,
            stride_q_n=idx_q.stride(0),
            stride_q_h=idx_q.stride(1),
            stride_q_d=idx_q.stride(2),
            stride_ik_blk=index_kv_cache.stride(0),
            stride_ik_pos=index_kv_cache.stride(1),
            stride_ik_d=index_kv_cache.stride(2),
            stride_f_h=topk_idx.stride(0),
            stride_f_n=topk_idx.stride(1),
            stride_f_t=topk_idx.stride(2),
            stride_bt_b=block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_BLOCK=SPARSE_BLOCK_SIZE,
            num_warps=TOPK_NUM_WARPS,
            num_stages=TOPK_NUM_STAGES,
        )
        return topk_idx

    partial_scores = torch.empty(
        (num_chunks, num_idx_heads, total_q, topk_width),
        dtype=torch.float32,
        device=idx_q.device,
    )
    partial_indices = torch.empty(
        (num_chunks, num_idx_heads, total_q, topk_width),
        dtype=torch.int32,
        device=idx_q.device,
    )

    _decode_fused_topk_partial_kernel[(total_q, num_chunks)](
        idx_q,
        index_kv_cache,
        partial_scores,
        partial_indices,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        TOPK_WIDTH=topk_width,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=sm_scale,
        decode_query_len=decode_query_len,
        chunk_blocks=chunk_blocks,
        stride_q_n=idx_q.stride(0),
        stride_q_h=idx_q.stride(1),
        stride_q_d=idx_q.stride(2),
        stride_ik_blk=index_kv_cache.stride(0),
        stride_ik_pos=index_kv_cache.stride(1),
        stride_ik_d=index_kv_cache.stride(2),
        stride_ps_c=partial_scores.stride(0),
        stride_ps_h=partial_scores.stride(1),
        stride_ps_n=partial_scores.stride(2),
        stride_ps_t=partial_scores.stride(3),
        stride_pi_c=partial_indices.stride(0),
        stride_pi_h=partial_indices.stride(1),
        stride_pi_n=partial_indices.stride(2),
        stride_pi_t=partial_indices.stride(3),
        stride_bt_b=block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_BLOCK=SPARSE_BLOCK_SIZE,
        num_warps=TOPK_NUM_WARPS,
        num_stages=TOPK_NUM_STAGES,
    )

    _, final_indices = _merge_topk_levels_decode(
        partial_scores,
        partial_indices,
        num_idx_heads,
        total_q,
        topk_width,
    )
    _decode_topk_finalize_kernel[(total_q, num_idx_heads)](
        final_indices,
        topk_idx,
        topk,
        final_indices.stride(0),
        final_indices.stride(1),
        final_indices.stride(2),
        final_indices.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        BLOCK_SIZE_T=topk_width,
        num_warps=TOPK_NUM_WARPS,
        num_stages=TOPK_NUM_STAGES,
    )
    return topk_idx

