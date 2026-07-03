# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone MiniMax M3 decode index block-score + top-k kernel + wrapper.

Extracted from vllm/models/minimax_m3/common/ops/index_topk.py — only
``minimax_m3_index_decode`` and its direct dependencies.

NPU-adapted: the bitonic top-k kernels are replaced with a chunk-wise
select/merge top-k (one ``_select_topk_pairs`` per chunk, then pairwise
``_merge_topk_pairs`` reductions), which avoids the reshape/bitcast ops the
Ascend BiShengIR backend does not support. The decode score kernel keeps its
split-K grid but drops PDL (CUDA SM9+), since this build runs on NPU.
"""

import torch
import triton
import triton.language as tl

SPARSE_BLOCK_SIZE = 128

TOPK_SELECTION_TILE = 128
TOPK_COMPUTE_MIN_TILE = 16
TOPK_NUM_WARPS = 4
TOPK_NUM_STAGES = 2


def _round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _as_triton_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Normalize an index KV cache to ``[num_blocks, 128, head_dim]``.

    Accepts an already-3-D tensor, a ``(k, v)`` pair/list (K is used), or a
    4-D/5-D tensor with a singleton/leading-2 head axis that Ascend's layout
    produces; squeezes it down to the 3-D shape the kernel expects.
    """
    if isinstance(index_kv_cache, (tuple, list)):
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 5 and index_kv_cache.shape[0] == 2:
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 4:
        if index_kv_cache.shape[2] != 1:
            raise ValueError(
                f"Unexpected index cache head dim: {tuple(index_kv_cache.shape)}"
            )
        index_kv_cache = index_kv_cache.squeeze(2)
    if index_kv_cache.ndim != 3:
        raise ValueError(f"Unexpected index cache ndim: {index_kv_cache.ndim}")
    return index_kv_cache


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
    off_t = tl.arange(0, topk_size)

    left_valid_i32 = (left_indices > 0).to(tl.int32)
    right_valid_i32 = (right_indices > 0).to(tl.int32)
    left_work = tl.where(left_valid_i32 != 0, left_scores, float("-inf"))
    right_work = tl.where(right_valid_i32 != 0, right_scores, float("-inf"))

    out_scores = tl.full((topk_size,), -1e30, dtype=tl.float32)
    out_indices = tl.full((topk_size,), 0, dtype=tl.int32)

    for rank in tl.static_range(0, topk_size):
        left_best = tl.max(left_work, axis=0)
        left_offset = tl.argmax(left_work, axis=0).to(tl.int32)
        right_best = tl.max(right_work, axis=0)
        right_offset = tl.argmax(right_work, axis=0).to(tl.int32)

        take_left_i32 = (left_best >= right_best).to(tl.int32)

        left_selected_i32 = (
            (off_t == left_offset).to(tl.int32) * left_valid_i32
        )
        right_selected_i32 = (
            (off_t == right_offset).to(tl.int32) * right_valid_i32
        )
        left_index = tl.sum(
            tl.where(left_selected_i32 != 0, left_indices, 0),
            axis=0,
        ).to(tl.int32)
        right_index = tl.sum(
            tl.where(right_selected_i32 != 0, right_indices, 0),
            axis=0,
        ).to(tl.int32)

        best_score = tl.where(take_left_i32 != 0, left_best, right_best)
        best_index = tl.where(take_left_i32 != 0, left_index, right_index)
        has_best_i32 = (best_index > 0).to(tl.int32)

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

        left_remove_i32 = left_selected_i32 * take_left_i32
        right_remove_i32 = right_selected_i32 * (1 - take_left_i32)
        left_valid_i32 = left_valid_i32 * (1 - left_remove_i32)
        right_valid_i32 = right_valid_i32 * (1 - right_remove_i32)
        left_work = tl.where(left_remove_i32 != 0, float("-inf"), left_work)
        right_work = tl.where(right_remove_i32 != 0, float("-inf"), right_work)

    return out_scores, out_indices


# ---------------------------------------------------------------------------
# Fused chunk-wise score + partial topk kernel for multi-chunk cases.
# Each program processes one chunk of blocks for a query token, maintaining
# a partial running topk per head. Partial results are stored to a
# [num_chunks, num_idx_heads, total_q, topk_width] tensor and merged
# downstream by _topk_pair_merge_kernel + _topk_finalize_kernel.
# Eliminates the separate score tensor and _decode_topk_partial_kernel launch.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_fused_chunk_topk_kernel(
    q_ptr,
    ik_cache_ptr,
    partial_scores_ptr,  # [num_chunks, num_idx_heads, total_q, topk_width]
    partial_indices_ptr,  # [num_chunks, num_idx_heads, total_q, topk_width]
    block_table_ptr,
    seq_lens,
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
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_n = tl.program_id(0)  # query token
    pid_c = tl.program_id(1)  # chunk id

    req_id = pid_n // decode_query_len
    q_offset = pid_n - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + BLOCK_SIZE_BLOCK - 1) // BLOCK_SIZE_BLOCK

    chunk_start = pid_c * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)
    if chunk_start >= chunk_end:
        return

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + req_id * stride_bt_b
    local_start = tl.maximum(0, num_blocks - local_blocks)

    # Q [D, H]
    q = tl.load(
        q_ptr
        + pid_n * stride_q_n
        + tl.arange(0, num_idx_heads) * stride_q_h
        + off_d[:, None] * stride_q_d,
    )  # [D, H]

    # Partial top-k per head: [num_idx_heads, TOPK_WIDTH]
    off_t = tl.arange(0, TOPK_WIDTH)
    topk_scores = tl.full((num_idx_heads, TOPK_WIDTH), -1e30, dtype=tl.float32)
    topk_indices = tl.full((num_idx_heads, TOPK_WIDTH), -1, dtype=tl.int32)

    for blk in tl.range(chunk_start, chunk_end):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos < kv_len
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[:, None] * stride_ik_pos
            + off_d * stride_ik_d,
        )  # [N, D]

        kq = tl.dot(k, q) * sm_scale_log2e  # [N, H]
        kq = tl.where(pos_mask[:, None], kq, float("-inf"))
        score = tl.max(kq, axis=0)  # [H]

        is_init = blk < init_blocks
        is_local = (blk >= local_start) & (blk < num_blocks)
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))

        # Running top-k per head: [H, TOPK_WIDTH]
        min_score = tl.min(topk_scores, axis=1)  # [H]
        min_pos = tl.argmin(topk_scores, axis=1)  # [H]

        should_replace = (score > min_score)  # [H]
        replace_mask = (
            (off_t[None, :] == min_pos[:, None]) & should_replace[:, None]
        )

        block_idx = (blk + 1).to(tl.int32)
        topk_scores = tl.where(replace_mask, score[:, None], topk_scores)
        topk_indices = tl.where(replace_mask, block_idx, topk_indices)

    # Store partial top-k scores and indices per head.
    off_h = tl.arange(0, num_idx_heads)
    tl.store(
        partial_scores_ptr
        + pid_c * stride_ps_c
        + off_h[:, None] * stride_ps_h
        + pid_n * stride_ps_n
        + off_t[None, :] * stride_ps_t,
        topk_scores,
    )
    tl.store(
        partial_indices_ptr
        + pid_c * stride_pi_c
        + off_h[:, None] * stride_pi_h
        + pid_n * stride_pi_n
        + off_t[None, :] * stride_pi_t,
        topk_indices,
    )


# ---------------------------------------------------------------------------
# Fused decode score + topk kernel for single-chunk cases.
# Processes all blocks for one query token in a single program, computing
# scores and maintaining a running topk per head. Eliminates the intermediate
# score tensor, the PyTorch topk call, and the CPU-based torch.where cleanup.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_fused_score_topk_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    topk_indices_ptr,  # [num_idx_heads, total_q, topk]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    TOPK: tl.constexpr,  # actual topk value (not padded)
    init_blocks,
    local_blocks,
    sm_scale,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_out_h,
    stride_out_n,
    stride_out_t,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_n = tl.program_id(0)  # flattened query-token id
    req_id = pid_n // decode_query_len
    q_offset = pid_n - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + req_id * stride_bt_b
    local_start = tl.maximum(0, num_blocks - local_blocks)

    # Q [D, H]
    q = tl.load(
        q_ptr
        + pid_n * stride_q_n
        + tl.arange(0, num_idx_heads) * stride_q_h
        + off_d[:, None] * stride_q_d,
    )  # [D, H]

    # Running top-k per head: [num_idx_heads, TOPK] layout.
    # Track exactly TOPK slots (not a padded width), so each slot always
    # belongs to one of the current top-TOPK candidates. After processing
    # all blocks the array contains the exact top-TOPK scores.
    off_t = tl.arange(0, TOPK)
    topk_scores = tl.full((num_idx_heads, TOPK), -1e30, dtype=tl.float32)
    topk_indices = tl.full((num_idx_heads, TOPK), -1, dtype=tl.int32)

    for blk in tl.range(0, num_blocks):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos < kv_len
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[:, None] * stride_ik_pos
            + off_d * stride_ik_d,
        )  # [N, D]

        kq = tl.dot(k, q) * sm_scale_log2e  # [N, H]
        kq = tl.where(pos_mask[:, None], kq, float("-inf"))
        score = tl.max(kq, axis=0)  # [H]

        # Apply init/local score overrides
        is_init = blk < init_blocks
        is_local = (blk >= local_start) & (blk < num_blocks)
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))

        # Insert score into running top-k per head.
        # topk_scores: [H, TOPK] — min/argmin over axis=1 gives [H]
        min_score = tl.min(topk_scores, axis=1)  # [H]
        min_pos = tl.argmin(topk_scores, axis=1)  # [H]

        # Replace mask: [H, TOPK]
        should_replace = (score > min_score)  # [H]
        replace_mask = (
            (off_t[None, :] == min_pos[:, None]) & should_replace[:, None]
        )

        block_idx = (blk + 1).to(tl.int32)
        topk_scores = tl.where(replace_mask, score[:, None], topk_scores)
        topk_indices = tl.where(replace_mask, block_idx, topk_indices)

    # Store final top-k indices: all TOPK slots map directly to the output
    # tensor's last dimension (both are exactly `topk` elements).
    off_h = tl.arange(0, num_idx_heads)
    output_vals = tl.where(
        topk_indices >= 0,
        topk_indices - 1,
        tl.full((num_idx_heads, TOPK), -1, dtype=tl.int32),
    )
    tl.store(
        topk_indices_ptr
        + off_h[:, None] * stride_out_h
        + pid_n * stride_out_n
        + off_t[None, :] * stride_out_t,
        output_vals,
    )


# ---------------------------------------------------------------------------
# Decode index-score kernel (split-K over seq blocks). Decode batches are
# flattened request-major, with a runtime query length used to map each query
# token back to its request metadata. Base-2 score scaling matches prefill.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["num_kv_chunks", "decode_query_len"])
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    sm_scale,
    decode_query_len,
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
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    num_kv_chunks,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_c = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-shape padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # Block-aligned fixed-count split.  The grid depends only on the launch
    # shape, while each program masks to its request's actual valid block range.
    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + req_id * stride_bt_b
    local_start = tl.maximum(0, num_blocks - local_blocks)

    q = tl.load(
        q_ptr
        + pid_b * stride_q_n
        + tl.arange(0, num_idx_heads) * stride_q_h
        + off_d[:, None] * stride_q_d,
    )  # [D, H]

    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos < kv_len
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[:, None] * stride_ik_pos
            + off_d * stride_ik_d,
        )  # [N, D]

        kq = tl.dot(k, q) * sm_scale_log2e  # [N, H]
        kq = tl.where(pos_mask[:, None], kq, float("-inf"))
        score = tl.max(kq, axis=0)  # [H]

        is_init = blk < init_blocks
        is_local = (blk >= local_start) & (blk < num_blocks)
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr
            + tl.arange(0, num_idx_heads) * stride_s_h
            + pid_b * stride_s_n
            + blk * stride_s_k,
            score,
        )


# ---------------------------------------------------------------------------
# Decode local top-k: one program per (query-token, head, chunk).
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["chunk_blocks", "decode_query_len"])
def _decode_topk_partial_kernel(
    s_ptr,
    scores_partial_ptr,
    indices_partial_ptr,
    seq_lens,
    chunk_blocks,
    decode_query_len,
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
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    req_id = pid_n // decode_query_len
    q_offset = pid_n - req_id * decode_query_len
    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + BLOCK_SIZE_BLOCK - 1) // BLOCK_SIZE_BLOCK

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    chunk_start = pid_chunk * chunk_blocks
    block_ids = chunk_start + off_k
    valid_i32 = (
        ((block_ids < num_blocks) & (off_k < chunk_blocks)).to(tl.int32)
    )

    scores = tl.load(
        s_ptr
        + pid_h * stride_s_h
        + pid_n * stride_s_n
        + block_ids * stride_s_k,
        mask=valid_i32 != 0,
        other=-1e30,
    ).to(tl.float32)
    scores = tl.where(scores != scores, -1e30, scores)
    indices = tl.where(valid_i32 != 0, block_ids + 1, 0).to(tl.int32)

    selected_scores, selected_indices = _select_topk_pairs(
        scores,
        indices,
        valid_i32 != 0,
        BLOCK_SIZE_T,
        BLOCK_SIZE_K,
    )

    tl.store(
        scores_partial_ptr
        + pid_chunk * stride_ps_c
        + pid_h * stride_ps_h
        + pid_n * stride_ps_n
        + off_t * stride_ps_t,
        selected_scores,
    )
    tl.store(
        indices_partial_ptr
        + pid_chunk * stride_pi_c
        + pid_h * stride_pi_h
        + pid_n * stride_pi_n
        + off_t * stride_pi_t,
        selected_indices,
    )


@triton.jit(do_not_specialize=["num_input_chunks"])
def _topk_pair_merge_kernel(
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
        in_scores_ptr
        + left_chunk * stride_is_c
        + pid_h * stride_is_h
        + pid_n * stride_is_n
        + off_t * stride_is_t,
    ).to(tl.float32)
    left_indices = tl.load(
        in_indices_ptr
        + left_chunk * stride_ii_c
        + pid_h * stride_ii_h
        + pid_n * stride_ii_n
        + off_t * stride_ii_t,
    ).to(tl.int32)
    right_scores = tl.load(
        in_scores_ptr
        + right_chunk * stride_is_c
        + pid_h * stride_is_h
        + pid_n * stride_is_n
        + off_t * stride_is_t,
        mask=right_exists_i32 != 0,
        other=-1e30,
    ).to(tl.float32)
    right_indices = tl.load(
        in_indices_ptr
        + right_chunk * stride_ii_c
        + pid_h * stride_ii_h
        + pid_n * stride_ii_n
        + off_t * stride_ii_t,
        mask=right_exists_i32 != 0,
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
    )
    tl.store(
        out_indices_ptr
        + pid_out_chunk * stride_oi_c
        + pid_h * stride_oi_h
        + pid_n * stride_oi_n
        + off_t * stride_oi_t,
        merged_indices,
    )


@triton.jit
def _topk_finalize_kernel(
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
        indices_partial_ptr
        + pid_h * stride_pi_h
        + pid_n * stride_pi_n
        + off_t * stride_pi_t,
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
        mask=off_t < topk,
    )


def _topk_compute_width(topk: int) -> int:
    return max(TOPK_COMPUTE_MIN_TILE, triton.next_power_of_2(topk))


def _topk_select_width(topk: int) -> int:
    return max(TOPK_SELECTION_TILE, _topk_compute_width(topk))


def _merge_topk_levels(
    scores: torch.Tensor,
    indices: torch.Tensor,
    num_heads: int,
    total_q: int,
    topk_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        _topk_pair_merge_kernel[(total_q, num_heads, next_chunks)](
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
    """Decode index block-score + top-k, both split-K (cudagraph-safe).

    Returns topk_idx [num_kv_heads, total_q, topk] (0-indexed block ids, -1 pad).
    """
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads
    assert total_q == seq_lens.shape[0] * decode_query_len
    assert topk > 0

    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    topk_width = _topk_compute_width(topk)
    select_width = _topk_select_width(topk)
    score_block_stride = _round_up(max_block, 16)
    num_chunks = max(1, triton.cdiv(score_block_stride, select_width))

    if num_chunks == 1:
        # Fast path: fused score + topk kernel.
        # Processes all blocks for each query token in a single program,
        # maintaining a running topk per head. Eliminates the intermediate
        # score tensor, the PyTorch topk call, and CPU-based cleanup.
        topk_idx = torch.full(
            (num_idx_heads, total_q, topk),
            -1,
            dtype=torch.int32,
            device=idx_q.device,
        )
        _decode_fused_score_topk_kernel[(total_q,)](
            idx_q,
            index_kv_cache,
            topk_idx,
            block_table,
            seq_lens,
            num_idx_heads,
            head_dim,
            TOPK=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sm_scale=sm_scale,
            decode_query_len=decode_query_len,
            stride_q_n=idx_q.stride(0),
            stride_q_h=idx_q.stride(1),
            stride_q_d=idx_q.stride(2),
            stride_ik_blk=index_kv_cache.stride(0),
            stride_ik_pos=index_kv_cache.stride(1),
            stride_ik_d=index_kv_cache.stride(2),
            stride_out_h=topk_idx.stride(0),
            stride_out_n=topk_idx.stride(1),
            stride_out_t=topk_idx.stride(2),
            stride_bt_b=block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        )
        return topk_idx

    # Multi-chunk path: fused chunk-wise score + partial topk kernel.
    # Each program processes one chunk of blocks, maintaining a partial
    # running topk per head. Partial results are then merged using the
    # existing topk merge pipeline.
    chunk_blocks = triton.cdiv(score_block_stride, num_chunks)

    partial_scores = torch.full(
        (num_chunks, num_idx_heads, total_q, topk_width),
        float("-inf"),
        dtype=torch.float32,
        device=idx_q.device,
    )
    partial_indices = torch.full(
        (num_chunks, num_idx_heads, total_q, topk_width),
        -1,
        dtype=torch.int32,
        device=idx_q.device,
    )
    _decode_fused_chunk_topk_kernel[(total_q, num_chunks)](
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
    )

    _, final_indices = _merge_topk_levels(
        partial_scores,
        partial_indices,
        num_idx_heads,
        total_q,
        topk_width,
    )
    topk_idx = torch.empty(
        (num_idx_heads, total_q, topk),
        dtype=torch.int32,
        device=idx_q.device,
    )
    _topk_finalize_kernel[(total_q, num_idx_heads)](
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
