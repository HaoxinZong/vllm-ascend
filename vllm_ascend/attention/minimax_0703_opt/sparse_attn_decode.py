# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone MiniMax M3 decode block-sparse GQA attention kernel + wrapper.

Extracted from vllm/models/minimax_m3/common/ops/sparse_attn.py — only
``minimax_m3_sparse_attn_decode`` and its direct dependencies.
"""

import torch
import triton
import triton.language as tl

SPARSE_BLOCK_SIZE = 128

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _split_triton_main_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an interleaved main KV cache into separate K/V tensors.

    Accepts either an already-split ``(k_cache, v_cache)`` pair/list, or a
    single interleaved tensor whose K/V axis is either ``shape[0] == 2`` or
    ``shape[1] == 2`` (Ascend's split cache layout keeps K/V on axis 1). The
    returned tensors are 4-D ``[num_blocks, 128, num_kv_heads, head_dim]``.
    """
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) < 2:
            raise ValueError("Main kv cache tuple must contain K and V tensors")
        k_cache, v_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.ndim != 5:
            raise ValueError(f"Unexpected main kv cache ndim: {kv_cache.ndim}")
        if kv_cache.shape[0] == 2:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        elif kv_cache.shape[1] == 2:
            k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
        else:
            raise ValueError(f"Unexpected main kv cache shape: {tuple(kv_cache.shape)}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "Unexpected split main kv cache shapes: "
            f"{tuple(k_cache.shape)}, {tuple(v_cache.shape)}"
        )
    return k_cache, v_cache


def _sparse_attn_num_stages_kwarg() -> dict:
    """Triton num_stages override for sparse-attn GEMM kernels.

    Only needed on AMD gfx942 (limited LDS). Returns empty dict otherwise.
    """
    return {}


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(
            args["gqa_group_size"]
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    t_ptr,
    o_ptr,
    lse_ptr,
    block_table_ptr,
    seq_lens,
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_k_blk,
    stride_k_pos,
    stride_k_h,
    stride_k_d,
    stride_v_blk,
    stride_v_pos,
    stride_v_h,
    stride_v_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    topk_idx = tl.load(idx_base + off_t * stride_tk, mask=off_t < max_topk, other=-1)
    real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    # Hoist loop-invariant address components to reduce per-iteration scalar work.
    # Round 12 profiling: 56% scalar ratio, largely from repeated address arithmetic
    # inside the topk loop. Precomputing these once saves 8+ scalar ops per iteration.
    k_head_base = k_cache_ptr + pid_kh * stride_k_h
    v_head_base = v_cache_ptr + pid_kh * stride_k_h
    k_pos_off = off_n[None, :] * stride_k_pos
    k_d_off = off_d[:, None] * stride_k_d
    v_pos_off = off_n[:, None] * stride_v_pos
    v_d_off = off_d[None, :] * stride_v_d
    d_mask_k = d_mask[:, None]
    d_mask_v = d_mask[None, :]

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < kv_len
        k = tl.load(
            k_head_base + page * stride_k_blk + k_pos_off + k_d_off,
            mask=d_mask_k & pos_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            v_head_base + page * stride_v_blk + v_pos_off + v_d_off,
            mask=pos_mask[:, None] & d_mask_v,
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,
    lse_ptr,
    out_ptr,
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_h_start = pid_m * BLOCK_M

    off_h = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_SIZE_D)

    d_mask = off_d < head_dim

    lse_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o_acc = tl.zeros((BLOCK_M, BLOCK_SIZE_D), dtype=tl.float32)

    for c in tl.range(NUM_TOPK_CHUNKS):
        o_chunk_base = (
            o_ptr + c * stride_o_c + pid_b * stride_o_b + pid_h_start * stride_o_h
        )
        o_chunk = tl.make_block_ptr(
            base=o_chunk_base,
            shape=(NUM_HEADS, head_dim),
            strides=(stride_o_h, stride_o_d),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_SIZE_D),
            order=(1, 0),
        )
        o_vals = tl.load(o_chunk, boundary_check=(0, 1), padding_option="zero")

        lse_chunk_base = (
            lse_ptr
            + c * stride_l_c
            + pid_b * stride_l_b
            + pid_h_start * stride_l_h
        )
        lse_vals = tl.load(
            lse_chunk_base + off_h * stride_l_h,
            mask=off_h < NUM_HEADS,
            other=float("-inf"),
        )

        new_lse_max = tl.maximum(lse_max, lse_vals)
        exp_scale = tl.exp2(lse_max - new_lse_max)
        o_acc = o_acc * exp_scale[:, None]
        sum_exp = sum_exp * exp_scale
        exp_new = tl.exp2(lse_vals - new_lse_max)
        o_acc = o_acc + o_vals * exp_new[:, None]
        sum_exp = sum_exp + exp_new
        lse_max = new_lse_max

    o_acc = o_acc / sum_exp[:, None]

    out_base = (
        out_ptr + pid_b * stride_out_n + pid_h_start * stride_out_h
    )
    out_store = tl.make_block_ptr(
        base=out_base,
        shape=(NUM_HEADS, head_dim),
        strides=(stride_out_h, stride_out_d),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(out_store, o_acc.to(out_ptr.dtype.element_ty), boundary_check=(0, 1))


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    k_cache, v_cache = _split_triton_main_kv_cache(kv_cache)
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = k_cache.dtype in _FP8_DTYPES or v_cache.dtype in _FP8_DTYPES
    # Hybrid dispatch: use fused single-pass when grid parallelism is sufficient;
    # use split-K with merge kernel when the grid is too small for good utilization.
    if num_kv_heads >= 2:
        # Fused: direct-to-output, no merge kernel needed
        o_ptr = output.unsqueeze(0)
        lse_ptr = torch.empty(
            1, total_q, num_heads, dtype=torch.float32, device=q.device
        )
        grid = (total_q, num_kv_heads)
        _gqa_sparse_decode_kernel[grid](
            q, k_cache, v_cache, topk_idx,
            o_ptr, lse_ptr,
            block_table, seq_lens,
            total_q, gqa_group_size, head_dim, max_topk, sm_scale,
            decode_query_len,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            o_ptr.stride(0), o_ptr.stride(1), o_ptr.stride(2), o_ptr.stride(3),
            lse_ptr.stride(0), lse_ptr.stride(1), lse_ptr.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            NUM_TOPK_CHUNKS=1,
            USE_FP8=use_fp8,
            **_sparse_attn_num_stages_kwarg(),
        )
        return

    # Split-K path for small grids: low parallelism, need chunk splitting
    TARGET_GRID = 16
    target = max(1, min(max_topk, TARGET_GRID // max(1, total_q * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, head_dim, dtype=q.dtype, device=q.device
    )
    lse_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device
    )
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        total_q,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        **_sparse_attn_num_stages_kwarg(),
    )
    BLOCK_M = 4
    merge_grid = (total_q, (num_heads + BLOCK_M - 1) // BLOCK_M)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        BLOCK_M=BLOCK_M,
        NUM_HEADS=num_heads,
    )
