import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.minimax_m3.ops import minimax_m3_triton as m


def _unwrap_triton_function(kernel):
    while hasattr(kernel, "fn"):
        kernel = kernel.fn
    return kernel


class Pointer:
    def __init__(self, tensor, offsets=0):
        self.tensor = tensor
        self.offsets = offsets

    @property
    def dtype(self):
        return SimpleNamespace(element_ty=self.tensor.dtype)

    def __add__(self, offsets):
        return Pointer(self.tensor, torch.as_tensor(self.offsets) + offsets)

    __radd__ = __add__


def ptr(tensor):
    return Pointer(tensor)


class FakeTL:
    int32 = torch.int32
    int64 = torch.int64
    float32 = torch.float32
    constexpr = object()

    def __init__(self):
        self.program_ids = (0, 0, 0)
        self.extra = SimpleNamespace(cuda=SimpleNamespace(gdc_wait=lambda: None, gdc_launch_dependents=lambda: None))

    def set_program_ids(self, *program_ids):
        self.program_ids = (*program_ids, 0, 0, 0)[:3]

    def program_id(self, axis):
        return self.program_ids[axis]

    @staticmethod
    def static_assert(condition, message=None):
        assert condition, message

    @staticmethod
    def arange(start, end):
        return torch.arange(start, end)

    @staticmethod
    def range(start, end, step=1):
        return range(int(start), int(end), int(step))

    @staticmethod
    def _access_mask(offsets, mask):
        if mask is None:
            return torch.ones_like(offsets, dtype=torch.bool)
        return torch.as_tensor(mask, dtype=torch.bool).broadcast_to(offsets.shape)

    @classmethod
    def load(cls, pointer, mask=None, other=0):
        offsets = torch.as_tensor(pointer.offsets, dtype=torch.long)
        access_mask = cls._access_mask(offsets, mask)
        safe_offsets = torch.where(access_mask, offsets, torch.zeros_like(offsets))
        values = pointer.tensor.reshape(-1)[safe_offsets]
        fallback = torch.as_tensor(other, dtype=values.dtype).broadcast_to(values.shape)
        return torch.where(access_mask, values, fallback)

    @classmethod
    def store(cls, pointer, value, mask=None):
        offsets = torch.as_tensor(pointer.offsets, dtype=torch.long)
        access_mask = cls._access_mask(offsets, mask)
        values = torch.as_tensor(value, dtype=pointer.tensor.dtype).broadcast_to(offsets.shape)
        pointer.tensor.reshape(-1)[offsets[access_mask]] = values[access_mask]

    @staticmethod
    def minimum(left, right):
        return torch.minimum(torch.as_tensor(left), torch.as_tensor(right))

    @staticmethod
    def maximum(left, right):
        return torch.maximum(torch.as_tensor(left), torch.as_tensor(right))

    @staticmethod
    def where(condition, left, right):
        return torch.where(torch.as_tensor(condition), torch.as_tensor(left), torch.as_tensor(right))

    @staticmethod
    def dot(left, right, out_dtype=None):
        result = torch.matmul(left, right)
        return result.to(out_dtype) if out_dtype is not None else result

    @staticmethod
    def max(value, axis):
        return torch.max(value, dim=axis).values

    @staticmethod
    def min(value, axis):
        return torch.min(value, dim=axis).values

    @staticmethod
    def sum(value, axis):
        return torch.sum(value, dim=axis)

    @staticmethod
    def floor(value):
        return torch.floor(value)


class KernelStub:
    def __init__(self, side_effect=None):
        self.side_effect = side_effect
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            if self.side_effect is not None:
                self.side_effect(grid, args, kwargs)

        return launch


def _all_launch_stubs(monkeypatch):
    stubs = {
        "_prefill_scalar_key_extrema_kernel": KernelStub(),
        "_prefill_scalar_index_score_kernel": KernelStub(),
        "_prefill_index_score_kernel": KernelStub(),
        "_prepare_prefill_topk_scores_kernel": KernelStub(),
        "_mask_prefill_topk_indices_kernel": KernelStub(),
        "_prepare_decode_score_masks_kernel": KernelStub(),
        "_decode_index_score_kernel": KernelStub(_fill_decode_scores),
        "_fill_decode_score_tail_kernel": KernelStub(),
        "_mask_decode_topk_indices_kernel": KernelStub(),
        "_gqa_sparse_fwd_kernel": KernelStub(),
        "_gqa_sparse_decode_kernel": KernelStub(),
        "_merge_topk_attn_out_kernel": KernelStub(),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(m, name, stub)
    return stubs


def _fill_decode_scores(_grid, args, _kwargs):
    score = args[2]
    score.fill_(float("-inf"))
    max_blocks = min(3, score.shape[-1])
    values = torch.arange(max_blocks, dtype=score.dtype, device=score.device)
    score[..., :max_blocks] = values


def test_as_triton_index_kv_cache_accepts_supported_layouts():
    cache = torch.randn(4, m.SPARSE_BLOCK_SIZE, 8)
    assert m._as_triton_index_kv_cache(cache) is cache
    assert m._as_triton_index_kv_cache((cache, torch.empty(0))) is cache
    assert m._as_triton_index_kv_cache([cache]) is cache

    stacked = torch.randn(2, 4, m.SPARSE_BLOCK_SIZE, 1, 8)
    assert m._as_triton_index_kv_cache(stacked).shape == (4, m.SPARSE_BLOCK_SIZE, 8)

    with_head_axis = torch.randn(4, m.SPARSE_BLOCK_SIZE, 1, 8)
    assert m._as_triton_index_kv_cache(with_head_axis).shape == (4, m.SPARSE_BLOCK_SIZE, 8)


def test_as_triton_index_kv_cache_rejects_unexpected_layouts():
    with pytest.raises(ValueError, match="head dim"):
        m._as_triton_index_kv_cache(torch.randn(4, m.SPARSE_BLOCK_SIZE, 2, 8))

    with pytest.raises(ValueError, match="ndim"):
        m._as_triton_index_kv_cache(torch.randn(1, 2, 3, 4, 5, 6))


def test_split_triton_main_kv_cache_accepts_tuple_and_packed_layouts():
    k_cache = torch.randn(3, m.SPARSE_BLOCK_SIZE, 2, 8)
    v_cache = torch.randn_like(k_cache)
    assert m._split_triton_main_kv_cache((k_cache, v_cache)) == (k_cache, v_cache)
    assert m._split_triton_main_kv_cache([k_cache, v_cache]) == (k_cache, v_cache)

    first_axis = torch.stack([k_cache, v_cache])
    split_k, split_v = m._split_triton_main_kv_cache(first_axis)
    assert torch.equal(split_k, k_cache)
    assert torch.equal(split_v, v_cache)

    second_axis = torch.stack([k_cache, v_cache], dim=1)
    split_k, split_v = m._split_triton_main_kv_cache(second_axis)
    assert torch.equal(split_k, k_cache)
    assert torch.equal(split_v, v_cache)


def test_split_triton_main_kv_cache_rejects_bad_layouts():
    with pytest.raises(ValueError, match="must contain"):
        m._split_triton_main_kv_cache((torch.empty(1),))

    with pytest.raises(ValueError, match="ndim"):
        m._split_triton_main_kv_cache(torch.empty(1, 2, 3))

    with pytest.raises(ValueError, match="shape"):
        m._split_triton_main_kv_cache(torch.empty(3, 4, 5, 6, 7))

    with pytest.raises(ValueError, match="split"):
        m._split_triton_main_kv_cache((torch.empty(1, 2, 3), torch.empty(1, 2, 3)))


def test_platform_helpers_cover_pdl_and_rocm_stage_cache(monkeypatch):
    monkeypatch.setattr(m, "current_platform", SimpleNamespace(device_name="npu", is_arch_support_pdl=lambda: True))
    assert m._is_arch_support_pdl() is False

    monkeypatch.setattr(m, "current_platform", SimpleNamespace(device_name="cuda", is_arch_support_pdl=lambda: True))
    assert m._is_arch_support_pdl() is True

    monkeypatch.setattr(m, "current_platform", SimpleNamespace(device_name="cuda"))
    assert m._is_arch_support_pdl() is False

    monkeypatch.setattr(m, "_SPARSE_ATTN_NUM_STAGES_KWARG", None)
    monkeypatch.setattr(m, "current_platform", SimpleNamespace(is_rocm=lambda: False))
    assert m._sparse_attn_num_stages_kwarg() == {}

    monkeypatch.setattr(m, "_SPARSE_ATTN_NUM_STAGES_KWARG", None)
    rocm_module = types.ModuleType("vllm.platforms.rocm")
    rocm_module.on_gfx942 = lambda: True
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm_module)
    monkeypatch.setattr(m, "current_platform", SimpleNamespace(is_rocm=lambda: True))
    assert m._sparse_attn_num_stages_kwarg() == {"num_stages": 1}


def test_prune_decode_score_configs_limits_split_k_chunks():
    configs = [SimpleNamespace(kwargs={"num_kv_chunks": value}) for value in (1, 2, 4, 8)]

    assert [c.kwargs["num_kv_chunks"] for c in m._prune_decode_score_configs(configs, {"num_reqs": 200})] == [1, 2]
    assert m._prune_decode_score_configs(configs, {"num_reqs": 1024}) == configs[:1]


def test_copy_topk_indices_reuses_converts_pads_and_slices_output():
    raw = torch.tensor([[[2, 1, 0], [3, 2, 1]]], dtype=torch.int64)
    reused = m._copy_topk_indices(raw, 3, None)
    assert reused.dtype == torch.int32
    assert reused.tolist() == raw.to(torch.int32).tolist()

    padded = m._copy_topk_indices(raw[..., :2], 4, None)
    assert padded.tolist() == [[[2, 1, -1, -1], [3, 2, -1, -1]]]

    out = torch.full((1, 4, 5), -99, dtype=torch.int32)
    sliced = m._copy_topk_indices(raw[..., :2], 4, out)
    assert sliced.shape == (1, 2, 4)
    assert sliced.tolist() == [[[2, 1, -1, -1], [3, 2, -1, -1]]]
    assert out[:, 2:].eq(-99).all()


def test_prefill_index_score_kernel_body_covers_full_boundary_and_empty_tile(monkeypatch):
    fake_tl = FakeTL()
    monkeypatch.setattr(m, "tl", fake_tl)
    kernel = _unwrap_triton_function(m._prefill_index_score_kernel)
    query = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    cache = torch.tensor([[[1.0, 0.0], [2.0, 0.0]], [[3.0, 0.0], [4.0, 0.0]]])
    score = torch.full((1, 2, 2), -99.0)
    block_table = torch.tensor([[0, 1]])
    cu_seqlens = torch.tensor([0, 2])
    seq_lens = torch.tensor([4])
    prefix_lens = torch.tensor([2])

    kernel(
        ptr(query),
        ptr(cache),
        ptr(score),
        ptr(block_table),
        ptr(cu_seqlens),
        ptr(seq_lens),
        ptr(prefix_lens),
        1,
        2,
        *query.stride(),
        *cache.stride(),
        *score.stride(),
        block_table.stride(0),
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_K=2,
    )
    assert torch.equal(score, torch.tensor([[[2.0, 3.0], [2.0, 4.0]]]))

    fake_tl.set_program_ids(1, 0)
    kernel(
        ptr(query),
        ptr(cache),
        ptr(score),
        ptr(block_table),
        ptr(cu_seqlens),
        ptr(seq_lens),
        ptr(prefix_lens),
        1,
        2,
        *query.stride(),
        *cache.stride(),
        *score.stride(),
        block_table.stride(0),
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_K=2,
    )
    assert torch.equal(score, torch.tensor([[[2.0, 3.0], [2.0, 4.0]]]))


def test_decode_score_and_topk_mask_kernel_bodies(monkeypatch):
    fake_tl = FakeTL()
    monkeypatch.setattr(m, "tl", fake_tl)
    score_kernel = _unwrap_triton_function(m._decode_index_score_kernel)
    query = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    cache = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    score = torch.full((1, 2, 1), -99.0)
    init_mask = torch.tensor([[True], [False]])
    local_mask = torch.tensor([[False], [True]])
    block_table = torch.tensor([[0]])
    seq_lens = torch.tensor([2])

    score_kernel(
        ptr(query),
        ptr(cache),
        ptr(score),
        ptr(init_mask),
        ptr(local_mask),
        ptr(block_table),
        ptr(seq_lens),
        1,
        2,
        1,
        2,
        *query.stride(),
        *cache.stride(),
        *score.stride(),
        *init_mask.stride(),
        block_table.stride(0),
        BLOCK_SIZE_K=2,
        BLOCK_SIZE_Q=2,
        num_kv_chunks=1,
        USE_PDL=True,
    )
    assert score[0, 0, 0] == pytest.approx(1e30)
    assert score[0, 1, 0] == pytest.approx(1e29)

    fake_tl.set_program_ids(0, 1)
    score_kernel(
        ptr(query),
        ptr(cache),
        ptr(score),
        ptr(init_mask),
        ptr(local_mask),
        ptr(block_table),
        ptr(seq_lens),
        1,
        2,
        1,
        2,
        *query.stride(),
        *cache.stride(),
        *score.stride(),
        *init_mask.stride(),
        block_table.stride(0),
        BLOCK_SIZE_K=2,
        BLOCK_SIZE_Q=2,
        num_kv_chunks=2,
        USE_PDL=False,
    )

    tail_kernel = _unwrap_triton_function(m._fill_decode_score_tail_kernel)
    tail_score = torch.zeros((1, 2, 4))
    short_seq_lens = torch.tensor([1])
    for query_id in (0, 1):
        fake_tl.set_program_ids(query_id, 0, 0)
        tail_kernel(ptr(tail_score), ptr(short_seq_lens), 2, 4, 2, 4, *tail_score.stride(), BLOCK_SIZE_K=4)
    assert torch.isneginf(tail_score[0, 0]).all()
    assert tail_score[0, 1, 0] == 0
    assert torch.isneginf(tail_score[0, 1, 1:]).all()

    mask_kernel = _unwrap_triton_function(m._mask_decode_topk_indices_kernel)
    indices = torch.tensor([[[0, 1, 2], [0, 3, -1]]], dtype=torch.int32)
    for query_id in (0, 1):
        fake_tl.set_program_ids(query_id, 0)
        mask_kernel(ptr(indices), ptr(short_seq_lens), 2, 3, 2, *indices.stride(), BLOCK_SIZE_T=4)
    assert indices.tolist() == [[[-1, -1, -1], [0, -1, -1]]]


def test_prefill_finalize_and_mask_kernel_bodies(monkeypatch):
    fake_tl = FakeTL()
    monkeypatch.setattr(m, "tl", fake_tl)
    finalize_kernel = _unwrap_triton_function(m._prepare_prefill_topk_scores_kernel)
    score = torch.zeros((1, 2, 4))
    cu_seqlens = torch.tensor([0, 2])
    prefix_lens = torch.tensor([1])

    finalize_kernel(
        ptr(score),
        ptr(cu_seqlens),
        ptr(prefix_lens),
        1,
        1,
        1,
        4,
        *score.stride(),
        2,
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_FORCE=2,
        BLOCK_SIZE_TAIL=2,
    )
    assert score[0, 0, 0] == pytest.approx(1e29)
    assert torch.isneginf(score[0, 0, 1:]).all()
    assert score[0, 1, 0] == pytest.approx(1e30)
    assert score[0, 1, 1] == pytest.approx(1e29)
    assert torch.isneginf(score[0, 1, 2:]).all()

    fake_tl.set_program_ids(1, 0)
    finalize_kernel(
        ptr(score),
        ptr(cu_seqlens),
        ptr(prefix_lens),
        1,
        0,
        0,
        4,
        *score.stride(),
        2,
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_FORCE=1,
        BLOCK_SIZE_TAIL=2,
    )

    mask_kernel = _unwrap_triton_function(m._mask_prefill_topk_indices_kernel)
    indices = torch.tensor([[[0, 1, 2], [1, 0, 3]]], dtype=torch.int32)
    fake_tl.set_program_ids(0, 0)
    mask_kernel(
        ptr(indices),
        ptr(cu_seqlens),
        ptr(prefix_lens),
        1,
        2,
        3,
        *indices.stride(),
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_T=4,
    )
    assert indices.tolist() == [[[0, -1, -1], [1, 0, -1]]]


def test_decode_score_mask_kernel_body(monkeypatch):
    fake_tl = FakeTL()
    monkeypatch.setattr(m, "tl", fake_tl)
    kernel = _unwrap_triton_function(m._prepare_decode_score_masks_kernel)
    init_mask = torch.zeros((2, 3), dtype=torch.bool)
    local_mask = torch.zeros_like(init_mask)
    seq_lens = torch.tensor([2])

    for query_id in (0, 1):
        fake_tl.set_program_ids(query_id, 0)
        kernel(
            ptr(init_mask),
            ptr(local_mask),
            ptr(seq_lens),
            2,
            3,
            2,
            3,
            1,
            1,
            *init_mask.stride(),
            BLOCK_SIZE_K=4,
        )
    assert init_mask.tolist() == [[True, False, False], [True, False, False]]
    assert local_mask.tolist() == [[True, False, False], [True, False, False]]

    fake_tl.set_program_ids(0, 1)
    kernel(
        ptr(init_mask),
        ptr(local_mask),
        ptr(seq_lens),
        2,
        3,
        2,
        3,
        1,
        1,
        *init_mask.stride(),
        BLOCK_SIZE_K=4,
    )


def test_minimax_m3_index_score_launches_scalar_and_vector_paths(monkeypatch):
    stubs = _all_launch_stubs(monkeypatch)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    seq_lens = torch.tensor([130, 260], dtype=torch.int32)
    prefix_lens = torch.tensor([0, 128], dtype=torch.int32)
    block_table = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int32)

    scalar_q = torch.randn(5, 2, 1)
    scalar_cache = torch.randn(3, m.SPARSE_BLOCK_SIZE, 1)
    scalar_score = m.minimax_m3_index_score(
        scalar_q,
        scalar_cache,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_query_len=3,
        max_seq_len=300,
        num_kv_heads=2,
    )

    assert scalar_score.shape == (2, 5, 16)
    assert stubs["_prefill_scalar_key_extrema_kernel"].calls
    assert stubs["_prefill_scalar_index_score_kernel"].calls

    vector_q = torch.randn(5, 2, 4)
    vector_cache = torch.randn(3, m.SPARSE_BLOCK_SIZE, 4)
    vector_score = m.minimax_m3_index_score(
        vector_q,
        vector_cache,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_query_len=3,
        max_seq_len=300,
        num_kv_heads=2,
        sm_scale=0.5,
    )

    assert vector_score.shape == (2, 5, 16)
    grid, args, kwargs = stubs["_prefill_index_score_kernel"].calls[-1]
    assert grid == (1, 4)
    assert args[7] == 2
    assert args[8] == 4
    assert kwargs["BLOCK_SIZE_Q"] == m.PREFILL_SCORE_QUERY_TILE_SIZE

    with pytest.raises(AssertionError, match="num_idx_heads"):
        m.minimax_m3_index_score(
            vector_q,
            vector_cache,
            block_table,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            max_query_len=3,
            max_seq_len=300,
            num_kv_heads=1,
        )


def test_minimax_m3_index_topk_launches_finalize_and_masks(monkeypatch):
    stubs = _all_launch_stubs(monkeypatch)
    score = torch.tensor(
        [
            [[0.1, 0.9, 0.2, -5.0], [3.0, 1.0, 2.0, 0.0], [0.0, 0.5, 0.4, 0.3]],
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 0.0, 2.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    out = torch.full((2, 5, 5), -9, dtype=torch.int32)
    topk = m.minimax_m3_index_topk(
        score,
        torch.tensor([0, 1, 3], dtype=torch.int32),
        torch.tensor([0, 128], dtype=torch.int32),
        max_query_len=2,
        topk=5,
        init_blocks=0,
        local_blocks=3,
        out=out,
    )

    assert topk.shape == (2, 3, 5)
    assert topk[0, 0].tolist()[:4] == [1, 2, 0, 3]
    assert topk[0, 0, 4].item() == -1
    assert stubs["_prepare_prefill_topk_scores_kernel"].calls[0][0] == (1, 4)
    assert stubs["_prepare_prefill_topk_scores_kernel"].calls[0][2]["BLOCK_SIZE_FORCE"] == 4
    assert stubs["_mask_prefill_topk_indices_kernel"].calls

    with pytest.raises(AssertionError):
        m.minimax_m3_index_topk(score, torch.tensor([0, 3]), torch.tensor([0]), 3, 0, 0, 0)


def test_minimax_m3_index_decode_launches_all_kernels_and_pads_topk(monkeypatch):
    stubs = _all_launch_stubs(monkeypatch)
    monkeypatch.setattr(m, "current_platform", SimpleNamespace(is_arch_support_pdl=lambda: True))
    idx_q = torch.randn(4, 2, 4)
    index_cache = torch.randn(3, m.SPARSE_BLOCK_SIZE, 4)
    block_table = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([130, 260], dtype=torch.int32)
    out = torch.full((2, 8, 5), -7, dtype=torch.int32)

    topk = m.minimax_m3_index_decode(
        idx_q,
        index_cache,
        block_table,
        seq_lens,
        max_seq_len=300,
        topk=5,
        init_blocks=1,
        local_blocks=2,
        num_kv_heads=2,
        decode_query_len=2,
        max_decode_query_len=4,
        out=out,
    )

    assert topk.shape == (2, 4, 5)
    assert topk[0, 0].tolist() == [2, 1, 0, -1, -1]
    assert stubs["_prepare_decode_score_masks_kernel"].calls[0][0] == (4, 16)
    decode_grid, decode_args, decode_kwargs = stubs["_decode_index_score_kernel"].calls[0]
    assert decode_grid({"num_kv_chunks": 8}) == (2, 8)
    assert decode_args[7:11] == (2, 4, 2, 2)
    assert decode_kwargs["USE_PDL"] is True
    assert decode_kwargs["launch_pdl"] is True
    assert stubs["_fill_decode_score_tail_kernel"].calls[0][0] == (4, 2, 8)
    assert stubs["_mask_decode_topk_indices_kernel"].calls[0][0] == (4, 2)

    with pytest.raises(AssertionError):
        m.minimax_m3_index_decode(
            idx_q,
            index_cache,
            block_table,
            seq_lens,
            300,
            1,
            0,
            0,
            2,
            decode_query_len=5,
            max_decode_query_len=4,
        )


def test_sparse_attn_launches_forward_kernel_for_split_tuple(monkeypatch):
    stubs = _all_launch_stubs(monkeypatch)
    monkeypatch.setattr(m, "_SPARSE_ATTN_NUM_STAGES_KWARG", {"num_stages": 1})
    q = torch.randn(3, 4, 8)
    k_cache = torch.randn(5, m.SPARSE_BLOCK_SIZE, 2, 8)
    v_cache = torch.randn_like(k_cache)
    output = torch.empty_like(q)

    m.minimax_m3_sparse_attn(
        q,
        (k_cache, v_cache),
        torch.tensor([[[0, 1], [1, -1], [0, 2]], [[0, 2], [1, 2], [2, -1]]], dtype=torch.int32),
        torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int32),
        torch.tensor([0, 1, 3], dtype=torch.int32),
        torch.tensor([130, 260], dtype=torch.int32),
        torch.tensor([0, 128], dtype=torch.int32),
        max_query_len=2,
        num_kv_heads=2,
        sm_scale=0.125,
        output=output,
    )

    grid, args, kwargs = stubs["_gqa_sparse_fwd_kernel"].calls[0]
    assert grid == (2, 2, 2)
    assert args[6].shape == (3,)
    assert args[10] == 2
    assert args[11] == 2
    assert kwargs["USE_FP8"] is False
    assert kwargs["num_stages"] == 1


def test_sparse_attn_decode_launches_decode_and_merge_with_pdl(monkeypatch):
    stubs = _all_launch_stubs(monkeypatch)
    monkeypatch.setattr(m, "_sparse_attn_num_stages_kwarg", lambda: {})
    monkeypatch.setattr(m, "_is_arch_support_pdl", lambda: True)
    q = torch.randn(4, 4, 8)
    kv_cache = torch.randn(2, 5, m.SPARSE_BLOCK_SIZE, 2, 8)
    output = torch.empty_like(q)

    m.minimax_m3_sparse_attn_decode(
        q,
        kv_cache,
        torch.tensor(
            [
                [[0, 1, 2], [1, -1, -1], [0, 2, -1], [1, 2, 0]],
                [[0, 1, 2], [2, 1, 0], [0, -1, -1], [1, 0, -1]],
            ],
            dtype=torch.int32,
        ),
        torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int32),
        torch.tensor([130, 260], dtype=torch.int32),
        num_kv_heads=2,
        sm_scale=0.125,
        output=output,
        decode_query_len=2,
    )

    decode_grid, decode_args, decode_kwargs = stubs["_gqa_sparse_decode_kernel"].calls[0]
    merge_grid, merge_args, merge_kwargs = stubs["_merge_topk_attn_out_kernel"].calls[0]
    assert decode_grid == (8, 2)
    assert decode_args[8:13] == (3, 4, 2, 8, 3)
    assert decode_kwargs["NUM_TOPK_CHUNKS"] == 2
    assert decode_kwargs["USE_PDL"] is True
    assert decode_kwargs["launch_pdl"] is True
    assert merge_grid == (4, 4)
    assert merge_args[2] is output
    assert merge_kwargs["USE_PDL"] is True

    with pytest.raises(AssertionError):
        m.minimax_m3_sparse_attn_decode(
            q,
            kv_cache,
            torch.empty(2, 4, 3, dtype=torch.int32),
            torch.empty(2, 3, dtype=torch.int32),
            torch.tensor([1, 2, 3], dtype=torch.int32),
            2,
            0.125,
            output,
            2,
        )
