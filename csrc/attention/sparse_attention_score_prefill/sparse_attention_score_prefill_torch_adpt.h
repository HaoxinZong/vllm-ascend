#ifndef SPARSE_ATTENTION_SCORE_PREFILL_TORCH_ADPT_H
#define SPARSE_ATTENTION_SCORE_PREFILL_TORCH_ADPT_H

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <acl/acl.h>

namespace vllm_ascend {

namespace sparse_attention_score_prefill {
namespace {

void CheckInt32Tensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kInt,
                name, " dtype must be int32, got ", tensor.scalar_type());
}

}  // namespace
}  // namespace sparse_attention_score_prefill

at::Tensor npu_sparse_attention_score_prefill(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &block_table,
    const at::Tensor &k2q_row_ptr,
    const at::Tensor &k2q_q_indices,
    const at::Tensor &k2q_slot_indices,
    int64_t num_key_value_heads, double scale_value, int64_t block_size,
    int64_t top_k, int64_t inner_precise,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv
    )
{
    TORCH_CHECK(query.dim() == 3,
                "query only supports TND layout [T,N,D], but got dim ", query.dim());
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 ||
                    query.scalar_type() == at::kFloat8_e4m3fn,
                "query dtype must be bfloat16 or float8_e4m3fn, got ",
                query.scalar_type());
    TORCH_CHECK(key.scalar_type() == query.scalar_type() &&
                    value.scalar_type() == query.scalar_type(),
                "query, key and value must have the same dtype");
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }

    sparse_attention_score_prefill::CheckInt32Tensor(block_table, "block_table");
    sparse_attention_score_prefill::CheckInt32Tensor(k2q_row_ptr, "k2q_row_ptr");
    sparse_attention_score_prefill::CheckInt32Tensor(k2q_q_indices, "k2q_q_indices");
    sparse_attention_score_prefill::CheckInt32Tensor(k2q_slot_indices, "k2q_slot_indices");
    TORCH_CHECK(actual_seq_lengths.has_value() && actual_seq_lengths.value().defined(),
                "actual_seq_lengths must be provided");
    TORCH_CHECK(actual_seq_lengths_kv.has_value() && actual_seq_lengths_kv.value().defined(),
                "actual_seq_lengths_kv must be provided");
    sparse_attention_score_prefill::CheckInt32Tensor(
        actual_seq_lengths.value(), "actual_seq_lengths");
    sparse_attention_score_prefill::CheckInt32Tensor(
        actual_seq_lengths_kv.value(), "actual_seq_lengths_kv");

    at::ScalarType out_dtype = query.scalar_type() == at::kFloat8_e4m3fn
                                   ? at::kBFloat16
                                   : query.scalar_type();
    at::Tensor output = at::empty(query.sizes(), query.options().dtype(out_dtype));

    EXEC_NPU_CMD(
        aclnnSparseAttentionScorePrefill,
        query,
        key,
        value,
        block_table,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_slot_indices,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        num_key_value_heads,
        scale_value,
        block_size,
        top_k,
        inner_precise,
        output
    );

    return output;
}
}
#endif
