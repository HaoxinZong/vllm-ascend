# SparseAttentionScore950

Sparse Attention 算子：根据外部传入的 TopK block 索引（selectIdx）+ blockTable（逻辑->物理映射），从 blocked KV cache 中 gather 对应 KV blocks，执行 FlashAttention 计算。

外层 `sparse_attention_score_950` 目录用于 A5 构建选择；实际算子位于
`sparse_attention_score950` 子目录，使 CANN 的算子名、kernel 源目录和 binary key 保持一致。

## 接口

```python
torch_npu.npu_sparse_attention_score(
    query,          # [T, N, D], fp16/bf16/fp8
    key,            # [blockNum, blockSize, KVHead, D]
    value,          # [blockNum, blockSize, KVHead, D]
    select_idx,     # [KVHead, maxQSeqlen, TopK], int32
    block_table,    # [batch, maxBlocksPerBatch], int32
    *,
    select_num_idx=None,        # [KVHead, maxQSeqlen], int32
    actual_seq_lengths=None,    # list[int]
    actual_seq_lengths_kv=None, # list[int]
    num_key_value_heads=1,
    scale_value=1.0,
    block_size=128,
    top_k=16,
) -> Tensor  # [T, N, D]
```

## 约束

- 平台：Ascend 950
- blockSize = 128
- 支持 GQA（numHeads 必须能被 numKeyValueHeads 整除）
- fp8 输入需提供 dequant scale
