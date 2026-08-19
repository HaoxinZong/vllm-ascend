/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef SPARSE_ATTENTION_SCORE_PREFILL_L0OP_H_
#define SPARSE_ATTENTION_SCORE_PREFILL_L0OP_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor *SparseAttentionScorePrefill(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockTable,
    const aclTensor *k2qRowPtr,
    const aclTensor *k2qQIndices,
    const aclTensor *k2qSlotIndices,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    int64_t numKeyValueHeads,
    double scaleValue,
    int64_t blockSize,
    int64_t topK,
    int64_t innerPrecise,
    aclOpExecutor *executor);

} // namespace l0op

#endif  // SPARSE_ATTENTION_SCORE_PREFILL_L0OP_H_
