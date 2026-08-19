/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "sparse_attention_score_prefill.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SparseAttentionScorePrefill);

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
    aclOpExecutor *executor)
{
    L0_DFX(SparseAttentionScorePrefill, query, key, value, blockTable,
           k2qRowPtr, k2qQIndices, k2qSlotIndices,
           actualSeqLengthsOptional, actualSeqLengthsKvOptional,
           numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise);

    DataType attentionOutDtype = query->GetDataType() == DataType::DT_FLOAT8_E4M3FN ?
        DataType::DT_BF16 : query->GetDataType();
    auto attentionOutTensor = executor->AllocTensor(attentionOutDtype,
        Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(SparseAttentionScorePrefill,
                           OP_INPUT(query, key, value, blockTable,
                                    k2qRowPtr, k2qQIndices, k2qSlotIndices,
                                    actualSeqLengthsOptional, actualSeqLengthsKvOptional),
                           OP_OUTPUT(attentionOutTensor),
                           OP_ATTR(static_cast<int64_t>(numKeyValueHeads),
                                   static_cast<float>(scaleValue),
                                   static_cast<int64_t>(blockSize),
                                   static_cast<int64_t>(topK),
                                   static_cast<int64_t>(innerPrecise)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "SparseAttentionScorePrefill infer shape failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(SparseAttentionScorePrefill,
                                OP_INPUT(query, key, value, blockTable,
                                         k2qRowPtr, k2qQIndices, k2qSlotIndices,
                                         actualSeqLengthsOptional, actualSeqLengthsKvOptional),
                                OP_OUTPUT(attentionOutTensor),
                                OP_ATTR(static_cast<int64_t>(numKeyValueHeads),
                                        static_cast<float>(scaleValue),
                                        static_cast<int64_t>(blockSize),
                                        static_cast<int64_t>(topK),
                                        static_cast<int64_t>(innerPrecise)));

    return attentionOutTensor;
}

} // namespace l0op
