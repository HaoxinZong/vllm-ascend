/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr uint32_t QUERY_INDEX = 0;
static constexpr uint32_t ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t TND_DIM_NUM = 3;
static constexpr int32_t UNKNOWN_DIMS = -2;

static ge::graphStatus InferShapeSparseAttentionScorePrefill(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("SparseAttentionScorePrefill", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    gert::Shape *attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);

    if (queryShape->GetDimNum() == 1 && queryShape->GetDim(0) == UNKNOWN_DIMS) {
        attentionOutShape->SetDimNum(1);
        (*attentionOutShape)[0] = UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    if (queryShape->GetDimNum() != TND_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "SparseAttentionScorePrefill only supports TND layout, queryDims(%zu) must be 3!",
                queryShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    *attentionOutShape = *queryShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSparseAttentionScorePrefill(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(QUERY_INDEX);
    if (dtype == ge::DT_FLOAT8_E4M3FN) {
        context->SetOutputDataType(ATTENTION_OUT_INDEX, ge::DT_BF16);
    } else {
        context->SetOutputDataType(ATTENTION_OUT_INDEX, dtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseAttentionScorePrefill)
    .InferShape(InferShapeSparseAttentionScorePrefill)
    .InferDataType(InferDataTypeSparseAttentionScorePrefill);

}  // namespace ops
