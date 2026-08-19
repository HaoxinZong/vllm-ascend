/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "sparse_attention_score_prefill_tiling.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <vector>
#include "log/log.h"
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "log/log.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int BLOCK_TABLE_INDEX = 3;
constexpr int K2Q_ROW_PTR_INDEX = 4;
constexpr int K2Q_Q_INDICES_INDEX = 5;
constexpr int K2Q_SLOT_INDICES_INDEX = 6;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 7;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 8;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;

constexpr int BLOCKED_KV_DIM_BLOCK_NUM = 0;
constexpr int BLOCKED_KV_DIM_BLOCK_SIZE = 1;
constexpr int BLOCKED_KV_DIM_KV_HEAD = 2;
constexpr int BLOCKED_KV_DIM_D = 3;

constexpr int ATTR_NUM_KV_HEADS_INDEX = 0;
constexpr int ATTR_SCALE_VALUE_INDEX = 1;
constexpr int ATTR_BLOCK_SIZE_INDEX = 2;
constexpr int ATTR_TOP_K_INDEX = 3;
constexpr int ATTR_INNER_PRECISE_INDEX = 4;

constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

namespace optiling {

static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) return 0;
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (align != 0) ? ((value + align - 1) / align * align) : value;
}

ge::graphStatus SASAPrefillTiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    blockDim_ = aicNum_;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    kvHeads_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_NUM_KV_HEADS_INDEX));
    scaleValue_ = static_cast<float>(*attrs->GetFloat(ATTR_SCALE_VALUE_INDEX));
    blockSize_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_BLOCK_SIZE_INDEX));
    topK_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_TOP_K_INDEX));
    innerPrecise_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_INNER_PRECISE_INDEX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::ParseInputTensors(gert::TilingContext *context)
{
    // printf("[SASAPrefillTiling] Enter ParseInputTensors\n");
    // Query: [total_q_tokens, num_q_heads, D]
    auto qShape = context->GetInputShape(QUERY_INDEX);
    if (qShape == nullptr) { printf("[SASAPrefillTiling] qShape is null!\n"); return ge::GRAPH_FAILED; }
    totalQTokens_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(TND_DIM_T));
    numHeads_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(TND_DIM_N));
    embeddingSize_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(TND_DIM_D));
    // printf("[SASAPrefillTiling] totalQTokens=%u numHeads=%u D=%u\n", totalQTokens_, numHeads_, embeddingSize_);

    // KV: [num_physical_blocks, blockSize, kvHeads, D]
    auto kShape = context->GetInputShape(KEY_INDEX);
    if (kShape == nullptr) { printf("[SASAPrefillTiling] kShape is null!\n"); return ge::GRAPH_FAILED; }
    if (kvHeads_ == 0) {
        kvHeads_ = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BLOCKED_KV_DIM_KV_HEAD));
    }
    groupSize_ = (kvHeads_ > 0) ? (numHeads_ / kvHeads_) : 1;
    // printf("[SASAPrefillTiling] kvHeads=%u groupSize=%u\n", kvHeads_, groupSize_);

    // block_table: [batch_size, max_num_blocks]
    auto btShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    if (btShape == nullptr) { printf("[SASAPrefillTiling] btShape is null!\n"); return ge::GRAPH_FAILED; }
    batch_ = static_cast<uint32_t>(btShape->GetStorageShape().GetDim(0));
    maxBlocksPerBatch_ = static_cast<uint32_t>(btShape->GetStorageShape().GetDim(1));
    // printf("[SASAPrefillTiling] batch=%u maxBlocksPerBatch=%u\n", batch_, maxBlocksPerBatch_);

    // Data type
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(QUERY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(KEY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(VALUE_INDEX));
    auto qTensor = context->GetInputDesc(QUERY_INDEX);
    dataType_ = qTensor->GetDataType();
    auto kTensor = context->GetInputDesc(KEY_INDEX);
    auto vTensor = context->GetInputDesc(VALUE_INDEX);
    if (kTensor->GetDataType() != dataType_ || vTensor->GetDataType() != dataType_) {
        OP_LOGE(context->GetNodeName(),
                "SparseAttentionScorePrefill requires query/key/value to share the same dtype.");
        return ge::GRAPH_FAILED;
    }

    // printf("[SASAPrefillTiling] ParseInputTensors done\n");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::ParseSeqlens(gert::TilingContext *context)
{
    // printf("[SASAPrefillTiling] Enter ParseSeqlens\n");
    // fflush(stdout);

    auto k2qRowPtrShape = context->GetInputShape(K2Q_ROW_PTR_INDEX);
    if (k2qRowPtrShape == nullptr) {
        printf("[SASAPrefillTiling] k2qRowPtr shape is null!\n");
        return ge::GRAPH_FAILED;
    }
    // k2qRowPtr is a plain int32 tensor Input (shape [kvHeads, totalRows+1] or flat 1D — either,
    // row-major so the linear layout is identical). flatSize = total element count; derive
    // totalRows = flatSize / kvHeads - 1. Shape-only: no host value-read.
    int64_t flatSize = k2qRowPtrShape->GetStorageShape().GetShapeSize();
    if (flatSize <= 0 || kvHeads_ == 0U ||
        static_cast<uint64_t>(flatSize) % static_cast<uint64_t>(kvHeads_) != 0U) {
        printf("[SASAPrefillTiling] k2qRowPtr flat size %lld not divisible by kvHeads %u!\n",
               static_cast<long long>(flatSize), kvHeads_);
        return ge::GRAPH_FAILED;
    }
    uint64_t perHead = static_cast<uint64_t>(flatSize) / static_cast<uint64_t>(kvHeads_);
    if (perHead == 0U) {
        printf("[SASAPrefillTiling] k2qRowPtr per-head size 0 (flatSize/kvHeads < 1)!\n");
        return ge::GRAPH_FAILED;
    }
    numKvBlocks_ = static_cast<uint32_t>(perHead - 1U);
    // numKvBlocks_ is derived from the k2qRowPtr Input SHAPE only (flatSize/kvHeads - 1). The
    // strided row-outer partition in CalculateTaskSplit needs NO row_ptr values on host (shape-
    // only); the kernel reads csrStart/csrEnd from the runtime GM tensor via GetValue.
    // fflush(stdout);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::CalculateReverseIndexMeta(gert::TilingContext *context)
{
    // k2qQIndices / k2qSlotIndices: [kvHeads, totalQ * topK]
    auto qIndicesShape = context->GetInputShape(K2Q_Q_INDICES_INDEX);
    if (qIndicesShape != nullptr && qIndicesShape->GetStorageShape().GetDimNum() >= 2) {
        k2qNnzUpperBound_ = static_cast<uint32_t>(qIndicesShape->GetStorageShape().GetDim(1));
    } else {
        k2qNnzUpperBound_ = totalQTokens_ * topK_;
    }
    if (k2qNnzUpperBound_ == 0) {
        k2qNnzUpperBound_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::CalculateTaskSplit(gert::TilingContext *context)
{
    // STRIDED row-outer head-split partition (Phase1). taskIdx = packedRow*kvHeads + kvHeadIdx
    // (row-outer); core coreIdx handles taskIdx in {coreIdx, coreIdx+coreNum, ...}. A row's
    // kvHeads heads are co-hot -> they land on kvHeads DIFFERENT cores, balancing the head
    // dimension for free (residual imbalance is row-level only; sim 1.44x on step3510, under the
    // Phase2 Vec bottleneck). Shape-only: NO row_ptr value read on host (drops the GetData + nnz
    // walk overhead the IntArray/const-fold path added). The kernel derives coreNum from
    // GetBlockNum(); no per-core boundary arrays are serialized.
    (void)context;
    uint32_t totalTasks = numKvBlocks_ * kvHeads_;        // Phase1 row-outer task count
    uint32_t totalTaskP2 = totalQTokens_ * kvHeads_;     // Phase2 strided task count
    // blockDim covers BOTH phases (strided): cores with no task simply skip the loop.
    uint32_t totalTaskMax = std::max(totalTasks, totalTaskP2);
    blockDim_ = (totalTaskMax == 0U) ? 1U : std::min(totalTaskMax, aicNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::CalculateWorkSpace(gert::TilingContext *context)
{
    (void)context;
    uint64_t slotOElems = static_cast<uint64_t>(groupSize_) * embeddingSize_;
    uint64_t slotStatElems = static_cast<uint64_t>(groupSize_);
    uint64_t taskSlots = static_cast<uint64_t>(totalQTokens_) * kvHeads_ * topK_;

    // accumOut | softmaxMax | softmaxSum (separate buffers).
    // O_partial: [totalQ, kvHeads, topK, groupSize, D] fp32, or bf16 when innerPrecise==1
    //   (F322BF16 fixpipe + Phase2 regbase cast). max/sum: [totalQ, kvHeads, topK, groupSize]
    //   fp32 (compact per slot) regardless.
    accumOutSize_ = taskSlots * slotOElems;
    lseStatSize_ = taskSlots * slotStatElems;
    // innerPrecise==1 halves the O_partial buffer (bf16=2B vs fp32=4B); lse stats stay fp32.
    // fp8 path (key 20003) always uses fp32 O_partial regardless of innerPrecise.
    const bool useBf16OPartial =
        (dataType_ != ge::DT_FLOAT8_E4M3FN) && (innerPrecise_ == 1U);
    uint64_t accumOutBytes = accumOutSize_ *
        (useBf16OPartial ? sizeof(uint16_t) : sizeof(float));
    uint64_t userWorkspaceSize = accumOutBytes + (lseStatSize_ * 2U) * sizeof(float);
    workSpaceSize_ = libapiSize_ + userWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

uint64_t SASAPrefillTiling::GenerateTilingKey()
{
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        return SASA_PREFILL_FP8_D128_BF16_TILING;
    }
    if (innerPrecise_ == 1U) {
        return SASA_PREFILL_BF16_D128_INNER_LOW_TILING;
    }
    return SASA_PREFILL_BF16_D128_TILING;
}

ge::graphStatus SASAPrefillTiling::FillTilingData(gert::TilingContext *context)
{
    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_groupSize(groupSize_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockSize(blockSize_);
    tilingData_->set_topK(topK_);
    tilingData_->set_totalQTokens(totalQTokens_);
    tilingData_->set_numKvBlocks(numKvBlocks_);
    tilingData_->set_maxBlocksPerBatch(maxBlocksPerBatch_);
    tilingData_->set_k2qNnzUpperBound(k2qNnzUpperBound_);
    tilingData_->set_totalTaskNumP1(numKvBlocks_ * kvHeads_);
    tilingData_->set_totalTaskNumP2(totalQTokens_ * kvHeads_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_innerPrecise(innerPrecise_);
    tilingData_->set_accumOutSize(accumOutSize_);
    tilingData_->set_lseStatSize(lseStatSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        if (embeddingSize_ != 128U || blockSize_ != 128U) {
            OP_LOGE(context->GetNodeName(),
                    "SparseAttentionScorePrefill fp8 path requires D=128 and blockSize=128.");
            return ge::GRAPH_FAILED;
        }
    }
    uint64_t tilingKeyVal = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKeyVal);
    context->SetTilingKey(tilingKeyVal);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::GetTiling(gert::TilingContext *context,
    SparseAttentionScorePrefillTilingData &tilingData)
{
    tilingData_ = &tilingData;

    auto ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = ParseAttrs(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = ParseInputTensors(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = ParseSeqlens(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = CalculateReverseIndexMeta(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::SetTilingData(gert::TilingContext *context,
    SparseAttentionScorePrefillTilingData &tilingData)
{
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);

    // Set block dim computed from both Phase1 and Phase2 task counts.
    context->SetBlockDim(blockDim_);

    // Set workspace size
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(workSpaceSize_);

    // Set tiling data
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingSparseAttentionScorePrefill(gert::TilingContext* context)
{
    // printf("[SASAPrefillTiling] ===== TilingSparseAttentionScorePrefill ENTER =====\n");
    if (context == nullptr) { printf("[SASAPrefillTiling] context is null!\n"); return ge::GRAPH_FAILED; }
    SparseAttentionScorePrefillTilingData tilingData;
    SASAPrefillTiling tiling;
    if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        // printf("[SASAPrefillTiling] GetTiling success, calling SetTilingData\n");
        tiling.SetTilingData(context, tilingData);
        // printf("[SASAPrefillTiling] ===== TilingSparseAttentionScorePrefill DONE =====\n");
        return ge::GRAPH_SUCCESS;
    }
    printf("[SASAPrefillTiling] GetTiling FAILED!\n");
    OP_LOGE(context->GetNodeName(), "GetTiling failed");
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForSparseAttentionScorePrefill(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseAttentionScorePrefill)
    .Tiling(TilingSparseAttentionScorePrefill)
    .TilingParse<SparseAttentionScorePrefillCompileInfo>(TilingPrepareForSparseAttentionScorePrefill);

}  // namespace optiling
