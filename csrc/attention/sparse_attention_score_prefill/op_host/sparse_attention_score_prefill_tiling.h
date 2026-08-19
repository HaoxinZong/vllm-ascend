/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef SPARSE_ATTENTION_SCORE_PREFILL_TILING_H
#define SPARSE_ATTENTION_SCORE_PREFILL_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

// Keep in sync with op_kernel/sparse_attention_score_prefill_tilingkey.h
constexpr uint64_t SASA_PREFILL_BASE_TILING = 20000;
constexpr uint64_t SASA_PREFILL_BF16_D128_TILING = 20001;
// innerPrecise==1: O_partial written/read as bf16 (Phase1 PV fixpipe F322BF16, Phase2
// regbase-cast bf16->fp32 before combine). Otherwise (e.g. innerPrecise==4) the fp32
// O_partial path is used byte-for-byte (SASA_PREFILL_BF16_D128_TILING).
constexpr uint64_t SASA_PREFILL_BF16_D128_INNER_LOW_TILING = 20002;
constexpr uint64_t SASA_PREFILL_FP8_D128_BF16_TILING = 20003;

BEGIN_TILING_DATA_DEF(SparseAttentionScorePrefillTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, groupSize);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, topK);
TILING_DATA_FIELD_DEF(uint32_t, totalQTokens);
TILING_DATA_FIELD_DEF(uint32_t, numKvBlocks);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, k2qNnzUpperBound);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNumP1);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNumP2);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, innerPrecise);
TILING_DATA_FIELD_DEF(uint64_t, accumOutSize);
TILING_DATA_FIELD_DEF(uint64_t, lseStatSize);
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseAttentionScorePrefill, SparseAttentionScorePrefillTilingData)

struct SparseAttentionScorePrefillCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
};

class SASAPrefillTiling {
public:
    SASAPrefillTiling() = default;
    ~SASAPrefillTiling() = default;

    ge::graphStatus GetTiling(gert::TilingContext *context,
                              SparseAttentionScorePrefillTilingData &tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext *context,
                                  SparseAttentionScorePrefillTilingData &tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext *context);
    ge::graphStatus ParseAttrs(gert::TilingContext *context);
    ge::graphStatus ParseInputTensors(gert::TilingContext *context);
    ge::graphStatus ParseSeqlens(gert::TilingContext *context);
    ge::graphStatus CalculateReverseIndexMeta(gert::TilingContext *context);
    ge::graphStatus CalculateTaskSplit(gert::TilingContext *context);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *context);
    uint64_t GenerateTilingKey();
    ge::graphStatus FillTilingData(gert::TilingContext *context);

private:
    uint32_t batch_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t embeddingSize_ = 0;
    uint32_t blockSize_ = 128;
    uint32_t topK_ = 8;
    uint32_t totalQTokens_ = 0;
    uint32_t numKvBlocks_ = 0;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t k2qNnzUpperBound_ = 0;
    float scaleValue_ = 0.0f;
    uint32_t innerPrecise_ = 0;

    const int32_t *qSeqLenList_ = nullptr;
    const int32_t *kvSeqLenList_ = nullptr;

    uint64_t workSpaceSize_ = 0;
    uint64_t accumOutSize_ = 0;
    uint64_t lseStatSize_ = 0;

    uint32_t blockDim_ = 20;
    uint32_t aivNum_ = 0;
    uint32_t aicNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t libapiSize_ = 0;

    ge::DataType dataType_ = ge::DT_BF16;

    SparseAttentionScorePrefillTilingData *tilingData_ = nullptr;
};

}  // namespace optiling

#endif  // SPARSE_ATTENTION_SCORE_PREFILL_TILING_H
