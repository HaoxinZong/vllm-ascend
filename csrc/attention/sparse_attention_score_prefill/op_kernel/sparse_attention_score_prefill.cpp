/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "kernel_operator.h"
#include "sparse_attention_score_prefill_tilingkey.h"
#include "sparse_attention_score_prefill_kernel_interface.cpp"

extern "C" __global__ __aicore__ void sparse_attention_score_prefill(
    __gm__ uint8_t* query,
    __gm__ uint8_t* key,
    __gm__ uint8_t* value,
    __gm__ uint8_t* blockTable,
    __gm__ uint8_t* k2qRowPtr,
    __gm__ uint8_t* k2qQIndices,
    __gm__ uint8_t* k2qSlotIndices,
    __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKv,
    __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* workspace,
    __gm__ uint8_t* tiling)
{
    if (TILING_KEY_VAR >= SASA_PREFILL_BASE_TILING) {
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 310)
        TILING_KEY_IS(SASA_PREFILL_BF16_D128_TILING);
#if TILING_KEY_VAR == SASA_PREFILL_BF16_D128_TILING
        SasaPrefillInferIntf<bfloat16_t, bfloat16_t, float>(
            query, key, value, blockTable,
            k2qRowPtr, k2qQIndices, k2qSlotIndices,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, user, tiling);
#endif
        TILING_KEY_IS(SASA_PREFILL_BF16_D128_INNER_LOW_TILING);
#if TILING_KEY_VAR == SASA_PREFILL_BF16_D128_INNER_LOW_TILING
        // innerPrecise==1: REDtype(bf16) => PV fixpipe F322BF16 + Phase2 regbase cast
        // (bf16 O_partial). fp32 path (above) stays byte-identical.
        SasaPrefillInferIntf<bfloat16_t, bfloat16_t, bfloat16_t>(
            query, key, value, blockTable,
            k2qRowPtr, k2qQIndices, k2qSlotIndices,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, user, tiling);
#endif
        TILING_KEY_IS(SASA_PREFILL_FP8_D128_BF16_TILING);
#if TILING_KEY_VAR == SASA_PREFILL_FP8_D128_BF16_TILING
        SasaPrefillInferIntf<fp8_e4m3fn_t, bfloat16_t, float>(
            query, key, value, blockTable,
            k2qRowPtr, k2qQIndices, k2qSlotIndices,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, user, tiling);
#endif
#endif
    }
}
