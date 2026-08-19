/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VF_CAST_TRANSDATA_DECONFLICT_PREFILL_H
#define VF_CAST_TRANSDATA_DECONFLICT_PREFILL_H

#include "kernel_tensor.h"

namespace AscendC
{
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp16Odd = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp16Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T1, typename T>
__simd_vf__ inline void CastTransdataDeconflictVFbf16(const uint32_t fullExeSize, uint64_t srcLocalInt, uint64_t dstLocalInt,
                                                         uint32_t blockStride, uint32_t repeatStride, uint32_t srcM)
{
    RegTensor<T> vregSrcEven;
    RegTensor<T> vregSrcOdd;
    RegTensor<bfloat16_t> vregCastEven;
    RegTensor<bfloat16_t> vregCastOdd;
    RegTensor<bfloat16_t> vregCastRes;
    MaskReg pregFullExe = CreateMask<T1, MaskPattern::ALL>();

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vregSrcEven, vregSrcOdd, ((__ubuf__ T *&)srcLocalInt), fullExeSize);
        Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
        Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
        Or((RegTensor<uint16_t> &)vregCastRes, (RegTensor<uint16_t> &)vregCastEven,
            (RegTensor<uint16_t> &)vregCastOdd, pregFullExe);
        StoreAlign<T1, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T1 *&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregFullExe);
    }
}

template <typename T1, typename T, uint32_t srcN>
__aicore__ inline void CastTransdataDeconflict(const LocalTensor<T1> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &selrIndexesTensor, uint32_t srcM)
{
    (void)selrIndexesTensor;
    const uint32_t blockSize = 32;
    const uint32_t blockN = blockSize / sizeof(T1);
    const uint32_t fullExeSize = srcN;
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint32_t blockStride = (srcM * blockN) * sizeof(T1) / blockSize + 1;
    uint32_t repeatStride = 1;

    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        CastTransdataDeconflictVFbf16<T1, T>(fullExeSize, srcLocalInt, dstLocalInt, blockStride, repeatStride, srcM);
    }
}
#else
template <typename T1, typename T, uint32_t srcN>
__aicore__ inline void CastTransdataDeconflict(const LocalTensor<T1> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &selrIndexesTensor, uint32_t srcM)
{
    (void)dstTensor;
    (void)srcTensor;
    (void)selrIndexesTensor;
    (void)srcM;
}
#endif
} // namespace AscendC

#endif // VF_CAST_TRANSDATA_DECONFLICT_PREFILL_H
