/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Block-level online softmax epilogue for prefill Phase1 (Arch35 / Atlas A5).
 *
 * Batch: ... -> CastTransdataDeconflict(ubP zN) -> rowMax/rowSum -> CopyUbZNToL1
 * Each AIV owns its local UB rows (QK FixPipe already split S across vectors).
 */
#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_PREFILL_ARCH35_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_PREFILL_ARCH35_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/cross_core_sync.hpp"
#include "../../../attn_infra/detail/alignment.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/layout/layout.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../arch35/vector_api/vf_cast_transdata_deconflict.h"
#include "kernel_operator.h"

namespace NpuArch::Epilogue::Block {

template <class InDtype>
class BlockEpilogue<
    EpilogueOnlineSoftmaxPrefillArch35,
    InDtype>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxPrefillArch35;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementS = float;
    using ElementP = InDtype;

    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t REDUCE_SCRATCH_OFFSET = 64;
    static constexpr uint32_t TV_UB_OFFSET = 128;
    static constexpr uint32_t ELE_NUM_PER_C0 = 32 / sizeof(ElementP);
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    // Must match blockSize (128) and CastTransdataDeconflict srcN template.
    static constexpr uint32_t P_CAST_COLS = 128;
    static constexpr uint32_t FRACTAL_NZ_C0 = 32 / sizeof(ElementP);
    static constexpr uint32_t INPUT_BLOCK_NUM = 32 / sizeof(ElementP);
    static constexpr float MIN_VALUE_FP32 = -3.389531390315715675e+38f;
    static constexpr float NEG_INF_ROW_MAX = -3.4028235e38f;

    __aicore__ inline
    BlockEpilogue() : scale_(0.0f) {}

    __aicore__ inline
    void Init(AscendC::LocalTensor<float> &ubTmpBuf,
              AscendC::LocalTensor<float> &ubRowMaxOut,
              AscendC::LocalTensor<float> &ubRowSumOut)
    {
        tmpBuf_ = ubTmpBuf;
        rowMaxBuf_ = ubRowMaxOut;
        rowSumBuf_ = ubRowSumOut;
    }

    __aicore__ inline
    void SetScale(float s) { scale_ = s; }

    // CastTransdataDeconflict writes zN to ubP; CopyUbZNToL1 matches FAG grad CopyUB2L1 layout.
    template <class TensorPL1>
    __aicore__ inline
    void CopyUbZNToL1(TensorPL1 const &l1PTile,
                      AscendC::LocalTensor<ElementP> const &ubPZn,
                      uint32_t numRows, uint32_t rowsAligned, uint32_t groupSize,
                      uint32_t validSize)
    {
        auto dstOffset = l1PTile.layout()(l1PTile.coord());
        AscendC::DataCopyParams params;
        params.blockCount = RoundUp(validSize, C0_NUM_PER_FRACTAL) / FRACTAL_NZ_C0;
        params.blockLen = static_cast<uint16_t>(numRows * FRACTAL_NZ_C0 / INPUT_BLOCK_NUM);
        params.srcStride = static_cast<uint16_t>(
            (rowsAligned + 1U - numRows) * FRACTAL_NZ_C0 / INPUT_BLOCK_NUM);
        params.dstStride = static_cast<uint16_t>(
            (RoundUp(groupSize, C0_NUM_PER_FRACTAL) - numRows) * FRACTAL_NZ_C0 / INPUT_BLOCK_NUM);
        AscendC::DataCopy(l1PTile.data()[dstOffset], ubPZn, params);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    template <uint32_t SRCN>
    __aicore__ inline
    void CastExpToUbZN(AscendC::LocalTensor<float> &expUb,
                       AscendC::LocalTensor<ElementP> &pUbZN,
                       uint32_t rowsAligned)
    {
        AscendC::LocalTensor<uint8_t> selDummy;
        AscendC::CastTransdataDeconflict<ElementP, float, SRCN>(pUbZN, expUb, selDummy, rowsAligned);
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetVecCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitVecCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <class TensorPL1>
    __aicore__ inline
    void operator()(AscendC::LocalTensor<float> &ubSBuf,
                    AscendC::LocalTensor<InDtype> &ubPBuf,
                    TensorPL1 &l1PTensor,
                    uint32_t groupSize,
                    uint32_t numRows,
                    uint32_t validSize,
                    uint32_t sColsAligned,
                    Arch::CrossCoreFlag &mm1ToSmFlag,
                    Arch::CrossCoreFlag &smToMm2Flag)
    {
        uint32_t ubSBufId = mm1ToSmFlag.id;
        uint32_t mCopyOffset = RoundUp(groupSize, FLOAT_BLOCK_SIZE) / 2;
        uint32_t m = groupSize < mCopyOffset ? groupSize : mCopyOffset;
        m = AscendC::GetSubBlockIdx() == 0 ? m : groupSize - m;

        WaitVecCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        if (numRows == 0U || m == 0U) {
            SetVecCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitVecCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetVecCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }
        //AscendC::DumpTensor(ubSBuf, 123, 128);

        uint32_t numRowsRound = RoundUp(numRows, FLOAT_BLOCK_SIZE);
        AscendC::Muls(
            ubSBuf, ubSBuf, scale_,
            static_cast<int32_t>(numRows * sColsAligned));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> tvUb = tmpBuf_[TV_UB_OFFSET];
        AscendC::LocalTensor<float> tvScratch = tmpBuf_[TV_UB_OFFSET + REDUCE_SCRATCH_OFFSET];

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2U);

        CalcRowMaxTail(ubSBuf, rowMaxBuf_, tvUb, tvScratch,
            numRows, numRowsRound, validSize, sColsAligned);
        SubtractRowMaxAndExp(ubSBuf, rowMaxBuf_, tvUb,
            numRows, numRowsRound, validSize, sColsAligned);
        CalcRowSumTail(ubSBuf, rowSumBuf_, tvUb, tvScratch,
            numRows, numRowsRound, validSize, sColsAligned);

        uint32_t n = validSize;
        auto l1PTile = GetTile(l1PTensor,
            tla::MakeCoord(AscendC::GetSubBlockIdx() * mCopyOffset, 0), tla::MakeShape(m, n));

        uint32_t rowsAligned = RoundUp(numRows, C0_NUM_PER_FRACTAL);
        //AscendC::DumpTensor(ubSBuf, 456, 128);
        if constexpr (P_CAST_COLS == 128) {
            CastExpToUbZN<P_CAST_COLS>(ubSBuf, ubPBuf, rowsAligned);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetVecCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        WaitVecCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        //AscendC::DumpTensor(ubPBuf, 123, 128);
        CopyUbZNToL1(l1PTile, ubPBuf, numRows, rowsAligned, groupSize, validSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2U);
        SetVecCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
    }

private:
    __aicore__ inline
    void SetVecMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < static_cast<int64_t>(temp); i++) {
            mask |= one << i;
        }
        if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, static_cast<uint64_t>(-1));
        } else if (len > 0) {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        } else {
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    }

    __aicore__ inline
    void SetBlockReduceMask(int32_t len)
    {
        if (len > 8 || len < 1) {
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            return;
        }
        uint64_t subMask = (static_cast<uint64_t>(1) << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask +
            (subMask << 56) + (subMask << 40) + (subMask << 24) + (subMask << 8);
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    __aicore__ inline
    void CalcRowMaxTail(const AscendC::LocalTensor<float> &srcUb,
                        const AscendC::LocalTensor<float> &rowmaxUb,
                        const AscendC::LocalTensor<float> &tvUb,
                        const AscendC::LocalTensor<float> &tvScratch,
                        uint32_t numRows, uint32_t numRowsRound,
                        uint32_t numElems, uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceMax<float, false>(
                tvUb, srcUb, numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float, false>(
                rowmaxUb, tvUb,
                CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t colIdx = 1; colIdx < static_cast<uint64_t>(numElems) / FLOAT_VECTOR_SIZE; ++colIdx) {
                AscendC::BlockReduceMax<float, false>(
                    tvUb, srcUb[colIdx * FLOAT_VECTOR_SIZE],
                    numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceMax<float, false>(
                    tvScratch, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Max<float, false>(
                    rowmaxUb, rowmaxUb, tvScratch,
                    static_cast<uint64_t>(0), 1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(static_cast<int32_t>(numElems % FLOAT_VECTOR_SIZE));
            AscendC::BlockReduceMax<float, false>(
                tvUb, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceMax<float, false>(
                    rowmaxUb, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
            } else {
                AscendC::BlockReduceMax<float, false>(
                    tvScratch, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Max<float, false>(
                    rowmaxUb, rowmaxUb, tvScratch,
                    static_cast<uint64_t>(0), 1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    }

    __aicore__ inline
    void CalcRowSumTail(const AscendC::LocalTensor<float> &srcUb,
                        const AscendC::LocalTensor<float> &rowsumUb,
                        const AscendC::LocalTensor<float> &tvUb,
                        const AscendC::LocalTensor<float> &tvScratch,
                        uint32_t numRows, uint32_t numRowsRound,
                        uint32_t numElems, uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceSum<float, false>(
                tvUb, srcUb, numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float, false>(
                rowsumUb, tvUb,
                CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                0, 1, 1, 8);

            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t colIdx = 1; colIdx < static_cast<uint64_t>(numElems) / FLOAT_VECTOR_SIZE; ++colIdx) {
                AscendC::BlockReduceSum<float, false>(
                    tvUb, srcUb[colIdx * FLOAT_VECTOR_SIZE],
                    numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceSum<float, false>(
                    tvScratch, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Add<float, false>(
                    rowsumUb, rowsumUb, tvScratch,
                    static_cast<uint64_t>(0), 1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(static_cast<int32_t>(numElems % FLOAT_VECTOR_SIZE));
            AscendC::BlockReduceSum<float, false>(
                tvUb, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceSum<float, false>(
                    rowsumUb, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
            } else {
                AscendC::BlockReduceSum<float, false>(
                    tvScratch, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Add<float, false>(
                    rowsumUb, rowsumUb, tvScratch,
                    static_cast<uint64_t>(0), 1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    }

    __aicore__ inline
    void SubtractRowMaxAndExp(const AscendC::LocalTensor<float> &srcUb,
                              const AscendC::LocalTensor<float> &rowmaxUb,
                              AscendC::LocalTensor<float> &tvUb,
                              uint32_t numRows, uint32_t numRowsRound,
                              uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::Brcb(
            tvUb.ReinterpretCast<uint32_t>(),
            rowmaxUb.ReinterpretCast<uint32_t>(),
            numRowsRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t colBlocks = numElemsAligned / FLOAT_BLOCK_SIZE;
        for (uint32_t colIdx = 0; colIdx < numElems / FLOAT_VECTOR_SIZE; ++colIdx) {
            AscendC::Sub<float, false>(
                srcUb[colIdx * FLOAT_VECTOR_SIZE],
                srcUb[colIdx * FLOAT_VECTOR_SIZE],
                tvUb,
                static_cast<uint64_t>(0),
                numRows,
                AscendC::BinaryRepeatParams(1, 1, 0, colBlocks, colBlocks, 1));
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(static_cast<int32_t>(numElems % FLOAT_VECTOR_SIZE));
            AscendC::Sub<float, false>(
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tvUb,
                static_cast<uint64_t>(0),
                numRows,
                AscendC::BinaryRepeatParams(1, 1, 0, colBlocks, colBlocks, 1));
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp<float, false>(
            srcUb, srcUb,
            static_cast<uint64_t>(0),
            CeilDiv(numRows * numElemsAligned, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    float scale_;
    AscendC::LocalTensor<float> tmpBuf_;
    AscendC::LocalTensor<float> rowMaxBuf_;
    AscendC::LocalTensor<float> rowSumBuf_;
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_PREFILL_ARCH35_HPP
