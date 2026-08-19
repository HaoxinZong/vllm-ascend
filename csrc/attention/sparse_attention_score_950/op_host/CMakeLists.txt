# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
get_filename_component(OP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
# build_aclnn.sh selects this source tree by its _950 directory name, while the
# operator and all generated artifacts intentionally retain their canonical name.
set_property(DIRECTORY "${OP_SOURCE_DIR}" PROPERTY ASCEND_OP_BUILD_NAME sparse_attention_score)
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        sparse_attention_score_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME SparseAttentionScore
            OPTIONS --cce-auto-sync=off
                    -DASCENDC_DUMP=1
                    -Wno-deprecated-declarations
                    -Werror
                    -fpermissive
    )
else()
    add_modules_sources(OPTYPE sparse_attention_score ACLNNTYPE aclnn_exclude)
endif()
