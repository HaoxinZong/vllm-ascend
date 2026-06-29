# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patch target: vllm/transformers_utils/config.py
# - vLLM 0.20.2 does not include MiniMax M3 config classes yet.
#

from vllm.logger import logger

from vllm_ascend.models.minimax_m3_config import (
    MiniMaxM3Config,
    MiniMaxM3MTPConfig,
    MiniMaxM3TextConfig,
)


def register_minimax_m3_configs() -> None:
    try:
        from vllm.transformers_utils import config as vllm_config
    except Exception:
        logger.debug("Skip MiniMax M3 config registration: vLLM config module is unavailable.")
        return

    registry = getattr(vllm_config, "_CONFIG_REGISTRY", None)
    if registry is not None:
        registry.setdefault("minimax_m3_vl", MiniMaxM3Config)
        registry.setdefault("minimax_m3_mtp", MiniMaxM3MTPConfig)
        registry.setdefault("minimax_m3_text", MiniMaxM3TextConfig)
        registry.setdefault("minimax_m3", MiniMaxM3TextConfig)

    auto_config = getattr(vllm_config, "AutoConfig", None)
    if auto_config is None:
        try:
            from transformers import AutoConfig as auto_config  # type: ignore[no-redef]
        except Exception:
            logger.debug("Skip MiniMax M3 AutoConfig registration: AutoConfig is unavailable.")
            return

    for model_type, config_cls in (
        ("minimax_m3_vl", MiniMaxM3Config),
        ("minimax_m3_mtp", MiniMaxM3MTPConfig),
        ("minimax_m3_text", MiniMaxM3TextConfig),
        ("minimax_m3", MiniMaxM3TextConfig),
    ):
        try:
            auto_config.register(model_type, config_cls, exist_ok=True)
        except TypeError:
            try:
                auto_config.register(model_type, config_cls)
            except ValueError:
                pass
        except ValueError:
            pass


register_minimax_m3_configs()
