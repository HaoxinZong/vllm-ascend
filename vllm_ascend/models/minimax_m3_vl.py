# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The MiniMax AI team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.

"""MiniMax M3 multimodal wrapper for Ascend.

The ViT implementation is reused from vLLM's common MiniMax M3 vision tower,
while the language model path remains the Ascend-native implementation in
``vllm_ascend.models.minimax_m3``.
"""

import sys
from collections.abc import Iterable, Mapping, Sequence
from types import ModuleType
from typing import Any, cast

import torch
from torch import nn
from transformers import BatchFeature, PretrainedConfig
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.minimax_m3 import MiniMaxM3SparseForCausalLM


def _install_fused_allreduce_norm_fallback() -> None:
    """Avoid importing vLLM's CUDA-only fusion module on Ascend."""
    module_name = "vllm.model_executor.layers.fused_allreduce_gemma_rms_norm"
    if module_name in sys.modules:
        return

    fallback_module = ModuleType(module_name)

    def fused_allreduce_gemma_rms_norm(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm: GemmaRMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if get_tensor_model_parallel_world_size() > 1:
            hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states)
        return norm(hidden_states, residual)

    cast(Any, fallback_module).fused_allreduce_gemma_rms_norm = fused_allreduce_gemma_rms_norm
    sys.modules[module_name] = fallback_module


_install_fused_allreduce_norm_fallback()

from vllm.models.minimax_m3.common.vision_tower import MiniMaxVLVisionModel  # noqa: E402


def _get_minimax_m3_image_size(image_processor: object) -> ImageSize:
    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    factor = patch_size * merge_size

    max_pixels = getattr(image_processor, "max_pixels", None)
    if not isinstance(max_pixels, int):
        size = getattr(image_processor, "size", None)
        if isinstance(size, Mapping):
            height = int(size.get("height") or size.get("shortest_edge") or 672)
            width = int(size.get("width") or size.get("longest_edge") or height)
            max_pixels = height * width
        elif isinstance(size, Sequence) and len(size) >= 2:
            max_pixels = int(size[0]) * int(size[1])
        else:
            max_pixels = 672 * 672

    side = max(factor, int(max_pixels**0.5) // factor * factor)
    return ImageSize(width=side, height=side)


def _get_minimax_m3_num_image_tokens(
    image_processor: object,
    *,
    image_width: int,
    image_height: int,
) -> int:
    merge_size = int(getattr(image_processor, "merge_size", 2))
    if hasattr(image_processor, "get_number_of_image_patches"):
        num_patches = image_processor.get_number_of_image_patches(image_height, image_width)
    else:
        patch_size = int(getattr(image_processor, "patch_size", 14))
        grid_h = image_height // patch_size
        grid_w = image_width // patch_size
        num_patches = grid_h * grid_w
    return int(num_patches) // (merge_size**2)


def _get_minimax_m3_num_video_tokens(
    video_processor: object,
    *,
    video_width: int,
    video_height: int,
    num_frames: int,
) -> int:
    patch_size = int(getattr(video_processor, "patch_size", 14))
    merge_size = int(getattr(video_processor, "merge_size", 2))
    temporal_patch_size = int(getattr(video_processor, "temporal_patch_size", 2))
    padded_frames = num_frames
    pad = -padded_frames % temporal_patch_size
    if pad:
        padded_frames += pad
    grid_t = max(padded_frames // temporal_patch_size, 1)
    grid_h = video_height // patch_size
    grid_w = video_width // patch_size
    return int(grid_t * grid_h * grid_w) // (merge_size**2)


class MiniMaxM3VLProcessingInfo(BaseProcessingInfo):
    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    VISION_START_TOKEN = "]<]start of image[>["
    VISION_END_TOKEN = "]<]end of image[>["

    def get_hf_processor(self, **kwargs: object):
        # Do not type-check the processor class here: released MiniMax-M3
        # checkpoints may expose remote-code MiniMaxVLProcessor, while
        # transformers main uses MiniMaxM3VLProcessor.
        return self.ctx.get_hf_processor(**kwargs)

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).video_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits: dict[str, int | None] = {"image": None}
        mm_config = self.ctx.get_mm_config()
        if "video" in mm_config.limit_per_prompt:
            limits["video"] = None
        return limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_tokens = {"image": self.get_max_image_tokens()}
        if mm_counts.get("video", 0) > 0:
            max_tokens["video"] = self.get_max_video_tokens(seq_len, mm_counts)
        return max_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        return _get_minimax_m3_image_size(self.get_image_processor())

    def get_max_image_tokens(self) -> int:
        image_processor = self.get_image_processor()
        size = self.get_image_size_with_most_features()
        return _get_minimax_m3_num_image_tokens(
            image_processor,
            image_width=size.width,
            image_height=size.height,
        )

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        video_processor = self.get_video_processor()
        size = self.get_image_size_with_most_features()
        patch_size = int(getattr(video_processor, "patch_size", 14))
        merge_size = int(getattr(video_processor, "merge_size", 2))
        temporal_patch_size = int(getattr(video_processor, "temporal_patch_size", 2))
        max_frames = int(getattr(video_processor, "max_frames", 768))

        grid_h = max(size.height // patch_size, 1)
        grid_w = max(size.width // patch_size, 1)
        tokens_per_temporal_grid = max(grid_h * grid_w // (merge_size**2), 1)

        num_images = mm_counts.get("image", 0)
        num_videos = max(mm_counts.get("video", 0), 1)
        remaining = max(seq_len - num_images * self.get_max_image_tokens(), 1)
        max_grid_t = max(remaining // tokens_per_temporal_grid // num_videos, 1)
        num_frames = max_grid_t * temporal_patch_size
        return max(min(num_frames, max_frames), temporal_patch_size)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        video_processor = self.get_video_processor()
        size = self.get_image_size_with_most_features()
        return _get_minimax_m3_num_video_tokens(
            video_processor,
            video_width=size.width,
            video_height=size.height,
            num_frames=self.get_num_frames_with_most_features(seq_len, mm_counts),
        )


class MiniMaxM3VLDummyInputsBuilder(BaseDummyInputsBuilder[MiniMaxM3VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.info.IMAGE_TOKEN * mm_counts.get("image", 0) + self.info.VIDEO_TOKEN * mm_counts.get("video", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        size = self.info.get_image_size_with_most_features()
        num_frames = self.info.get_num_frames_with_most_features(seq_len, mm_counts)
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=size.width,
                height=size.height,
                num_frames=num_frames,
                num_videos=mm_counts.get("video", 0),
                overrides=mm_options.get("video"),
            ),
        }


class MiniMaxM3VLMultiModalProcessor(BaseMultiModalProcessor[MiniMaxM3VLProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # MiniMax-M3's processor default sets videos_kwargs.do_resize=False,
        # which assumes pre-resized frames. vLLM passes raw video frames, so
        # resize by default unless the user explicitly overrides it.
        merged_kwargs = {"do_resize": True}
        merged_kwargs.update(mm_kwargs)
        merged_kwargs.update(tok_kwargs)
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            merged_kwargs,
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        video_processor = self.info.get_video_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        image_token = getattr(hf_processor, "image_token", MiniMaxM3VLProcessingInfo.IMAGE_TOKEN)
        video_token = getattr(hf_processor, "video_token", MiniMaxM3VLProcessingInfo.VIDEO_TOKEN)
        start_token = getattr(
            hf_processor,
            "VISION_START_TOKEN",
            MiniMaxM3VLProcessingInfo.VISION_START_TOKEN,
        )
        end_token = getattr(
            hf_processor,
            "VISION_END_TOKEN",
            MiniMaxM3VLProcessingInfo.VISION_END_TOKEN,
        )

        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]
        start_token_id = vocab[start_token]
        end_token_id = vocab[end_token]
        image_merge_length = int(getattr(image_processor, "merge_size", 2)) ** 2
        video_merge_length = int(getattr(video_processor, "merge_size", 2)) ** 2
        temporal_patch_size = int(getattr(video_processor, "temporal_patch_size", 2))
        fps = float(getattr(video_processor, "fps", 1.0) or 1.0)

        def get_image_replacement(item_idx: int):
            grid_thw = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod().item()) // image_merge_length
            full = [start_token_id] + [image_token_id] * num_tokens + [end_token_id]
            return PromptUpdateDetails.select_token_id(full, image_token_id)

        def get_video_replacement(item_idx: int):
            grid_thw = out_mm_kwargs["video"][item_idx]["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            grid_t, grid_h, grid_w = [int(x) for x in grid_thw.tolist()]
            frame_seqlen = grid_h * grid_w // video_merge_length

            full: list[int] = []
            for frame_idx in range(grid_t):
                timestamp = frame_idx * temporal_patch_size / fps
                full.extend(
                    tokenizer.encode(
                        f"]<]{timestamp:.1f} seconds[>[",
                        add_special_tokens=False,
                    )
                )
                full.append(start_token_id)
                full.extend([video_token_id] * frame_seqlen)
                full.append(end_token_id)
            return PromptUpdateDetails.select_token_id(full, video_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=[video_token_id],
                replacement=get_video_replacement,
            ),
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            image_grid_sizes = torch.empty(0, dtype=torch.long)
        else:
            image_grid_sizes = image_grid_thw.prod(-1)

        video_grid_thw = hf_inputs.get("video_grid_thw")
        if video_grid_thw is None:
            video_grid_sizes = torch.empty(0, dtype=torch.long)
        else:
            video_grid_sizes = video_grid_thw.prod(-1)

        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
            "image_grid_thw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "pixel_values_videos": MultiModalFieldConfig.flat_from_sizes("video", video_grid_sizes),
            "video_grid_thw": MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        }


class MiniMaxM3VLModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            raise ValueError("MiniMax-M3 VL requires config.vision_config.")

        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig.from_dict(vision_config)

        projector_hidden_size = getattr(config, "projector_hidden_size", None)
        self.vision_tower = MiniMaxVLVisionModel(
            config=vision_config,
            text_hidden_size=text_config.hidden_size,
            projector_hidden_size=projector_hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )
        self.language_model = MiniMaxM3SparseForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder,
)
class MiniMaxM3SparseForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP, SupportsEagle3):
    supports_encoder_tp_data = True

    packed_modules_mapping = {
        **MiniMaxM3SparseForCausalLM.packed_modules_mapping,
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.language_model.model.",
            "language_model.model.": "model.language_model.model.",
            "language_model.lm_head.": "model.language_model.lm_head.",
            "model.vision_tower.": "model.vision_tower.",
            "vision_tower.": "model.vision_tower.",
            "multi_modal_projector.": ("model.vision_tower.multi_modal_projector."),
            "patch_merge_mlp.": "model.vision_tower.patch_merge_mlp.",
            "lm_head.": "model.language_model.lm_head.",
        },
        orig_to_new_substr={
            ".mlp.fc1.": ".fc1.",
            ".mlp.fc2.": ".fc2.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "]<]image[>["
        if modality == "video":
            return "]<]video[>["
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model_config = vllm_config.model_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = (
            self.multimodal_config is not None and self.multimodal_config.mm_encoder_tp_mode == "data"
        )

        with self._mark_composite_model(
            vllm_config,
            language_targets=MiniMaxM3SparseForCausalLM,
            tower_targets={"image": MiniMaxVLVisionModel, "video": MiniMaxVLVisionModel},
        ):
            self.model = MiniMaxM3VLModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        self.vision_tower = self.model.vision_tower
        self.language_model = self.model.language_model
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        image_embeds = kwargs.pop("image_embeds", None)
        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None:
            return {
                "type": "pixel_values",
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        return {
            "type": "image_embeds",
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
        }

    def _parse_and_validate_video_input(self, **kwargs: object) -> dict | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        video_embeds = kwargs.pop("video_embeds", None)
        if pixel_values_videos is None and video_embeds is None:
            return None
        if pixel_values_videos is not None:
            return {
                "type": "pixel_values_videos",
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            }
        return {
            "type": "video_embeds",
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
        }

    def _process_image_input(self, image_input: dict) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw is not None and grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.vision_tower.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.vision_tower.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.vision_tower,
                    pixel_values,
                    grid_thw_list,
                    rope_type="rope_3d",
                )
            image_embeds = self.vision_tower(
                pixel_values=pixel_values,
                grid_thw=grid_thw_list,
            )

        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input: dict) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw is not None and grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.vision_tower.dtype)
        else:
            pixel_values = video_input["pixel_values_videos"].type(self.vision_tower.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.vision_tower,
                    pixel_values,
                    grid_thw_list,
                    rope_type="rope_3d",
                )
            video_embeds = self.vision_tower(
                pixel_values=pixel_values,
                grid_thw=grid_thw_list,
            )

        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict[str, dict]:
        mm_input_by_modality: dict[str, dict] = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is not None:
                    mm_input_by_modality["image"] = image_input
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                video_input = self._parse_and_validate_video_input(**kwargs)
                if video_input is not None:
                    mm_input_by_modality["video"] = video_input
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: list[torch.Tensor] = []
        for modality, multimodal_input in mm_input_by_modality.items():
            if modality == "image":
                multimodal_embeddings.extend(self._process_image_input(multimodal_input))
            elif modality == "video":
                multimodal_embeddings.extend(self._process_video_input(multimodal_input))
        return tuple(multimodal_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return SupportsMultiModal.embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_default_aux_hidden_state_layers()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        raw_tensors = 0
        prefix_counts: dict[str, int] = {}

        def counted_weights() -> Iterable[tuple[str, torch.Tensor]]:
            nonlocal raw_tensors
            for name, weight in weights:
                raw_tensors += 1
                if name.startswith("language_model."):
                    bucket = "language_model"
                elif name.startswith("vision_tower."):
                    bucket = "vision_tower"
                elif name.startswith("multi_modal_projector."):
                    bucket = "multi_modal_projector"
                elif name.startswith("patch_merge_mlp."):
                    bucket = "patch_merge_mlp"
                else:
                    bucket = name.split(".", 1)[0]
                prefix_counts[bucket] = prefix_counts.get(bucket, 0) + 1
                yield name, weight

        logger.warning("MiniMax M3 VL load_weights entered")
        loaded_params = loader.load_weights(counted_weights(), mapper=self.hf_to_vllm_mapper)
        logger.warning(
            "MiniMax M3 VL load_weights saw %d checkpoint tensors by prefix %s; returned %d loaded parameter names",
            raw_tensors,
            prefix_counts,
            len(loaded_params),
        )
        return loaded_params
