from vllm import ModelRegistry


def register_model():
    from vllm_ascend.patch.platform.patch_minimax_m3_config import register_minimax_m3_configs

    register_minimax_m3_configs()
    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "DeepSeekV4MTPModel",
        "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP",
    )
