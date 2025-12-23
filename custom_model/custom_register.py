from functools import partial

from swift.llm import (Model, ModelGroup, ModelMeta, get_model_tokenizer, get_model_tokenizer_with_flash_attn,
                       register_model)
from swift.llm import MODEL_MAPPING
from swift.megatron.model.constant import MegatronModelType
from swift.megatron.model.register import MEGATRON_MODEL_MAPPING, MegatronModelMeta, register_megatron_model
from swift.llm.model.model_arch import ModelArch
from swift.llm.template.constant import TemplateType

from modeling_llama import LlamaForCausalLM
from modeling_qwen3 import Qwen3ForCausalLM

print("[custom_register_path] loaded:", __file__)


def _register_megatron_gpt_models(model_types):
    """Extend Megatron GPT mapping without overwriting existing registrations."""
    gpt_meta = MEGATRON_MODEL_MAPPING.get(MegatronModelType.gpt)
    if gpt_meta is not None:
        for model_type in model_types:
            if model_type not in gpt_meta.model_types:
                gpt_meta.model_types.append(model_type)
            if model_type in MODEL_MAPPING:
                MODEL_MAPPING[model_type].support_megatron = True
        return
    register_megatron_model(
        MegatronModelMeta(
            MegatronModelType.gpt,
            model_types,
        ),
        exist_ok=True,
    )


register_model(
    ModelMeta(
        "llama3_2_plt",
        [
            ModelGroup([
                Model('LLM-Research/Llama-3.2-1B', 'meta-llama/Llama-3.2-1B'),
                Model('LLM-Research/Llama-3.2-3B', 'meta-llama/Llama-3.2-3B'),
                Model('LLM-Research/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct'),
                Model('LLM-Research/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct'),
            ])
        ],
        TemplateType.llama3_2,
        partial(get_model_tokenizer_with_flash_attn, automodel_class=LlamaForCausalLM),
        architectures=['LlamaForCausalLM'],
        requires=['transformers>=4.43'],
        model_arch=ModelArch.llama,
    ))

register_model(
    ModelMeta(
        "qwen3_plt",
        [
            ModelGroup([
                Model('Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen3-0.6B-Base'),
                Model('Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3-1.7B-Base'),
                Model('Qwen/Qwen3-4B-Base', 'Qwen/Qwen3-4B-Base'),
                Model('Qwen/Qwen3-8B-Base', 'Qwen/Qwen3-8B-Base'),
                Model('Qwen/Qwen3-14B-Base', 'Qwen/Qwen3-14B-Base'),
                Model('Qwen/Qwen3-32B-Base', 'Qwen/Qwen3-32B-Base'),
                # instruct
                Model('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-0.6B'),
                Model('Qwen/Qwen3-1.7B', 'Qwen/Qwen3-1.7B'),
                Model('Qwen/Qwen3-4B', 'Qwen/Qwen3-4B'),
                Model('Qwen/Qwen3-8B', 'Qwen/Qwen3-8B'),
                Model('Qwen/Qwen3-14B', 'Qwen/Qwen3-14B'),
                Model('Qwen/Qwen3-32B', 'Qwen/Qwen3-32B'),
                # fp8
                Model('Qwen/Qwen3-0.6B-FP8', 'Qwen/Qwen3-0.6B-FP8'),
                Model('Qwen/Qwen3-1.7B-FP8', 'Qwen/Qwen3-1.7B-FP8'),
                Model('Qwen/Qwen3-4B-FP8', 'Qwen/Qwen3-4B-FP8'),
                Model('Qwen/Qwen3-8B-FP8', 'Qwen/Qwen3-8B-FP8'),
                Model('Qwen/Qwen3-14B-FP8', 'Qwen/Qwen3-14B-FP8'),
                Model('Qwen/Qwen3-32B-FP8', 'Qwen/Qwen3-32B-FP8'),
                # awq
                Model('Qwen/Qwen3-4B-AWQ', 'Qwen/Qwen3-4B-AWQ'),
                Model('Qwen/Qwen3-8B-AWQ', 'Qwen/Qwen3-8B-AWQ'),
                Model('Qwen/Qwen3-14B-AWQ', 'Qwen/Qwen3-14B-AWQ'),
                Model('Qwen/Qwen3-32B-AWQ', 'Qwen/Qwen3-32B-AWQ'),
            ])
        ],
        TemplateType.qwen3,
        partial(get_model_tokenizer_with_flash_attn, automodel_class=Qwen3ForCausalLM),
        architectures=['Qwen3ForCausalLM'],
        requires=['transformers>=4.51'],
        model_arch=ModelArch.llama,
    ))

_register_megatron_gpt_models(['llama3_2_plt', 'qwen3_plt'])

if __name__ == '__main__':
    # Test and debug
    model, tokenizer = get_model_tokenizer('meta-llama/Llama-3.2-1B', model_type='llama3_2_plt', use_hf=True)
