from functools import partial

from swift.llm import (Model, ModelGroup, ModelMeta, get_model_tokenizer, get_model_tokenizer_with_flash_attn,
                       register_model)
from swift.megatron.model.constant import MegatronModelType
from swift.megatron.model.register import MegatronModelMeta, register_megatron_model
from swift.llm.model.model_arch import ModelArch
from swift.llm.template.constant import TemplateType

from modeling_llama import LlamaForCausalLM

print("[custom_register_path] loaded:", __file__)

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

register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.gpt,
        ['llama3_2_plt'],
    ),
    exist_ok=True,
)

if __name__ == '__main__':
    # Test and debug
    model, tokenizer = get_model_tokenizer('meta-llama/Llama-3.2-1B', model_type='llama3_2_plt', use_hf=True)
