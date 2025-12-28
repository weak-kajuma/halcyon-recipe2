# CUDA_VISIBLE_DEVICES=0 \
# USE_HF=1 \
# swift export \
#     --use_hf true \
#     --custom_register_path custom_model/custom_register.py \
#     --model Qwen/Qwen3-0.6B-Base \
#     --model_type qwen3_plt \
#     --to_mcore true \
#     --torch_dtype bfloat16 \
#     --output_dir Qwen3-0.6B-Base_PLT \
#     --test_convert_precision true


# for baseline
CUDA_VISIBLE_DEVICES=0 \
USE_HF=1 \
swift export \
    --use_hf true \
    --model Qwen/Qwen3-0.6B-Base \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen3-0.6B-Base-mcore \
    --test_convert_precision true
