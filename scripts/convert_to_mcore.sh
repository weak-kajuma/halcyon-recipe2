CUDA_VISIBLE_DEVICES=0 \
USE_HF=1 \
swift export \
    --use_hf true \
    --custom_register_path custom_model/llama.py \
    --model ./Llama-3.2-1B \
    --model_type llama3_2_plt \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir llama_plt \
    --test_convert_precision true
