PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
megatron pt \
    --load llama_plt \
    --custom_register_path custom_model/llama.py \
    --model_type llama3_2_plt \
    --patch_size 4 \
    --dataset 'kajuma/training_01-09_token'\
    --micro_batch_size 2 \
    --global_batch_size 256 \
    --attention_backend flash \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 1 \
    --train_iters 1000 \
    --save megatron_output/llama3_2_plt \
    --save_interval 50 \
    --max_length 8192 \
    --truncation_strategy right \
    --no_save_optim true \
    --no_save_rng true \
    --packing true \
    --wandb_project qwen3 \
    --wandb_exp_name plt_test \
    --log_interval 1 \
    --logging_level 20 \
    --num_workers 12 \
    --dataset_num_proc 24 \
    --use_hf true
