PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
megatron pt \
    --load Qwen3-0.6B-Base_PLT \
    --custom_register_path custom_model/custom_register.py \
    --model_type qwen3_plt \
    --patch_size 4 \
    --cached_dataset ./2of3 \
    --micro_batch_size 1 \
    --global_batch_size 256 \
    --attention_backend flash \
    --use_precision_aware_optimizer true \
    --overlap_grad_reduce True \
    --overlap_param_gather True \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 3e-6 \
    --train_iters 6350 \
    --save megatron_output/patch\
    --save_interval 2 \
    --max_length 16384 \
    --truncation_strategy right \
    --packing true \
    --wandb_project plt \
    --wandb_exp_name plt_1 \
    --log_interval 1 \
    --logging_level 20 \
    --num_workers 12 \
    --dataset_num_proc 24 \
    --use_hf true
    # --recompute_num_layers 1 \
    # --recompute_granularity full \
    # --recompute_method uniform \
    # --no_save_optim true \
    # --no_save_rng true \