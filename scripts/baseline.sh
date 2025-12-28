PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
megatron pt \
    --load Qwen3-0.6B-Base-mcore \
    --cached_dataset ./full \
    --micro_batch_size 1 \
    --global_batch_size 259 \
    --attention_backend flash \
    --use_precision_aware_optimizer true \
    --overlap_grad_reduce True \
    --overlap_param_gather True \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 3e-6 \
    --train_iters 38100 \
    --save megatron_output/qwen3-baseline \
    --save_interval 100 \
    --max_length 4096 \
    --truncation_strategy right \
    --packing true \
    --wandb_project plt \
    --wandb_exp_name baseline \
    --log_interval 1 \
    --logging_level 20 \
    --num_workers 12 \
    --dataset_num_proc 24 \
    --use_hf true
    # --train_iters 1000 \
    # --no_save_optim true \
    # --no_save_rng true \
    # --recompute_num_layers 1 \
    # --recompute_granularity full \
    # --recompute_method uniform \