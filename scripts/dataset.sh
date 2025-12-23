HF_HOME="/mnt/hdd1/huggingface/" \
swift export \
  --to_cached_dataset true \
  --use_hf true \
  --model Qwen/Qwen3-0.6B-Base \
  --dataset /mnt/hdd1/dataset_messages_parquet/messages_2of3.parquet \
  --output_dir /mnt/hdd1/dataset_cached/2of3/ \
  --max_length 4096 \
  --template_mode train \
  --truncation_strategy right \
  --dataset_num_proc 24; echo $?

HF_HOME="/mnt/hdd1/huggingface/" \
swift export \
  --to_cached_dataset true \
  --use_hf true \
  --model Qwen/Qwen3-0.6B-Base \
  --dataset /mnt/hdd1/dataset_messages_parquet/messages_full.parquet \
  --output_dir /mnt/hdd1/dataset_cached/full/ \
  --max_length 4096 \
  --template_mode train \
  --truncation_strategy right \
  --dataset_num_proc 24; echo $?
