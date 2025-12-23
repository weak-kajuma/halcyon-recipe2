import os
from datasets import load_dataset, concatenate_datasets

DATASET_NAME = "kajuma/40B"
SPLIT = "train"
TEXT_COL = "text"

OUT_DIR = "/mnt/hdd1/dataset_messages_parquet"
FULL_PATH = os.path.join(OUT_DIR, "messages_full.parquet")
ONE_THIRD_PATH = os.path.join(OUT_DIR, "messages_1of3.parquet")
TWO_THIRD_PATH = os.path.join(OUT_DIR, "messages_2of3.parquet")


def to_messages(example):
    return {"messages": [{"role": "assistant", "content": example[TEXT_COL]}]}


def convert_and_write(ds, out_path, desc):
    # shard後に map する（無駄な変換を減らす）
    ds2 = ds.map(
        to_messages,
        remove_columns=ds.column_names,
        num_proc=24,  # メモリ不安なら控えめ推奨
        desc=desc,
    )
    ds2.to_parquet(out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = load_dataset(DATASET_NAME, split=SPLIT)

    # 重要: contiguous=False (デフォルト) = mod3 分割
    shard0 = ds.shard(num_shards=3, index=0, contiguous=False)
    shard1 = ds.shard(num_shards=3, index=1, contiguous=False)
    shard2 = ds.shard(num_shards=3, index=2, contiguous=False)

    # 2/3 は残り2つを結合（重複なし）
    shard_2of3 = concatenate_datasets([shard1, shard2])

    convert_and_write(ds, FULL_PATH, "FULL -> messages")
    convert_and_write(shard0, ONE_THIRD_PATH, "1/3 -> messages")
    convert_and_write(shard_2of3, TWO_THIRD_PATH, "2/3 -> messages")

    print("DONE")
    print(" -", FULL_PATH)
    print(" -", ONE_THIRD_PATH)
    print(" -", TWO_THIRD_PATH)
    print("rows:", len(ds), len(shard0), len(shard_2of3))


if __name__ == "__main__":
    main()
