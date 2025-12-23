from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

# ----------------------------
# 設定
# ----------------------------
text_col = "text"

token_budget_1 = 36_000_000_000  # ds1からだいたいこのトークン数ぶん取る
token_budget_2 = 4_000_000_000  # ds2からだいたいこのトークン数ぶん取る

batch_size = 2048
shuffle = False
seed = 42

# mapの並列設定
num_proc = 24          # CPUに合わせて
map_batch_size = 2048 # 大きめが速いことが多い

# ここが重要：部分的に処理する単位
chunk_size = 200_000   # まずは大きめで。RAMがきつければ下げる

# ----------------------------
# tokenizer（worker内で遅延初期化）
# ----------------------------
_TOK = None
def _count_tokens_batch(batch):
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", use_fast=True)

    enc = _TOK(
        batch[text_col],
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
    )
    return {"n_tokens": [len(x) for x in enc["input_ids"]]}

# ----------------------------
# chunkごとにmapして予算まで取る
# ----------------------------
def take_by_token_budget_chunked(
    ds,
    budget_tokens: int,
    *,
    chunk_size: int,
    allow_overflow: bool = True,
    num_proc: int = 8,
    map_batch_size: int = 2048,
    seed: int = 0,
    shuffle: bool = False,
):
    if shuffle:
        ds = ds.shuffle(seed=seed)

    picked_chunks = []
    total = 0
    n = len(ds)

    for start in range(0, n, chunk_size):
        if total >= budget_tokens:
            break

        end = min(start + chunk_size, n)
        chunk = ds.select(range(start, end))

        # このchunkだけ並列map（全件にmapしない）
        chunk_tok = chunk.map(
            _count_tokens_batch,
            batched=True,
            batch_size=map_batch_size,
            num_proc=num_proc,
            desc=f"count tokens [{start}:{end}]",
            load_from_cache_file=False,  # キャッシュ肥大化を避けたいならFalse
        )

        lens = chunk_tok["n_tokens"]
        remain = budget_tokens - total

        # このchunkから何件取るか決める
        take_k = 0
        s = 0
        for L in lens:
            if s >= remain:
                break
            if (not allow_overflow) and (s + L > remain) and (take_k > 0):
                break
            s += L
            take_k += 1

        # 何も取れない & allow_overflow=True なら最低1件は取る（ざっくり運用）
        if take_k == 0 and allow_overflow and len(lens) > 0:
            take_k = 1
            s = lens[0]

        picked_chunks.append(chunk_tok.select(range(take_k)))
        total += s

        if total >= budget_tokens:
            break

    picked = concatenate_datasets(picked_chunks) if picked_chunks else ds.select([])
    return picked, total

# ----------------------------
# 実行
# ----------------------------
ds2 = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train").select_columns(["text"])

sub2, total2 = take_by_token_budget_chunked(
    ds2, token_budget_2,
    chunk_size=chunk_size,
    allow_overflow=True,
    num_proc=num_proc,
    map_batch_size=map_batch_size,
    shuffle=shuffle,
    seed=seed,
)

ds1 = load_dataset("hotchpotch/fineweb-2-edu-japanese", split="train", num_proc=24).select_columns(["text"])

sub1, total1 = take_by_token_budget_chunked(
    ds1, token_budget_1,
    chunk_size=chunk_size,
    allow_overflow=True,
    num_proc=num_proc,
    map_batch_size=map_batch_size,
    shuffle=shuffle,
    seed=seed,
)

merged = concatenate_datasets([sub1, sub2])
merged = merged.select_columns(["text"])

print(f"ds1: docs={len(sub1):,}, tokens≈{total1:,}")
print(f"ds2: docs={len(sub2):,}, tokens≈{total2:,}")
print(f"merged: docs={len(merged):,}, columns={merged.column_names}")

merged.push_to_hub("kajuma/40B", max_shard_size="10GB")
