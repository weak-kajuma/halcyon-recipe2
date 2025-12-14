# Megatron-SWIFT で Llama を PLT（Patch-Level Training）対応にする変更点まとめ

このドキュメントは、論文 *Beyond Next Token Prediction: Patch-Level Training for Large Language Models* で提唱された PLT を、Megatron-SWIFT（megatron-swift）で Llama に適用するために行った改造点を **日本語で** まとめたものです。

> 注: `config` に `patch_size` を追加・引き回す程度の小変更は、本ドキュメントでは詳細説明の対象外とします（ご要望により省略）。

---

## 目的と前提

PLT では、入力トークン列を `patch_size` 個ずつの「パッチ」にまとめ、**埋め込み後の表現をパッチ単位に縮約**して Transformer に入力します。
同時に、**loss もパッチ単位**で計算します（本 repo では `custom_model/modeling_llama.py` の `# CHANGED` 部分と同等の挙動）。

Megatron-SWIFT では通常、

- データ側で `labels` を 1-token 右シフト（`torch.roll(labels, -1)`）
- モデル側（Megatron GPTModel）が損失（per-token loss）を返す（mcore の場合）
- trainer 側が `loss_func()` で集約・ログ出し

という流れになっています。PLT ではこれが合わないため、**「embedding 後の処理」と「loss 計算」だけを差し替える**方針で整合性を取っています。

---

## 変更点（重要）

### 1) `swift/megatron/model/gpt_model.py`：embedding 後にパッチ化 + attention/position をパッチ化

**狙い**

- Transformer 本体に入れる `hidden_states` を、トークン長 `S` からパッチ長 `P=S/patch_size` に縮約した表現へ変換する
- RoPE/position・causal mask も **パッチ長**に整合させる
- Megatron の pipeline / mcore の呼び出し流儀を壊さずに PLT を差し込む

**実装の要点**

- `GPTModel.__init__` に `self.patch_size` を追加し、`config.patch_size`（なければ 1）を保持
- `_preprocess(...)` のシグネチャに `attention_mask` を追加し、embedding 生成後に以下を実施:
  - **パッチ化**: `decoder_input` を `[B, P, patch_size, H]` に reshape して平均し、`[P, B, H]` へ
  - **position_ids の切り詰め**: `position_ids[..., :P]`
  - **attention_mask の作り直し**: 元の causal+padding マスクから、有効トークン長を推定し、パッチ長の causal mask + padding を生成
  - `seq_len_tokens % patch_size != 0` の場合は明示的に `ValueError`（PLT では割り切れない長さを許容しない）
- `_preprocess` が `attention_mask` を更新して返すようにし、`forward()` 側で `decoder(..., attention_mask=...)` に patch 化済みの mask を渡すように変更

**追加された補助関数（同ファイル内）**

- `_merge_patch_embeddings(...)`
  - `decoder_input` の形状が `[S, B, H]`（通常の token-wise）で来た場合に、patch 平均して `[P, B, H]` に変換
  - すでに `[P, B, H]` の場合はそのまま通す
- `_build_patch_attention_mask(...)`
  - 既存の `attention_mask`（`[B,1,S,S]`）から、対角成分を利用して「有効トークン数」を推定し、`patch_size` で割って `patch_len` を得る
  - `[B,1,P,P]` の causal mask を作り、`patch_len` より後ろの query 行を padding として True（mask）にする

---

### 2) `swift/megatron/trainers/trainer.py`：PLT 用 loss 分岐を追加

**狙い**

- mcore の GPTModel が通常返す「per-token loss」ルートではなく、PLT では **logits から trainer 側で loss を算出**する
- `custom_model/modeling_llama.py` の `ForCausalLMLossPLT` と同等の計算（patch 内の各 token を平均）を Megatron 側でも再現する

**実装の要点**

- `loss_func(...)` 冒頭で `patch_size = getattr(args, 'patch_size', 1)` を取得し、`patch_size > 1` の場合だけ PLT loss を計算
- `output_tensor` を logits として扱い、次を実施:
  - `shift_logits = logits[:, :-1, :]`（パッチ単位の next-patch 予測に対応）
  - `target_labels = labels[..., patch_size:]` を取り、`[-1, patch_size]` に reshape
  - `log_probs = log_softmax(shift_logits)` を作って、
    - `i=0..patch_size-1` それぞれについて `nll_loss(..., ignore_index=-100, reduction='sum')` を計算
    - 有効ラベル数で割って平均、最後に `patch_size` でも平均
  - 戻り値は Megatron の期待形式（`loss = [loss_sum, token_count]`）に合わせて `torch.stack([loss_val*count, count])` を構築
- `patch_size==1` の場合は従来の per-token loss 集約ロジックをそのまま使用

**備考**

- この実装は「パッチ内 token の loss を平均」する点で、添付の `ForCausalLMLossPLT` と整合します。
- channel_loss / dft_loss などの追加機構は、PLT 分岐では現時点で適用していません（既存の per-token losses 前提のため）。

---

### 3) `swift/megatron/trainers/utils.py`：PLT 時の labels シフト抑止 + position_ids のパッチ化

**狙い**

- 従来 Megatron-SWIFT は trainer 取り込み時に `labels = roll(-1)` しており、PLT の「labels[..., patch_size:]」前提と衝突する
- また、Megatron pipeline 側では `position_ids` がそのまま shape 推定などに使われるため、**position_ids もパッチ長に合わせる**必要がある

**実装の要点**

- `get_batch_on_this_tp_rank(...)` 内で `patch_size` を取得し、
  - `patch_size == 1` のときだけ従来通り `labels/loss_scale` を `torch.roll(..., -1)`
  - `patch_size > 1` の場合は **roll しない**
- `patch_size > 1` の場合、`position_ids` を `[..., :num_patches]` に切り詰め
  - `position_ids` 長が `patch_size` で割り切れない場合は `ValueError`（PLT の前提違反）

---

### 4) `swift/megatron/utils/utils.py`：pipeline shape 推定・padding の patch 対応

**狙い**

- pipeline parallel の `forward_step_helper()` は、最初の PP stage で次 stage へ送るテンソル形状を `position_ids` 長から推定します。
- PLT では埋め込み後に `S -> P` に縮むため、**shape 推定も P に合わせないと PP で不整合が起きる**。
- さらに、PLT では入力長が patch_size の倍数である必要があるため、padding の方針を patch_size と整合させる必要がある。

**実装の要点**

- `forward_step_helper(...)` の `seq_length` 推定で、`patch_size > 1` の場合 `seq_length //= patch_size` に変更
  - ただし割り切れない場合は `ValueError`
- `get_padding_to(args)` で、既存の SP/CP/FP8 由来の `padding_to` と `patch_size` の **最小公倍数**（`math.lcm`）を取るように変更
  - 結果として、データ collator が「patch_size の倍数」になるよう padding しやすくなる

---

### 5) `custom_model/llama.py`：`llama3_2_plt` の Megatron 登録

**狙い**

- `--custom_register_path custom_model/llama.py` で HF 側の `LlamaForCausalLM`（PLT 版）をロードするだけでは、Megatron 側のモデルマッピングに `llama3_2_plt` が存在しないため、Megatron トレーニングで model_type 解決に失敗しうる。
- そこで、**Megatron 側にも `llama3_2_plt` を GPT 系として登録**し、通常の Llama と同じ GPTModel/bridge を使いつつ PLT の挙動だけ差し込めるようにする。

**実装の要点**

- `register_megatron_model(..., exist_ok=True)` を追加し、`MegatronModelType.gpt` に `['llama3_2_plt']` を紐付け
- これにより `args.model_type == llama3_2_plt` の場合でも、Megatron 側は GPTModel 系として構築される

---

## 変更点（省略対象）

以下は「`config patch_size` 程度の変更」に該当するため、本ドキュメントでは詳細説明を省略します。

- `swift/megatron/utils/config.py` の HF config → Megatron config 変換に `patch_size` を追加
- `swift/megatron/argument/megatron_args.py` に `patch_size` フィールドを追加し、Megatron CLI へ流さず Swift 側だけで保持する調整
- `swift/megatron/model/model_provider.py` で `config.patch_size` を args から注入

---

## 注意点・運用上の条件

- **シーケンス長が `patch_size` の倍数である必要があります。**
  - そうでない場合、PP shape 推定や patch 平均が成立しないため `ValueError` を出すようにしています。
  - 実運用では `--max_length` / `--packing_length`、および padding 設定がこの条件を満たすようにしてください。
- PLT では logits を返して trainer 側で loss を計算するため、従来の「モデルが per-token loss を返す」前提の機能（例: 一部の loss フック）はそのままでは適用されません。

---

## 使い方（実行手順）

ここでは「Llama を `llama3_2_plt` として PLT 学習する」ための最小手順を示します。

### 1) 事前準備

- `megatron` コマンドは `setup.py` の console_scripts により提供されます（`swift` とは別コマンドです）。
- `megatron pt/sft/...` は torchrun 前提のため、最低限 `NPROC_PER_NODE` 等の環境変数が必要です（例は下記）。

### 2) `patch_size` の指定方法

PLT を有効にするには **学習時に `--patch_size` を指定**してください。

- 例: `--patch_size 4`
- `--max_length`（および `--packing_length` を使う場合はそれも）が **`patch_size` の倍数**になるように設定してください。

### 3) （推奨）HF → mcore へ変換してから Megatron 学習

repo 既存の運用（`--load <mcore_dir>`）に合わせ、まず mcore へ変換します。

```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
  --use_hf true \
  --custom_register_path custom_model/llama.py \
  --model meta-llama/Llama-3.2-1B \
  --model_type llama3_2_plt \
  --to_mcore true \
  --torch_dtype bfloat16 \
  --output_dir llama3_2_plt_mcore
```

### 4) Megatron で PLT 学習（PT 例）

```bash
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron pt \
  --load llama3_2_plt_mcore \
  --custom_register_path custom_model/llama.py \
  --model_type llama3_2_plt \
  --patch_size 4 \
  --dataset swift/chinese-c4 \
  --streaming true \
  --packing true \
  --tensor_model_parallel_size 4 \
  --micro_batch_size 1 \
  --global_batch_size 16 \
  --train_iters 1000 \
  --finetune true \
  --lr 1e-5 \
  --min_lr 1e-6 \
  --save megatron_output/llama3_2_plt \
  --save_interval 200 \
  --max_length 8192 \
  --sequence_parallel true \
  --attention_backend flash
```

ポイント:

- `--model_type llama3_2_plt` と `--custom_register_path custom_model/llama.py` をセットで指定します。
- `--patch_size` を指定しない場合、従来の token-level 学習（`patch_size==1` 相当）として動作します。
