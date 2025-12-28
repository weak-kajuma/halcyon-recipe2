import os
import time
import re
import shutil
from huggingface_hub import HfApi

WATCH_DIR = "/home/kazuma/codes/halcyon-recipe2/megatron_output/qwen3-baseline/v1-20251218-031023"
REPO_ID = "kajuma/patch"
REPO_TYPE = "model"      # "model" / "dataset" / "space"
PREFIX = "iter_"
KEEP_LATEST = 2
SCAN_INTERVAL_SEC = 300    # 5分ごと（適宜）

api = HfApi()  # huggingface-cli login 済み前提


def iter_number(name: str):
    """
    iter_XXXX からXXXXを数値として取り出す。
    例: iter_100 -> 100, iter_000200 -> 200
    形式が崩れていれば None
    """
    if not name.startswith(PREFIX):
        return None
    m = re.fullmatch(r"iter_(\d+)", name)
    return int(m.group(1)) if m else None


def list_iter_folders():
    items = []
    for name in os.listdir(WATCH_DIR):
        if not name.startswith(PREFIX):
            continue
        if name.startswith(".uploading__"):
            continue
        path = os.path.join(WATCH_DIR, name)
        if not os.path.isdir(path):
            continue

        k = iter_number(name)
        if k is None:
            # iter_123 以外（例: iter_latest 等）は無視
            continue

        items.append((name, path, k))
    # 連番でソート
    items.sort(key=lambda x: x[2])
    return items


def upload_then_delete(name: str, src_path: str):
    staging_name = f".uploading__{name}"
    staging_path = os.path.join(WATCH_DIR, staging_name)

    # 二重処理防止（同一FSなら原子的）
    try:
        os.rename(src_path, staging_path)
    except Exception:
        return

    try:
        api.upload_folder(
            folder_path=staging_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_in_repo=os.path.join("checkpoints", name),
            commit_message=f"Add folder: {name}",
        )
    except Exception as e:
        # 失敗したら戻して保持
        try:
            os.rename(staging_path, src_path)
        except Exception:
            pass
        print(f"[ERROR] upload failed, kept local: {name} ({e})")
        return

    # 成功したら削除
    try:
        shutil.rmtree(staging_path)
        print(f"[OK] uploaded & deleted: {name}")
    except Exception as e:
        print(f"[WARN] uploaded but failed to delete local: {name} ({e})")


def main():
    print("start")
    while True:
        items = list_iter_folders()
        if len(items) > KEEP_LATEST:
            keep = items[-KEEP_LATEST:]
            old = items[:-KEEP_LATEST]
            print(f"[KEEP] {KEEP_LATEST}: {[x[0] for x in keep]}")
            for name, path, _ in old:
                upload_then_delete(name, path)
        time.sleep(SCAN_INTERVAL_SEC)


if __name__ == "__main__":
    main()
