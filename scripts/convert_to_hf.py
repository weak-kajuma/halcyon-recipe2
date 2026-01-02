#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, hf_hub_download


# =========================
# 設定
# =========================
SRC_REPO_ID = "kajuma/qwen2.5-baseline"
DST_NAMESPACE = "plt4cpt"

TMP_DIR = Path("/mnt/hdd1/qwen2.5-baseline")              # 一時フォルダ（最後に全削除）
OUT_ROOT = Path("/mnt/hdd1/models")                       # swift export 出力先ルート（out_dir自体は作らない）
LOCAL_CACHE_ROOT = Path("/mnt/hdd1/qwen2.5-baseline")    # ローカルにある可能性
LOCAL_CKPT_ROOT = LOCAL_CACHE_ROOT / "checkpoints"       # /mnt/hdd1/qwen2.5-baseline/checkpoints/

ITERS = list(range(400, 29200 + 1, 400))
TORCH_DTYPE = "bfloat16"

# iter内のファイルを「2分割」して並列DL
FILE_DL_POOL = ThreadPoolExecutor(max_workers=2)


def run(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_rmtree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)


def copy_or_link(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # 同一FSなら高速
    except OSError:
        shutil.copy2(src, dst)


def ensure_args_json(tmp_dir: Path) -> None:
    """
    swift exportに必要な args.json を確保（TMP_DIR直下）。
    優先順位:
      1) TMP_DIR/args.json
      2) /mnt/hdd1/qwen2.5-baseline/args.json
      3) HF からDL（キャッシュ利用）
    """
    ensure_dir(tmp_dir)
    dst = tmp_dir / "args.json"
    if dst.exists():
        return

    src = LOCAL_CACHE_ROOT / "args.json"
    if src.exists():
        shutil.copy2(src, dst)
        return

    cached = hf_hub_download(
        repo_id=SRC_REPO_ID,
        repo_type="model",
        filename="args.json",
    )
    copy_or_link(Path(cached), dst)
    if not dst.exists():
        raise FileNotFoundError("args.json が取得できませんでした。")


def list_iter_files(all_repo_files: List[str], it_str: str) -> List[str]:
    prefix = f"checkpoints/iter_{it_str}/"
    return [p for p in all_repo_files if p.startswith(prefix) and not p.endswith("/")]


def split_two_groups(items: List[str]) -> (List[str], List[str]):
    mid = (len(items) + 1) // 2
    return items[:mid], items[mid:]


def download_group(paths: List[str], tmp_dir: Path) -> None:
    """
    paths: repo内相対パス (e.g., checkpoints/iter_0000400/xxx)
    各ファイルをHFキャッシュにDLし、TMP_DIR配下に配置（ハードリンク優先）
    """
    for rel in paths:
        subfolder, filename = rel.rsplit("/", 1)
        dst = tmp_dir / subfolder / filename
        if dst.exists():
            continue
        cached = hf_hub_download(
            repo_id=SRC_REPO_ID,
            repo_type="model",
            subfolder=subfolder,
            filename=filename,
        )
        copy_or_link(Path(cached), dst)


def stage_iter_checkpoint(tmp_dir: Path, it_str: str, all_repo_files: List[str]) -> Path:
    """
    まず TMP_DIR/checkpoints/iter_{it_str} に配置する。
    優先順位:
      1) /mnt/hdd1/qwen2.5-baseline/checkpoints/iter_{it_str} があれば symlink（失敗時はcopy）
      2) HF から iter配下ファイルを列挙して、2分割並列DL
    戻り値: TMP_DIR/checkpoints/iter_{it_str}
    """
    dst_iter_dir = tmp_dir / "checkpoints" / f"iter_{it_str}"
    if dst_iter_dir.exists():
        return dst_iter_dir

    ensure_dir(dst_iter_dir.parent)

    src_local = LOCAL_CKPT_ROOT / f"iter_{it_str}"
    if src_local.exists():
        try:
            os.symlink(str(src_local), str(dst_iter_dir), target_is_directory=True)
            return dst_iter_dir
        except OSError:
            shutil.copytree(src_local, dst_iter_dir, dirs_exist_ok=True)
            return dst_iter_dir

    files = list_iter_files(all_repo_files, it_str)
    if not files:
        raise FileNotFoundError(f"repo内に checkpoints/iter_{it_str}/ が見つかりません。")

    g1, g2 = split_two_groups(files)
    f1 = FILE_DL_POOL.submit(download_group, g1, tmp_dir)
    f2 = FILE_DL_POOL.submit(download_group, g2, tmp_dir)
    f1.result()
    f2.result()

    if not dst_iter_dir.exists():
        raise FileNotFoundError(f"checkpoints/iter_{it_str} の配置に失敗しました。")

    return dst_iter_dir


def move_iter_from_checkpoints_to_root(tmp_dir: Path, it_str: str) -> Path:
    """
    追加要件:
      TMP_DIR/checkpoints/iter_{it_str} から TMP_DIR/iter_{it_str} に移動する
    """
    src = tmp_dir / "checkpoints" / f"iter_{it_str}"
    dst = tmp_dir / f"iter_{it_str}"

    if not src.exists():
        raise FileNotFoundError(f"移動元が存在しません: {src}")

    # 既存があれば削除してから移動
    safe_rmtree(dst)

    try:
        os.rename(src, dst)  # 同一FSなら高速（symlinkでもOK）
    except OSError:
        shutil.move(str(src), str(dst))

    if not dst.exists():
        raise RuntimeError(f"iter_{it_str} の作成に失敗しました: {dst}")

    return dst


class DownloadHandle:
    def __init__(self, it: int):
        self.it = it
        self.event = threading.Event()
        self.err: Optional[BaseException] = None
        self.thread: Optional[threading.Thread] = None


def start_download_thread(
    handle: DownloadHandle,
    it_str: str,
    tmp_dir: Path,
    all_repo_files: List[str],
) -> None:
    """
    ダウンロード（or ローカル参照のsymlink/コピー）を別スレッドで実行。
    - TMP_DIR/checkpoints/iter_{it_str} を準備（2分割並列DL）
    - 「rootへ移動」は export側（メインスレッド）で行う
    """
    def _worker():
        try:
            safe_rmtree(tmp_dir / "checkpoints" / f"iter_{it_str}")
            stage_iter_checkpoint(tmp_dir, it_str, all_repo_files)
        except BaseException as e:
            handle.err = e
        finally:
            handle.event.set()

    t = threading.Thread(target=_worker, daemon=True)
    handle.thread = t
    t.start()


def main() -> int:
    if shutil.which("swift") is None:
        print("ERROR: `swift` が見つかりません (PATHにありません)。", file=sys.stderr)
        return 1

    if not OUT_ROOT.exists():
        print(f"ERROR: OUT_ROOT が存在しません: {OUT_ROOT}", file=sys.stderr)
        return 1

    api = HfApi()

    # resetは最後。ルート作成は可
    ensure_dir(TMP_DIR)
    ensure_dir(TMP_DIR / "checkpoints")

    # args.json は一度だけ確保（最後にTMP_DIRごと削除）
    ensure_args_json(TMP_DIR)

    # repo内ファイル一覧は一度取得して使い回す
    all_repo_files = api.list_repo_files(repo_id=SRC_REPO_ID, repo_type="model")

    # 先に最初のiterをDL開始
    cur_handle = DownloadHandle(ITERS[0])
    start_download_thread(cur_handle, f"{ITERS[0]:07d}", TMP_DIR, all_repo_files)

    for idx, it in enumerate(ITERS):
        it_str = f"{it:07d}"
        dst_repo_id = f"{DST_NAMESPACE}/baseline_{it_str}"
        out_dir = OUT_ROOT / f"qwen2.5-baseline_{it_str}"

        # このiterのDL完了待ち
        cur_handle.event.wait()
        if cur_handle.err is not None:
            raise cur_handle.err

        # 次iterのDLを先に開始（download律速を隠す）
        next_handle: Optional[DownloadHandle] = None
        if idx + 1 < len(ITERS):
            nxt = ITERS[idx + 1]
            next_handle = DownloadHandle(nxt)
            start_download_thread(next_handle, f"{nxt:07d}", TMP_DIR, all_repo_files)

        print(f"\n===== ITER {it_str} =====", flush=True)

        # output_dir は作らない（swiftに作らせる）。既存は削除。
        safe_rmtree(out_dir)

        # checkpoints/iter_{it_str} -> iter_{it_str} に移動（swiftがroot側を読む想定）
        safe_rmtree(TMP_DIR / f"iter_{it_str}")
        move_iter_from_checkpoints_to_root(TMP_DIR, it_str)

        # latest_checkpointed_iteration.txt（swift exportに必要）
        (TMP_DIR / "latest_checkpointed_iteration.txt").write_text(f"{it}\n", encoding="utf-8")

        # アップロード先repo作成（存在していればOK）
        api.create_repo(repo_id=dst_repo_id, repo_type="model", exist_ok=True)

        # swift export
        run([
            "swift", "export",
            "--mcore_model", str(TMP_DIR),
            "--to_hf", "true",
            "--torch_dtype", TORCH_DTYPE,
            "--output_dir", str(out_dir),
        ])

        # upload
        api.upload_folder(
            repo_id=dst_repo_id,
            repo_type="model",
            folder_path=str(out_dir),
            path_in_repo="",
            commit_message=f"Add HF export for iter_{it_str}",
        )

        # export後に削除（生成物 + このiterのcheckpoint）
        safe_rmtree(out_dir)
        safe_rmtree(TMP_DIR / f"iter_{it_str}")

        # 次へ
        if next_handle is not None:
            cur_handle = next_handle

    # 最後に reset（全削除）
    safe_rmtree(TMP_DIR)

    FILE_DL_POOL.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
