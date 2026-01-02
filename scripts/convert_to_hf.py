#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


# =========================
# 設定
# =========================
SRC_REPO_ID = "kajuma/qwen2.5-baseline"
DST_NAMESPACE = "plt4cpt"

TMP_DIR = Path("/mnt/hdd1/qwen2.5-baseline")              # 一時フォルダ（最後に全削除）
OUT_ROOT = Path("/mnt/hdd1/models")                       # swift export 出力先ルート（out_dir自体は作らない）
LOCAL_CACHE_ROOT = Path("/mnt/hdd1/qwen2.5-baseline")    # ローカルにある可能性
LOCAL_CKPT_ROOT = LOCAL_CACHE_ROOT / "checkpoints"       # /mnt/hdd1/qwen2.5-baseline/checkpoints/

# HFキャッシュもTMP配下に閉じ込めて最後に全削除
CACHE_DIR = TMP_DIR / ".hf_cache"

ITERS = list(range(36800, 14000-1, -400))
TORCH_DTYPE = "bfloat16"

# snapshot_download 内部の並列数（「split 2 groups」は廃止）
SNAPSHOT_MAX_WORKERS = 2


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
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def ensure_args_json(tmp_dir: Path) -> None:
    """
    swift exportに必要な args.json を確保（TMP_DIR直下）。
    優先順位:
      1) TMP_DIR/args.json
      2) LOCAL_CACHE_ROOT/args.json
      3) HF からDL（CACHE_DIRを使用）
    """
    ensure_dir(tmp_dir)
    ensure_dir(CACHE_DIR)

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
        cache_dir=str(CACHE_DIR),
    )
    copy_or_link(Path(cached), dst)
    if not dst.exists():
        raise FileNotFoundError("args.json が取得できませんでした。")


def stage_iter_checkpoint(tmp_dir: Path, it_str: str) -> Path:
    """
    まず TMP_DIR/checkpoints/iter_{it_str} に配置する。
    優先順位:
      1) LOCAL_CKPT_ROOT/iter_{it_str} が存在し、かつ別パスなら symlink（失敗時copy）
      2) HF から allow_patterns で iter配下のみダウンロード（snapshot_download の内部並列を使用）
    """
    ensure_dir(tmp_dir / "checkpoints")
    ensure_dir(CACHE_DIR)

    dst_iter_dir = tmp_dir / "checkpoints" / f"iter_{it_str}"
    if dst_iter_dir.exists():
        return dst_iter_dir

    src_local = LOCAL_CKPT_ROOT / f"iter_{it_str}"
    if src_local.exists():
        # 自己参照（同一パス）を避ける
        try:
            if src_local.resolve() != dst_iter_dir.resolve():
                try:
                    os.symlink(str(src_local), str(dst_iter_dir), target_is_directory=True)
                    return dst_iter_dir
                except OSError:
                    shutil.copytree(src_local, dst_iter_dir, dirs_exist_ok=True)
                    return dst_iter_dir
        except FileNotFoundError:
            # resolve で失敗するケースは無視して次へ
            pass

    snapshot_download(
        repo_id=SRC_REPO_ID,
        repo_type="model",
        local_dir=str(tmp_dir),
        allow_patterns=[f"checkpoints/iter_{it_str}/**"],
        cache_dir=str(CACHE_DIR),
        max_workers=SNAPSHOT_MAX_WORKERS,
        resume_download=True,
    )

    if not dst_iter_dir.exists():
        raise FileNotFoundError(f"checkpoints/iter_{it_str} の配置に失敗しました。")

    return dst_iter_dir


def move_iter_from_checkpoints_to_root(tmp_dir: Path, it_str: str) -> Path:
    """
    TMP_DIR/checkpoints/iter_{it_str} から TMP_DIR/iter_{it_str} に移動する
    """
    src = tmp_dir / "checkpoints" / f"iter_{it_str}"
    dst = tmp_dir / f"iter_{it_str}"

    if not src.exists():
        raise FileNotFoundError(f"移動元が存在しません: {src}")

    safe_rmtree(dst)
    try:
        os.rename(src, dst)
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
) -> None:
    """
    - TMP_DIR/checkpoints/iter_{it_str} を準備（snapshot_download内部並列）
    - rootへ移動はメイン側で実施
    """
    def _worker():
        try:
            safe_rmtree(tmp_dir / "checkpoints" / f"iter_{it_str}")
            stage_iter_checkpoint(tmp_dir, it_str)
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

    # 先に最初のiterをDL開始
    cur_handle = DownloadHandle(ITERS[0])
    start_download_thread(cur_handle, f"{ITERS[0]:07d}", TMP_DIR)

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
            start_download_thread(next_handle, f"{nxt:07d}", TMP_DIR)

        print(f"\n===== ITER {it_str} =====", flush=True)

        # output_dir は作らない（swiftに作らせる）。既存は削除。
        safe_rmtree(out_dir)

        # checkpoints/iter_{it_str} -> iter_{it_str} に移動
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

        if next_handle is not None:
            cur_handle = next_handle

    # 最後に reset（全削除：HFキャッシュも含む）
    safe_rmtree(TMP_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
