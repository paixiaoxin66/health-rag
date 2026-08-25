"""将 HF 缓存的 Embedding 模型导出到 ./models/，供 Docker 卷挂载使用。

背景：本地开发从 HF 缓存加载 embedding 模型；容器内没有 HF 缓存，
必须把模型放到 ./models/ 并通过 EMBEDDING_MODEL_PATH 指定。

用法：
    python scripts/export_embedding_model.py
"""

import shutil
import sys
from pathlib import Path

MODEL_ID = "BAAI/bge-small-zh-v1.5"
TARGET = Path("models") / "bge-small-zh-v1.5"


def find_hf_snapshot(model_id: str) -> Path | None:
    """在 HF 缓存中定位模型快照目录。"""
    cache_home = Path.home() / ".cache" / "huggingface" / "hub"
    dir_name = "models--" + model_id.replace("/", "--")
    snapshot_dir = cache_home / dir_name / "snapshots"
    if not snapshot_dir.is_dir():
        return None
    snaps = sorted(snapshot_dir.iterdir())
    return snaps[-1] if snaps else None


def main() -> None:
    if TARGET.exists() and any(TARGET.iterdir()):
        print(f"[跳过] 目标已存在: {TARGET}")
        return

    src = find_hf_snapshot(MODEL_ID)
    if src is None:
        print(
            f"[错误] HF 缓存中未找到 {MODEL_ID}，"
            "请先运行 scripts/download_model.py"
        )
        sys.exit(1)

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, TARGET)
    print(f"[完成] {src}")
    print(f"       -> {TARGET}")
    print("下一步：在 .env 设置 EMBEDDING_MODEL_PATH=models/bge-small-zh-v1.5")


if __name__ == "__main__":
    main()
