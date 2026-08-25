"""下载并补全 Embedding 模型到本地目录。

用法：
    python scripts/download_model.py

结果：
    models/bge-small-zh-v1.5/  ← 完整模型（含 2_Normalize/config.json）
"""

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "BAAI/bge-small-zh-v1.5"
LOCAL_DIR = Path("models/bge-small-zh-v1.5")

# 如果本地已有，跳过下载
if LOCAL_DIR.exists() and (LOCAL_DIR / "model.safetensors").exists():
    print(f"模型已存在: {LOCAL_DIR.resolve()}")
else:
    print(f"正在下载 {MODEL_ID} ...")
    cache_dir = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"下载完成: {LOCAL_DIR.resolve()}")

# 补上缺失的 2_Normalize/config.json（上游仓库不含此文件）
missing_dir = LOCAL_DIR / "2_Normalize"
missing_file = missing_dir / "config.json"
if not missing_file.exists():
    missing_dir.mkdir(parents=True, exist_ok=True)
    missing_file.write_text("{}", encoding="utf-8")
    print(f"已补丁: {missing_file}")

# 验证
config_path = LOCAL_DIR / "config.json"
if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"模型验证通过: {config.get('_name_or_path', MODEL_ID)}")

print()
print("=" * 60)
print("请在 .env 中设置:")
print(f"  EMBEDDING_MODEL_PATH={LOCAL_DIR.resolve()}")
print("=" * 60)