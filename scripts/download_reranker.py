"""下载 Reranker ONNX 模型到本地目录（断点续传 + 自动重试）。

用法：
    python scripts/download_reranker.py

结果：
    models/bge-reranker-base/  ← 含 onnx/model.onnx 及 tokenizer 文件

说明：
    - 通过 hf-mirror 镜像下载，规避 huggingface_hub 的 HEAD 校验问题
    - 支持断点续传：中断后重跑会从上次位置继续
"""

import os
import tempfile
import time
from pathlib import Path

import requests

BASE_URL = "https://hf-mirror.com/BAAI/bge-reranker-base/resolve/main"
LOCAL_DIR = Path("models/bge-reranker-base")

# 需要下载的文件（镜像相对路径 → 本地相对路径）
FILES = [
    "model.safetensors",
    "config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

RETRY_LIMIT = 20


def download_file(url: str, dest: Path) -> bool:
    """断点续传下载单个文件。返回是否成功。"""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 已存在且非空则跳过
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  已存在: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return True

    # 已有部分下载 → 断点续传
    resume_from = dest.stat().st_size if dest.exists() else 0

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            headers = {"Range": f"bytes={resume_from}-"}
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(10, 60),
            ) as resp:
                if resp.status_code == 416:  # 已下载完整
                    print(f"  完成: {dest.name}")
                    return True
                if resp.status_code not in (200, 206):
                    print(
                        f"  {dest.name} HTTP {resp.status_code}, "
                        f"重试 {attempt}/{RETRY_LIMIT}"
                    )
                    time.sleep(2)
                    continue

                mode = "ab" if resp.status_code == 206 else "wb"
                with open(dest, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)

                print(f"  完成: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
                return True

        except requests.exceptions.RequestException as e:
            resume_from = dest.stat().st_size if dest.exists() else 0
            print(
                f"  {dest.name} 中断 ({str(e)[:50]}), "
                f"已下载 {resume_from/1e6:.1f} MB, "
                f"重试 {attempt}/{RETRY_LIMIT}"
            )
            time.sleep(2)

    print(f"  失败: {dest.name} (重试 {RETRY_LIMIT} 次仍失败)")
    return False


def main():
    print(f"目标目录: {LOCAL_DIR.resolve()}")
    print(f"镜像: {BASE_URL}")
    print()

    # 设置代理（如果存在）
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if key not in os.environ:
            proxy = "http://127.0.0.1:10808"
            os.environ[key] = proxy
            print(f"设置 {key}={proxy}")

    success = True
    for rel in FILES:
        url = f"{BASE_URL}/{rel}"
        dest = LOCAL_DIR / rel
        ok = download_file(url, dest)
        success = success and ok

    if success:
        print()
        print("=" * 60)
        print("模型下载完成!")
        print(f"目录: {LOCAL_DIR.resolve()}")
        print("Reranker 将使用 PyTorch 后端（model.safetensors）推理")
        print("=" * 60)
    else:
        print()
        print("部分文件下载失败，请重跑本脚本续传")
        raise SystemExit(1)


if __name__ == "__main__":
    main()