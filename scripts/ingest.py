"""一键摄入知识库语料。

用法：
    python scripts/ingest.py [目录路径]

默认目录: data/raw/
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_rag.config.settings import get_settings
from health_rag.embedding.embedding import get_embedding_service
from health_rag.ingestion.pipeline import IngestionPipeline
from health_rag.vectorstore.chroma import ChromaVectorStore


def main():
    settings = get_settings()

    # 目录：命令行参数 > 默认值
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    dir_path = Path(directory)

    if not dir_path.is_dir():
        print(f"错误: 目录不存在: {dir_path.resolve()}")
        print("用法: python scripts/ingest.py [目录路径]")
        sys.exit(1)

    print("=" * 60)
    print(f"摄入目录: {dir_path.resolve()}")
    print(f"模型: {settings.embedding_model}")
    print(f"向量库: {settings.vector_store_path}")
    print(f"chunk_size: {settings.chunk_size}")
    print("=" * 60)

    # 初始化
    embedding_service = get_embedding_service()
    vector_store = ChromaVectorStore(
        persist_directory=settings.vector_store_path,
        collection_name="health_knowledge",
    )

    # 摄入（loader 递归扫描全部支持格式：md/txt/pdf/docx）
    pipeline = IngestionPipeline(embedding_service, vector_store)
    total = pipeline.ingest_directory(directory)

    print("=" * 60)
    print(f"完成: {total} 个 chunk 已写入向量库")
    print(f"向量库路径: {Path(settings.vector_store_path).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()