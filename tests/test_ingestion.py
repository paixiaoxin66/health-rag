"""端到端测试：Ingestion Pipeline 完整流程。"""

import tempfile
from pathlib import Path

from health_rag.embedding.embedding import get_embedding_service
from health_rag.ingestion.pipeline import IngestionPipeline
from health_rag.vectorstore.chroma import ChromaVectorStore


def test_ingest_single_file(tmp_path):
    """测试单个文件摄入流程。"""
    # 准备测试文件
    test_file = tmp_path / "test_health.md"
    test_file.write_text(
        "# 健康饮食指南\n\n"
        "均衡饮食是维持人体健康的重要基础。\n\n"
        "## 蛋白质\n\n"
        "蛋白质是人体重要的营养素之一。\n\n"
        "## 碳水化合物\n\n"
        "碳水化合物是人体重要的能量来源。\n\n",
        encoding="utf-8",
    )

    # 创建向量库
    embedding_service = get_embedding_service()
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="ingestion_test",
    )

    # 执行摄入
    pipeline = IngestionPipeline(embedding_service, vector_store)
    chunk_count = pipeline.ingest_file(test_file)

    # 验证
    assert chunk_count > 0, "应至少产生 1 个 chunk"
    assert vector_store.count() == chunk_count, "向量库数量应与 chunk 数一致"


def test_ingest_metadata(tmp_path):
    """测试摄入后 metadata 是否完整。"""
    test_file = tmp_path / "健康饮食.md"
    test_file.write_text(
        "# 健康饮食\n\n"
        "均衡饮食是维持健康的基础。\n\n",
        encoding="utf-8",
    )

    embedding_service = get_embedding_service()
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="metadata_test",
    )

    pipeline = IngestionPipeline(embedding_service, vector_store)
    pipeline.ingest_file(test_file)

    # 检索验证 metadata
    query_embedding = embedding_service.embed_query("健康饮食")
    results = vector_store.search(query_embedding, top_k=1)

    metadata = results["metadatas"][0][0]
    assert metadata["source"] == "健康饮食.md"
    assert metadata["title"] == "健康饮食"
    assert metadata["chunk_id"].startswith("健康饮食_")
    assert metadata["category"] == "health"


def test_ingest_empty_directory(tmp_path):
    """测试空目录摄入不报错。"""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    embedding_service = get_embedding_service()
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="empty_test",
    )

    pipeline = IngestionPipeline(embedding_service, vector_store)
    total = pipeline.ingest_directory(empty_dir)

    assert total == 0
    assert vector_store.count() == 0