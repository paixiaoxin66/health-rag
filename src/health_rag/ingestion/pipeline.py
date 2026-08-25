"""Knowledge Ingestion Pipeline.

编排文档摄入全流程：
    文件 → 加载 → 切分 → 元数据 → 向量化 → 写入向量库
"""

from collections import defaultdict
from pathlib import Path

from health_rag.config.settings import get_settings
from health_rag.embedding.embedding import EmbeddingService
from health_rag.ingestion.loader import (
    load_document,
    load_documents_from_directory,
)
from health_rag.ingestion.splitter import split_documents
from health_rag.vectorstore.chroma import ChromaVectorStore


class IngestionPipeline:
    """健康知识库 Ingestion 流水线。

    Usage:
        pipeline = IngestionPipeline(embedding_service, vector_store)
        n = pipeline.ingest_file("data/raw/01-基础营养.md")
        total = pipeline.ingest_directory("data/raw/")
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.settings = get_settings()

    def ingest_file(
        self,
        file_path: str | Path,
    ) -> int:
        """摄入单个文件：加载 → 切分 → 元数据 → 向量化 → 入库。

        Returns:
            摄入的 chunk 数量。
        """
        path = Path(file_path)

        # 1. Load
        documents = load_document(path)

        # 2. Split
        chunks = split_documents(documents)

        # 3. Enrich metadata
        stem = path.stem  # 文件名去后缀，如 "01-基础营养-宏量营养素"
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = path.name
            chunk.metadata["title"] = stem
            chunk.metadata["chunk_id"] = f"{stem}_{i:03d}"
            chunk.metadata["category"] = "health"

        # 4. Embed
        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_service.embed_documents(texts)

        # 5. Store
        ids = [c.metadata["chunk_id"] for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self.vector_store.add_documents(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def ingest_directory(
        self,
        directory: str | Path,
    ) -> int:
        """摄入目录下所有支持格式的文档（递归）。

        用 loader 统一加载（含错误隔离），批量切分 → 元数据 → 向量化 → 入库。

        Returns:
            总摄入 chunk 数。
        """
        # 1. Load（loader 内部已做错误隔离 + 递归）
        documents = load_documents_from_directory(directory)
        if not documents:
            return 0

        # 2. Split
        chunks = split_documents(documents)

        # 3. Enrich metadata（按 source 分组，chunk_id 每组内独立编号）
        by_source = defaultdict(list)
        for chunk in chunks:
            by_source[chunk.metadata.get("source", "unknown")].append(chunk)

        for source, src_chunks in by_source.items():
            stem = Path(source).stem
            for i, chunk in enumerate(src_chunks):
                chunk.metadata["source"] = Path(source).name
                chunk.metadata["title"] = stem
                chunk.metadata["chunk_id"] = f"{stem}_{i:03d}"
                chunk.metadata["category"] = "health"

        # 4. Embed（一次批量向量化）
        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_service.embed_documents(texts)

        # 5. Store（一次批量入库）
        ids = [c.metadata["chunk_id"] for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self.vector_store.add_documents(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)