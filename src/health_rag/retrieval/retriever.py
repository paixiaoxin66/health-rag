from dataclasses import dataclass
from typing import Any

from health_rag.config.settings import get_settings
from health_rag.embedding.embedding import EmbeddingService
from health_rag.vectorstore.chroma import ChromaVectorStore


@dataclass
class RetrievedDocument:
    """检索结果。"""

    id: str
    content: str
    score: float
    metadata: dict[str, Any]


class HealthRetriever:
    """健康知识库检索器。

    支持两种模式：
    - 无 Reranker：向量检索直接取 top_k（向后兼容）
    - 有 Reranker：先向量召回 recall_k 个候选，再重排取 top_k
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        top_k: int | None = None,
        reranker: Any | None = None,
        recall_k: int | None = None,
    ):
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        settings = get_settings()
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k or settings.top_k
        self.reranker = reranker

        # 有 Reranker 时，先召回 recall_k 个候选再重排
        self.recall_k = (
            recall_k if recall_k is not None else settings.recall_k
        )

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        """根据用户查询检索相关知识。"""
        if not query or not query.strip():
            raise ValueError("查询内容不能为空")

        query_embedding = self.embedding_service.embed_query(query)

        # 向量检索：有 Reranker 召回更多候选，否则直接用 top_k
        candidate_k = self.recall_k if self.reranker else self.top_k

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

        documents = results["documents"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        retrieved_documents = []

        for doc_id, document, distance, metadata in zip(
            ids,
            documents,
            distances,
            metadatas,
        ):
            retrieved_documents.append(
                RetrievedDocument(
                    id=doc_id,
                    content=document,
                    score=1 - distance,
                    metadata=metadata or {},
                )
            )

        # 有 Reranker：重排后取 top_k
        if self.reranker is not None:
            return self.reranker.rerank(
                query,
                retrieved_documents,
                top_k=self.top_k,
            )

        return retrieved_documents[: self.top_k]