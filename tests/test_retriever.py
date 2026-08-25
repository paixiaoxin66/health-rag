import pytest

from health_rag.retrieval.retriever import (
    HealthRetriever,
    RetrievedDocument,
)
from health_rag.vectorstore.chroma import ChromaVectorStore


def test_retrieve(tmp_path, embedding_service):
    """测试检索器能够找到相关文档。"""
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="retrieval_test",
    )

    documents = [
        "高血压患者应该限制钠盐摄入。",
        "糖尿病患者需要注意血糖控制。",
        "成年人应该保持适量运动。",
    ]

    embeddings = embedding_service.embed_documents(documents)

    vector_store.add_documents(
        ids=[
            "doc_001",
            "doc_002",
            "doc_003",
        ],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {"source": "health.pdf", "page": 1},
            {"source": "health.pdf", "page": 2},
            {"source": "health.pdf", "page": 3},
        ],
    )

    retriever = HealthRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=2,
    )

    results = retriever.retrieve(
        "高血压饮食应该注意什么？"
    )

    assert len(results) == 2
    assert isinstance(results[0], RetrievedDocument)
    assert results[0].id == "doc_001"
    assert results[0].content == documents[0]
    assert results[0].score > 0
    assert results[0].metadata["source"] == "health.pdf"
    assert results[0].metadata["page"] == 1


def test_empty_query(tmp_path, embedding_service):
    """测试空查询。"""
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="empty_query_test",
    )

    retriever = HealthRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")


class FakeReranker:
    """模拟 Reranker：按分数升序排序（故意反转，验证确实走了重排）。"""

    def __init__(self):
        self.last_top_k = None

    def rerank(self, query, documents, top_k=5):
        self.last_top_k = top_k
        # 故意反转：把分数最低的排到最前，验证集成逻辑确实调用了重排
        return sorted(
            documents,
            key=lambda d: d.score,
        )[:top_k]


def test_retrieve_with_reranker(tmp_path, embedding_service):
    """测试有 Reranker 时：先召回 recall_k 候选，再重排取 top_k。"""
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="reranker_test",
    )

    # 5 篇文档，向量检索默认会按相似度排
    documents = [
        "高血压患者应该限制钠盐摄入。",
        "糖尿病患者需要注意血糖控制。",
        "成年人应该保持适量运动。",
        "高血压人群应该每天测量血压。",
        "高血压患者应保持低盐饮食。",
    ]

    embeddings = embedding_service.embed_documents(documents)

    vector_store.add_documents(
        ids=[f"doc_{i:03d}" for i in range(5)],
        documents=documents,
        embeddings=embeddings,
    )

    fake_reranker = FakeReranker()
    retriever = HealthRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=2,
        reranker=fake_reranker,
        recall_k=5,
    )

    results = retriever.retrieve("高血压应该注意什么？")

    # FakeReranker 反转了顺序，所以第一个结果应是向量分数最低的
    assert len(results) == 2
    assert fake_reranker.last_top_k == 2
    # 确认返回的是重排后的结果（不是向量检索的原始 top_k）
    assert results[0].score <= results[1].score