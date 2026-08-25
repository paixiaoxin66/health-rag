from health_rag.pipeline.context import ContextBuilder
from health_rag.retrieval.retriever import RetrievedDocument


def test_build_context():
    """测试上下文构建。"""

    documents = [
        RetrievedDocument(
            id="doc_001",
            content="高血压患者应该限制钠盐摄入。",
            score=0.95,
            metadata={
                "source": "health.pdf",
                "page": 12,
            },
        ),
        RetrievedDocument(
            id="doc_002",
            content="成年人应该保持适量运动。",
            score=0.82,
            metadata={
                "source": "health.pdf",
                "page": 15,
            },
        ),
    ]

    builder = ContextBuilder()

    context = builder.build(documents)

    assert "高血压患者应该限制钠盐摄入。" in context
    assert "成年人应该保持适量运动。" in context
    assert "health.pdf" in context
    assert "12" in context
    assert "15" in context


def test_empty_documents():
    """测试空检索结果。"""

    builder = ContextBuilder()

    context = builder.build([])

    assert context == ""


def test_max_documents():
    """测试最大文档数量限制。"""

    documents = [
        RetrievedDocument(
            id=f"doc_{i}",
            content=f"测试内容 {i}",
            score=0.9,
            metadata={"source": "test.pdf", "page": i},
        )
        for i in range(5)
    ]

    builder = ContextBuilder(max_documents=2)

    context = builder.build(documents)

    assert "测试内容 0" in context
    assert "测试内容 1" in context
    assert "测试内容 2" not in context