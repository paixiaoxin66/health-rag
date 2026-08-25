from unittest.mock import MagicMock

import pytest

from health_rag.pipeline.rag import RAGPipeline


def test_rag_pipeline_ask():
    """测试完整 RAG Pipeline。"""

    rag = RAGPipeline()

    # Mock 检索器
    rag.retriever = MagicMock()

    rag.retriever.retrieve.return_value = [
        MagicMock(
            content="高血压患者应该限制钠盐摄入。",
            score=0.95,
            metadata={
                "source": "health.pdf",
                "page": 12,
            },
            id="doc_001",
        )
    ]

    # Mock LLM
    rag.llm = MagicMock()
    rag.llm.generate.return_value = "高血压患者应该限制钠盐摄入。"

    answer = rag.ask("高血压患者应该怎么饮食？")

    assert answer == "高血压患者应该限制钠盐摄入。"

    rag.retriever.retrieve.assert_called_once_with(
        "高血压患者应该怎么饮食？"
    )

    rag.llm.generate.assert_called_once()


def test_empty_query():
    """测试空问题。"""

    rag = RAGPipeline()

    with pytest.raises(ValueError):
        rag.ask("")


def test_no_context():
    """测试知识库没有检索到内容。"""

    rag = RAGPipeline()

    rag.retriever = MagicMock()
    rag.retriever.retrieve.return_value = []

    answer = rag.ask("一个知识库中不存在的问题")

    assert "没有找到" in answer

    rag.llm = MagicMock()
    rag.llm.generate.assert_not_called()