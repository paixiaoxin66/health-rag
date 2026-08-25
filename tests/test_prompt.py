import pytest

from health_rag.pipeline.prompt import PromptBuilder


def test_build_prompt():
    """测试 RAG Prompt 构建。"""
    builder = PromptBuilder()

    result = builder.build(
        query="高血压患者应该怎么饮食？",
        context="""
[资料 1]
来源：health.pdf
页码：12

内容：
高血压患者应该限制钠盐摄入。
""",
    )

    assert "高血压患者应该怎么饮食？" in result["user"]
    assert "限制钠盐摄入" in result["user"]
    assert "健康知识助手" in result["system"]


def test_empty_query():
    """测试空问题。"""
    builder = PromptBuilder()

    with pytest.raises(ValueError):
        builder.build(
            query="",
            context="测试内容",
        )


def test_empty_context():
    """测试空 Context。"""
    builder = PromptBuilder()

    with pytest.raises(ValueError):
        builder.build(
            query="高血压应该注意什么？",
            context="",
        )