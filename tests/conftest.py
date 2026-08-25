import pytest

from health_rag.embedding.embedding import get_embedding_service


@pytest.fixture(scope="session")
def embedding_service():
    """整个测试会话只加载一次 Embedding 模型。"""
    return get_embedding_service()