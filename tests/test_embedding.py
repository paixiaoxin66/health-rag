import pytest


def test_embedding_dimension(embedding_service):
    """测试 Embedding 向量维度。"""
    vector = embedding_service.embed_query(
        "高血压患者应该注意什么？"
    )
    assert len(vector) == 512


def test_embed_documents(embedding_service):
    """测试批量文档 Embedding。"""
    texts = [
        "高血压患者应该减少钠盐摄入。",
        "成年人应该保持适量运动。",
    ]
    vectors = embedding_service.embed_documents(texts)

    assert len(vectors) == 2
    assert all(len(vector) == 512 for vector in vectors)


def test_embedding_dimension_property(embedding_service):
    """测试向量维度属性。"""
    assert embedding_service.dimension == 512


def test_empty_query(embedding_service):
    """测试空查询应该抛出异常。"""
    with pytest.raises(ValueError):
        embedding_service.embed_query("")