from health_rag.vectorstore.chroma import ChromaVectorStore


def test_add_and_search(tmp_path, embedding_service):
    """测试写入向量并进行相似度检索。"""
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )

    documents = [
        "高血压患者应该限制钠盐摄入。",
        "糖尿病患者需要注意血糖控制。",
        "成年人应该保持适量运动。",
    ]

    ids = [
        "doc_001",
        "doc_002",
        "doc_003",
    ]

    metadatas = [
        {"source": "health_guide.pdf", "page": 1},
        {"source": "health_guide.pdf", "page": 2},
        {"source": "health_guide.pdf", "page": 3},
    ]

    embeddings = embedding_service.embed_documents(documents)

    vector_store.add_documents(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    query = "高血压饮食应该注意什么？"
    query_embedding = embedding_service.embed_query(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=2,
    )

    assert len(results["ids"][0]) == 2
    assert results["documents"][0][0] == documents[0]


def test_count(tmp_path, embedding_service):
    """测试文档数量统计。"""
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="count_test",
    )

    documents = [
        "高血压患者应该限制钠盐摄入。",
        "成年人应该保持适量运动。",
    ]

    embeddings = embedding_service.embed_documents(documents)

    vector_store.add_documents(
        ids=["doc_001", "doc_002"],
        documents=documents,
        embeddings=embeddings,
    )

    assert vector_store.count() == 2


def test_delete(tmp_path, embedding_service):
    """测试删除文档。"""
    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="delete_test",
    )

    documents = [
        "高血压患者应该限制钠盐摄入。",
        "成年人应该保持适量运动。",
    ]

    embeddings = embedding_service.embed_documents(documents)

    vector_store.add_documents(
        ids=["doc_001", "doc_002"],
        documents=documents,
        embeddings=embeddings,
    )

    assert vector_store.count() == 2

    vector_store.delete(["doc_001"])

    assert vector_store.count() == 1