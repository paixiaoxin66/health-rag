from pathlib import Path

import chromadb


class ChromaVectorStore:
    """Chroma 向量存储封装。"""

    def __init__(
        self,
        persist_directory: str = ".chroma",
        collection_name: str = "health_knowledge",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """添加文档、向量和 metadata。"""

        if not ids:
            raise ValueError("ids 不能为空")

        if not documents:
            raise ValueError("documents 不能为空")

        if len(ids) != len(documents):
            raise ValueError("ids 和 documents 数量必须一致")

        if len(documents) != len(embeddings):
            raise ValueError("documents 和 embeddings 数量必须一致")

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> dict:
        """根据查询向量进行相似度搜索。"""

        if not query_embedding:
            raise ValueError("query_embedding 不能为空")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def delete(self, ids: list[str]) -> None:
        """删除指定文档。"""

        if not ids:
            raise ValueError("ids 不能为空")

        self.collection.delete(ids=ids)

    def count(self) -> int:
        """返回当前 collection 中的文档数量。"""

        return self.collection.count()