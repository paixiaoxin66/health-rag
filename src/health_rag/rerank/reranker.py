"""Reranker 模块：对召回候选文档进行语义重排序。"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from health_rag.config.settings import get_settings


class Reranker:
    """基于 CrossEncoder 的重排序器。

    用法：
        reranker = Reranker()
        pairs = [(query, doc.content) for doc in candidates]
        scores = reranker.score(pairs)     # 返回相关性分数列表
        ranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str | None = None,
        local_files_only: bool | None = None,
    ):
        settings = get_settings()

        self.model_name = model_name or settings.reranker_model

        self._local_files_only = (
            local_files_only
            if local_files_only is not None
            else settings.reranker_local_only
        )

        self.model = CrossEncoder(
            self.model_name,
            local_files_only=self._local_files_only,
        )

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """对 (query, document) 对批量打分，返回相关性分数。"""
        if not pairs:
            return []

        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        documents: list,
        top_k: int = 5,
    ) -> list:
        """对候选文档重排序，返回排序后的文档列表（前 top_k 个）。

        Args:
            query: 用户查询
            documents: 候选文档列表（RetrievedDocument 或带 .content 的对象）
            top_k: 返回数量

        Returns:
            按相关性分数降序的文档列表（保留原对象，附加 score 属性）
        """
        if not query or not query.strip():
            raise ValueError("查询内容不能为空")

        if not documents:
            return []

        # 构造 (query, doc) 对
        pairs = [(query, doc.content) for doc in documents]
        scores = self.score(pairs)

        # 按分数降序排序
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # 附加重排分数到文档对象
        for doc, score in ranked:
            doc.rerank_score = score  # type: ignore[attr-defined]

        return [doc for doc, _ in ranked[:top_k]]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """进程级单例：整个程序只加载一次 Reranker 模型。"""
    return Reranker()