from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer

from health_rag.config.settings import get_settings


class EmbeddingService:
    """BGE Embedding 服务封装。

    - 优先使用本地模型路径（EMBEDDING_MODEL_PATH），
      不存在时回退到 HF 模型名从缓存加载
    - 默认 local_files_only=True（离线加载，不联网校验）
    - 通过 get_embedding_service() 获取进程级单例
    """

    def __init__(
        self,
        model_name: str | None = None,
        model_path: str | None = None,
        local_files_only: bool | None = None,
    ):
        settings = get_settings()

        # 离线模式：构造参数 > settings > 默认 True
        self._local_files_only = (
            local_files_only
            if local_files_only is not None
            else settings.embedding_local_only
        )

        # 模型来源：本地路径 > 构造参数 > settings > HF 模型名
        _path = model_path or settings.embedding_model_path
        if _path and Path(_path).exists():
            self.model_name = _path
            self.model = SentenceTransformer(
                _path,
                local_files_only=self._local_files_only,
            )
        else:
            self.model_name = model_name or settings.embedding_model
            self.model = SentenceTransformer(
                self.model_name,
                local_files_only=self._local_files_only,
            )

    def embed_query(self, text: str) -> list[float]:
        """将单条查询文本转换为向量。"""
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """将多条文档文本转换为向量。"""
        if not texts:
            raise ValueError("文档列表不能为空")

        if any(not text or not text.strip() for text in texts):
            raise ValueError("文档中不能包含空文本")

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return vectors.tolist()

    @property
    def dimension(self) -> int:
        """返回 Embedding 向量维度。"""
        return self.model.get_embedding_dimension()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """进程级单例：整个程序只加载一次 Embedding 模型。"""
    return EmbeddingService()