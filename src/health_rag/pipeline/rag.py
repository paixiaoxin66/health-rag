"""完整 RAG 流程（含分步追踪）。

设计目标：让每次问答"有迹可循"——每一步（检索/上下文/Prompt/LLM）
都记录状态、耗时与细节，便于线上排查与质量观测。
"""

import logging
import time

from health_rag.config.settings import get_settings
from health_rag.embedding.embedding import EmbeddingService
from health_rag.vectorstore.chroma import ChromaVectorStore
from health_rag.retrieval.retriever import HealthRetriever
from health_rag.pipeline.context import ContextBuilder
from health_rag.pipeline.prompt import PromptBuilder
from health_rag.generation.llm import LLMService

logger = logging.getLogger(__name__)


class RAGPipeline:
    """完整 RAG 流程。

    支持分步追踪（ask_with_trace），生产环境默认启用 Reranker
    （加载失败时自动降级为无重排，不阻塞服务启动）。
    """

    def __init__(self, enable_reranker: bool = True):
        settings = get_settings()
        self.settings = settings

        # 1. Embedding
        self.embedding = EmbeddingService()

        # 2. Vector Store
        self.vectorstore = ChromaVectorStore(
            persist_directory=settings.vector_store_path,
        )

        # 3. Reranker（可选，加载失败降级）
        self.reranker = None
        if enable_reranker:
            try:
                from health_rag.rerank.reranker import get_reranker

                self.reranker = get_reranker()
                logger.info("Reranker 已加载: %s", settings.reranker_model)
            except Exception as e:
                logger.warning("Reranker 加载失败，降级为无重排: %s", e)

        # 4. Retriever
        self.retriever = HealthRetriever(
            embedding_service=self.embedding,
            vector_store=self.vectorstore,
            top_k=settings.top_k,
            reranker=self.reranker,
            recall_k=settings.recall_k,
        )

        # 5. Context Builder
        self.context_builder = ContextBuilder()

        # 6. Prompt Builder
        self.prompt_builder = PromptBuilder()

        # 7. LLM
        self.llm = LLMService()

    def ask(self, query: str) -> str:
        """执行完整 RAG 问答，只返回答案（向后兼容）。"""
        return self.ask_with_trace(query)["answer"]

    def ask_with_trace(self, query: str) -> dict:
        """执行完整 RAG 问答，返回答案 + 来源 + 分步追踪。

        Returns:
            {
                "answer": str,
                "sources": [{"source", "page", "score", "snippet"}],
                "steps": [{"step", "status", "elapsed_ms", "detail"}],
            }
        """
        if not query or not query.strip():
            raise ValueError("查询内容不能为空")

        steps = []

        def run_step(name: str, fn, detail: str | None = None):
            """执行一步并记录状态/耗时（有迹可循）。"""
            start = time.perf_counter()
            try:
                result = fn()
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                steps.append(
                    {
                        "step": name,
                        "status": "ok",
                        "elapsed_ms": elapsed_ms,
                        "detail": detail,
                    }
                )
                logger.info(
                    "rag_step",
                    extra={
                        "step": name,
                        "status": "ok",
                        "elapsed_ms": elapsed_ms,
                        "detail": detail,
                    },
                )
                return result
            except Exception as e:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                steps.append(
                    {
                        "step": name,
                        "status": f"error: {e}",
                        "elapsed_ms": elapsed_ms,
                        "detail": detail,
                    }
                )
                logger.error(
                    "rag_step",
                    extra={
                        "step": name,
                        "status": f"error: {e}",
                        "elapsed_ms": elapsed_ms,
                        "detail": detail,
                    },
                    exc_info=True,
                )
                raise

        # 1. 检索（向量召回 recall_k → 重排取 top_k）
        docs = run_step(
            "retrieve",
            lambda: self.retriever.retrieve(query),
            detail=(
                f"recall_k={self.settings.recall_k} "
                f"top_k={self.settings.top_k} "
                f"reranker={'on' if self.reranker else 'off'}"
            ),
        )

        # 2. 构建上下文
        context = run_step(
            "build_context",
            lambda: self.context_builder.build(docs),
            detail=f"docs={len(docs)}",
        )

        # 无检索结果：不调用 LLM
        if not context:
            steps.append(
                {
                    "step": "llm_generate",
                    "status": "skipped (无检索结果)",
                    "elapsed_ms": 0.0,
                    "detail": None,
                }
            )
            return {
                "answer": "抱歉，知识库中暂时没有找到与该问题相关的信息。",
                "sources": [],
                "steps": steps,
            }

        # 3. 构建 Prompt
        prompt = run_step(
            "build_prompt",
            lambda: self.prompt_builder.build(query=query, context=context),
        )

        # 4. 调用 LLM 生成答案
        answer = run_step(
            "llm_generate",
            lambda: self.llm.generate(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
                temperature=0.2,
            ),
            detail=f"model={self.settings.llm_model}",
        )

        # 来源列表（供前端引用）
        sources = []
        for d in docs:
            sources.append(
                {
                    "source": d.metadata.get("source", "未知来源"),
                    "page": d.metadata.get("page"),
                    "score": round(d.score, 4),
                    "snippet": d.content[:200],
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "steps": steps,
        }
