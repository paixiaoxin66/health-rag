"""FastAPI 应用入口：健康知识库 RAG API。

启动：
    uvicorn health_rag.api.main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from health_rag.api import schemas
from health_rag.api.middleware import RequestLoggingMiddleware
from health_rag.config.logging import setup_logging
from health_rag.config.settings import get_settings
from health_rag.pipeline.rag import RAGPipeline

logger = logging.getLogger(__name__)

# 全局单例：应用启动时加载一次（模型/向量库/Reranker）
_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动初始化日志 + 加载模型，关闭释放。"""
    global _pipeline
    setup_logging()
    logger.info("正在初始化 RAG Pipeline（加载模型）...")
    _pipeline = RAGPipeline()
    logger.info("RAG Pipeline 就绪")
    yield
    _pipeline = None


app = FastAPI(
    title="Health RAG API",
    description="健康知识库检索增强生成 API（含分步追踪）",
    version="0.1.0",
    lifespan=lifespan,
)

# 请求日志中间件：request_id + 访问日志
app.add_middleware(RequestLoggingMiddleware)


def get_pipeline() -> RAGPipeline:
    """FastAPI 依赖：返回全局 pipeline 单例。"""
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="服务尚未就绪，请稍后再试",
        )
    return _pipeline


@app.get("/", tags=["meta"])
def root():
    """服务信息。"""
    return {
        "service": "health-rag",
        "version": "0.1.0",
        "endpoints": ["/health", "/ask", "/retrieve"],
    }


@app.get("/health", response_model=schemas.HealthResponse, tags=["meta"])
def health(pipeline: RAGPipeline = Depends(get_pipeline)):
    """健康检查：服务状态 + 知识库规模 + 模型信息。"""
    settings = get_settings()
    return schemas.HealthResponse(
        status="ok",
        vector_store=str(settings.vector_store_path),
        chunk_count=pipeline.vectorstore.count(),
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        llm_model=settings.llm_model,
    )


@app.post("/ask", response_model=schemas.AskResponse, tags=["rag"])
def ask(
    req: schemas.AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """RAG 问答：返回答案 + 引用来源 + 分步追踪。"""
    try:
        result = pipeline.ask_with_trace(req.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("问答失败: %s", req.question)
        raise HTTPException(status_code=500, detail=f"内部错误: {e}")

    return schemas.AskResponse(
        answer=result["answer"],
        sources=[schemas.SourceDoc(**s) for s in result["sources"]],
        steps=[schemas.TraceStep(**s) for s in result["steps"]],
    )


@app.post("/retrieve", response_model=schemas.RetrieveResponse, tags=["rag"])
def retrieve(
    req: schemas.RetrieveRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """仅检索（不做 LLM 生成），用于调试检索质量。"""
    try:
        docs = pipeline.retriever.retrieve(req.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("检索失败: %s", req.question)
        raise HTTPException(status_code=500, detail=f"内部错误: {e}")

    return schemas.RetrieveResponse(
        documents=[
            schemas.RetrievedDoc(
                id=d.id,
                source=d.metadata.get("source", "未知来源"),
                page=d.metadata.get("page"),
                score=round(d.score, 4),
                content=d.content,
            )
            for d in docs
        ]
    )
