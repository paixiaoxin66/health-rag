"""API 请求/响应数据模型（Pydantic）。"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """问答请求。"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="用户问题",
        examples=["高血压人群应该怎么饮食？"],
    )


class SourceDoc(BaseModel):
    """回答引用的来源文档。"""

    source: str = Field(description="来源文件名")
    page: int | None = Field(default=None, description="页码（PDF）")
    score: float = Field(description="相关度得分")
    snippet: str = Field(description="内容片段（前 200 字符）")


class TraceStep(BaseModel):
    """分步追踪：记录每一步的状态与耗时。"""

    step: str = Field(description="步骤名（retrieve/build_context/build_prompt/llm_generate）")
    status: str = Field(description="状态（ok / error / skipped）")
    elapsed_ms: float = Field(description="耗时（毫秒）")
    detail: str | None = Field(default=None, description="细节（如模型名、候选数）")


class AskResponse(BaseModel):
    """问答响应（含分步追踪，有迹可循）。"""

    answer: str = Field(description="生成的回答")
    sources: list[SourceDoc] = Field(description="引用的来源")
    steps: list[TraceStep] = Field(description="分步执行记录")


class RetrieveRequest(BaseModel):
    """检索请求。"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="检索问题",
    )


class RetrievedDoc(BaseModel):
    """单条检索结果。"""

    id: str = Field(description="chunk id")
    source: str = Field(description="来源文件名")
    page: int | None = Field(default=None, description="页码（PDF）")
    score: float = Field(description="相关度得分")
    content: str = Field(description="chunk 内容")


class RetrieveResponse(BaseModel):
    """检索响应。"""

    documents: list[RetrievedDoc] = Field(description="检索到的文档列表")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(description="服务状态")
    vector_store: str = Field(description="向量库路径")
    chunk_count: int = Field(description="知识库 chunk 数")
    embedding_model: str = Field(description="Embedding 模型")
    reranker_model: str = Field(description="Reranker 模型")
    llm_model: str = Field(description="LLM 模型")
