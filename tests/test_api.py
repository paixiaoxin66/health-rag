"""API 层测试（FastAPI TestClient，Fake Pipeline，不调用真实 LLM/模型）。"""

import pytest
from fastapi.testclient import TestClient

from health_rag.api import main
from health_rag.api.main import app, get_pipeline
from health_rag.retrieval.retriever import RetrievedDocument

FAKE_SOURCE = "10-特殊人群营养策略与恢复.md"


class FakeVectorStore:
    def count(self) -> int:
        return 89


class FakeRetriever:
    def retrieve(self, question: str):
        return [
            RetrievedDocument(
                id="c_001",
                content="高血压患者应限制钠盐摄入。",
                score=0.95,
                metadata={"source": FAKE_SOURCE, "page": None},
            )
        ]


class FakePipeline:
    """不加载任何模型/不发真实 LLM 请求的假 Pipeline。"""

    def __init__(self):
        self.vectorstore = FakeVectorStore()
        self.retriever = FakeRetriever()

    def ask_with_trace(self, question: str) -> dict:
        return {
            "answer": "高血压患者应限制钠盐摄入。",
            "sources": [
                {
                    "source": FAKE_SOURCE,
                    "page": None,
                    "score": 0.95,
                    "snippet": "高血压患者应限制钠盐摄入。",
                }
            ],
            "steps": [
                {"step": "retrieve", "status": "ok", "elapsed_ms": 45.0, "detail": "recall_k=20 top_k=5 reranker=on"},
                {"step": "build_context", "status": "ok", "elapsed_ms": 1.0, "detail": "docs=1"},
                {"step": "build_prompt", "status": "ok", "elapsed_ms": 0.5, "detail": None},
                {"step": "llm_generate", "status": "ok", "elapsed_ms": 1200.0, "detail": "model=deepseek-v4-flash"},
            ],
        }


@pytest.fixture()
def client():
    """覆盖依赖注入为 FakePipeline；不用 lifespan（避免加载真实模型）。"""
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "health-rag"
    assert "endpoints" in data


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["chunk_count"] == 89
    assert data["embedding_model"]
    assert data["reranker_model"]
    assert data["llm_model"]


def test_ask(client):
    resp = client.post("/ask", json={"question": "高血压人群应该怎么饮食？"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "高血压患者应限制钠盐摄入。"
    # 来源
    assert data["sources"][0]["source"] == FAKE_SOURCE
    assert data["sources"][0]["score"] == 0.95
    # 分步追踪（有迹可循）
    steps = [s["step"] for s in data["steps"]]
    assert steps == ["retrieve", "build_context", "build_prompt", "llm_generate"]
    assert all(s["status"] == "ok" for s in data["steps"])
    assert all(s["elapsed_ms"] >= 0 for s in data["steps"])


def test_ask_empty_question(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422  # Pydantic 校验：min_length=1


def test_ask_missing_field(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_retrieve(client):
    resp = client.post("/retrieve", json={"question": "高血压"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["documents"]) == 1
    assert data["documents"][0]["source"] == FAKE_SOURCE
    assert data["documents"][0]["score"] == 0.95


def test_health_503_when_not_ready():
    """pipeline 未初始化时应返回 503。"""
    main._pipeline = None
    app.dependency_overrides.clear()
    c = TestClient(app)
    resp = c.get("/health")
    assert resp.status_code == 503
