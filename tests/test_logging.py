"""日志系统测试：配置、request_id 上下文、JSON 格式化、中间件。"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from health_rag.api.main import app, get_pipeline
from health_rag.config.logging import (
    JsonFormatter,
    TextFormatter,
    get_request_id,
    set_request_id,
    setup_logging,
)


def test_setup_logging_creates_handlers():
    """setup_logging 应创建控制台 + 文件两个 handler。"""
    setup_logging(force=True)
    root = logging.getLogger()
    handler_types = [type(h).__name__ for h in root.handlers]
    assert "StreamHandler" in handler_types
    assert "RotatingFileHandler" in handler_types


def test_setup_logging_idempotent():
    """重复调用不应重复添加 handler。"""
    setup_logging(force=True)
    n_before = len(logging.getLogger().handlers)
    setup_logging()  # 幂等：不再新增
    assert len(logging.getLogger().handlers) == n_before


def test_request_id_contextvars():
    """request_id 上下文设置/读取。"""
    assert get_request_id() == ""
    set_request_id("req-abc-123")
    assert get_request_id() == "req-abc-123"
    # 清理，避免污染其他测试
    set_request_id("")


def test_text_formatter_injects_request_id():
    """文本格式应注入 request_id。"""
    set_request_id("req-xyz")
    fmt = TextFormatter("%(request_id)s|%(message)s")
    record = logging.LogRecord("t", logging.INFO, "", 0, "hello", None, None)
    out = fmt.format(record)
    assert "req-xyz" in out
    set_request_id("")


def test_json_formatter_valid_json():
    """JSON 格式输出应为合法 JSON，且含关键字段。"""
    set_request_id("req-json-1")
    fmt = JsonFormatter()
    record = logging.LogRecord("t", logging.INFO, "", 0, "hello", None, None)
    record.elapsed_ms = 12.5
    record.status = 200
    data = json.loads(fmt.format(record))
    assert data["message"] == "hello"
    assert data["level"] == "INFO"
    assert data["request_id"] == "req-json-1"
    assert data["elapsed_ms"] == 12.5
    set_request_id("")


def test_request_middleware_adds_header_and_logs(caplog):
    """请求应带 X-Request-ID 响应头，且产生访问日志。"""

    class FakePipeline:
        def __init__(self):
            self.vectorstore = type("VS", (), {"count": lambda self: 1})()
            self.retriever = type("R", (), {"retrieve": lambda self, q: []})()

        def ask_with_trace(self, question):
            return {"answer": "x", "sources": [], "steps": []}

    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    caplog.set_level(logging.INFO)
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID"), "响应头应包含 request_id"

    # 访问日志应产生
    messages = [r.getMessage() for r in caplog.records]
    assert any(m == "access" for m in messages)
    app.dependency_overrides.clear()


def test_request_id_echoes_client_header(caplog):
    """调用方传入 X-Request-ID 时应原样回传。"""

    class FakePipeline:
        def __init__(self):
            self.vectorstore = type("VS", (), {"count": lambda self: 1})()
            self.retriever = type("R", (), {"retrieve": lambda self, q: []})()

        def ask_with_trace(self, question):
            return {"answer": "x", "sources": [], "steps": []}

    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    resp = client.get("/health", headers={"X-Request-ID": "client-trace-42"})
    assert resp.headers.get("X-Request-ID") == "client-trace-42"
    app.dependency_overrides.clear()
