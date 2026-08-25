"""API 中间件：request_id 注入 + 访问日志。

实现为纯 ASGI 中间件（而非 BaseHTTPMiddleware），确保 request_id
通过 contextvars 在同一请求上下文内正确贯穿到下游处理器。
"""

import logging
import time
import uuid

from health_rag.config.logging import set_request_id

logger = logging.getLogger("health_rag.api.access")


class RequestLoggingMiddleware:
    """为每个请求生成/复用 request_id，记录访问日志，响应头回传。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 提取或生成 request_id（优先复用调用方传入的 X-Request-ID）
        headers = dict(scope.get("headers", []))
        raw_rid = headers.get(b"x-request-id")
        request_id = (
            raw_rid.decode("utf-8") if raw_rid else uuid.uuid4().hex[:12]
        )
        set_request_id(request_id)

        start = time.perf_counter()
        status_code = [0]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code[0] = message["status"]
                # 响应头回传 request_id，便于调用方追溯
                message = dict(message)
                message["headers"] = list(message["headers"]) + [
                    (b"x-request-id", request_id.encode("utf-8"))
                ]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "unhandled_error",
                extra={
                    "method": scope.get("method", ""),
                    "path": scope.get("path", ""),
                    "request_id": request_id,
                },
            )
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "access",
                extra={
                    "method": scope.get("method", ""),
                    "path": scope.get("path", ""),
                    "status": status_code[0],
                    "elapsed_ms": elapsed_ms,
                    "request_id": request_id,
                },
            )
