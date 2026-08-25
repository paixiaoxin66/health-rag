"""日志系统：控制台（可读文本）+ 轮转文件（JSON 结构化）。

设计要点：
- 控制台：人类可读，开发时直观。
- 文件：JSON Lines，机器可解析，可对接 ELK/Loki 等日志平台（生产）。
- request_id：通过 contextvars 贯穿单次请求的所有日志，实现"有迹可循"。
- 文件日志按大小轮转（默认 5MB × 5 份），防止无限膨胀。
"""

import contextvars
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from health_rag.config.settings import get_settings

# ---------------------------------------------------------------------------
# request_id 上下文（contextvars：单次请求内全局可见，跨请求隔离）
# ---------------------------------------------------------------------------

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def set_request_id(request_id: str) -> None:
    """在当前请求上下文设置 request_id。"""
    _request_id_var.set(request_id)


def get_request_id() -> str:
    """读取当前请求的 request_id（无则为空串）。"""
    return _request_id_var.get("")


def _inject_request_id(record: logging.LogRecord) -> None:
    """把当前 request_id 注入日志记录（缺省用 '-'）。"""
    record.request_id = get_request_id() or "-"


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class TextFormatter(logging.Formatter):
    """控制台可读格式。"""

    def format(self, record: logging.LogRecord) -> str:
        _inject_request_id(record)
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """文件 JSON Lines 格式（结构化，含 request_id/耗时等附加字段）。"""

    def format(self, record: logging.LogRecord) -> str:
        _inject_request_id(record)
        data = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": record.request_id,
            "message": record.getMessage(),
        }
        # 附加字段：访问日志/步骤追踪等通过 extra 注入
        for key in ("method", "path", "status", "elapsed_ms", "step", "detail"):
            value = getattr(record, key, None)
            if value is not None:
                data[key] = value
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

_configured = False


def setup_logging(force: bool = False) -> None:
    """配置根 logger：控制台 + 轮转 JSON 文件。

    Args:
        force: True 时强制重新配置（清空已有 handler），主要用于测试。
    """
    global _configured

    if _configured and not force:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    # 1. 控制台（可读文本）
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        TextFormatter(
            fmt="%(asctime)s | %(levelname)-7s | req=%(request_id)s | %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(console)

    # 2. 文件（JSON Lines，轮转）
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    _configured = True
