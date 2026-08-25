# ============================================================
# health-rag API 镜像
#
# 设计要点：
# - python:3.12-slim 与本地开发版本一致，杜绝"环境不一致"
# - 模型/向量库/日志均通过 volume 挂载（镜像不含运行数据，体积可控）
# - 非 root 用户运行（安全基线）
# - 健康检查探针（容器就绪判定）
# - pip 镜像源可配置（默认清华，适配国内网络）
# ============================================================

FROM python:3.12-slim

# --- 构建参数 ---
# pip 镜像源（国内默认清华，海外部署可用 ARG 覆盖为官方源）
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# --- 基础环境 ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# --- 先装依赖（利用 Docker 层缓存：requirements 不变则不重装） ---
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --index-url ${PIP_INDEX_URL} \
    --trusted-host ${PIP_TRUSTED_HOST} \
    -r requirements.txt

# --- 拷贝运行所需源码与资源 ---
COPY src ./src
COPY prompts ./prompts
COPY pyproject.toml .

# --- 非 root 用户（安全基线） ---
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# --- 健康检查（API 就绪探针） ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

EXPOSE 8000

CMD ["uvicorn", "health_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
