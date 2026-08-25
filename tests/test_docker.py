"""Docker 配置冒烟测试：验证 Dockerfile / .dockerignore / compose 配置正确性。"""

import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_exists_and_key_directives():
    """Dockerfile 应包含关键指令：基础镜像/工作目录/非root/健康检查/端口。"""
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in df, "基础镜像应为 python:3.12-slim"
    assert "WORKDIR /app" in df
    assert "USER appuser" in df, "应为非 root 用户运行"
    assert "HEALTHCHECK" in df, "应有健康检查"
    assert "EXPOSE 8000" in df
    assert "uvicorn" in df and "health_rag.api.main:app" in df


def test_dockerfile_references_requirements():
    """Dockerfile 应安装 requirements.txt。"""
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY requirements.txt ." in df
    assert "-r requirements.txt" in df


def test_dockerfile_copies_runtime_sources():
    """Dockerfile 应拷贝运行所需 src/ 与 prompts/。"""
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY src ./src" in df
    assert "COPY prompts ./prompts" in df


def test_dockerignore_excludes_sensitive():
    """敏感/运行数据必须被 .dockerignore 排除。"""
    di = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for entry in (".env", "models/", "data/", "logs/", ".venv/", "__pycache__/"):
        assert entry in di, f".dockerignore 缺少 {entry}"


def test_dockerignore_keeps_requirements_and_src():
    """构建必需文件不应被误排除。"""
    di = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "requirements.txt" not in di
    assert "src/" not in di
    assert "prompts/" not in di


def test_docker_compose_valid_yaml_and_config():
    """docker-compose.yml 应为合法 YAML，且端口/env_file/卷配置正确。"""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "api" in compose["services"]
    api = compose["services"]["api"]
    assert api["ports"] == ["8000:8000"]
    assert api["env_file"] == [".env"]
    assert "restart" in api
    # 模型/数据/日志三卷挂载
    volumes = " ".join(api.get("volumes", []))
    for v in ("/app/models", "/app/data", "/app/logs"):
        assert v in volumes, f"缺少挂载 {v}"
    # 构建上下文
    assert api["build"]["context"] == "."
    assert api["build"]["dockerfile"] == "Dockerfile"
