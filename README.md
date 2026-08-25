# health-rag

个人健康知识库检索增强生成（RAG）系统。基于本地知识库（营养/疾病/运动等主题文档）构建向量检索 + 重排 + LLM 生成，为健康管理系统提供可信、可追溯的知识问答能力。

## 特性

- 🧠 **本地 Embedding**：`BAAI/bge-small-zh-v1.5`（512 维，CPU 离线，零网络依赖）
- 📚 **生产级加载器**：多格式（txt/md/pdf/docx）+ 编码自动检测（utf-8/gbk）+ 错误隔离 + 递归
- 🔍 **召回 + 重排**：向量召回 20 候选 → `bge-reranker-base` 重排 → Top-5
- 💬 **LLM 生成**：OpenAI 兼容接口接入阿里云百炼 DashScope（`deepseek-v4-flash`）
- 📊 **RAG 评估**：30 条 QA 评测集，Recall@K / MRR / NDCG / 忠实度 / 相关性
- 📡 **FastAPI 服务**：`/ask`（含分步追踪）、`/retrieve`、`/health`
- 🪵 **结构化日志**：JSON Lines + request_id 贯穿全链路（有迹可循）
- 🧪 **68 个自动化测试**全覆盖

## 技术栈

Python 3.12 · LangChain · Sentence Transformers · ChromaDB · FastAPI · Uvicorn · Pydantic · DashScope(DeepSeek)

## 项目结构

```text
health-rag/
├── data/
│   ├── raw/                  # 语料（10 篇 .md，gitignore）
│   └── vector_store/         # ChromaDB 向量库（gitignore）
├── evaluation/
│   ├── questions.json        # 30 条评测集
│   ├── metrics.py            # 指标函数
│   ├── retrieval_eval.py     # 检索层评估（本地）
│   ├── generation_eval.py    # 生成层评估（LLM-as-judge）
│   └── results/              # 评估报告（质量基线）
├── logs/                     # 日志（JSON Lines，gitignore）
├── models/                   # 本地模型（gitignore）
├── prompts/                  # system / rag prompt 模板
├── scripts/
│   ├── ingest.py             # 一键摄入语料
│   ├── download_model.py     # 下载 embedding 模型
│   └── download_reranker.py  # 下载重排模型
├── src/health_rag/
│   ├── api/                  # main.py / schemas.py / middleware.py
│   ├── config/               # settings.py / logging.py
│   ├── embedding/            # EmbeddingService
│   ├── generation/           # LLMService
│   ├── ingestion/            # loader / splitter / pipeline
│   ├── pipeline/             # context / prompt / rag（含分步追踪）
│   ├── rerank/               # Reranker
│   ├── retrieval/            # HealthRetriever
│   └── vectorstore/          # ChromaVectorStore
├── tests/                    # 68 个测试
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

## RAG 流程

```
用户问题 → 向量召回(recall_k=20) → Reranker 重排 → Top-5
       → ContextBuilder 拼上下文 → PromptBuilder 构建提示
       → LLM(DashScope) 生成答案 → 返回 [答案 + 来源 + 分步 trace]
```

每次问答返回的 `steps` 记录每一步的状态与耗时（retrieve/build_context/build_prompt/llm_generate），配合日志中的 `request_id` 实现全链路可追溯。

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env   # 填写 DEEPSEEK_API_KEY（百炼 API Key）
```

### 3. 下载模型

```bash
python scripts/download_model.py     # bge-small-zh-v1.5
python scripts/download_reranker.py  # bge-reranker-base（约 1.1GB）
```

> 国内网络可在 `.env` 配置镜像源或按脚本提示操作。

### 4. 摄入语料

```bash
python scripts/ingest.py data/raw/
```

### 5. 启动 API

```bash
uvicorn health_rag.api.main:app --host 0.0.0.0 --port 8000
```

打开 http://127.0.0.1:8000/docs 查看交互式文档。

## API 端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 服务信息 |
| `/health` | GET | 健康检查 + 知识库规模 + 模型信息 |
| `/ask` | POST | RAG 问答：`{"question": "..."}` → 答案 + 来源 + 分步 trace |
| `/retrieve` | POST | 仅检索（调试检索质量，不调 LLM） |

## 测试

```bash
python -m pytest          # 68 个测试
```

## 评估

```bash
python evaluation/retrieval_eval.py    # 检索层（本地，不需要 LLM）
python evaluation/generation_eval.py   # 生成层（需要 DEEPSEEK_API_KEY）
```

当前质量基线（见 `evaluation/results/`）：

- 检索层：Recall@K=1.0 · MRR=0.983 · NDCG@K=0.988
- 生成层：Faithfulness=0.998 · Answer Relevance=1.0 · Context Relevance=0.99

## 日志

- 控制台：人类可读文本
- `logs/app.log`：JSON Lines 结构化日志（5MB 轮转 × 5 份）
- 每个请求自动生成 `request_id`，客户端可通过 `X-Request-ID` 头传入并原样回传

## Roadmap

- [x] Embedding 本地化
- [x] 配置系统
- [x] Ingestion Pipeline（多格式）
- [x] Reranker
- [x] RAG 评估
- [x] FastAPI 服务化
- [x] 日志系统
- [ ] Docker 容器化
- [ ] CI/CD
- [ ] 部署
- [ ] 与饮食管理系统集成
