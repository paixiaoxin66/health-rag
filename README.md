# health-rag
个人健康管理系统，用于记录和管理日常健康数据，包括体检指标、运动记录、饮食日志等，帮助用户直观了解自身健康状况变化趋势

## Tech Stack

- Python 3.12
- LangChain
- Sentence Transformers
- ChromaDB
- FastAPI
- DeepSeek
- Docker

## Project Structure

```text
health-rag/
├── data/
├── docs/
├── evaluation/
├── prompts/
├── scripts/
├── src/
│   └── health_rag/
│       ├── api/
│       ├── config/
│       ├── generation/
│       ├── ingestion/
│       ├── pipeline/
│       └── retrieval/
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md