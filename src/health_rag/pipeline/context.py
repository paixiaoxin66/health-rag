from health_rag.retrieval.retriever import RetrievedDocument


class ContextBuilder:
    """将检索结果构建为 LLM 可使用的上下文。"""

    def __init__(self, max_documents: int = 5):
        if max_documents <= 0:
            raise ValueError("max_documents 必须大于 0")

        self.max_documents = max_documents

    def build(self, documents: list[RetrievedDocument]) -> str:
        """将检索结果转换为结构化上下文。"""

        if not documents:
            return ""

        selected_documents = documents[: self.max_documents]

        context_parts = []

        for index, document in enumerate(selected_documents, start=1):
            source = document.metadata.get("source", "未知来源")
            page = document.metadata.get("page", "未知页码")

            context_parts.append(
                f"""[资料 {index}]
来源：{source}
页码：{page}
相关度：{document.score:.4f}

内容：
{document.content}
"""
            )

        return "\n".join(context_parts)