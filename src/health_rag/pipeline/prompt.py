from pathlib import Path


class PromptBuilder:
    """RAG Prompt 构建器。"""

    def __init__(
        self,
        system_prompt_path: str = "prompts/system.txt",
        rag_prompt_path: str = "prompts/rag.txt",
    ):
        self.system_prompt = Path(system_prompt_path).read_text(
            encoding="utf-8"
        )

        self.rag_prompt = Path(rag_prompt_path).read_text(
            encoding="utf-8"
        )

    def build(
        self,
        query: str,
        context: str,
    ) -> dict[str, str]:
        """构建发送给 LLM 的消息。"""

        if not query or not query.strip():
            raise ValueError("query 不能为空")

        if not context or not context.strip():
            raise ValueError("context 不能为空")

        user_prompt = self.rag_prompt.format(
            context=context,
            query=query,
        )

        return {
            "system": self.system_prompt,
            "user": user_prompt,
        }