from openai import OpenAI

from health_rag.config.settings import get_settings


class LLMService:
    """LLM 服务封装。"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        settings = get_settings()

        self.model = model or settings.llm_model
        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 未配置，请检查 .env 文件"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """调用 LLM 生成回答。"""

        if not system_prompt.strip():
            raise ValueError("system_prompt 不能为空")

        if not user_prompt.strip():
            raise ValueError("user_prompt 不能为空")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
        )

        return response.choices[0].message.content or ""