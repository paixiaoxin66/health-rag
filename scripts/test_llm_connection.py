from health_rag.config.settings import get_settings
from health_rag.generation.llm import LLMService


def main():
    settings = get_settings()

    print("=" * 50)
    print("LLM 连接测试")
    print("=" * 50)

    print(f"Provider : {settings.llm_provider}")
    print(f"Model    : {settings.llm_model}")
    print(f"Base URL : {settings.deepseek_base_url}")
    print(
        f"API Key  : "
        f"{'已配置' if settings.deepseek_api_key else '未配置'}"
    )

    print("\n正在调用 LLM...")

    service = LLMService()

    answer = service.generate(
        system_prompt="你是一个专业、简洁的健康知识助手。",
        user_prompt="请用一句话解释什么是高血压。",
    )

    print("\n模型回答：")
    print(answer)

    print("\n" + "=" * 50)
    print("LLM API 测试成功")
    print("=" * 50)


if __name__ == "__main__":
    main()