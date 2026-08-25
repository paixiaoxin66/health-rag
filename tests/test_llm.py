from unittest.mock import MagicMock, patch

import pytest

from health_rag.generation.llm import LLMService


def test_llm_generate():
    """测试 LLM 生成逻辑。"""

    mock_response = MagicMock()

    mock_response.choices[0].message.content = (
        "高血压患者应该注意控制钠盐摄入。"
    )

    with patch(
        "health_rag.generation.llm.OpenAI"
    ) as mock_openai:

        mock_client = mock_openai.return_value

        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        service = LLMService(
            api_key="test-api-key"
        )

        result = service.generate(
            system_prompt="你是健康知识助手。",
            user_prompt="高血压患者应该注意什么？",
        )

        assert result == "高血压患者应该注意控制钠盐摄入。"

        mock_client.chat.completions.create.assert_called_once()


def test_empty_system_prompt():
    """测试空 System Prompt。"""

    service = LLMService(
        api_key="test-api-key"
    )

    with pytest.raises(ValueError):
        service.generate(
            system_prompt="",
            user_prompt="测试问题",
        )


def test_empty_user_prompt():
    """测试空 User Prompt。"""

    service = LLMService(
        api_key="test-api-key"
    )

    with pytest.raises(ValueError):
        service.generate(
            system_prompt="你是健康助手。",
            user_prompt="",
        )