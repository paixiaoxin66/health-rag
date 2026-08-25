from langchain_core.documents import Document

from health_rag.config.settings import get_settings
from health_rag.ingestion.splitter import split_documents

LONG_TEXT = (
    "健康饮食是维持人体健康的重要基础。"
    "合理的饮食应该包含适量的碳水化合物、蛋白质、脂肪，"
    "同时摄入足够的维生素、矿物质和膳食纤维。"
    "蛋白质是人体重要的营养素之一，主要参与肌肉、皮肤和其他组织的构成与修复。"
    "常见的蛋白质来源包括鱼类、禽肉、蛋类、奶类、豆类等。"
    "碳水化合物是人体重要的能量来源。"
    "全谷物、薯类、水果和部分豆类可以提供碳水化合物，"
    "同时也能够提供膳食纤维和其他营养成分。"
    "膳食纤维有助于维持正常的消化系统功能。"
    "日常饮食中可以通过增加蔬菜、水果、全谷物和豆类来获得膳食纤维。"
    "健康饮食并不是完全禁止某一种食物，"
    "而是强调食物种类多样、营养比例合理以及适量摄入。"
    "成年人每天应保证适量运动，配合均衡饮食以维持健康体重。"
    "控制钠盐摄入有助于降低高血压风险，建议每天不超过5克食盐。"
) * 5  # 重复 5 次，确保文本足够长，无论 chunk_size 如何都能切出多个 chunk


def test_split_documents():
    settings = get_settings()
    documents = [Document(page_content=LONG_TEXT)]

    chunks = split_documents(documents)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert len(chunk.page_content) <= settings.chunk_size
