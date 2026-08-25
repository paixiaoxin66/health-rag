"""评估指标计算（纯函数，无重依赖，可独立单测）。"""

import json
import math
import re


def ndcg_at_k(
    hits: list[bool],
    k: int | None = None,
    total_relevant: int | None = None,
) -> float:
    """NDCG@K（二值相关性 0/1）。

    Args:
        hits: 每个检索位置的命中标记（True/False），按排序顺序。
        k: 截断位置，默认取全部。
        total_relevant: 该问题真实的全部相关文档数（用于理想排序
            IDCG）。默认用 hits 中的命中数——仅在无法获得真实相关
            文档总数时的简化。

    Returns:
        0.0 ~ 1.0
    """
    k = k or len(hits)
    dcg = 0.0
    for i, hit in enumerate(hits[:k]):
        if hit:
            dcg += 1.0 / math.log2(i + 2)
    n_relevant = (
        total_relevant if total_relevant is not None else sum(hits)
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    retrieved_sources: list[str],
    relevant_sources: list[str],
    k: int | None = None,
) -> float:
    """Recall@K：检索结果中去重后的相关文档命中率。

    Args:
        retrieved_sources: 检索到的文档来源（按排序顺序，可含重复 chunk）。
        relevant_sources: 该问题的相关文档来源。

    Returns:
        0.0 ~ 1.0
    """
    retrieved = retrieved_sources[:k] if k else retrieved_sources
    relevant = set(relevant_sources)
    found = set(src for src in retrieved if src in relevant)
    return len(found) / len(relevant) if relevant else 0.0


def precision_at_k(
    retrieved_sources: list[str],
    relevant_sources: list[str],
    k: int | None = None,
) -> float:
    """Precision@K：Top-K 中相关文档（去重）占比。"""
    retrieved = retrieved_sources[:k] if k else retrieved_sources
    relevant = set(relevant_sources)
    found = set(src for src in retrieved if src in relevant)
    return len(found) / len(retrieved) if retrieved else 0.0


def mrr(chunk_hits: list[bool]) -> float:
    """MRR：第一个命中位置倒数的平均（单条问题）。"""
    for i, hit in enumerate(chunk_hits):
        if hit:
            return 1.0 / (i + 1)
    return 0.0


def parse_scores(text: str) -> dict:
    """从 LLM-as-judge 输出解析三个分数，尽量健壮。

    Returns:
        {"faithfulness": float, "answer_relevance": float, "context_relevance": float}
    """
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return {
                "faithfulness": float(data.get("faithfulness", 0.0)),
                "answer_relevance": float(data.get("answer_relevance", 0.0)),
                "context_relevance": float(data.get("context_relevance", 0.0)),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # 兜底：按关键字就近提取数字
    result = {}
    for key in ("faithfulness", "answer_relevance", "context_relevance"):
        km = re.search(key + r"[^0-9]*([\d.]+)", text, re.IGNORECASE)
        result[key] = float(km.group(1)) if km else 0.0
    return result