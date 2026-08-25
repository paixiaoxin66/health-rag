"""评估指标纯函数测试（metrics.py，无重依赖）。"""

import math

from evaluation.metrics import (
    mrr,
    ndcg_at_k,
    parse_scores,
    precision_at_k,
    recall_at_k,
)


def test_ndcg_perfect_ranking():
    """全部命中时 NDCG = 1.0。"""
    assert ndcg_at_k([True, True, True]) == 1.0


def test_ndcg_first_only():
    """只有第一个命中，且真实相关文档共 3 个时，NDCG 与理想排序之比较低。"""
    dcg = 1.0 / math.log2(2)  # 第一个位置贡献
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    expected = dcg / idcg
    assert abs(ndcg_at_k([True, False, False], total_relevant=3) - expected) < 1e-9


def test_ndcg_single_relevant_found_late():
    """真实相关文档只有 1 个，但排在第 3 位：NDCG < 1。"""
    # 理想排序 = [True]（1 个相关文档）
    ndcg = ndcg_at_k([False, False, True], total_relevant=1)
    expected = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
    assert abs(ndcg - expected) < 1e-9


def test_ndcg_default_total_relevant():
    """未指定 total_relevant 时退化为用命中数（简化）。"""
    assert ndcg_at_k([True, False, False]) == 1.0  # 1 个命中即理想排序


def test_ndcg_truncated_k():
    """K 截断：只计算前 K 个位置。"""
    hits = [False, False, True, True, True]
    assert abs(ndcg_at_k(hits, k=2) - 0.0) < 1e-9  # 前2位都未命中
    assert ndcg_at_k(hits, k=5) > 0.0


def test_ndcg_no_hits():
    assert ndcg_at_k([False, False, False]) == 0.0


def test_recall_dedup_same_source():
    """同一文档多个 chunk 只算一次命中（去重）。"""
    retrieved = ["a.md", "a.md", "a.md", "b.md", "b.md"]
    assert recall_at_k(retrieved, ["a.md"]) == 1.0
    assert recall_at_k(retrieved, ["a.md", "b.md"]) == 1.0


def test_recall_partial():
    retrieved = ["a.md", "x.md", "y.md"]
    assert recall_at_k(retrieved, ["a.md", "b.md"]) == 0.5


def test_recall_miss():
    assert recall_at_k(["x.md"], ["a.md"]) == 0.0


def test_recall_truncated_k():
    retrieved = ["a.md", "x.md", "y.md"]
    assert recall_at_k(retrieved, ["a.md", "b.md"], k=1) == 0.5  # 前1位命中 a


def test_precision_basic():
    retrieved = ["a.md", "x.md", "y.md", "z.md", "w.md"]
    assert precision_at_k(retrieved, ["a.md"]) == 0.2  # 1/5


def test_precision_truncated_k():
    retrieved = ["a.md", "x.md", "y.md"]
    assert precision_at_k(retrieved, ["a.md"], k=1) == 1.0


def test_mrr_first_position():
    assert mrr([True, True, True]) == 1.0


def test_mrr_second_position():
    assert mrr([False, True, True]) == 0.5


def test_mrr_miss():
    assert mrr([False, False, False]) == 0.0


def test_parse_scores_json():
    text = '{"faithfulness": 0.9, "answer_relevance": 0.8, "context_relevance": 0.7}'
    scores = parse_scores(text)
    assert scores["faithfulness"] == 0.9
    assert scores["answer_relevance"] == 0.8
    assert scores["context_relevance"] == 0.7


def test_parse_scores_json_with_prose():
    """JSON 混在文字里也能解析。"""
    text = '评估结果如下：{"faithfulness": 1.0, "answer_relevance": 0.95, "context_relevance": 0.88}，完毕。'
    scores = parse_scores(text)
    assert scores["faithfulness"] == 1.0
    assert scores["answer_relevance"] == 0.95
    assert scores["context_relevance"] == 0.88


def test_parse_scores_fallback():
    """非 JSON 输出时按关键字兜底提取。"""
    text = "faithfulness: 0.75\nanswer_relevance 0.6\ncontext_relevance = 0.5"
    scores = parse_scores(text)
    assert scores["faithfulness"] == 0.75
    assert scores["answer_relevance"] == 0.6
    assert scores["context_relevance"] == 0.5


def test_parse_scores_garbage():
    """无法解析时返回全 0，不抛异常。"""
    scores = parse_scores("完全没有分数信息")
    assert scores == {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_relevance": 0.0,
    }