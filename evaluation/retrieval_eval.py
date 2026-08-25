"""检索层评估：Recall@K / Precision@K / MRR / NDCG@K。

完全本地运行（不需要 LLM）：
    python evaluation/retrieval_eval.py
"""

import json
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_rag.config.settings import get_settings
from health_rag.embedding.embedding import get_embedding_service
from health_rag.retrieval.retriever import HealthRetriever
from health_rag.rerank.reranker import get_reranker
from health_rag.vectorstore.chroma import ChromaVectorStore

from evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_questions() -> list[dict]:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return data["questions"]


def main():
    settings = get_settings()
    print("=" * 64)
    print("检索层评估 (Retrieval Evaluation)")
    print(f"向量库: {settings.vector_store_path}")
    print(f"模型: {settings.embedding_model}")
    print(f"top_k: {settings.top_k}, recall_k: {settings.recall_k}")
    print("=" * 64)

    # 初始化
    embedder = get_embedding_service()
    reranker = get_reranker()
    vector_store = ChromaVectorStore(
        persist_directory=settings.vector_store_path,
        collection_name="health_knowledge",
    )
    retriever = HealthRetriever(
        embedding_service=embedder,
        vector_store=vector_store,
        top_k=settings.top_k,
        reranker=reranker,
        recall_k=settings.recall_k,
    )

    questions = load_questions()
    if not questions:
        print("评测集为空！")
        return

    per_q = []
    agg_recall, agg_precision, agg_mrr, agg_ndcg = [], [], [], []

    print(f"\n共 {len(questions)} 条问题\n")
    for q in questions:
        relevant = list(q["sources"])
        results = retriever.retrieve(q["question"])

        retrieved_sources = [r.metadata.get("source", "") for r in results]
        chunk_hits = [src in relevant for src in retrieved_sources]

        # NDCG 用去重命中：每个相关文档只算首次出现（避免同源多 chunk 重复计分）
        seen: set[str] = set()
        unique_hits = []
        for src in retrieved_sources:
            if src in relevant and src not in seen:
                unique_hits.append(True)
                seen.add(src)
            else:
                unique_hits.append(False)

        recall = recall_at_k(retrieved_sources, relevant)
        precision = precision_at_k(retrieved_sources, relevant)
        mrr_val = mrr(chunk_hits)
        ndcg = ndcg_at_k(unique_hits, total_relevant=len(relevant))

        agg_recall.append(recall)
        agg_precision.append(precision)
        agg_mrr.append(mrr_val)
        agg_ndcg.append(ndcg)

        per_q.append(
            {
                "id": q["id"],
                "question": q["question"],
                "relevant": sorted(relevant),
                "retrieved": retrieved_sources,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "mrr": round(mrr_val, 4),
                "ndcg": round(ndcg, 4),
            }
        )

        status = "[HIT]" if any(chunk_hits) else "[MISS]"
        print(
            f"{status} [{q['id']}] Recall@{settings.top_k}={recall:.2f} "
            f"MRR={mrr_val:.2f}  {q['question'][:28]}"
        )

    # 汇总
    n = len(questions)
    summary = {
        "num_questions": n,
        "top_k": settings.top_k,
        "recall_k": settings.recall_k,
        "Recall@K": round(sum(agg_recall) / n, 4),
        "Precision@K": round(sum(agg_precision) / n, 4),
        "MRR": round(sum(agg_mrr) / n, 4),
        "NDCG@K": round(sum(agg_ndcg) / n, 4),
        "hit_questions": sum(1 for r in agg_recall if r > 0),
    }

    print("\n" + "=" * 64)
    print("评估汇总")
    print("=" * 64)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 保存结果
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "summary": summary,
        "details": per_q,
    }
    out_path = RESULTS_DIR / "retrieval_report.json"
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n详细报告已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()