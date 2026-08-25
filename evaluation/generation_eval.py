"""生成层评估：Faithfulness / Answer Relevance / Context Relevance。

使用 LLM-as-judge（需配置 DEEPSEEK_API_KEY）：
    python evaluation/generation_eval.py            # 跑全部 30 条
    python evaluation/generation_eval.py --limit 3  # 只跑前 3 条（先验证连通性）
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_rag.config.settings import get_settings
from health_rag.embedding.embedding import get_embedding_service
from health_rag.retrieval.retriever import HealthRetriever
from health_rag.rerank.reranker import get_reranker
from health_rag.vectorstore.chroma import ChromaVectorStore
from health_rag.pipeline.context import ContextBuilder
from health_rag.pipeline.prompt import PromptBuilder
from health_rag.generation.llm import LLMService

from evaluation.metrics import parse_scores

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

JUDGE_SYSTEM = """你是一个严格的 RAG 质量评估裁判。请根据【问题】、【检索资料】和【回答】，从三个维度各打一个 0 到 1 之间的分数：
- faithfulness（忠实度）：回答是否完全基于检索资料、没有编造资料中不存在的关键信息。
- answer_relevance（回答相关性）：回答是否直接针对问题、答有所问。
- context_relevance（资料相关性）：检索资料是否包含回答该问题所需的足够相关信息。
只输出一个 JSON 对象，格式严格如下：
{"faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    args = parser.parse_args()

    settings = get_settings()
    print("=" * 64)
    print("生成层评估 (Generation Evaluation)")
    print(f"模型: {settings.llm_model}")
    print(f"Base URL: {settings.deepseek_base_url}")
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
    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()
    llm = LLMService()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    if args.limit > 0:
        questions = questions[: args.limit]

    print(f"\n共 {len(questions)} 条问题\n")

    agg = {"faithfulness": [], "answer_relevance": [], "context_relevance": []}
    details = []
    errors = []

    for i, q in enumerate(questions, 1):
        query = q["question"]
        try:
            # 1. 检索
            docs = retriever.retrieve(query)
            context = context_builder.build(docs)

            # 2. 生成
            prompt = prompt_builder.build(query=query, context=context)
            answer = llm.generate(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
                temperature=0.2,
            )

            # 3. LLM-as-judge 打分
            judge_user = (
                f"【问题】\n{query}\n\n【检索资料】\n{context}\n\n【回答】\n{answer}"
            )
            judge_text = llm.generate(
                system_prompt=JUDGE_SYSTEM,
                user_prompt=judge_user,
                temperature=0.0,
            )
            scores = parse_scores(judge_text)

            for key in agg:
                agg[key].append(scores[key])

            details.append(
                {
                    "id": q["id"],
                    "question": query,
                    "answer": answer,
                    "scores": scores,
                }
            )
            print(
                f"[{i}/{len(questions)}] {q['id']} "
                f"faith={scores['faithfulness']:.2f} "
                f"ans_rel={scores['answer_relevance']:.2f} "
                f"ctx_rel={scores['context_relevance']:.2f}  {query[:20]}"
            )
        except Exception as e:
            errors.append({"id": q["id"], "error": str(e)})
            print(f"[{i}/{len(questions)}] {q['id']} ERROR: {e}")

    # 汇总
    print("\n" + "=" * 64)
    print("评估汇总")
    print("=" * 64)
    summary = {"num_questions": len(questions), "num_evaluated": len(agg["faithfulness"])}
    for key in agg:
        vals = agg[key]
        summary[key] = round(sum(vals) / len(vals), 4) if vals else None
        print(f"  {key}: {summary[key]:.4f}" if summary[key] is not None else f"  {key}: 无数据")
    if errors:
        print(f"  errors: {len(errors)} 条失败")
        for e in errors:
            print(f"    {e['id']}: {e['error'][:80]}")

    # 保存
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"summary": summary, "details": details, "errors": errors}
    out_path = RESULTS_DIR / "generation_report.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细报告已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()