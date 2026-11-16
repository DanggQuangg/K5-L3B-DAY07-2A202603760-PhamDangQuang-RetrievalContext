"""Script to run and benchmark all retrieval configurations on Dev and Test splits.
Generates comprehensive comparison metrics and traces for Day 7 report.
"""
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
    evaluate_retriever,
)


def run_benchmark():
    chunks_path = Path("data/chunks.json")
    questions_path = Path("data/questions.json")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    dev_questions = [q for q in questions if q.get("split") == "dev"]
    test_questions = [q for q in questions if q.get("split") == "test"]

    print(f"Loaded {len(chunks)} chunks, {len(dev_questions)} dev questions, {len(test_questions)} test questions.")

    # Initialize retrievers
    print("Initializing retrievers...")
    tfidf_retriever = TfidfRetriever(chunks)
    bm25_retriever = BM25Retriever(chunks)
    dense_retriever = DenseRetriever()
    hybrid_retriever = HybridRRFRetriever(dense_retriever, bm25_retriever, rrf_k=60, candidate_pool=10)

    results_dir = Path("results/retrieval")
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        # --- Baseline on Dev ---
        {
            "name": "Baseline (TF-IDF)",
            "retriever": tfidf_retriever,
            "questions": dev_questions,
            "split": "dev",
            "top_k": 3,
            "budget": 800,
            "use_filter": False,
            "hypothesis": "TF-IDF baseline provides lexical matching without dense semantic context or metadata filtering."
        },
        # --- Method 2 on Dev ---
        {
            "name": "Method 2: Dense Retrieval (Qdrant + BGE-Small)",
            "retriever": dense_retriever,
            "questions": dev_questions,
            "split": "dev",
            "top_k": 3,
            "budget": 800,
            "use_filter": False,
            "hypothesis": "Dense semantic embedding matches paraphrased questions better than exact lexical terms."
        },
        # --- Method 3 on Dev ---
        {
            "name": "Method 3: Hybrid Retrieval (Dense + BM25 via RRF)",
            "retriever": hybrid_retriever,
            "questions": dev_questions,
            "split": "dev",
            "top_k": 3,
            "budget": 800,
            "use_filter": False,
            "hypothesis": "Combining lexical BM25 and dense semantic search via RRF captures both technical terminology and conceptual queries."
        },
        # --- Improvement 1 on Dev ---
        {
            "name": "Improvement 1: Dense + Metadata Filter",
            "retriever": dense_retriever,
            "questions": dev_questions,
            "split": "dev",
            "top_k": 3,
            "budget": 800,
            "use_filter": True,
            "hypothesis": "Filtering on structured section metadata eliminates cross-section distractor chunks for scoped questions."
        },
        # --- Improvement 2 on Dev ---
        {
            "name": "Improvement 2: Hybrid RRF + Budget 1600 chars",
            "retriever": hybrid_retriever,
            "questions": dev_questions,
            "split": "dev",
            "top_k": 3,
            "budget": 1600,
            "use_filter": False,
            "hypothesis": "Expanding context budget from 800 to 1600 characters allows packing top-2/top-3 chunks, significantly improving Context Evidence Rate."
        },
    ]

    all_summaries = []

    for exp in experiments:
        print(f"Running experiment: {exp['name']} on {exp['split']}...")
        summary, traces = evaluate_retriever(
            retriever=exp["retriever"],
            questions=exp["questions"],
            strategy_name=exp["name"],
            top_k=exp["top_k"],
            budget=exp["budget"],
            use_filter=exp["use_filter"]
        )
        summary["hypothesis"] = exp["hypothesis"]
        all_summaries.append(summary)

        # Save individual traces
        safe_name = exp["name"].replace(" ", "_").replace(":", "").replace("+", "plus").replace("(", "").replace(")", "").lower()
        trace_file = results_dir / f"trace_{safe_name}.json"
        trace_file.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

    # Evaluate best chosen configuration and baseline on TEST
    # After analyzing dev results, chosen configuration: Improvement 2 (Hybrid RRF + Budget 1600)
    print("\nRunning Baseline on TEST split...")
    test_base_summary, test_base_traces = evaluate_retriever(
        retriever=tfidf_retriever,
        questions=test_questions,
        strategy_name="Baseline (TF-IDF) [Test]",
        top_k=3,
        budget=800,
        use_filter=False
    )
    test_base_summary["hypothesis"] = "Baseline on unseen test set"
    all_summaries.append(test_base_summary)
    (results_dir / "trace_test_baseline.json").write_text(json.dumps(test_base_traces, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Running Chosen Configuration (Hybrid RRF + Budget 1600) on TEST split...")
    test_chosen_summary, test_chosen_traces = evaluate_retriever(
        retriever=hybrid_retriever,
        questions=test_questions,
        strategy_name="Chosen Config: Hybrid RRF + Budget 1600 [Test]",
        top_k=3,
        budget=1600,
        use_filter=False
    )
    test_chosen_summary["hypothesis"] = "Chosen optimal config evaluated on independent test set"
    all_summaries.append(test_chosen_summary)
    (results_dir / "trace_test_chosen.json").write_text(json.dumps(test_chosen_traces, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save summary
    summary_file = results_dir / "summary.json"
    summary_file.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll experiments completed! Results saved to {summary_file}")

    # Print table
    print("\n" + "=" * 115)
    print(f"{'Configuration':<42} | {'Split':<5} | {'Hit@1':<14} | {'Hit@3':<14} | {'Context Rate':<14} | {'Median ms':<9}")
    print("-" * 115)
    for s in all_summaries:
        print(f"{s['strategy']:<42} | {s['split']:<5} | {s['hit1_rate']:<14} | {s['hit3_rate']:<14} | {s['context_hit_rate']:<14} | {s['median_ms']:<9.1f}")
    print("=" * 115)


if __name__ == "__main__":
    run_benchmark()
