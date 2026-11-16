"""Evaluation and comparison of VisRAG Multimodal Agent vs Text-Only Agent.

Benchmarking specifically on visual-intensive, structural, and diagram-centric queries
from NIST AI RMF 1.0 (Figures 1, 2, 3 and Core Function Tables).
"""
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visrag import VisualDocumentInspector
from src.visrag_agent import VisRAGController, run_visrag_agent
from src.retrieval import HybridRRFRetriever, DenseRetriever, BM25Retriever


def run_visrag_benchmark():
    chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever()
    hybrid = HybridRRFRetriever(dense, bm25)

    inspector = VisualDocumentInspector()
    controller = VisRAGController()

    visual_eval_cases = [
        {
            "id": "vis_q01",
            "type": "diagram_figure_3",
            "question": "What best practice is recommended for separating AI actor roles in the AI Model dimension (Figure 3)?",
            "expected_key": "separate",
            "target_page": 16
        },
        {
            "id": "vis_q02",
            "type": "diagram_figure_1",
            "question": "According to Figure 1 (Examples of potential harms), how are potential harms categorized?",
            "expected_key": "harm to people",
            "target_page": 10
        },
        {
            "id": "vis_q03",
            "type": "diagram_figure_2",
            "question": "What two inner circles and outer circle are shown in Figure 2 representing the AI system lifecycle?",
            "expected_key": "dimensions",
            "target_page": 15
        },
        {
            "id": "vis_q04",
            "type": "table_core_functions",
            "question": "Which four core functions are organized across Tables 1 through 4 in the AI RMF Core?",
            "expected_key": "Govern",
            "target_page": 27
        },
        {
            "id": "vis_q05",
            "type": "visual_abstain",
            "question": "What color-coded circuit schematic diagram is provided in NIST AI 100-1 for hardware GPU wiring?",
            "expected_key": "abstain",
            "target_page": None
        }
    ]

    output_dir = Path("results/visrag")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("STARTING VISRAG MULTIMODAL EVALUATION SUITE")
    print(f"Total Visual Evaluation Cases: {len(visual_eval_cases)}")
    print("=" * 90)

    results = []
    latencies = []

    for idx, case in enumerate(visual_eval_cases, start=1):
        qid = case["id"]
        q = case["question"]
        print(f"\n[{idx}/{len(visual_eval_cases)}] Evaluating {qid} ({case['type']}):")
        print(f"  Question: {q}")

        start_t = time.perf_counter()
        res = run_visrag_agent(
            question=q,
            controller=controller,
            text_retriever=hybrid,
            visual_inspector=inspector,
            budget=2200
        )
        elapsed = (time.perf_counter() - start_t) * 1000
        latencies.append(elapsed)

        status = res["status"]
        answer = res.get("answer", "")
        citations = res.get("citations", [])
        visual_pages = res.get("visual_pages_inspected", [])
        events = res.get("events", [])

        # Check correctness
        exp_key = case["expected_key"].lower()
        if exp_key == "abstain":
            is_correct = (status == "abstain")
        else:
            is_correct = (exp_key in answer.lower()) and bool(citations)

        print(f"  -> Status: {status} | Correct: {is_correct}")
        print(f"  -> Visual Pages Inspected: {visual_pages}")
        print(f"  -> Citations: {citations}")
        print(f"  -> Elapsed: {elapsed:.0f}ms")

        results.append({
            "id": qid,
            "type": case["type"],
            "question": q,
            "status": status,
            "is_correct": is_correct,
            "answer": answer,
            "citations": citations,
            "visual_pages_inspected": visual_pages,
            "events_count": len(events),
            "elapsed_ms": elapsed,
            "trace_events": events
        })

    # Summary
    correct_count = sum(r["is_correct"] for r in results)
    visual_inspections_total = sum(len(r["visual_pages_inspected"]) for r in results)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    summary = {
        "suite": "VisRAG Multimodal Agent Evaluation",
        "cases": len(results),
        "accuracy": f"{correct_count}/{len(results)} ({correct_count / len(results) * 100:.1f}%)",
        "visual_pages_inspected_total": visual_inspections_total,
        "avg_latency_ms": round(avg_latency, 1),
        "results": results
    }

    (output_dir / "visrag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 90)
    print(f"VISRAG EVALUATION FINISHED: {correct_count}/{len(results)} ({correct_count / len(results) * 100:.1f}%)")
    print(f"Total Visual Page Inspections: {visual_inspections_total}")
    print(f"Results saved to {output_dir / 'visrag_summary.json'}")
    print("=" * 90)


if __name__ == "__main__":
    run_visrag_benchmark()
