---
name: visrag-agent
description: Multimodal Visual Document RAG (VisRAG) and Self-Reflective Agentic Retrieval for complex documents containing figures, tables, flowcharts, and text.
---

# VisRAG & Multimodal Agentic Retrieval Skill

This skill extends standard text-based RAG into a **Dual-Modality (Text + Vision) Agentic System** capable of retrieving and reasoning over both text chunks and high-resolution visual PDF pages.

## Why VisRAG?

Standard text RAG pipelines flatten PDF documents into plain text, discarding:
1. **Spatial Relationships**: Hierarchies in flowcharts, inner/outer concentric circles in lifecycle diagrams (e.g., NIST AI RMF Figure 2).
2. **Actor & Arrow Flow**: Cross-stage connections in complex diagrams (e.g., Figure 3 AI Actors across lifecycle stages).
3. **Complex Tables**: Multi-column tabular alignments and subcategory groupings (e.g., Tables 1-4 for GOVERN, MAP, MEASURE, MANAGE).

VisRAG preserves visual fidelity by rendering physical PDF pages on demand and invoking Vision Language Models (VLMs) to inspect target visual regions.

---

## Architecture & Tool Suite

The VisRAG Agent operates with 4 primary tools:

```text
               +----------------------------------------------------+
               |                User Question                       |
               +----------------------------------------------------+
                                         |
                                         v
               +----------------------------------------------------+
               |            VisRAG Agent Controller                 |
               |      (ReAct Reasoning & Tool Routing)              |
               +----------------------------------------------------+
                        |                                |
                        | (Textual Query)                | (Diagram/Table/Figure)
                        v                                v
       +---------------------------------+  +---------------------------------+
       |     Hybrid RRF Text Retriever   |  |   Visual Document Inspector     |
       |  - FastEmbed BGE-Small (Dense)  |  |  - pypdfium2 150 DPI Rendering |
       |  - Okapi BM25 (Lexical)         |  |  - VLM Visual Reasoning (GPT-4o)|
       +---------------------------------+  +---------------------------------+
                        |                                |
                        +----------------+---------------+
                                         |
                                         v
               +----------------------------------------------------+
               |          Cumulative Context Packaging              |
               |   - Deduplicated Text Chunks ([chunk_id])          |
               |   - Visual Inspection Chunks ([vis_page_XX])       |
               +----------------------------------------------------+
                                         |
                                         v
               +----------------------------------------------------+
               |            Grounded Answer Synthesis               |
               |  - Strict Dual-Source Citations                    |
               |  - Self-Reflective Abstention if unanswerable      |
               +----------------------------------------------------+
```

### 1. `search(query, filters)`
Performs Reciprocal Rank Fusion ($k=60$) over Dense Vector Search (Qdrant + `BAAI/bge-small-en-v1.5`) and BM25 Okapi lexical search.

### 2. `inspect_visual_page(page, visual_query)`
Renders the physical PDF page into high-resolution JPEG bytes using `pypdfium2`, caches it in `artifacts/page_cache/`, and queries the Vision Language Model (`gpt-4o-mini`) to extract diagram structures, node labels, and visual tables.

### 3. `answer(answer, citations)`
Outputs a grounded answer where every statement is backed by either text chunk IDs (`[nist_ai_rmf_1_0_B_...]`) or visual page IDs (`[vis_page_XX]`).

### 4. `abstain(reason)`
Provides principled refusal (*"Không đủ bằng chứng để trả lời."*) for out-of-scope or unanswerable queries without hallucination.

---

## CLI Execution

```bash
# 1. Run unit tests including VisRAG tests
python -m unittest discover -s tests -v

# 2. Run standard retrieval benchmark
python src/evaluate_retrieval.py

# 3. Run standard agent benchmark (Dev & Test)
python src/evaluate_agent.py

# 4. Run VisRAG multimodal benchmark suite
python src/evaluate_visrag.py
```

## Programmatic Usage

```python
from src.retrieval import HybridRRFRetriever, DenseRetriever, BM25Retriever
from src.visrag import VisualDocumentInspector
from src.visrag_agent import VisRAGController, run_visrag_agent
import json

# Initialize components
chunks = json.load(open("data/chunks.json", encoding="utf-8"))
hybrid = HybridRRFRetriever(DenseRetriever(), BM25Retriever(chunks))
inspector = VisualDocumentInspector("data/NIST_AI_100_1.pdf")
controller = VisRAGController()

# Run query
res = run_visrag_agent(
    question="What two inner circles and outer circle are shown in Figure 2 representing the AI system lifecycle?",
    controller=controller,
    text_retriever=hybrid,
    visual_inspector=inspector,
    budget=2200
)

print("Answer:", res["answer"])
print("Citations:", res["citations"])
print("Visual Pages:", res["visual_pages_inspected"])
```
