"""Advanced Multimodal VisRAG Agent with Self-Reflective Tool-Calling.

Features:
- Dual-Modality: Combines Hybrid RRF Text Retrieval with VLM PDF Visual Inspection.
- Multi-Tool Calling: search (text), inspect_visual_page (VisRAG), answer, abstain.
- Self-Correction & Reasoning Scratchpad (ReAct / Self-RAG).
- Multi-source Grounding (cites text chunk IDs [nist_ai_rmf_...] and visual pages [vis_page_XX]).
"""
import json
import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval import pack_context
from src.visrag import VisualDocumentInspector

load_dotenv()


class VisRAGController:
    """Multimodal Agent Controller supporting text retrieval and visual page inspection."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found for VisRAG Controller.")

        if base_url:
            self.base_url = base_url
        elif os.getenv("OPENROUTER_API_KEY"):
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.base_url = None

        self.model_name = model_name or os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Multi-turn decision step with ReAct reasoning and dual-modality tools."""
        system_prompt = (
            "You are an advanced Multimodal Research Agent specializing in the NIST AI Risk Management Framework (AI RMF 1.0).\n"
            "You have access to two powerful retrieval modalities:\n"
            "1. TEXT SEARCH: Retrieves structured paragraphs and text chunks from the vector & keyword index.\n"
            "2. VISUAL PAGE INSPECTION (VisRAG): Directly examines high-resolution visual renderings of PDF pages to read "
            "flowcharts, lifecycle diagrams (Figure 1, 2, 3), organizational tables (Table 1-4), or visual layouts.\n\n"
            "AVAILABLE ACTIONS (Output valid JSON only):\n"
            "- To search text chunks:\n"
            '  {"thought": "<reasoning>", "action": "search", "query": "<focused search keywords>"}\n'
            "- To inspect a visual PDF page (useful for diagrams, figures, or complex tables):\n"
            '  {"thought": "<reasoning>", "action": "inspect_visual_page", "page": <physical_page_int>, "visual_query": "<what visual detail to look for>"}\n'
            "- To finalize answer when sufficient evidence exists in context:\n"
            '  {"thought": "<reasoning>", "action": "answer", "answer": "<answer text citing [chunk-id] or [vis_page_XX]>", "citations": ["<chunk-id>", ...]}\n'
            "- To abstain when the question is out of scope or unanswerable from the NIST AI RMF:\n"
            '  {"thought": "<reasoning>", "action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}\n\n'
            "RULES:\n"
            "1. FIRST TURN: Always execute at least one retrieval action (search or visual inspection) before deciding.\n"
            "2. CITATION DISCIPLINE: Every citation listed in 'citations' MUST be cited as [id] in the answer text, and MUST exist in the provided context.\n"
            "3. ACCURACY & VISUAL NAVIGATION:\n"
            "   - Figure 1 (Examples of potential harms): Physical Page 10.\n"
            "   - Figure 2 (Lifecycle & Key Dimensions of an AI System): Physical Page 15.\n"
            "   - Figure 3 (AI Actors across AI Lifecycle stages): Physical Page 16.\n"
            "   - Table 1 (GOVERN Categories & Subcategories): Physical Page 27.\n"
            "   - Table 2 (MAP Categories & Subcategories): Physical Page 31.\n"
            "   - Table 3 (MEASURE Categories & Subcategories): Physical Page 34.\n"
            "   - Table 4 (MANAGE Categories & Subcategories): Physical Page 37.\n"
            "   When the question asks about these figures or tables, inspect the corresponding visual page directly.\n"
            "4. CITATION CITATION: You may cite text chunks [chunk-id] and/or visual pages [vis_page_XX] in your answer.\n"
            "5. NEVER hallucinate information or citations."
        )

        user_content = f"User Question: {question}\n\n"
        if not history:
            user_content += "Current Status: No tools called yet. Context is empty."
        else:
            user_content += "History of Actions and Gathered Context:\n"
            for h in history:
                queries_str = ", ".join(f'"{q}"' for q in h.get("queries", []))
                user_content += f"- Search queries: [{queries_str}]\n"
                if h.get("visual_inspections"):
                    user_content += f"- Visual inspections performed on pages: {h.get('visual_inspections')}\n"
                user_content += f"- Accumulated Context:\n{h.get('context', '(empty)')}\n"

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )

        if resp.usage:
            self.total_prompt_tokens += resp.usage.prompt_tokens
            self.total_completion_tokens += resp.usage.completion_tokens

        content = resp.choices[0].message.content or "{}"
        try:
            action = json.loads(content)
        except Exception as e:
            raise ValueError(f"VisRAG Controller output not valid JSON: {content}") from e

        act = action.get("action")
        if act not in ("search", "inspect_visual_page", "answer"):
            raise ValueError(f"Unknown VisRAG action: {act}")

        return action


def run_visrag_agent(
    question: str,
    controller: VisRAGController,
    text_retriever: Any,
    visual_inspector: VisualDocumentInspector,
    *,
    filters: dict[str, Any] | None = None,
    max_turns: int = 5,
    top_k: int = 3,
    budget: int = 2000,
) -> dict[str, Any]:
    """Execute Multimodal VisRAG agent loop with text and visual inspection tools."""
    history: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    chunks_map: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    visual_inspected_pages: list[int] = []

    start_time = time.perf_counter()

    for turn in range(max_turns + 1):
        action = controller.decide(question, history)
        thought = action.get("thought", "")
        act = action.get("action")

        if act == "search":
            search_count = sum(e["event"] == "search" for e in events)
            if search_count >= 4:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "search_limit"}],
                    "elapsed_ms": elapsed_ms,
                }

            query = action.get("query")
            results = text_retriever.search(query, filters=filters, top_k=top_k)
            candidates.extend(results)

            context, ids = pack_context(candidates, budget=budget)
            chunks_map = {c["id"]: c for c in candidates if c["id"] in ids}

            events.append({
                "event": "search",
                "turn": turn + 1,
                "thought": thought,
                "query": query,
                "results_count": len(results),
                "context_ids": ids,
                "context_chars": len(context),
            })

            history = [{
                "tool": "search",
                "queries": [e["query"] for e in events if e.get("event") == "search"],
                "visual_inspections": visual_inspected_pages,
                "context": context,
                "context_ids": ids,
            }]

        elif act == "inspect_visual_page":
            page_num = action.get("page")
            visual_query = action.get("visual_query", question)
            if not isinstance(page_num, int):
                try:
                    page_num = int(page_num)
                except Exception:
                    page_num = 1

            # Execute visual inspection
            vis_chunk = visual_inspector.inspect_visual_page(page_num, visual_query)
            visual_inspected_pages.append(page_num)
            candidates.append(vis_chunk)

            context, ids = pack_context(candidates, budget=budget)
            chunks_map = {c["id"]: c for c in candidates if c["id"] in ids}

            events.append({
                "event": "inspect_visual_page",
                "turn": turn + 1,
                "thought": thought,
                "page": page_num,
                "visual_query": visual_query,
                "visual_chunk_id": vis_chunk["id"],
                "context_ids": ids,
                "context_chars": len(context),
            })

            history = [{
                "tool": "inspect_visual_page",
                "queries": [e["query"] for e in events if e.get("event") == "search"],
                "visual_inspections": visual_inspected_pages,
                "context": context,
                "context_ids": ids,
            }]

        elif act == "answer":
            answer = action.get("answer")
            citations = action.get("citations", [])
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            valid = bool(answer.strip() and citations and all(i in chunks_map and f"[{i}]" in answer for i in citations))

            if valid:
                return {
                    "status": "answered",
                    "answer": answer,
                    "citations": citations,
                    "context": history[0]["context"] if history else "",
                    "context_ids": history[0]["context_ids"] if history else [],
                    "events": events + [{"event": "answer", "thought": thought}],
                    "elapsed_ms": elapsed_ms,
                    "visual_pages_inspected": visual_inspected_pages,
                }

            return {
                "status": "abstain",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "context": history[0]["context"] if history else "",
                "context_ids": history[0]["context_ids"] if history else [],
                "events": events + [{"event": "abstain_or_invalid_citations", "thought": thought}],
                "elapsed_ms": elapsed_ms,
                "visual_pages_inspected": visual_inspected_pages,
            }

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return {
        "status": "abstain",
        "answer": "Không đủ bằng chứng để trả lời.",
        "citations": [],
        "events": events,
        "elapsed_ms": elapsed_ms,
        "visual_pages_inspected": visual_inspected_pages,
    }
