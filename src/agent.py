"""Production Agent and LLM Controller for Day 7 Agentic RAG.

Connects real LLM (gpt-4o-mini via OpenRouter / OpenAI API) and real retrieval adapters.
Supports multi-turn tool calling, context management, strict citation grounding,
and principled abstention when evidence is insufficient.
"""
import json
import os
import time
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval import pack_context

load_dotenv()


class LLMController:
    """Real LLM Controller using OpenAI / OpenRouter API with JSON mode."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY.")

        if base_url:
            self.base_url = base_url
        elif os.getenv("OPENROUTER_API_KEY"):
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.base_url = None

        self.model_name = model_name or os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call LLM with query and retrieval history, returning normalized action."""
        system_prompt = (
            "You are a strict, grounded AI research agent answering questions based SOLELY on retrieved "
            "context from the NIST AI Risk Management Framework (AI RMF 1.0).\n\n"
            "OPERATING RULES:\n"
            "1. FIRST TURN REQUIREMENT: If history is empty (no searches performed yet), you MUST ALWAYS output:\n"
            '   {"action": "search", "query": "<focused search keywords based on the question>"}\n'
            "   You must NEVER answer or abstain without searching first.\n"
            "2. AFTER RETRIEVAL:\n"
            "   a) If the retrieved context contains sufficient, verifiable evidence to answer the question, output:\n"
            '      {"action": "answer", "answer": "<your clear answer citing [chunk-id] for every claim>", "citations": ["<chunk-id>", ...]}\n'
            "      IMPORTANT: Every citation in the 'citations' list MUST appear in the 'answer' text as [chunk-id], "
            "and MUST be present in the provided context.\n"
            "   b) If the retrieved context is partially relevant but missing key details, and you want to search again with refined keywords, output:\n"
            '      {"action": "search", "query": "<refined query keywords>"}\n'
            "   c) If the retrieved context shows that the document does NOT contain the answer, or if the question asks about facts/regulations outside NIST AI RMF 1.0 "
            "(e.g., monetary penalties/fines, mandatory commercial licensing, compute thresholds in FLOPs, specific benchmark cutoffs), you MUST abstain by outputting:\n"
            '      {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}\n'
            "3. NEVER invent facts, hallucinate citations, or make unsupported claims.\n"
            "4. Respond ONLY with a valid JSON object matching the schema above."
        )

        user_content = f"User Question: {question}\n\n"
        if not history:
            user_content += "History: No searches executed yet. Context is empty."
        else:
            user_content += "Retrieval History:\n"
            for h in history:
                queries_str = ", ".join(f'"{q}"' for q in h.get("queries", []))
                user_content += f"- Executed queries: [{queries_str}]\n"
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

        # Track tokens
        if resp.usage:
            self.total_prompt_tokens += resp.usage.prompt_tokens
            self.total_completion_tokens += resp.usage.completion_tokens

        content = resp.choices[0].message.content or "{}"
        try:
            action = json.loads(content)
        except Exception as e:
            raise ValueError(f"Model output is not valid JSON: {content}") from e

        act = action.get("action")
        if act not in ("search", "answer"):
            raise ValueError(f"Invalid model action: {act}")

        return action


def run_agent(
    question: str,
    controller: Any,
    retriever: Any,
    *,
    filters: dict[str, Any] | None = None,
    max_searches: int = 4,
    top_k: int = 3,
    budget: int = 1600,
) -> dict[str, Any]:
    """Execute multi-turn agent retrieval loop conforming to Day 7 contract."""
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError("Invalid question, max_searches, top_k or budget")

    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    start_time = time.perf_counter()

    for step in range(max_searches + 1):
        action = controller.decide(question, history)

        if action.get("action") == "search":
            search_count = sum(e["event"] == "search" for e in events)
            if search_count >= max_searches:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "search_limit"}],
                    "elapsed_ms": elapsed_ms,
                }

            query = action.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("search query must be non-empty string")

            # Perform real retrieval
            results = retriever.search(query, filters=filters, top_k=top_k)
            candidates.extend(results)

            context, ids = pack_context(candidates, budget=budget)
            chunks = {c["id"]: c for c in candidates if c["id"] in ids}

            events.append({
                "event": "search",
                "step": step + 1,
                "query": query,
                "filter": filters,
                "results_count": len(results),
                "context_ids": ids,
                "context_chars": len(context),
            })

            # Pass latest cumulative context to history
            history = [{
                "tool": "search",
                "queries": [e["query"] for e in events if e.get("event") == "search"],
                "context": context,
                "context_ids": ids,
            }]

        elif action.get("action") == "answer":
            answer = action.get("answer")
            citations = action.get("citations")
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError("Invalid answer or citations format")

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Grounding check: all citations must exist in packed context and be referenced in answer as [id]
            valid = bool(answer.strip() and citations and all(i in chunks and f"[{i}]" in answer for i in citations))

            if valid:
                return {
                    "status": "answered",
                    "answer": answer,
                    "citations": citations,
                    "context": history[0]["context"] if history else "",
                    "context_ids": history[0]["context_ids"] if history else [],
                    "events": events + [{"event": "answer"}],
                    "elapsed_ms": elapsed_ms,
                }

            # If no valid citations or empty answer -> principled abstain
            return {
                "status": "abstain",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "context": history[0]["context"] if history else "",
                "context_ids": history[0]["context_ids"] if history else [],
                "events": events + [{"event": "invalid_or_missing_citations"}],
                "elapsed_ms": elapsed_ms,
            }
        else:
            raise ValueError(f"Unknown action: {action.get('action')}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return {
        "status": "abstain",
        "answer": "Không đủ bằng chứng để trả lời.",
        "citations": [],
        "events": events,
        "elapsed_ms": elapsed_ms,
    }
