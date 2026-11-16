"""Retrieval engines and benchmarking suite for Day 7 Agentic RAG.

Includes:
- Lexical TF-IDF Retriever
- Lexical BM25 Retriever
- Dense Vector Retriever (Qdrant + BAAI/bge-small-en-v1.5 via FastEmbed)
- Hybrid RRF Retriever (Dense + BM25 via Reciprocal Rank Fusion)
- Evaluation and trace extraction tools
"""
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


def tokenize(text: str) -> list[str]:
    """Tokenize and normalize text."""
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Exact substring and page/source check conforming to Day 7 benchmark standard."""
    return any(
        chunk.get('source_id') == g.get('source_id')
        and chunk.get('page') == g.get('page')
        and g.get('evidence', '') in chunk.get('text', '')
        for g in gold
    )


def pack_context(results: list[dict[str, Any]], budget: int = 800) -> tuple[str, list[str]]:
    """Pack deduplicated results into context blocks within character budget."""
    blocks, ids, seen = [], [], set()
    for c in results:
        cid = c.get('id')
        if not cid or cid in seen:
            continue
        seen.add(cid)
        block = f"[{cid} | {c.get('source')} | page {c.get('page')}]\n{c.get('text')}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(cid)
    return '\n\n'.join(blocks), ids


class TfidfRetriever:
    """Standard TF-IDF Lexical Retriever."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {
            term: math.log((1 + len(chunks)) / (1 + count)) + 1
            for term, count in df.items()
        }
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {term: v / norm for term, v in vector.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_vec = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if filters:
                meta = chunk.get('metadata', {})
                if any(meta.get(k) != v for k, v in filters.items()):
                    continue
            score = sum(val * vector.get(term, 0.0) for term, val in q_vec.items())
            if score > 0:
                results.append({**chunk, 'score': score, 'retrieval_method': 'tfidf'})
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


class BM25Retriever:
    """Okapi BM25 Lexical Retriever."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        self.doc_tokens = [tokenize(c['text']) for c in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)

        df = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        self.idf = {
            term: math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        results = []
        for idx, (chunk, tokens, doc_len) in enumerate(zip(self.chunks, self.doc_tokens, self.doc_lens)):
            if filters:
                meta = chunk.get('metadata', {})
                if any(meta.get(k) != v for k, v in filters.items()):
                    continue
            tf = Counter(tokens)
            score = 0.0
            for term in q_tokens:
                if term in tf:
                    term_tf = tf[term]
                    term_idf = self.idf.get(term, 0.0)
                    denom = term_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += term_idf * (term_tf * (self.k1 + 1.0)) / denom
            if score > 0:
                results.append({**chunk, 'score': score, 'retrieval_method': 'bm25'})
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


class DenseRetriever:
    """Dense Vector Retriever using FastEmbed and Qdrant database."""

    def __init__(
        self,
        qdrant_path: str = r"c:\Study\VinAI\K5-L3B-DAY06-2A202603760-DataIngestion\artifacts\qdrant_db",
        collection_name: str = "collection_strategy_b",
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        self.embedder = TextEmbedding(model_name=model_name)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_vec = list(self.embedder.embed([query]))[0].tolist()

        q_filter = None
        if filters:
            must_conditions = []
            for k, v in filters.items():
                must_conditions.append(
                    qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                )
            q_filter = qmodels.Filter(must=must_conditions)

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True
        )

        results = []
        for p in res.points:
            pl = p.payload or {}
            results.append({
                'id': pl.get('chunk_id'),
                'source_id': pl.get('document_id'),
                'source': pl.get('source'),
                'page': pl.get('page'),
                'text': pl.get('text'),
                'metadata': {
                    'section': pl.get('section'),
                    'section_title': pl.get('section_title'),
                    'part': pl.get('part'),
                    'chunk_size': pl.get('chunk_size'),
                    'strategy': pl.get('strategy')
                },
                'score': float(p.score),
                'retrieval_method': 'dense'
            })
        return results


class HybridRRFRetriever:
    """Hybrid Retriever combining Dense and BM25 via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        candidate_pool: int = 10
    ) -> None:
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        dense_hits = self.dense.search(query, filters=filters, top_k=self.candidate_pool)
        bm25_hits = self.bm25.search(query, filters=filters, top_k=self.candidate_pool)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, c in enumerate(dense_hits, start=1):
            cid = c['id']
            chunk_map[cid] = c
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, c in enumerate(bm25_hits, start=1):
            cid = c['id']
            chunk_map[cid] = c
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        ranked_cids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))
        results = []
        for cid in ranked_cids[:top_k]:
            c = dict(chunk_map[cid])
            c['score'] = round(rrf_scores[cid], 5)
            c['retrieval_method'] = 'hybrid_rrf'
            results.append(c)
        return results


def evaluate_retriever(
    retriever: Any,
    questions: list[dict[str, Any]],
    strategy_name: str,
    top_k: int = 3,
    budget: int = 800,
    use_filter: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Benchmark a retriever across questions and compute Hit@1, Hit@3, Context Hit."""
    traces = []
    latencies = []

    for q in questions:
        q_filter = q.get('filter') if use_filter else None
        start = time.perf_counter()
        results = retriever.search(q['question'], filters=q_filter, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        context, ids = pack_context(results, budget=budget)
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]

        is_hit1 = bool(marked and marked[0]['hit'])
        is_hit3 = any(c['hit'] for c in marked[:3])
        is_context_hit = any(c['hit'] and c['id'] in ids for c in marked)

        traces.append({
            'query_id': q['id'],
            'split': q.get('split'),
            'question': q['question'],
            'strategy': strategy_name,
            'top_k': top_k,
            'budget_chars': budget,
            'filter_used': q_filter,
            'gold': q['gold'],
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': is_hit1,
            'hit3': is_hit3,
            'context_hit': is_context_hit
        })

    n = len(questions)
    hit1_count = sum(t['hit1'] for t in traces)
    hit3_count = sum(t['hit3'] for t in traces)
    context_hit_count = sum(t['context_hit'] for t in traces)

    latencies_sorted = sorted(latencies)
    median_ms = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0.0
    avg_context_chars = sum(t['context_chars'] for t in traces) / max(1, n)

    summary = {
        'strategy': strategy_name,
        'split': questions[0].get('split', 'dev') if questions else 'unknown',
        'n': n,
        'top_k': top_k,
        'budget': budget,
        'filter_used': use_filter,
        'hit1': hit1_count,
        'hit1_rate': f"{hit1_count}/{n} ({hit1_count / n * 100:.1f}%)",
        'hit3': hit3_count,
        'hit3_rate': f"{hit3_count}/{n} ({hit3_count / n * 100:.1f}%)",
        'context_hit': context_hit_count,
        'context_hit_rate': f"{context_hit_count}/{n} ({context_hit_count / n * 100:.1f}%)",
        'avg_context_chars': round(avg_context_chars, 1),
        'median_ms': round(median_ms, 2),
        'service_cost_usd': 0.0
    }
    return summary, traces
