"""
RAG evaluation harness.

Measures the StudyMate pipeline over a test set of questions with known answers,
so you can report real numbers (retrieval accuracy, faithfulness, latency)
instead of eyeballing a few queries. Run it before and after a change (e.g.
increasing retrieval depth) to *prove* the change helped.

Metrics per question:
  - retrieval_hit       : did a correct chunk reach the top-n? (by page or keyword)
  - support_score       : faithfulness of the answer (from HallucinationDetector)
  - hallucination_risk  : fraction of answer claims the context contradicts
  - answer_recall       : fraction of expected keywords present in the answer
  - latency             : retrieval + generation time
"""
import time
from statistics import mean


# ---------- pure metric functions (no models needed; unit-testable) ----------

def retrieval_hit(chunks, expected_pages, expected_keywords):
    """True if a retrieved chunk matches the known answer location/content."""
    pages = {c.get("page") for c in chunks}
    text = " ".join(c.get("chunk_text", "") for c in chunks).lower()
    page_hit = bool(expected_pages) and any(p in pages for p in expected_pages)
    kw_hit = bool(expected_keywords) and any(k.lower() in text for k in expected_keywords)
    return bool(page_hit or kw_hit)


def answer_recall(answer, expected_keywords):
    """Fraction of expected key terms that appear in the answer (rough correctness)."""
    if not expected_keywords:
        return None
    a = (answer or "").lower()
    hits = sum(1 for k in expected_keywords if k.lower() in a)
    return hits / len(expected_keywords)


def summarize(rows):
    """Aggregate per-question rows into headline metrics."""
    n = len(rows)
    if n == 0:
        return {}
    recalls = [r["answer_recall"] for r in rows if r["answer_recall"] is not None]
    return {
        "questions": n,
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in rows) / n * 100, 1),
        "avg_support_score": round(mean(r["support_score"] for r in rows), 1),
        "avg_hallucination_risk": round(mean(r["hallucination_risk"] for r in rows), 1),
        "avg_answer_recall": round(mean(recalls) * 100, 1) if recalls else None,
        "avg_latency_seconds": round(mean(r["total_seconds"] for r in rows), 2),
    }


# ---------- orchestration (runs the real pipeline over the test set) ----------

class RAGEvaluator:
    def __init__(self, embedder, hybrid, reranker, generator, hallucination,
                 retrieve_k=10, top_n=3):
        self.embedder = embedder
        self.hybrid = hybrid
        self.reranker = reranker
        self.generator = generator
        self.hallucination = hallucination
        self.retrieve_k = retrieve_k
        self.top_n = top_n

    def _retrieve(self, question):
        q_emb = self.embedder.model.encode([question], normalize_embeddings=True)[0]
        results = self.hybrid.search(question, q_emb, k=self.retrieve_k)
        results = self.reranker.rerank(question, results)
        return results

    def evaluate(self, testset):
        rows = []
        for item in testset:
            q = item["question"]
            expected_pages = item.get("expected_pages", [])
            expected_keywords = item.get("expected_keywords", [])

            t0 = time.perf_counter()
            results = self._retrieve(q)
            top = results[: self.top_n]
            t1 = time.perf_counter()

            response = self.generator.answer(q, top)
            answer = response["answer"]
            t2 = time.perf_counter()

            chunks = [r["document"] for r in top]
            contexts = [c["chunk_text"] for c in chunks]
            faith = self.hallucination.check(answer, contexts)

            rows.append({
                "question": q,
                "answer": answer,
                "retrieval_hit": retrieval_hit(chunks, expected_pages, expected_keywords),
                "support_score": faith["support_score"],
                "hallucination_risk": faith["hallucination_risk"],
                "answer_recall": answer_recall(answer, expected_keywords),
                "retrieval_seconds": round(t1 - t0, 3),
                "generation_seconds": round(t2 - t1, 3),
                "total_seconds": round(t2 - t0, 3),
            })

        return rows, summarize(rows)