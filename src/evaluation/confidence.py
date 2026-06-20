import math


class ConfidenceScorer:
    """Heuristic answer confidence in [0, 100].

    Driven mainly by faithfulness (support_score), because the cross-encoder's
    rerank scores are uncalibrated *ranking logits*, not probabilities. The old
    version averaged sigmoid(logit) over all chunks, which let weaker tail
    chunks (and slightly-negative logits) drag confidence far below the answer's
    real quality. Fixes: use the single best (top) chunk's relevance, and weight
    the now-reliable support score most heavily.
    """

    def calculate(self, retrieved_results, support_score):
        if not retrieved_results:
            return 0.0

        support = support_score / 100.0

        # Relevance of the BEST supporting chunk (top reranked), as a (0,1) value.
        top = max(r.get("rerank_score", 0.0) for r in retrieved_results)
        retrieval_quality = 1.0 / (1.0 + math.exp(-top))

        # Citation coverage: how many supporting chunks we have (cap at 3).
        citation_quality = min(len(retrieved_results) / 3.0, 1.0)

        final = support * 0.6 + retrieval_quality * 0.25 + citation_quality * 0.15
        return round(final * 100, 2)