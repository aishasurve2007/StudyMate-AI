import re
from transformers import pipeline


# Formatting the LLM adds that aren't factual claims (from the structured prompt).
_LABELS = re.compile(
    r"^\s*(answer|explanation|in simple terms|simplified view|summary|sources?)\s*:",
    re.I,
)
_BULLET = re.compile(r"^\s*([-*\u2022]|\d+[.)])\s+")


def to_claims(answer: str) -> list[str]:
    """Turn a (possibly structured / bulleted) answer into real claim sentences.

    Strips section labels ("Answer:", "Explanation:"...) and list markers, then
    splits into sentences and drops fragments shorter than 4 words so headings
    and stray labels don't get scored as unsupported 'claims'.
    """
    lines = []
    for raw in answer.splitlines():
        line = _LABELS.sub("", raw)
        line = _BULLET.sub("", line)
        line = line.strip()
        if line:
            lines.append(line)
    text = " ".join(lines)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= 4]


def aggregate(claims, preds, owner, entail_threshold=0.4, contradict_threshold=0.5):
    """Score faithfulness from NLI predictions.

    For each claim we keep the verdict from its *best-entailing* chunk, then:
      - contradicted (context disagrees) -> counts as a hallucination
      - entailed     (context supports)  -> full support
      - neutral      (rephrase / extra detail, not contradicted) -> half support
    Hallucination risk is the *contradicted* fraction only, NOT 'lack of
    entailment' -- which is what made faithful paraphrases score badly before.
    """
    total = len(claims)
    if total == 0:
        return {"support_score": 0.0, "hallucination_risk": 0.0, "details": []}

    best = [{"ent": 0.0, "con": 0.0} for _ in range(total)]
    for i, pred in enumerate(preds):
        scores = {p["label"].lower(): p["score"] for p in pred}
        ent = scores.get("entailment", 0.0)
        con = scores.get("contradiction", 0.0)
        ci = owner[i]
        if ent > best[ci]["ent"]:
            best[ci] = {"ent": ent, "con": con}

    supported = neutral = contradicted = 0
    details = []
    for claim, b in zip(claims, best):
        if b["con"] >= contradict_threshold and b["con"] > b["ent"]:
            verdict, contradicted = "contradicted", contradicted + 1
        elif b["ent"] >= entail_threshold:
            verdict, supported = "supported", supported + 1
        else:
            verdict, neutral = "neutral", neutral + 1
        details.append({
            "sentence": claim,
            "best_entailment": round(b["ent"], 3),
            "supported": verdict == "supported",
            "entailment": round(b["ent"], 3),
            "contradiction": round(b["con"], 3),
            "verdict": verdict,
        })

    support_score = (supported + 0.5 * neutral) / total * 100
    hallucination_risk = contradicted / total * 100
    return {
        "support_score": round(support_score, 2),
        "hallucination_risk": round(hallucination_risk, 2),
        "details": details,
    }


class HallucinationDetector:
    """Faithfulness check via NLI. A claim is a hallucination only when the
    retrieved context *contradicts* it -- strict entailment is too harsh because
    correct paraphrases register as 'neutral'."""

    def __init__(self, entail_threshold=0.4, contradict_threshold=0.5):
        self.checker = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-base",
            top_k=None,  # all label scores per pair
        )
        self.entail_threshold = entail_threshold
        self.contradict_threshold = contradict_threshold

    def check(self, answer, contexts):
        if isinstance(contexts, str):
            contexts = [contexts]
        claims = to_claims(answer)
        if not claims:
            return {"support_score": 0.0, "hallucination_risk": 0.0, "details": []}

        pairs, owner = [], []
        for ci, claim in enumerate(claims):
            for chunk in contexts:
                # premise = chunk (the source), hypothesis = claim (the answer)
                pairs.append({"text": chunk, "text_pair": claim})
                owner.append(ci)

        preds = self.checker(pairs, truncation=True, max_length=512, batch_size=16)
        return aggregate(claims, preds, owner,
                         self.entail_threshold, self.contradict_threshold)