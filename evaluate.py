"""
Run the RAG evaluation harness over a document + test set.

Usage (from the project root, venv active):
    python evaluate.py --pdf data/raw_documents/OS.pdf --testset data/eval/testset.json

Defaults to the adopted config (retrieve 20, top 5). Override with --retrieve /
--top to compare configurations -- e.g. --retrieve 10 --top 3 reproduces the
baseline. Generation must be deterministic (temperature=0 in llm.py) for the
numbers to be reproducible.
"""
import argparse
import json
import os

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.metadata import add_metadata
from src.chunking.chunker import DocumentChunker
from src.intelligence.analyzer import DocumentAnalyzer
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Retriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.answer_generator import AnswerGenerator
from src.evaluation.hallucination import HallucinationDetector
from src.evaluation.evaluator import RAGEvaluator


def build_pipeline(pdf_path, retrieve_k, top_n):
    documents = add_metadata(PDFLoader(pdf_path).load())
    chunks = DocumentChunker().split_documents(documents)
    chunks = DocumentAnalyzer().analyze(chunks)   # match app.py: adds section/keywords

    embedder = Embedder()
    embeddings = embedder.create_embeddings([c["chunk_text"] for c in chunks])

    vector_store = VectorStore()
    vector_store.build(embeddings, chunks)
    hybrid = HybridRetriever(vector_store, BM25Retriever(chunks))

    return RAGEvaluator(
        embedder=embedder,
        hybrid=hybrid,
        reranker=Reranker(),
        generator=AnswerGenerator(),
        hallucination=HallucinationDetector(),
        retrieve_k=retrieve_k,
        top_n=top_n,
    ), len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="PDF to evaluate against")
    ap.add_argument("--testset", required=True, help="JSON list of test questions")
    ap.add_argument("--retrieve", type=int, default=20, help="candidates retrieved before rerank")
    ap.add_argument("--top", type=int, default=5, help="chunks passed to the LLM after rerank")
    ap.add_argument("--out", default="data/eval/results.json", help="where to save full results")
    args = ap.parse_args()

    with open(args.testset, encoding="utf-8") as f:
        testset = json.load(f)

    print(f"Building index for {args.pdf} ...")
    evaluator, n_chunks = build_pipeline(args.pdf, args.retrieve, args.top)
    print(f"Indexed {n_chunks} chunks. Running {len(testset)} questions "
          f"(retrieve={args.retrieve}, top={args.top})...\n")

    rows, summary = evaluator.evaluate(testset)

    for r in rows:
        flag = "HIT " if r["retrieval_hit"] else "MISS"
        print(f"[{flag}] support {r['support_score']:5.1f}%  "
              f"halluc {r['hallucination_risk']:4.1f}%  "
              f"{r['total_seconds']:.2f}s  | {r['question'][:60]}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Questions ............. {summary['questions']}")
    print(f"Retrieval hit-rate .... {summary['retrieval_hit_rate']}%")
    print(f"Avg support (faithful)  {summary['avg_support_score']}%")
    print(f"Avg hallucination risk  {summary['avg_hallucination_risk']}%")
    if summary["avg_answer_recall"] is not None:
        print(f"Avg answer recall ..... {summary['avg_answer_recall']}%")
    print(f"Avg latency ........... {summary['avg_latency_seconds']}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"settings": vars(args), "summary": summary, "rows": rows},
                  f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()