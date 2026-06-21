import os
import hashlib
import streamlit as st

st.set_page_config(page_title="StudyMate AI", page_icon="📚")

# Bridge HF/Streamlit secrets to env. Safe even if neither exists.
try:
    if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

st.title("StudyMate AI")
st.write("Upload a study document (PDF) and ask questions grounded in it.")


# ---- models: loaded once, cached across the whole session ----
@st.cache_resource(show_spinner="Loading AI models (first run only)...")
def load_models():
    from src.embeddings.embedder import Embedder
    from src.retrieval.reranker import Reranker
    from src.generation.answer_generator import AnswerGenerator
    from src.evaluation.confidence import ConfidenceScorer
    from src.evaluation.hallucination import HallucinationDetector
    return {
        "embedder": Embedder(),
        "reranker": Reranker(),
        "generator": AnswerGenerator(),
        "confidence": ConfidenceScorer(),
        "hallucination": HallucinationDetector(),
    }


# ---- parse + chunk + analyze, cached per uploaded file ----
@st.cache_data(show_spinner="Processing PDF...")
def process_pdf(file_bytes, file_name):
    from src.ingestion.pdf_loader import PDFLoader
    from src.ingestion.metadata import add_metadata
    from src.chunking.chunker import DocumentChunker
    from src.intelligence.analyzer import DocumentAnalyzer

    os.makedirs("data/raw_documents", exist_ok=True)
    file_hash = hashlib.md5(file_bytes).hexdigest()
    path = f"data/raw_documents/{file_hash}_{file_name}"
    with open(path, "wb") as f:
        f.write(file_bytes)

    docs = add_metadata(PDFLoader(path).load())
    chunks = DocumentChunker().split_documents(docs)
    chunks = DocumentAnalyzer().analyze(chunks)
    return chunks, docs


# ---- embeddings, cached per file (leading-underscore args are NOT hashed) ----
@st.cache_data(show_spinner="Computing embeddings...")
def compute_embeddings(file_hash, _texts, _embedder):
    return _embedder.create_embeddings(_texts)


# ---- search index, cached per file (underscore args skip hashing) ----
@st.cache_resource(show_spinner="Building search index...")
def build_index(file_hash, _chunks, _embeddings):
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_search import BM25Retriever
    from src.retrieval.hybrid_search import HybridRetriever
    vs = VectorStore()
    vs.build(_embeddings, _chunks)
    return HybridRetriever(vs, BM25Retriever(_chunks))


uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    models = load_models()
    chunks, documents = process_pdf(file_bytes, uploaded_file.name)

    # memory guard for the free CPU tier
    MAX_CHUNKS = 250
    if len(chunks) > MAX_CHUNKS:
        st.warning(f"Large document — using the first {MAX_CHUNKS} chunks to stay within memory.")
        chunks = chunks[:MAX_CHUNKS]

    texts = [c["chunk_text"] for c in chunks]
    embeddings = compute_embeddings(file_hash, texts, models["embedder"])
    hybrid = build_index(file_hash, chunks, embeddings)

    st.success("Document ready — ask a question below.")

    query = st.text_input("Ask something")
    if query:
        query_emb = models["embedder"].create_embeddings([query])[0]
        results = hybrid.search(query, query_emb, k=20)
        results = models["reranker"].rerank(query, results)
        answer = models["generator"].answer(query, results[:5])

        contexts = [r["document"]["chunk_text"] for r in results[:5]]
        hallucination = models["hallucination"].check(answer["answer"], contexts)
        confidence = models["confidence"].calculate(results[:5], hallucination["support_score"])

        st.subheader("Answer")
        st.write(answer["answer"])

        st.subheader("AI Reliability")
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{confidence:.1f}%")
        c2.metric("Support", f"{hallucination['support_score']:.1f}%")
        c3.metric("Hallucination risk", f"{hallucination['hallucination_risk']:.1f}%")

        with st.expander("Faithfulness details"):
            for item in hallucination["details"]:
                st.write("**Sentence:**")
                st.info(item["sentence"])
                st.write(f"**Entailment:** `{item['best_entailment']}` · Supported: {item['supported']}")
                st.divider()

        st.subheader("Sources")
        for s in answer["sources"]:
            with st.expander(f"📄 {s['source']} | Page {s['page']}"):
                st.write("**Section:**")
                st.info(s.get("section", "Unknown"))
                st.write("**Evidence:**")
                st.code(s.get("evidence", ""))

    st.caption(f"Indexed {len(chunks)} chunks from {len(documents)} pages.")