import streamlit as st
from src.ingestion.metadata import add_metadata
from src.ingestion.pdf_loader import PDFLoader

# --- REMOVED HEAVY IMPORTS FROM THE GLOBAL SCOPE ---
# These will be imported dynamically inside functions to prevent the joblib/atexit crash.


@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    # Dynamically import heavy ML modules inside the cached function
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


@st.cache_resource(show_spinner="Processing and Analyzing PDF...")
def process_pdf_to_chunks(file_bytes, file_name):
    """
    Cache only the heavy data-processing steps (parsing, chunking, analyzing).
    We return the data structures, NOT the active search engines.
    """
    import os
    from src.chunking.chunker import DocumentChunker
    from src.intelligence.analyzer import DocumentAnalyzer

    # Ensure directory exists
    os.makedirs("data/raw_documents", exist_ok=True)
    file_path = "data/raw_documents/" + file_name
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    loader = PDFLoader(file_path)
    documents = add_metadata(loader.load())

    chunker = DocumentChunker()
    chunks = chunker.split_documents(documents)
    
    analyzer = DocumentAnalyzer()
    chunks = analyzer.analyze(chunks)
    
    return chunks, documents


# --- UI SETUP ---
st.title("StudyMate AI")
st.write("Upload your study documents")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    # 1. Load your models safely
    models = load_models()
    
    # 2. Get your processed chunks (cached)
    chunks, documents = process_pdf_to_chunks(
        uploaded_file.getvalue(),
        uploaded_file.name,
    )

    # 3. Build the Search Index in live memory (Uncached, fast, and completely safe from thread crashes)
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_search import BM25Retriever
    from src.retrieval.hybrid_search import HybridRetriever

    texts = [c["chunk_text"] for c in chunks]
    embeddings = models["embedder"].create_embeddings(texts)

    vector_store = VectorStore()
    vector_store.build(embeddings, chunks)

    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vector_store, bm25)

    # --- REST OF YOUR UI LOGIC ---
    with st.expander("Document Intelligence Test"):
        st.json(chunks[0])

    st.success("PDF processed successfully")

    query = st.text_input("Ask something")

    if query:
        query_embedding = models["embedder"].model.encode(
            [query], normalize_embeddings=True
        )

        results = hybrid.search(query, query_embedding[0], k=20)
        
        reranker = models["reranker"]
        generator = models["generator"]

        results = reranker.rerank(query, results)

        response = generator.answer(query, results[:5])

        contexts = []
        for r in results[:5]:
            contexts.append(r["document"]["chunk_text"])

        hallucination = models["hallucination"].check(
            response["answer"], contexts
        )
        
        confidence = models["confidence"].calculate(
            results[:8], hallucination["support_score"]
        )

        st.subheader("Answer")
        st.write(response["answer"])

        st.subheader("AI Reliability")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{confidence:.2f}%")
        with col2:
            st.metric("Support Score", f"{hallucination['support_score']:.2f}%")
        with col3:
            st.metric("Hallucination Risk", f"{hallucination['hallucination_risk']:.2f}%")

        with st.expander("Faithfulness Details"):
            for item in hallucination["details"]:
                st.write("Sentence:")
                st.write(item["sentence"])
                st.write("Entailment:", item["best_entailment"])
                st.write("Supported:", item["supported"])
                st.divider()

        st.subheader("Sources")
        for s in response["sources"]:
            with st.expander(f"📄 {s['source']} | Page {s['page']}"):
                st.write("Section:")
                st.info(s.get("section", "Unknown"))
                st.write("Evidence:")
                st.code(s.get("evidence", ""))

    st.write(f"Chunks created: {len(chunks)}")
    with st.expander("View Chunks"):
        st.json(chunks[:5])

    st.write(f"Pages extracted: {len(documents)}")
    with st.expander("View extracted JSON"):
        st.json(documents)