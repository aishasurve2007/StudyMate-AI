import streamlit as st
import os
import hashlib
import numpy as np

# --- INITIAL UI CONFIGURATION ---
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚"
)

st.write("Application loaded ✅")

st.title("StudyMate AI")
st.write("Upload your study documents")

# Environment setup
if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# --- DETERMINISTIC CACHE UTILITIES ---
def get_file_hash(file_bytes):
    """Generates a stable unique cryptographic key for the uploaded file."""
    return hashlib.md5(file_bytes).hexdigest()


def make_deterministic_key(chunks):
    """Creates an entirely hashable, sorted tuple structure from chunks."""
    return tuple(
        tuple(sorted(c.items()))
        for c in chunks
    )


# --- 1. MODEL LOADING (SAFE & CACHED) ---
@st.cache_resource(show_spinner="Loading foundational AI models...")
def load_models():
    from src.embeddings.embedder import Embedder
    from src.retrieval.reranker import Reranker
    from src.generation.answer_generator import AnswerGenerator
    from src.evaluation.confidence import ConfidenceScorer

    return {
        "embedder": Embedder(),
        "reranker": Reranker(),
        "generator": AnswerGenerator(),
        "confidence": ConfidenceScorer(),
    }


# --- 2. PDF TO CHUNKS (CACHED PER FILE HASH) ---
@st.cache_data(show_spinner="Processing and Analyzing PDF...")
def process_pdf(file_bytes, file_name, file_hash_str):
    import os
    from src.chunking.chunker import DocumentChunker
    from src.ingestion.pdf_loader import PDFLoader
    from src.ingestion.metadata import add_metadata
    from src.intelligence.analyzer import DocumentAnalyzer

    os.makedirs("data/raw_documents", exist_ok=True)
    path = f"data/raw_documents/{file_hash_str}_{file_name}"
    with open(path, "wb") as f:
        f.write(file_bytes)

    loader = PDFLoader(path)
    docs = add_metadata(loader.load())

    chunks = DocumentChunker().split_documents(docs)
    chunks = DocumentAnalyzer().analyze(chunks)

    return chunks, docs


# --- 3. EMBEDDINGS (CLEAN & SIMPLE) ---
@st.cache_data(show_spinner=False)
def compute_embeddings(texts_tuple, embedder):
    embeddings = []
    with st.spinner("Computing embeddings..."):
        for i in range(0, len(texts_tuple), 16):
            batch = list(texts_tuple[i:i+16])
            embeddings.extend(embedder.create_embeddings(batch))
    return embeddings


# --- 4. RETRIEVAL ENGINE (CACHED PER STABLE HASH FILE STRUCTURE) ---
@st.cache_resource(show_spinner="Indexing Hybrid Search Components...")
def build_retriever(file_hash_str, chunks_key, embeddings_key):
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_search import BM25Retriever
    from src.retrieval.hybrid_search import HybridRetriever

    # Safe local reconstruction from stable tuple keys
    chunks = [dict(item) for item in chunks_key]
    embeddings = [np.array(e) for e in embeddings_key]

    vector_store = VectorStore()
    
    # Internal class responsibility boundary (Rule 3)
    vector_store.build_once(embeddings, chunks)

    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vector_store, bm25)

    return hybrid


# --- 5. HALLUCINATION DETECTOR (CACHED SEPARATE PIPELINE STATE) ---
@st.cache_resource(show_spinner="Compiling NLI Evaluator Graph...")
def load_detector():
    from src.evaluation.hallucination import HallucinationDetector
    return HallucinationDetector()


# --- 6. APPLICATION WORKFLOW EXECUTION LAYER ---
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    # Read payload bytes and immediately hash them for state keys
    file_bytes = uploaded_file.getvalue()
    file_hash_str = get_file_hash(file_bytes)
    
    # Initialize basic tooling variables
    models = load_models()

    # Document extraction process
    chunks, documents = process_pdf(file_bytes, uploaded_file.name, file_hash_str)
    texts = [c["chunk_text"] for c in chunks]

    # Guard Rails: Memory protective size ceiling optimization for stable HF tiers
    MAX_CHUNKS = 250
    if len(chunks) > MAX_CHUNKS:
        st.warning(f"⚠️ Large document detected! Truncating to the first {MAX_CHUNKS} chunks to protect memory stability.")
        chunks = chunks[:MAX_CHUNKS]
        texts = texts[:MAX_CHUNKS]

    # Compute high-density vectors directly using standard parameters (Rule 1 & 2)
    embeddings = compute_embeddings(tuple(texts), models["embedder"])

    # Create immutable and completely deterministic primitives
    chunks_key = make_deterministic_key(chunks)
    embeddings_key = tuple(tuple(e) for e in embeddings)

    # Establish long lived retrieval cache space mapped to the unique file hash
    hybrid = build_retriever(file_hash_str, chunks_key, embeddings_key)

    st.success("Document infrastructure initialized and loaded successfully!")

    # --- ACTION ABSTRACTION LAYER ---
    query = st.text_input("Ask something")

    if query:
        query_emb = models["embedder"].create_embeddings([query])[0]

        # Process standard search extraction commands
        results = hybrid.search(query, query_emb, k=20)
        results = models["reranker"].rerank(query, results)
        answer = models["generator"].answer(query, results[:5])

        # Lazy compilation check
        detector = load_detector()
        contexts = [r["document"]["chunk_text"] for r in results[:5]]
        hallucination = detector.check(answer["answer"], contexts)

        # Map correct precision numeric metrics out to final calculations
        confidence = models["confidence"].calculate(
            results[:8],
            hallucination["support_score"]
        )

        # --- GRAPHICAL UI WRITING ---
        st.subheader("Answer")
        st.write(answer["answer"])

        st.subheader("AI Reliability Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence Score", f"{confidence:.2f}%")
        with col2:
            st.metric("Support Validation Score", f"{hallucination['support_score']:.2f}%")
        with col3:
            st.metric("Hallucination Vulnerability Risk", f"{hallucination['hallucination_risk']:.2f}%")

        with st.expander("Detailed NLI Text Verification Breakdown"):
            for item in hallucination["details"]:
                st.write("**Sentence Analyzed:**")
                st.info(item["sentence"])
                st.write(f"**Entailment Status:** `{item['best_entailment']}`")
                st.write(f"**Supported:** {item['supported']}")
                st.divider()

        st.subheader("Context Verification Sources")
        for s in answer["sources"]:
            with st.expander(f"📄 {s['source']} | Page Reference {s['page']}"):
                st.write("**Section Header Reference:**")
                st.info(s.get("section", "Unknown"))
                st.write("**Textual Evidence Grounding:**")
                st.code(s.get("evidence", ""))

    st.write(f"Total processed chunk components: {len(chunks)}")
    with st.expander("Inspect Sample Fragment Structures"):
        st.json(chunks[:5])

    st.write(f"Total extracted page nodes: {len(documents)}")
    with st.expander("Inspect Parsed Source JSON Output"):
        st.json(documents)