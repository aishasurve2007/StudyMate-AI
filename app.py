import streamlit as st
import os

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

# --- REMOVED HEAVY IMPORTS FROM THE GLOBAL SCOPE ---
# These will be imported dynamically inside functions to prevent the joblib/atexit crash.


@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    # Dynamically import heavy ML modules inside the cached function
    from src.embeddings.embedder import Embedder
    from src.retrieval.reranker import Reranker
    from src.generation.answer_generator import AnswerGenerator
    from src.evaluation.confidence import ConfidenceScorer

    return {
        "embedder": Embedder(),
        "reranker": Reranker(),
        "generator": AnswerGenerator(),
        "confidence": ConfidenceScorer(),
        "hallucination": None  # Set to None to defer heavy NLI initialization
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
    
    from src.ingestion.metadata import add_metadata
    from src.ingestion.pdf_loader import PDFLoader

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


# --- UI INTERACTION ---
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
    embeddings = []
for i in range(0, len(texts), 16):
    batch = texts[i:i+16]
    embeddings.extend(models["embedder"].create_embeddings(batch))

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

        # --- LAZY LOADING HALLUCINATION DETECTOR ON USER QUERY ---
        from src.evaluation.hallucination import HallucinationDetector

        detector = HallucinationDetector()

        hallucination = detector.check(
            response["answer"], contexts
        )
        
        confidence = models["confidence"].calculate(
            results[:8], hallucination["support_score"]
        )

        st.subheader("Answer")
        st.write(response)