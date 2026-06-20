from sentence_transformers import SentenceTransformer


class Embedder:


    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


    def create_embeddings(self, texts):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings