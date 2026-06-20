import faiss
import numpy as np



class VectorStore:


    def __init__(self):

        self.index = None
        self.documents = []



    def build(self, embeddings, documents):


        dimension = embeddings.shape[1]


        self.index = faiss.IndexFlatIP(
            dimension
        )


        self.index.add(
            np.array(embeddings)
        )


        self.documents = documents



    def search(self, query_embedding, k=3):


        scores, indexes = self.index.search(
            np.array([query_embedding]),
            k
        )


        results=[]


        for score, idx in zip(
            scores[0],
            indexes[0]
        ):

            results.append(
                {
                "score":float(score),

                "document":
                self.documents[idx]
                }
            )


        return results