from rank_bm25 import BM25Okapi


class BM25Retriever:


    def __init__(self, chunks):

        self.chunks = chunks


        documents = [
            c["chunk_text"].split()
            for c in chunks
        ]


        self.bm25 = BM25Okapi(
            documents
        )



    def search(self, query, k=5):

        tokens = query.split()


        scores = self.bm25.get_scores(
            tokens
        )


        results = []


        top_indexes = scores.argsort()[-k:][::-1]


        for idx in top_indexes:

            results.append(
                {
                "score":float(scores[idx]),

                "document":
                self.chunks[idx]
                }
            )


        return results