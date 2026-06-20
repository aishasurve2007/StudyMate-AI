class HybridRetriever:


    def __init__(
        self,
        vector_store,
        bm25
    ):

        self.vector_store = vector_store
        self.bm25 = bm25



    def search(
        self,
        query,
        query_embedding,
        k=5
    ):


        semantic_results = (
            self.vector_store.search(
                query_embedding,
                k
            )
        )


        keyword_results = (
            self.bm25.search(
                query,
                k
            )
        )


        combined=[]


        for item in semantic_results:

            combined.append(
                {
                "document":
                item["document"],

                "score":
                item["score"] * 0.7
                }
            )


        for item in keyword_results:

            combined.append(
                {
                "document":
                item["document"],

                "score":
                item["score"] * 0.3
                }
            )


        combined.sort(
            key=lambda x:x["score"],
            reverse=True
        )


        return combined[:k]