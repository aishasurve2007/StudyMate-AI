from sentence_transformers import CrossEncoder



class Reranker:


    def __init__(self):

        self.model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )



    def rerank(
        self,
        query,
        results
    ):


        pairs=[]


        for r in results:

            pairs.append(
                [
                query,
                r["document"]["chunk_text"]
                ]
            )


        scores = self.model.predict(
            pairs
        )


        for r,score in zip(
            results,
            scores
        ):

            r["rerank_score"]=float(score)



        results.sort(
            key=lambda x:x["rerank_score"],
            reverse=True
        )


        return results