from langchain.vectorstores import Pinecone

class VectorDB(Pinecone):
    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        return distance