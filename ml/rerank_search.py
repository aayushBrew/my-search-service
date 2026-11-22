
from sentence_transformers import CrossEncoder

from hybrid_search import search as hybrid_search


# --------------------------------------------------
# 1. Load cross-encoder model
# --------------------------------------------------

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------------------------
# 2. Rerank retrieved candidates
# --------------------------------------------------

def rerank(query, candidates, top_k=5):

    # Create query-document pairs.
    pairs = [
        [query, candidate["document"]]
        for candidate in candidates
    ]

    # Predict relevance scores for all pairs.
    scores = model.predict(pairs)

    # Attach cross-encoder scores to candidates.
    reranked_results = []

    for candidate, score in zip(candidates, scores):

        reranked_results.append({
            "doc_id": candidate["doc_id"],
            "score": float(score),
            "document": candidate["document"]
        })

    # Sort by cross-encoder relevance score.
    reranked_results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return reranked_results[:top_k]


# --------------------------------------------------
# 3. Test retrieve + rerank pipeline
# --------------------------------------------------

if __name__ == "__main__":

    query = "machine learning models"

    # Stage 1: Retrieve candidates using BM25 + FAISS + RRF.
    candidates = hybrid_search(query, top_k=20)

    print("\nHYBRID RESULTS")

    for candidate in candidates[:5]:
        print(
            f"ID: {candidate['doc_id']} | "
            f"RRF Score: {candidate['score']:.6f}"
        )

    # Stage 2: Rerank candidates using cross-encoder.
    results = rerank(query, candidates, top_k=5)

    print("\nCROSS-ENCODER RERANKED RESULTS")

    for result in results:
        print(
            f"ID: {result['doc_id']} | "
            f"Relevance Score: {result['score']:.4f} | "
            f"{result['document']}"
        )
