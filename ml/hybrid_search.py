
from bm25_search import search as bm25_search
from search_engine import search as dense_search


# --------------------------------------------------
# 1. Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    result_lists,
    top_k=5,
    rrf_k=60
):

    fused_scores = {}

    # Process each retriever's ranked results.
    for results in result_lists:

        # Rank starts from 1, not 0.
        for rank, result in enumerate(results, start=1):

            doc_id = result["doc_id"]

            # Add this retriever's contribution.
            score = 1 / (rrf_k + rank)

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0) + score
            )

    # Sort documents by their combined RRF scores.
    ranked_documents = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Retrieve document text from either result list.
    document_lookup = {}

    for results in result_lists:
        for result in results:
            document_lookup[result["doc_id"]] = result["document"]

    final_results = []

    for doc_id, score in ranked_documents[:top_k]:

        final_results.append({
            "doc_id": doc_id,
            "score": score,
            "document": document_lookup[doc_id]
        })

    return final_results


# --------------------------------------------------
# 2. Hybrid search
# --------------------------------------------------

def search(query, top_k=5):

    # Retrieve candidates from both retrievers.
    bm25_results = bm25_search(query, top_k=10)
    dense_results = dense_search(query, top_k=10)

    # Fuse their rankings.
    hybrid_results = reciprocal_rank_fusion(
        [bm25_results, dense_results],
        top_k=top_k
    )

    return hybrid_results


# --------------------------------------------------
# 3. Test
# --------------------------------------------------

if __name__ == "__main__":

    query = "machine learning models"

    results = search(query)

    print(f"\nHybrid search results for: {query}\n")

    for result in results:

        print(
            f"ID: {result['doc_id']} | "
            f"RRF Score: {result['score']:.6f} | "
            f"{result['document']}"
        )
