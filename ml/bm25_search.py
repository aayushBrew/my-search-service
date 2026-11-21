import json
import re

from pathlib import Path
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# 1. Load documents
# --------------------------------------------------

index_dir = Path("ml/index")

with open(index_dir / "documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)


# --------------------------------------------------
# 2. Tokenization
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


tokenized_documents = [
    tokenize(document)
    for document in documents
]


# --------------------------------------------------
# 3. Build BM25 index
# --------------------------------------------------

bm25 = BM25Okapi(tokenized_documents)


# --------------------------------------------------
# 4. Search
# --------------------------------------------------

def search(query, top_k=5):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append({
            "doc_id": int(idx),
            "score": float(scores[idx]),
            "document": documents[idx]
        })

    return results


# --------------------------------------------------
# 5. Test
# --------------------------------------------------

if __name__ == "__main__":

    results = search("machine learning models")

    for result in results:
        print(
            f"ID: {result['doc_id']} | "
            f"Score: {result['score']:.4f} | "
            f"{result['document']}"
        )