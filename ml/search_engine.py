import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load documents and FAISS index
# --------------------------------------------------

index_dir = Path("ml/index")

with open(index_dir / "documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

index = faiss.read_index(str(index_dir / "faiss.index"))


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------------------------
# 3. Dense search
# --------------------------------------------------

def search(query, top_k=5):

    query_embedding = model.encode([query])

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "doc_id": int(idx),
            "score": float(score),
            "document": documents[idx]
        })

    return results


# --------------------------------------------------
# 4. Test
# --------------------------------------------------

if __name__ == "__main__":

    results = search("machine learning models")

    for result in results:
        print(
            f"ID: {result['doc_id']} | "
            f"Score: {result['score']:.4f} | "
            f"{result['document']}"
        )