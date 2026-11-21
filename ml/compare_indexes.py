import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Load documents
index_dir = Path("ml/index")
with open(index_dir / "documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

# 3. Load both indexes
exact_index = faiss.read_index(str(index_dir / "faiss.index"))
hnsw_index = faiss.read_index(str(index_dir / "faiss_hnsw.index"))


def search(index, query_embedding, top_k=3):
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((float(score), documents[idx]))
    return results


queries = [
    "Which sport does Virat Kohli play?",
    "What is machine learning?",
    "Tell me about programming languages.",
]

# Pre-compute and normalize query embeddings once for efficiency
query_embeddings = model.encode(queries)
query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
faiss.normalize_L2(query_embeddings)

# 4. Compare Exact Search vs. HNSW across different efSearch values
for ef in (4, 8, 16, 32, 64):
    hnsw_index.hnsw.efSearch = ef
    print("\n" + "#" * 60)
    print(f"TESTING efSearch = {ef}")
    print("#" * 60)

    for query, q_emb in zip(queries, query_embeddings):
        # Reshape single embedding to (1, dimensions) for FAISS
        q_emb_2d = q_emb.reshape(1, -1)

        exact_results = search(exact_index, q_emb_2d)
        hnsw_results = search(hnsw_index, q_emb_2d)

        print("\n" + "=" * 60)
        print("QUERY:", query)

        print("\nEXACT SEARCH")
        for score, document in exact_results:
            print(f"  {score:.4f} | {document}")

        print("\nHNSW SEARCH")
        for score, document in hnsw_results:
            print(f"  {score:.4f} | {document}")
