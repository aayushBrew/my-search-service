
import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load saved documents
index_dir = Path("ml/index")

with open(index_dir / "documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

# Load FAISS index
index = faiss.read_index(str(index_dir / "faiss.index"))

print("FAISS index loaded!")
print("Total vectors:", index.ntotal)


def search(query, top_k=3):

    # Generate query embedding
    query_embedding = model.encode([query])

    # Convert to float32
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    # Normalize query vector
    faiss.normalize_L2(query_embedding)

    # Search FAISS index
    scores, indices = index.search(query_embedding, top_k)

    print(f"\nQuery: {query}")

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        print(f"{score:.4f} | {documents[idx]}")


# Test searches
search("Which sport does Virat Kohli play?")
search("What is machine learning?")
search("Tell me about programming languages.")
