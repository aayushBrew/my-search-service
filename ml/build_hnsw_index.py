
import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Machine learning helps computers learn from data.",
    "AI systems learn patterns from data.",
    "Cricket is a bat and ball sport played between two teams.",
    "Virat Kohli is an Indian international cricketer known for batting.",
    "Python is a popular programming language."
]

# Generate embeddings
embeddings = model.encode(documents)
embeddings = np.asarray(embeddings, dtype=np.float32)

# Normalize vectors for cosine similarity
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# Create HNSW index
M = 16
index = faiss.IndexHNSWFlat(
    dimension,
    M,
    faiss.METRIC_INNER_PRODUCT
)

# Search quality vs speed parameter
index.hnsw.efConstruction = 100

# Add document vectors
index.add(embeddings)

# Save index and documents
index_dir = Path("ml/index")
index_dir.mkdir(parents=True, exist_ok=True)

with open(index_dir / "documents.json", "w", encoding="utf-8") as file:
    json.dump(documents, file, indent=4)

faiss.write_index(index, str(index_dir / "faiss_hnsw.index"))

print("HNSW index built successfully!")
print("Vectors:", index.ntotal)
print("Dimensions:", dimension)
print("M:", M)
print("efConstruction:", index.hnsw.efConstruction)
