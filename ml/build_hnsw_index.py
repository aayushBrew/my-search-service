
import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

index_dir = Path("ml/index")

# Load the same documents used by exact search
with open(index_dir / "documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

print("Loaded documents:", len(documents))

# Generate embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

embeddings = np.asarray(embeddings, dtype=np.float32)

# Normalize vectors
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# Build HNSW index
M = 16

index = faiss.IndexHNSWFlat(
    dimension,
    M,
    faiss.METRIC_INNER_PRODUCT
)

index.hnsw.efConstruction = 100

index.add(embeddings)

# Save HNSW index
faiss.write_index(
    index,
    str(index_dir / "faiss_hnsw.index")
)

print("\nHNSW index built successfully!")
print("Documents:", index.ntotal)
print("Dimensions:", dimension)
print("M:", M)
print("efConstruction:", index.hnsw.efConstruction)
