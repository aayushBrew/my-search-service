
import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths
index_dir = Path("ml/index")
documents_path = index_dir / "documents.json"

# Load documents
with open(documents_path, "r", encoding="utf-8") as file:
    documents = json.load(file)

print("Loaded documents:", len(documents))

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Generate document embeddings
embeddings = model.encode(
    documents,
    show_progress_bar=True
)

embeddings = np.asarray(embeddings, dtype=np.float32)

# Normalize vectors for cosine similarity
faiss.normalize_L2(embeddings)

# Build exact FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index
faiss.write_index(
    index,
    str(index_dir / "faiss.index")
)

print("\nExact index built successfully!")
print("Documents:", index.ntotal)
print("Dimensions:", dimension)
