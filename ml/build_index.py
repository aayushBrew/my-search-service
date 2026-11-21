
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

# Generate document embeddings
embeddings = model.encode(documents)

# Convert to float32 (FAISS expects float32 vectors)
embeddings = np.asarray(embeddings, dtype=np.float32)

# Normalize vectors so inner product equals cosine similarity
faiss.normalize_L2(embeddings)

# Get embedding dimension
dimension = embeddings.shape[1]

# Create FAISS exact-search index
index = faiss.IndexFlatIP(dimension)

# Add document vectors to index
index.add(embeddings)

# Prepare storage directory
index_dir = Path("ml/index")
index_dir.mkdir(parents=True, exist_ok=True)

# Save original documents
with open(index_dir / "documents.json", "w", encoding="utf-8") as file:
    json.dump(documents, file, indent=4)

# Save FAISS index
faiss.write_index(index, str(index_dir / "faiss.index"))

print("FAISS index built successfully!")
print("Number of vectors:", index.ntotal)
print("Vector dimension:", dimension)
