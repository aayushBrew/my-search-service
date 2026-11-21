
from sentence_transformers import SentenceTransformer

# Load a pretrained embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Sample documents
documents = [
    "Machine learning helps computers learn from data.",
    "Search engines retrieve relevant information for users.",
    "Python is a popular programming language.",
]

# Convert documents into numerical vectors
embeddings = model.encode(documents)

print("Number of documents:", len(documents))
print("Embedding shape:", embeddings.shape)
print("First document embedding (first 10 values):")
print(embeddings[0][:10])
