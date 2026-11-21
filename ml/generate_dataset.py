
import json
from pathlib import Path

topics = [
    ("Machine Learning", "Machine learning models learn patterns from data."),
    ("Deep Learning", "Deep learning uses neural networks with multiple layers."),
    ("Search Engineering", "Search engines retrieve relevant documents for user queries."),
    ("Information Retrieval", "Information retrieval finds useful information from large collections."),
    ("Python", "Python is a popular programming language."),
    ("Java", "Java is used to build scalable backend applications."),
    ("Cricket", "Cricket is a bat and ball sport played between two teams."),
    ("Football", "Football is a team sport played by kicking a ball."),
    ("Databases", "Databases store and organize structured information."),
    ("Artificial Intelligence", "Artificial intelligence enables machines to perform intelligent tasks.")
]

documents = []

for i in range(500):
    for topic, description in topics:
        documents.append(
            f"{topic}: {description} Example document number {i}."
        )

index_dir = Path("ml/index")
index_dir.mkdir(parents=True, exist_ok=True)

with open(index_dir / "documents.json", "w", encoding="utf-8") as file:
    json.dump(documents, file, indent=2)

print("Dataset generated!")
print("Total documents:", len(documents))
