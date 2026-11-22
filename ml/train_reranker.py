
import json

from pathlib import Path
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


# --------------------------------------------------
# 1. Load training triplets
# --------------------------------------------------

script_dir = Path(__file__).resolve().parent

data_path = script_dir / "training_data.json"
model_path = script_dir / "models" / "reranker-finetuned"

with open(data_path, "r", encoding="utf-8") as file:
    triplets = json.load(file)

print("Loaded triplets:", len(triplets))


# --------------------------------------------------
# 2. Convert triplets into labeled pairs
# --------------------------------------------------

training_examples = []

for item in triplets:

    query = item["query"]

    training_examples.append(
        InputExample(
            texts=[query, item["positive"]],
            label=1.0
        )
    )

    training_examples.append(
        InputExample(
            texts=[query, item["negative"]],
            label=0.0
        )
    )

print("Training examples:", len(training_examples))


# --------------------------------------------------
# 3. Load pretrained model
# --------------------------------------------------

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    num_labels=1
)


# --------------------------------------------------
# 4. Prepare DataLoader
# --------------------------------------------------

train_dataloader = DataLoader(
    training_examples,
    shuffle=True,
    batch_size=4
)


# --------------------------------------------------
# 5. Fine-tune
# --------------------------------------------------

model.fit(
    train_dataloader=train_dataloader,
    epochs=3,
    warmup_steps=2
)


# --------------------------------------------------
# 6. Explicitly save trained model
# --------------------------------------------------

model_path.mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(model_path))

print("\nTraining completed!")
print("Model saved at:", model_path)

# Verify that files were actually created.
saved_files = list(model_path.iterdir())

if not saved_files:
    raise RuntimeError("Model directory is empty!")

print("\nSaved files:")

for file in saved_files:
    print(file.name)
