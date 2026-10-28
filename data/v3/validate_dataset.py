import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

def load(name):
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)

documents = load("documents.json")["documents"]
train = load("train.json")["pairs"]
validation = load("validation.json")["pairs"]
test = load("test.json")["pairs"]

ids = [d["id"] for d in documents]
assert len(documents) == 100, f"Expected 100 docs, got {len(documents)}"
assert len(set(ids)) == len(ids), "Duplicate document IDs"
id_set = set(ids)
assert len(train) == 800, f"Expected 800 train pairs, got {len(train)}"
assert len(validation) == 200, f"Expected 200 validation pairs, got {len(validation)}"
assert len(test) == 200, f"Expected 200 test pairs, got {len(test)}"

for split_name, pairs in [("train", train), ("validation", validation), ("test", test)]:
    queries = [p["query"].strip().casefold() for p in pairs]
    assert all(queries), f"Empty query in {split_name}"
    assert len(queries) == len(set(queries)), f"Duplicate query in {split_name}"
    assert all(p["document_id"] in id_set for p in pairs), f"Unknown document ID in {split_name}"

query_sets = [
    {p["query"].strip().casefold() for p in pairs}
    for pairs in [train, validation, test]
]
assert query_sets[0].isdisjoint(query_sets[1]), "Train/validation query leakage"
assert query_sets[0].isdisjoint(query_sets[2]), "Train/test query leakage"
assert query_sets[1].isdisjoint(query_sets[2]), "Validation/test query leakage"

for split_name, pairs in [("train", train), ("validation", validation), ("test", test)]:
    counts = Counter(p["document_id"] for p in pairs) if False else None

print("Dataset checks passed.")
print(f"Documents: {len(documents)}")
print(f"Training pairs: {len(train)}")
print(f"Validation pairs: {len(validation)}")
print(f"Test pairs: {len(test)}")
print(f"Unique queries across splits: {sum(map(len, query_sets))}")
