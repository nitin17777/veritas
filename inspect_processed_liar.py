
import json
from collections import Counter
from pathlib import Path

base = Path("data/processed/liar")

for split in ("train", "validation", "test"):
    path = base / f"{split}.jsonl"
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"{split}: invalid JSON on line {line_number}: {exc}")

    ids = [record["id"] for record in records]
    duplicates = [
        claim_id
        for claim_id, count in Counter(ids).items()
        if count > 1
    ]

    print(f"\n{split}:")
    print("Records:", len(records))
    print("Unique IDs:", len(set(ids)))
    print("Duplicate IDs:", len(duplicates))
    print("First duplicate IDs:", duplicates[:10])
