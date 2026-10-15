
import csv
from pathlib import Path

data_dir = Path("data/raw/liar")
file_path = data_dir / "train.tsv"

column_names = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_fire_counts",
    "context",
]

with file_path.open("r", encoding="utf-8", newline="") as file:
    reader = csv.reader(file, delimiter="\t")
    first_row = next(reader)

print(f"Expected columns: {len(column_names)}")
print(f"Actual columns:   {len(first_row)}")
print("\nColumn mapping:")

for index, (name, value) in enumerate(zip(column_names, first_row)):
    print(f"{index:2}: {name:24} = {value!r}")
