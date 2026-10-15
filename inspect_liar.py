
import csv
from pathlib import Path

data_dir = Path("data/raw/liar")

for file_path in sorted(data_dir.glob("*.tsv")):
    with file_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        rows = list(reader)

    print(f"\nFile: {file_path.name}")
    print(f"Rows: {len(rows)}")

    if rows:
        print(f"Columns in first row: {len(rows[0])}")
        print(f"First row preview: {rows[0][:5]}")
