
import csv
from pathlib import Path

path = Path("data/raw/liar/train.tsv")

with path.open("r", encoding="utf-8", newline="") as file:
    reader = csv.reader(file, delimiter="\t")
    for row_number, row in enumerate(reader, start=1):
        if len(row) != 14:
            print(
                f"Malformed row {row_number}: "
                f"{len(row)} fields; {row[:3]!r}"
            )
            if row_number > 100:
                break
