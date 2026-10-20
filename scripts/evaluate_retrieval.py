#!/usr/bin/env python
"""CLI script to evaluate VERITAS retrieval (BM25, Dense, or Hybrid RRF).

Usage:
    python scripts/evaluate_retrieval.py --smoke-test --mode hybrid
    python scripts/evaluate_retrieval.py --claims-file data/processed/rawfc/test.jsonl --mode hybrid
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from veritas.retrieval.evaluate import main

if __name__ == "__main__":
    main()
