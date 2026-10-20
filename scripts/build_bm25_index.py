#!/usr/bin/env python
"""CLI script to build or refresh the VERITAS BM25 evidence retrieval index.

Usage:
    python scripts/build_bm25_index.py --config configs/retrieval_bm25.yaml
    python scripts/build_bm25_index.py --smoke-test --query "vaccine hospitalization"
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from veritas.retrieval.build_index import main

if __name__ == "__main__":
    main()
