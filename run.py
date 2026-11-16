#!/usr/bin/env python
"""Single-command runner for VERITAS.

Verifies dependencies and frontend build, then boots the REST API and
React application on http://localhost:8000.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main() -> None:
    print("=" * 60)
    print("   Starting VERITAS Fact Verification Platform")
    print("=" * 60)

    dist_index = ROOT / "frontend" / "dist" / "index.html"
    if not dist_index.exists():
        print("[1/2] Compiling frontend assets...")
        frontend_dir = ROOT / "frontend"
        subprocess.run(["npm", "run", "build"], cwd=frontend_dir, shell=True, check=True)
    else:
        print("[1/2] Frontend assets verified (frontend/dist/).")

    print("[2/2] Launching server on http://localhost:8000 ...")
    print("\nAccess the web application at:\n  -> http://localhost:8000\n")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.api_server:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    try:
        subprocess.run(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        print("\nVERITAS server stopped.")

if __name__ == "__main__":
    main()
