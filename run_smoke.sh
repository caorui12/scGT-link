#!/usr/bin/env bash
# Quick smoke test (2 epochs, CPU-friendly).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/src"
python main.py --epochs 2 --cpu --output_dir "$ROOT/out/smoke"
