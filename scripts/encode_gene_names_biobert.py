#!/usr/bin/env python3
"""Deprecated entry point: use encode_gene_symbols.py --backend biobert."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def main() -> None:
    here = Path(__file__).resolve().parent
    target = here / "encode_gene_symbols.py"
    cmd = [sys.executable, str(target), "--backend", "biobert", *sys.argv[1:]]
    print(
        "encode_gene_names_biobert.py → encode_gene_symbols.py --backend biobert",
        file=sys.stderr,
    )
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
