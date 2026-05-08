#!/usr/bin/env python3
"""
Legacy shim: forwards CLI to **scGREAT** ``benchmark_train_suite.py``.

Primary workflow — run inside your **scGREAT** clone (no Graph_Transformer needed)::

  cd /path/to/scGREAT
  export SCGREAT_BENCHMARK_ROOT="/path/to/Benchmark Dataset"
  python benchmark_train_suite.py --epochs 80

This wrapper exists only so old paths keep working. Override repo location::

  export SCGREAT_REPO=/path/to/scGREAT
  python scripts/benchmark_train_scgreat_suite.py ...
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_GT_ROOT = _HERE.parent
_DEFAULT_SCGREAT = _GT_ROOT.parent / "scGREAT"


def main() -> None:
    suite = Path(os.environ.get("SCGREAT_REPO", str(_DEFAULT_SCGREAT))).expanduser().resolve() / "benchmark_train_suite.py"
    if not suite.is_file():
        print(
            "Cannot find scGREAT benchmark driver:\n"
            f"  {suite}\n"
            "Clone scGREAT or set SCGREAT_REPO.\n"
            "Primary command:\n"
            "  cd /path/to/scGREAT && python benchmark_train_suite.py ...\n",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
    rc = subprocess.run([sys.executable, str(suite)] + sys.argv[1:], cwd=str(suite.parent)).returncode
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
