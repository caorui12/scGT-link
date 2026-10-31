#!/usr/bin/env python3
"""Build mouse→human gene-symbol map (HomoloGene via mygene) for scGPT rematching.

Writes JSON with key ``map``: mouse_symbol -> human_symbol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _collect_mouse_genes(benchmark_root: Path) -> list[str]:
    from grn_data import load_expression_for_grn

    genes: set[str] = set()
    for exp in sorted(benchmark_root.rglob("BL--ExpressionData.csv")):
        parts = exp.parts
        if not any(x in parts for x in ("mESC", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L")):
            continue
        _, names, _ = load_expression_for_grn(exp)
        genes.update(str(g) for g in names)
    return sorted(genes)


def _query_mygene(symbols: list[str], batch_size: int = 1000) -> dict[str, str]:
    import mygene

    mg = mygene.MyGeneInfo()
    mouse_to_human_gid: dict[str, int] = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        hits = mg.querymany(
            batch,
            scopes="symbol",
            fields="symbol,homologene",
            species="mouse",
            as_dataframe=False,
            returnall=False,
            verbose=False,
        )
        for h in hits:
            q = str(h.get("query", "")).strip()
            if not q or h.get("notfound"):
                continue
            hg = h.get("homologene") or {}
            for pair in hg.get("genes") or []:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                if int(pair[0]) == 9606:
                    mouse_to_human_gid[q] = int(pair[1])
                    break
        print(
            f"  homologene batch {i // batch_size + 1}: "
            f"{len(mouse_to_human_gid)} mouse genes with human gid",
            flush=True,
        )

    human_gids = sorted(set(mouse_to_human_gid.values()))
    gid_to_symbol: dict[int, str] = {}
    for i in range(0, len(human_gids), batch_size):
        batch = human_gids[i : i + batch_size]
        hits = mg.getgenes(batch, fields="symbol", species="human", as_dataframe=False)
        if isinstance(hits, dict):
            hits = [hits]
        for h in hits or []:
            if not isinstance(h, dict):
                continue
            eid = h.get("_id") or h.get("entrezgene")
            sym = h.get("symbol")
            if eid is not None and sym:
                gid_to_symbol[int(eid)] = str(sym)
        print(
            f"  human symbol batch {i // batch_size + 1}: {len(gid_to_symbol)} symbols",
            flush=True,
        )

    out: dict[str, str] = {}
    for mouse_sym, gid in mouse_to_human_gid.items():
        hs = gid_to_symbol.get(gid)
        if hs:
            out[mouse_sym] = hs
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--benchmark_root",
        type=str,
        default=str(
            Path("/Users/drruicao/Desktop/deeplearning/benchmark/Dataset/Benchmark Dataset")
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "revision for bib" / "tables" / "mouse_human_ortholog_map.json"),
    )
    args = p.parse_args()

    root = Path(args.benchmark_root)
    genes = _collect_mouse_genes(root)
    print(f"Unique mouse panel genes: {len(genes)}", flush=True)
    mapping = _query_mygene(genes)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_query_genes": len(genes),
        "n_mapped": len(mapping),
        "source": "mygene HomoloGene (mouse→human symbol)",
        "map": mapping,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(mapping)} mappings to {out}", flush=True)
    # quick sanity
    for k in ("TRP53", "Trp53", "MYC", "Myc"):
        if k in mapping:
            print(f"  example {k} -> {mapping[k]}", flush=True)


if __name__ == "__main__":
    main()
