#!/usr/bin/env python3
"""Precompute scgpt_gene_emb_ortholog.pt for all mouse BEELINE panels.

Uses HomoloGene mouse→human symbol map only for genes that miss the human scGPT vocab
under exact/upper/lower matching (same pad fallback otherwise).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from grn_data import load_expression_for_grn
from scgpt_loader import load_scgpt_gene_embedding_matrix, _vocab_lookup


def _mouse_expr_dirs(benchmark_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for exp in sorted(benchmark_root.rglob("BL--ExpressionData.csv")):
        parts = exp.parts
        if not any(x in parts for x in ("mESC", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L")):
            continue
        dirs.append(exp.parent)
    return dirs


def _load_vocab(model_dir: Path):
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in ("<pad>", "<cls>", "<eoc>"):
        if s not in vocab:
            vocab.append_token(s)
    return vocab


def match_stats(gene_names: list[str], vocab, omap: dict[str, str]) -> dict:
    n = len(gene_names)
    direct = 0
    via_ortho = 0
    still_miss = 0
    recovered: list[tuple[str, str]] = []
    for g in gene_names:
        if _vocab_lookup(vocab, g) is not None:
            direct += 1
            continue
        human = omap.get(g) or omap.get(g.upper()) or omap.get(g.lower())
        if human and _vocab_lookup(vocab, str(human)) is not None:
            via_ortho += 1
            recovered.append((g, str(human)))
        else:
            still_miss += 1
    return {
        "n_genes": n,
        "direct_match": direct,
        "ortholog_recovered": via_ortho,
        "still_miss": still_miss,
        "direct_frac": direct / n if n else 0.0,
        "with_ortholog_frac": (direct + via_ortho) / n if n else 0.0,
        "recovered_examples": recovered[:20],
    }


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
        "--scgpt_model_dir",
        type=str,
        default=str(_ROOT / "scGPT_human"),
    )
    p.add_argument(
        "--ortholog_map",
        type=str,
        default=str(_ROOT / "revision for bib" / "tables" / "mouse_human_ortholog_map.json"),
    )
    p.add_argument(
        "--stats_out",
        type=str,
        default=str(
            _ROOT / "revision for bib" / "tables" / "mouse_scgpt_ortholog_match_stats.json"
        ),
    )
    args = p.parse_args()

    omap_obj = json.loads(Path(args.ortholog_map).read_text(encoding="utf-8"))
    omap: dict[str, str] = dict(omap_obj.get("map") or omap_obj)
    model_dir = Path(args.scgpt_model_dir)
    vocab = _load_vocab(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_stats = []
    dirs = _mouse_expr_dirs(Path(args.benchmark_root))
    print(f"Mouse panels: {len(dirs)}", flush=True)
    for data_dir in dirs:
        _, gene_names, _ = load_expression_for_grn(data_dir / "BL--ExpressionData.csv")
        st = match_stats(gene_names, vocab, omap)
        st["data_dir"] = str(data_dir)
        all_stats.append(st)

        mat, dim = load_scgpt_gene_embedding_matrix(
            model_dir, gene_names, device=device, ortholog_map=omap
        )
        out = {
            "embeddings": mat,
            "gene_names": gene_names,
            "scgpt_dim": dim,
            "scgpt_model_dir": str(model_dir.resolve()),
            "ortholog_map_path": str(Path(args.ortholog_map).resolve()),
            "match_stats": {k: st[k] for k in st if k != "recovered_examples"},
        }
        out_path = data_dir / "scgpt_gene_emb_ortholog.pt"
        torch.save(out, out_path)
        print(
            f"  {data_dir.name}: direct={st['direct_frac']:.3f} "
            f"+ortho={st['with_ortholog_frac']:.3f} "
            f"(+{st['ortholog_recovered']} genes) -> {out_path}",
            flush=True,
        )

    stats_path = Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_panels": len(all_stats),
        "mean_direct_frac": sum(s["direct_frac"] for s in all_stats) / max(len(all_stats), 1),
        "mean_with_ortholog_frac": sum(s["with_ortholog_frac"] for s in all_stats)
        / max(len(all_stats), 1),
        "total_ortholog_recovered": sum(s["ortholog_recovered"] for s in all_stats),
        "panels": all_stats,
    }
    stats_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Stats -> {stats_path}", flush=True)
    print(
        f"Mean coverage direct={summary['mean_direct_frac']:.4f} "
        f"with_ortholog={summary['mean_with_ortholog_frac']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
