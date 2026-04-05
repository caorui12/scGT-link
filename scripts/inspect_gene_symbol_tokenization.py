#!/usr/bin/env python3
"""
Summarize tokenizer stats for gene names (UNK, truncation vs scripts/encode_gene_symbols.py).

Requires: pip install transformers pandas

Example:
  python scripts/inspect_gene_symbol_tokenization.py --backend biobert \\
    --expression_csv data/hESC/TFs+500/BL--ExpressionData.csv

  python scripts/inspect_gene_symbol_tokenization.py --backend gena-lm \\
    --manifest_json data/hESC/TFs+500/gene_symbols_gena.manifest.json --list_unk
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Keep in sync with encode_gene_symbols.ENCODER_PRESETS
ENCODER_PRESETS: dict[str, dict[str, Any]] = {
    "biobert": {
        "model_name": "dmis-lab/biobert-base-cased-v1.2",
        "trust_remote_code": False,
        "default_max_length": 32,
    },
    "gena-lm": {
        "model_name": "AIRI-Institute/gena-lm-bert-base",
        "trust_remote_code": True,
        "default_max_length": 64,
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect HuggingFace tokenizer on gene names (match encode_gene_symbols.py)"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--expression_csv",
        type=str,
        help="BL--ExpressionData.csv (gene names = index column)",
    )
    src.add_argument(
        "--manifest_json",
        type=str,
        help="*.manifest.json from encode_gene_symbols (uses gene_names; optional model hints)",
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=list(ENCODER_PRESETS.keys()) + ["hf"],
        default="biobert",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Override model id (required if --backend hf)",
    )
    p.add_argument(
        "--text_template",
        type=str,
        default="{symbol}",
        help="Same as encode script",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="For custom --model_name (e.g. GENA) when not using a preset manifest",
    )
    p.add_argument("--max_length", type=int, default=0, help="0 = preset default")
    p.add_argument(
        "--list_unk",
        action="store_true",
        help="Print each gene with at least one UNK token id",
    )
    p.add_argument(
        "--top_truncated",
        type=int,
        default=0,
        metavar="K",
        help="Print up to K genes longest before truncation (truncation=False length)",
    )
    return p.parse_args()


def resolve_model_args(
    args: argparse.Namespace, manifest_data: dict | None
) -> tuple[str, bool, int]:
    mn_user = args.model_name.strip()
    if mn_user:
        model_name = mn_user
        trust = bool(args.trust_remote_code)
        default_ml = 32
    elif manifest_data and isinstance(manifest_data.get("model_name"), str) and manifest_data[
        "model_name"
    ].strip():
        model_name = manifest_data["model_name"].strip()
        trust = bool(manifest_data.get("trust_remote_code", False))
        ml = manifest_data.get("max_length")
        default_ml = int(ml) if ml is not None else 32
    elif args.backend == "hf":
        raise SystemExit("--backend hf requires --model_name (or a manifest with model_name)")
    else:
        preset = ENCODER_PRESETS[args.backend]
        model_name = preset["model_name"]
        trust = bool(preset["trust_remote_code"]) or bool(args.trust_remote_code)
        default_ml = int(preset["default_max_length"])
    max_length = int(args.max_length) if args.max_length > 0 else default_ml
    return model_name, trust, max_length


def load_gene_names(args, manifest_data: dict | None) -> list[str]:
    if args.expression_csv:
        import pandas as pd

        df = pd.read_csv(Path(args.expression_csv), index_col=0)
        return [str(x).strip() for x in df.index.tolist()]
    data = manifest_data
    if data is None:
        with open(Path(args.manifest_json)) as f:
            data = json.load(f)
    names = data.get("gene_names")
    if not isinstance(names, list):
        raise SystemExit("manifest missing gene_names list")
    return [str(x).strip() for x in names]


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    manifest_data: dict | None = None
    if args.manifest_json:
        with open(Path(args.manifest_json)) as f:
            manifest_data = json.load(f)
    model_name, trust_remote_code, max_length = resolve_model_args(args, manifest_data)

    gene_names = load_gene_names(args, manifest_data)
    n = len(gene_names)
    if n < 1:
        raise SystemExit("No gene names")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    unk_id = tok.unk_token_id
    pad_id = tok.pad_token_id

    texts = [args.text_template.format(symbol=g if g else "<redacted_UNK>") for g in gene_names]

    n_any_unk = 0
    n_truncated = 0
    total_unk_in_seq = 0
    max_tokens_trunc = 0
    unk_examples: list[tuple[str, int, list[str]]] = []

    for g, text in zip(gene_names, texts):
        full = tok(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        full_ids = full["input_ids"]
        trunc = tok(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_tensors=None,
        )
        t_ids = trunc["input_ids"]
        max_tokens_trunc = max(max_tokens_trunc, len(t_ids))

        unk_here = sum(1 for i in t_ids if unk_id is not None and i == unk_id)
        total_unk_in_seq += unk_here
        if unk_here > 0:
            n_any_unk += 1
            if args.list_unk:
                toks = tok.convert_ids_to_tokens(t_ids)
                unk_examples.append((g, unk_here, toks))

        if len(full_ids) > len(t_ids):
            n_truncated += 1

    print(f"model_name={model_name} trust_remote_code={trust_remote_code}")
    print(f"text_template={args.text_template!r} max_length={max_length}")
    print(f"n_genes={n}")
    print(f"genes_with_any_unk_token={n_any_unk} ({100 * n_any_unk / n:.2f}%)")
    print(f"sum_unk_token_ids_across_genes={total_unk_in_seq}")
    print(f"genes_truncated_to_max_length={n_truncated} ({100 * n_truncated / n:.2f}%)")
    print(f"max_input_ids_len_after_truncation={max_tokens_trunc}")
    print(f"tokenizer.unk_token_id={unk_id} pad_token_id={pad_id}")

    if args.list_unk and unk_examples:
        print("\n--- genes with UNK (truncated encoding) ---")
        for g, k, toks in sorted(unk_examples, key=lambda x: -x[1])[:500]:
            print(f"{g}\tunk_count={k}\t{' '.join(toks)}")
        if len(unk_examples) > 500:
            print(f"... ({len(unk_examples) - 500} more omitted; increase code limit if needed)")

    if args.top_truncated > 0:
        lengths: list[tuple[int, str]] = []
        for g, text in zip(gene_names, texts):
            full = tok(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            lengths.append((len(full["input_ids"]), g))
        lengths.sort(reverse=True)
        print(f"\n--- top {args.top_truncated} by token length (no truncation) ---")
        for L, g in lengths[: args.top_truncated]:
            print(f"len={L}\t{g}")


if __name__ == "__main__":
    main()
