#!/usr/bin/env python3
"""
Encode gene symbols with a HuggingFace encoder → (n_genes, hidden) for train_grn_graph.py --gene_bert_emb.

Backends (see --backend):
  biobert   — dmis-lab/biobert-base-cased-v1.2 (biomedical text; good default for symbols)
  gena-lm   — AIRI-Institute/gena-lm-bert-base (DNA LM; using ASCII symbols is experimental)
  hf        — any AutoModel id via --model_name (you must pass --model_name)

Requires: pip install transformers torch pandas

Examples:
  python scripts/encode_gene_symbols.py --backend biobert \\
    --expression_csv data/hESC/TFs+500/BL--ExpressionData.csv \\
    --output data/hESC/TFs+500/gene_symbols_biobert.pt

  python scripts/encode_gene_symbols.py --backend gena-lm \\
    --expression_csv data/hESC/TFs+500/BL--ExpressionData.csv \\
    --output data/hESC/TFs+500/gene_symbols_gena.pt

  python scripts/encode_gene_symbols.py --backend hf \\
    --model_name some-org/some-bert \\
    --expression_csv ... --output ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

# Presets: default HuggingFace id + whether remote modeling code is needed
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
        description="Gene symbol → encoder [CLS] embeddings (expression CSV row order)"
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=list(ENCODER_PRESETS.keys()) + ["hf"],
        default="biobert",
        help="Encoder preset, or 'hf' with --model_name",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Override HuggingFace model id (required if --backend hf)",
    )
    p.add_argument(
        "--expression_csv",
        type=str,
        required=True,
        help="BL--ExpressionData.csv (gene names = index column, same order as training)",
    )
    p.add_argument("--output", type=str, required=True, help="Output .pt path")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--text_template",
        type=str,
        default="{symbol}",
        help="Format each gene name; e.g. 'Gene: {symbol}'",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=0,
        help="Tokenizer max_length (0 = use preset default)",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass to from_pretrained (use with --backend hf for models that ship custom code)",
    )
    return p.parse_args()


def resolve_model_args(args: argparse.Namespace) -> tuple[str, bool, int, str]:
    if args.backend == "hf":
        if not args.model_name.strip():
            raise SystemExit("--backend hf requires --model_name")
        model_name = args.model_name.strip()
        trust = bool(args.trust_remote_code)
        default_ml = 32
        backend_key = "hf"
    else:
        preset = ENCODER_PRESETS[args.backend]
        model_name = args.model_name.strip() or preset["model_name"]
        trust = bool(preset["trust_remote_code"]) or bool(args.trust_remote_code)
        default_ml = int(preset["default_max_length"])
        backend_key = args.backend
    max_length = int(args.max_length) if args.max_length > 0 else default_ml
    return model_name, trust, max_length, backend_key


def encoder_last_hidden(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """BertModel, or BertForPreTraining-style (use inner .bert)."""
    batch = {k: v for k, v in batch.items() if v is not None}
    inner = getattr(model, "bert", None)
    mod = inner if inner is not None else model
    out = mod(**batch)
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return out[0]
    raise RuntimeError(f"Unexpected model output type: {type(out)}")


def main():
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer

    import pandas as pd

    model_name, trust_remote_code, max_length, backend_key = resolve_model_args(args)

    csv_path = Path(args.expression_csv)
    df = pd.read_csv(csv_path, index_col=0)
    gene_names = [str(x).strip() for x in df.index.tolist()]
    n = len(gene_names)
    if n < 1:
        raise SystemExit("No genes in CSV index")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.eval()
    model.to(device)

    texts = [
        args.text_template.format(symbol=g if g else "<redacted_UNK>") for g in gene_names
    ]
    out_list = []
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, n, bs):
            batch = texts[i : i + bs]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = encoder_last_hidden(model, enc)
            cls = hidden[:, 0, :].cpu()
            out_list.append(cls)
    embeddings = torch.cat(out_list, dim=0).float()
    assert embeddings.shape[0] == n

    h = hashlib.sha256("\n".join(gene_names).encode()).hexdigest()[:16]
    payload = {
        "embeddings": embeddings,
        "gene_names": gene_names,
        "bert_dim": embeddings.shape[1],
        "encoder_backend": backend_key,
        "model_name": model_name,
        "trust_remote_code": trust_remote_code,
        "text_template": args.text_template,
        "max_length": max_length,
        "gene_order_sha256_16": h,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    manifest = {k: v for k, v in payload.items() if k != "embeddings"}
    manifest["n_genes"] = n
    with open(out_path.with_suffix(".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"Saved {out_path} shape={tuple(embeddings.shape)} "
        f"backend={backend_key} model={model_name} manifest={out_path.with_suffix('.manifest.json')}"
    )


if __name__ == "__main__":
    main()
