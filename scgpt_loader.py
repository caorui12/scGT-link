"""
Load scGPT per-gene embeddings (data-independent) from a released checkpoint folder.

Only the ``GeneEncoder`` (embedding + LayerNorm) weights are loaded — no full forward pass.

Requires: ``pip install scgpt`` and a downloaded model directory with ``args.json``, ``vocab.json``,
and ``best_model.pt`` (or ``model.pt``). See https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch


def _pick_model_file(model_dir: Path) -> Path:
    for name in ("best_model.pt", "model.pt", "checkpoint.pt"):
        p = model_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No model weights in {model_dir} (tried best_model.pt, model.pt, checkpoint.pt)"
    )


def _vocab_lookup(vocab, g: str) -> Optional[int]:
    if g in vocab:
        return int(vocab[g])
    if g.upper() in vocab:
        return int(vocab[g.upper()])
    if g.lower() in vocab:
        return int(vocab[g.lower()])
    return None


def load_scgpt_gene_embedding_matrix(
    model_dir: str | Path,
    gene_names: list[str],
    device: torch.device,
    ortholog_map: Optional[dict[str, str]] = None,
) -> tuple[torch.Tensor, int]:
    """
    Return ``(G, embsize)`` gene token embeddings from the pretrained scGPT ``GeneEncoder``.

    Genes missing from the vocabulary use the padding row.
    If ``ortholog_map`` is provided (e.g. mouse symbol → human symbol via HomoloGene),
    unmatched genes are remapped before falling back to pad.
    """
    try:
        try:
            from scgpt.model.model import GeneEncoder
        except ImportError:
            from scgpt.model import GeneEncoder  # type: ignore
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
    except ImportError as e:
        raise ImportError(
            "scgpt is required for --expr_encoder scgpt without --scgpt_emb_pt. "
            "Install with: pip install scgpt"
        ) from e

    model_dir = Path(model_dir).resolve()
    vocab_path = model_dir / "vocab.json"
    cfg_path = model_dir / "args.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing {vocab_path}")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    vocab = GeneVocab.from_file(vocab_path)
    for s in ("<pad>", "<cls>", "<eoc>"):
        if s not in vocab:
            vocab.append_token(s)

    pad_token = str(cfg.get("pad_token", "<pad>"))
    if pad_token not in vocab:
        pad_token = "<pad>"
    pad_id = int(vocab[pad_token])

    embsize = int(cfg["embsize"])
    ntokens = len(vocab)

    encoder = GeneEncoder(ntokens, embsize, padding_idx=pad_id)
    model_path = _pick_model_file(model_dir)
    state = torch.load(model_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format in {model_path}")

    enc_state = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            enc_state[k[len("encoder.") :]] = v
    if not enc_state:
        raise ValueError(
            f"No keys starting with 'encoder.' in {model_path}; cannot load GeneEncoder"
        )
    encoder.load_state_dict(enc_state, strict=True)
    encoder = encoder.to(device)
    encoder.eval()

    omap = ortholog_map or {}
    ids: list[int] = []
    for g in gene_names:
        gid = _vocab_lookup(vocab, g)
        if gid is None and omap:
            human = omap.get(g) or omap.get(str(g).upper()) or omap.get(str(g).lower())
            if human:
                gid = _vocab_lookup(vocab, str(human))
        ids.append(int(gid if gid is not None else pad_id))

    id_t = torch.tensor(ids, dtype=torch.long, device=device)
    with torch.no_grad():
        mat = encoder(id_t)

    return mat.detach().cpu().float(), embsize


def load_scgpt_emb_from_pt(
    path: str | Path,
    gene_names: list[str],
) -> tuple[torch.Tensor, int]:
    """Load precomputed tensor from ``scripts/precompute_scgpt_gene_emb.py``."""
    path = Path(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        emb = obj["embeddings"]
        stored = obj.get("gene_names")
        dim = int(obj.get("scgpt_dim", emb.shape[1]))
    else:
        emb = obj
        stored = None
        dim = emb.shape[1]
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    if emb.dim() != 2:
        raise ValueError(f"embeddings must be 2D, got {tuple(emb.shape)}")
    if emb.shape[0] != len(gene_names):
        raise ValueError(
            f"Embedding rows {emb.shape[0]} != len(gene_names) {len(gene_names)}"
        )
    if stored is not None and list(stored) != list(gene_names):
        raise ValueError("gene_names in embedding file do not match expression matrix order")
    return emb.float(), dim
