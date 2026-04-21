"""Gene-level encoders: expression + graph structural PE → per-gene embeddings for GraphTransformer."""

import torch
import torch.nn as nn


class ScGPTGeneExpressionEncoder(nn.Module):
    """
    scGPT pretrained per-gene token embeddings (data-independent) projected to ``d_model``,
    plus optional light Transformer refinement. ``gene_expr`` passed to ``forward`` is ignored
    (kept for API compatibility with ``GeneExpressionTransformer``).
    """

    def __init__(
        self,
        scgpt_gene_emb: torch.Tensor,
        d_model: int,
        refine_layers: int = 0,
        refine_nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if scgpt_gene_emb.dim() != 2:
            raise ValueError(f"scgpt_gene_emb must be (G, D), got {tuple(scgpt_gene_emb.shape)}")
        self.d_model = d_model
        self.register_buffer("scgpt_gene_emb", scgpt_gene_emb.detach().float())
        self.proj = nn.Linear(scgpt_gene_emb.shape[1], d_model)
        if refine_layers > 0:
            if d_model % refine_nhead != 0:
                raise ValueError(
                    f"d_model ({d_model}) must be divisible by refine_nhead ({refine_nhead})"
                )
            enc = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=refine_nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.refine = nn.TransformerEncoder(enc, num_layers=refine_layers)
        else:
            self.refine = None

    def forward(self, gene_expr: torch.Tensor, structural_pe: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.scgpt_gene_emb.to(dtype=structural_pe.dtype)) + structural_pe
        if self.refine is not None:
            x = self.refine(x.unsqueeze(0)).squeeze(0)
        return x


class ScGPTExprConcatEncoder(nn.Module):
    """
    Gene expression: ``Linear(n_cells→d_model) + Lap PE`` then ``TransformerEncoder`` over genes
    (same recipe as ``GeneExpressionTransformer``). scGPT: linear projection to ``d_model``.
    Concatenate ``[h_expr, h_scgpt]`` and fuse with ``Linear(2*d_model → d_model)``.
    """

    def __init__(
        self,
        scgpt_gene_emb: torch.Tensor,
        n_cells: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        if scgpt_gene_emb.dim() != 2:
            raise ValueError(f"scgpt_gene_emb must be (G, D), got {tuple(scgpt_gene_emb.shape)}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.register_buffer("scgpt_gene_emb", scgpt_gene_emb.detach().float())
        self.scgpt_proj = nn.Linear(scgpt_gene_emb.shape[1], d_model)
        self.input_proj = nn.Linear(n_cells, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.expr_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fusion = nn.Linear(2 * d_model, d_model)

    def forward(self, gene_expr: torch.Tensor, structural_pe: torch.Tensor) -> torch.Tensor:
        h_expr = self.expr_encoder(
            self.input_proj(gene_expr).unsqueeze(0) + structural_pe.unsqueeze(0)
        ).squeeze(0)
        h_scgpt = self.scgpt_proj(self.scgpt_gene_emb.to(dtype=structural_pe.dtype))
        return self.fusion(torch.cat([h_expr, h_scgpt], dim=-1))


class GeneExpressionTransformer(nn.Module):
    """
    Expression (and lap_pe on the correlation graph) → per-gene vectors for GraphTransformer.
    Each gene is one token; structural position is graph-based (e.g. dgl.lap_pe), not CSV row index.
    """

    def __init__(
        self,
        n_cells: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_cells, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, gene_expr: torch.Tensor, structural_pe: torch.Tensor) -> torch.Tensor:
        """
        gene_expr: (n_genes, n_cells)
        structural_pe: (n_genes, d_model), e.g. lap_pe on the Pearson correlation graph
        returns: (n_genes, d_model)
        """
        x = self.input_proj(gene_expr).unsqueeze(0) + structural_pe.unsqueeze(0)
        out = self.encoder(x)
        return out.squeeze(0)


# Backward compatibility for old checkpoints / imports
GeneTransformerGRN = GeneExpressionTransformer
