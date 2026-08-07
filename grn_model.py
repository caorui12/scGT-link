"""Gene-level encoders for scGT-Link (v2 default: scGPT + LapPE; expression optional).

``ScGPTGeneExpressionEncoder`` (``--expr_encoder scgpt``) ignores ``gene_expr`` and uses
pretrained scGPT token embeddings + structural PE. Dual-encoder ``scgpt_concat`` remains available.
"""

import torch
import torch.nn as nn


def _make_transformer_encoder(enc_layer: nn.TransformerEncoderLayer, num_layers: int) -> nn.Module:
    """PyTorch TransformerEncoder fails on num_layers=0; use Identity for linear-only ablations."""
    if num_layers <= 0:
        return nn.Identity()
    return nn.TransformerEncoder(enc_layer, num_layers=num_layers)


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


class GeneTransformerOnly(nn.Module):
    """
    Gene-level ``TransformerEncoder`` on raw expression profiles (no input Linear, no LapPE).
    Each gene is one token; token dimension = ``n_cells``.
    """

    def __init__(
        self,
        n_cells: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.padded_cells = ((n_cells + nhead - 1) // nhead) * nhead
        if self.padded_cells % nhead != 0:
            raise ValueError(
                f"padded_cells ({self.padded_cells}) must be divisible by nhead ({nhead})"
            )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.padded_cells,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = _make_transformer_encoder(enc_layer, num_layers)

    def forward(self, gene_expr: torch.Tensor) -> torch.Tensor:
        """gene_expr: (n_genes, n_cells) -> (n_genes, padded_cells)"""
        if gene_expr.shape[1] < self.padded_cells:
            pad = gene_expr.new_zeros(gene_expr.shape[0], self.padded_cells - gene_expr.shape[1])
            gene_expr = torch.cat([gene_expr, pad], dim=1)
        elif gene_expr.shape[1] > self.padded_cells:
            gene_expr = gene_expr[:, : self.padded_cells]
        return self.encoder(gene_expr.unsqueeze(0)).squeeze(0)


class ScGPTExprParallelEncoder(nn.Module):
    """
    Expression: ``GeneTransformerOnly`` only. scGPT: linear projection to ``d_model``.
    Returns ``(h_expr, h_scgpt)`` for parallel fusion in ``GraphTransformer`` (with LapPE).
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
        self.d_model = d_model
        self.register_buffer("scgpt_gene_emb", scgpt_gene_emb.detach().float())
        self.expr_transformer = GeneTransformerOnly(
            n_cells=n_cells,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.expr_out_dim = self.expr_transformer.padded_cells
        self.scgpt_proj = nn.Linear(scgpt_gene_emb.shape[1], d_model)
        self.expr_encoder_mode = "scgpt_parallel"

    def forward(
        self, gene_expr: torch.Tensor, structural_pe: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del structural_pe  # LapPE fused in GraphTransformer, not here
        h_expr = self.expr_transformer(gene_expr)
        h_scgpt = self.scgpt_proj(self.scgpt_gene_emb.to(dtype=gene_expr.dtype))
        return h_expr, h_scgpt


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
        self.expr_encoder = _make_transformer_encoder(enc_layer, num_layers)
        self.fusion = nn.Linear(2 * d_model, d_model)

    def forward(self, gene_expr: torch.Tensor, structural_pe: torch.Tensor) -> torch.Tensor:
        h_expr = self.expr_encoder(
            self.input_proj(gene_expr).unsqueeze(0) + structural_pe.unsqueeze(0)
        ).squeeze(0)
        h_scgpt = self.scgpt_proj(self.scgpt_gene_emb.to(dtype=structural_pe.dtype))
        return self.fusion(torch.cat([h_expr, h_scgpt], dim=-1))


class ScGPTExprTripleBranchEncoder(nn.Module):
    """
    Three independent branches fused before GraphTransformer (LapPE not inside expr path):

    * Expression: ``Linear(n_cells→d_model)`` + ``TransformerEncoder`` (no LapPE on expr).
    * scGPT: linear projection to ``d_model``.
    * LapPE: graph positional encoding passed through as its own branch.

    ``concat([h_expr, h_scgpt, pe]) → Linear(3*d_model → d_model)``.
    GraphTransformer entry should use ``skip_pos_at_gt=True`` (no second LapPE add).
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
        self.expr_encoder = _make_transformer_encoder(enc_layer, num_layers)
        self.fusion = nn.Linear(3 * d_model, d_model)

    def forward(self, gene_expr: torch.Tensor, structural_pe: torch.Tensor) -> torch.Tensor:
        h_expr = self.expr_encoder(self.input_proj(gene_expr).unsqueeze(0)).squeeze(0)
        h_scgpt = self.scgpt_proj(self.scgpt_gene_emb.to(dtype=structural_pe.dtype))
        return self.fusion(torch.cat([h_expr, h_scgpt, structural_pe], dim=-1))


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
        self.encoder = _make_transformer_encoder(enc_layer, num_layers)
        self.d_model = d_model

    def forward(self, gene_expr: torch.Tensor, structural_pe: torch.Tensor) -> torch.Tensor:
        """
        gene_expr: (n_genes, n_cells)
        structural_pe: (n_genes, d_model), e.g. lap_pe on the Pearson correlation graph
        returns: (n_genes, d_model)
        """
        x = self.input_proj(gene_expr).unsqueeze(0) + structural_pe.unsqueeze(0)
        return self.encoder(x).squeeze(0)


# Backward compatibility for old checkpoints / imports
GeneTransformerGRN = GeneExpressionTransformer
