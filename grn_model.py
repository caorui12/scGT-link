"""Gene-level Transformer encoder: expression + graph structural PE → per-gene embeddings."""

import torch
import torch.nn as nn


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
