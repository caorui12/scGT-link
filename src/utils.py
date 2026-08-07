from __future__ import annotations

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc



# --- Link Prediction Decoder ---
class LinkPredictor(nn.Module):
    """Edge scorer for directed TF→target links.

    Modes
    -----
    concat
        Baseline ``[h_i || h_j] → MLP`` (state_dict keys match older checkpoints).
    rich
        Fuse four channels before the MLP (parameter-light):
        - concat ``[h_i || h_j]`` projected
        - Hadamard ``h_i ⊙ h_j`` projected
        - absolute difference ``|h_i − h_j|`` projected
        - low-rank bilinear ``(W_L h_i) ⊙ (W_R h_j)``
    """

    def __init__(
        self,
        hidden_dim: int,
        mode: str = "concat",
        dropout: float = 0.1,
        bilinear_dim: int | None = None,
    ):
        super().__init__()
        mode = str(mode).lower().strip()
        if mode not in ("concat", "rich"):
            raise ValueError(f"link decoder mode must be 'concat' or 'rich', got {mode!r}")
        self.mode = mode
        self.hidden_dim = int(hidden_dim)
        if mode == "concat":
            self.decoder = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.bilinear_dim = 0
        else:
            # Keep rich head near baseline capacity: full nn.Bilinear(d,d,k) overfits
            # hard-negative splits (~20k edges). Use projections + low-rank bilinear.
            proj = max(16, hidden_dim // 2)
            bdim = int(bilinear_dim) if bilinear_dim is not None else max(8, hidden_dim // 8)
            self.bilinear_dim = bdim
            self.concat_proj = nn.Linear(2 * hidden_dim, proj)
            self.had_proj = nn.Linear(hidden_dim, proj)
            self.diff_proj = nn.Linear(hidden_dim, proj)
            self.bil_left = nn.Linear(hidden_dim, bdim, bias=False)
            self.bil_right = nn.Linear(hidden_dim, bdim, bias=False)
            fuse_in = 3 * proj + bdim
            self.decoder = nn.Sequential(
                nn.Linear(fuse_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        if self.mode == "concat":
            x = torch.cat([src, dst], dim=-1)
            return self.decoder(x)
        cat = self.concat_proj(torch.cat([src, dst], dim=-1))
        had = self.had_proj(src * dst)
        diff = self.diff_proj((src - dst).abs())
        bil = self.bil_left(src) * self.bil_right(dst)
        x = torch.cat([cat, had, diff, bil], dim=-1)
        return self.decoder(x)


# --- Train Function ---
def train_link_prediction(model, predictor, graph, pos_edges, neg_edges, optimizer, loss_fn):
    model.train()
    predictor.train()

    optimizer.zero_grad()


    h = model(graph, graph.ndata["feat"], graph.ndata["PE"])
    pos_logits = predictor(h, pos_edges)
    neg_logits = predictor(h, neg_edges)
    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)

    loss = loss_fn(pos_logits, pos_labels) + loss_fn(neg_logits, neg_labels)
    loss.backward()
    optimizer.step()
    return loss.item(), h

def evaluate(
    model,
    predictor,
    graph,
    pos_edges,
    neg_edges,
):
    model.eval()
    predictor.eval()

    with torch.no_grad():
        h = model(graph, graph.ndata["feat"], graph.ndata["PE"])
        pos_logits = predictor(h, pos_edges)
        neg_logits = predictor(h, neg_edges)

    logits = torch.cat([pos_logits, neg_logits])

    
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    return roc_auc_score(labels.cpu(), logits.cpu()), average_precision_score(labels.cpu(), logits.cpu()), labels.cpu(), logits.cpu()
