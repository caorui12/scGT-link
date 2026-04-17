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
    def __init__(self, hidden_dim, semantic_dim: int | None = None):
        super().__init__()
        # semantic_dim set → concat [h_src, h_dst, sem_src, sem_dst] (method 2: link-level fusion)
        self.semantic_dim = semantic_dim
        in_dim = 2 * hidden_dim + (2 * semantic_dim if semantic_dim else 0)
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, edge_index, sem: torch.Tensor | None = None):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        if self.semantic_dim and sem is not None:
            s_src = sem[edge_index[0]]
            s_dst = sem[edge_index[1]]
            x = torch.cat([src, dst, s_src, s_dst], dim=1)
        else:
            x = torch.cat([src, dst], dim=1)
        x = self.decoder(x)
        return x

# --- Train Function ---
def train_link_prediction(model, predictor, graph, pos_edges, neg_edges, optimizer, loss_fn):
    model.train()
    predictor.train()

    optimizer.zero_grad()


    h = model(
        graph,
        graph.ndata["feat"],
        graph.ndata["PE"],
        graph.ndata["semantic"],
        True,
    )
    pos_logits = predictor(h, pos_edges, None)
    neg_logits = predictor(h, neg_edges, None)
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
    use_semantic_gt: bool = True,
    sem_link: torch.Tensor | None = None,
):
    model.eval()
    predictor.eval()

    with torch.no_grad():
        h = model(
            graph,
            graph.ndata["feat"],
            graph.ndata["PE"],
            graph.ndata["semantic"],
            use_semantic_gt,
        )
        pos_logits = predictor(h, pos_edges, sem_link)
        neg_logits = predictor(h, neg_edges, sem_link)

    logits = torch.cat([pos_logits, neg_logits])

    
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    return roc_auc_score(labels.cpu(), logits.cpu()), average_precision_score(labels.cpu(), logits.cpu()), labels.cpu(), logits.cpu()