import torch
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# set dgl backend to pytorch
import os
os.environ['DGLBACKEND'] = 'pytorch'


# sparse attention module (without dgl.sparse)
class SparseMHA(nn.Module):
    def __init__(self,hidden_dim=80,num_heads=8):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads

        self.linear_q=nn.Linear(hidden_dim,hidden_dim)
        self.linear_k=nn.Linear(hidden_dim,hidden_dim)
        self.linear_v=nn.Linear(hidden_dim,hidden_dim)

        # projection of output
        self.out_proj=nn.Linear(hidden_dim,hidden_dim)

    def forward(self,g,h):
        # g: DGL graph, h: [N,hidden_dim]
        N=len(h)
        nh=self.num_heads
        dh=self.hidden_dim//nh
        device=h.device

        # Get edges from graph
        src, dst = g.edges()

        q=self.linear_q(h).reshape(N,dh,nh)  # [N,dh,nh]
        k=self.linear_k(h).reshape(N,dh,nh)  # [N,dh,nh]
        v=self.linear_v(h).reshape(N,dh,nh)  # [N,dh,nh]

        # Compute attention scores for edges only
        edge_q = q[src]  # [E,dh,nh]
        edge_k = k[dst]  # [E,dh,nh]
        edge_attention = torch.sum(edge_q * edge_k, dim=1)  # [E,nh]

        # Compute attention weights per source node
        # We need to softmax over neighbors for each source node
        # Group by source and apply softmax

        # Get unique source nodes
        unique_src, inverse_indices = torch.unique(src, return_inverse=True)

        # Initialize output
        out = torch.zeros_like(h).reshape(N, dh, nh)

        for i, node in enumerate(unique_src):
            # Get all outgoing edges from this node
            edge_mask = (src == node)
            node_dst = dst[edge_mask]
            node_edge_attn = edge_attention[edge_mask]  # [num_edges, nh]

            # Softmax over neighbors
            attn_weights = F.softmax(node_edge_attn, dim=0)  # [num_edges, nh]

            # Aggregate values from neighbors
            neighbor_v = v[node_dst]  # [num_edges, dh, nh]

            # Compute weighted sum
            # attn_weights: [num_edges, nh], neighbor_v: [num_edges, dh, nh]
            # out[node, k, h] = sum_j attn_weights[j, h] * neighbor_v[j, k, h]
            weighted_v = attn_weights.unsqueeze(1) * neighbor_v  # [num_edges, dh, nh]
            aggregated = torch.sum(weighted_v, dim=0)  # [dh, nh]

            out[node] = aggregated

        out = out.reshape(N, -1)  # [N, hidden_dim]

        return self.out_proj(out)


class GTLayer(nn.Module):
    def __init__(self,hidden_dim=80,num_heads=8):
        super().__init__()

        self.attention=SparseMHA(hidden_dim,num_heads)
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads

        self.bn1=nn.BatchNorm1d(hidden_dim)
        self.bn2=nn.BatchNorm1d(hidden_dim)

        self.ffn=nn.Sequential(nn.Linear(hidden_dim,2*hidden_dim),
                              nn.ReLU(),
                              nn.Linear(2*hidden_dim,hidden_dim))
    def forward(self,A,h):

        h1=self.attention(A,h) # [N,hidden_dim]
        h=self.bn1(h+h1)

        h2=self.ffn(h)
        h=self.bn2(h+h2)

        return h


class GraphTransformer(nn.Module):
    def __init__(self, in_dim ,hidden_dim, num_heads, num_layers):
        super().__init__()

        self.encoder = nn.Linear(in_dim, hidden_dim)

        self.pos_linear=nn.Linear(in_dim,hidden_dim)

        # BioBERT (or other) gene semantics projected to hidden_dim; same slot as former global_emb
        self.semantic_linear = nn.Linear(in_dim, hidden_dim)

        # stack graph transformer layers
        self.layers=nn.ModuleList(
            [GTLayer(hidden_dim,num_heads) for _ in range(num_layers)]
        )

    def forward(self, g, X, pos_enc, semantic_feat):
        # semantic_feat: (N, in_dim); use zeros when no BioBERT (bias-only contribution from semantic_linear)
        h = (
            self.encoder(X)
            + self.pos_linear(pos_enc)
            + self.semantic_linear(semantic_feat)
        )

        for layer in self.layers:
            h=layer(g,h)

        return h
