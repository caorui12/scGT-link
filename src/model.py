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


def _grouped_softmax(logits: torch.Tensor, group_index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Softmax over edges that share the same ``group_index`` (e.g. destination node).

    logits: [E, H], group_index: [E]
    """
    # numerical stability via per-group max
    neg_inf = torch.finfo(logits.dtype).min
    group_max = torch.full(
        (n_groups, logits.size(1)),
        neg_inf,
        device=logits.device,
        dtype=logits.dtype,
    )
    idx = group_index.unsqueeze(1).expand_as(logits)
    group_max.scatter_reduce_(0, idx, logits, reduce="amax", include_self=True)
    exp = torch.exp(logits - group_max[group_index])
    group_sum = torch.zeros_like(group_max)
    group_sum.scatter_add_(0, idx, exp)
    return exp / (group_sum[group_index].clamp_min(1e-9))


# sparse attention module (without dgl.sparse)
class SparseMHA(nn.Module):
    """Sparse multi-head attention on prior-graph edges.

    Parameters
    ----------
    aggregate:
        ``src`` — legacy: each source aggregates values from its destinations
        (message flow opposite to edge arrow).
        ``dst`` — each destination aggregates from sources (standard directed MP:
        on TF→target edges, targets receive TF messages).
    use_edge_message:
        If True, add an MLP embedding of ``[w, dir]`` into the value message.
    """

    def __init__(
        self,
        hidden_dim=80,
        num_heads=8,
        use_edge_weight_bias: bool = True,
        use_directed_edge_bias: bool = False,
        *,
        aggregate: str = "src",
        use_edge_message: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        aggregate = str(aggregate).lower().strip()
        if aggregate not in ("src", "dst"):
            raise ValueError(f"aggregate must be 'src' or 'dst', got {aggregate!r}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_edge_weight_bias = use_edge_weight_bias
        self.use_directed_edge_bias = use_directed_edge_bias
        self.aggregate = aggregate
        self.use_edge_message = use_edge_message
        self.attn_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        if use_edge_weight_bias:
            # Add Pearson r_ij (from g.edata['w']) to attention logits before softmax.
            self.edge_logit_scale = nn.Parameter(torch.tensor(1.0))
        if use_directed_edge_bias:
            # +1/−1 dir bias, plus binary TF→target (tf2t) bias.
            self.edge_dir_scale = nn.Parameter(torch.tensor(1.0))
            self.edge_tf2t_scale = nn.Parameter(torch.tensor(1.0))
        if use_edge_message:
            # Edge conditioner: [w, dir] → hidden, added to value messages.
            self.edge_msg_mlp = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def _edge_features(self, g, device, dtype, n_edges: int):
        if "w" in g.edata:
            w = g.edata["w"].to(device=device, dtype=dtype).reshape(-1)
        else:
            w = torch.zeros(n_edges, device=device, dtype=dtype)
        if "dir" in g.edata:
            ddir = g.edata["dir"].to(device=device, dtype=dtype).reshape(-1)
        else:
            ddir = torch.zeros(n_edges, device=device, dtype=dtype)
        if "tf2t" in g.edata:
            tf2t = g.edata["tf2t"].to(device=device, dtype=dtype).reshape(-1)
        elif "dir" in g.edata:
            # Derive on the fly for older graphs that only stored dir.
            tf2t = (ddir > 0).to(dtype=dtype)
        else:
            tf2t = torch.zeros(n_edges, device=device, dtype=dtype)
        return w, ddir, tf2t

    def forward(self, g, h):
        # g: DGL graph, h: [N, hidden_dim]
        N = h.size(0)
        nh = self.num_heads
        dh = self.hidden_dim // nh
        device = h.device

        src, dst = g.edges()
        E = src.numel()

        q = self.linear_q(h).reshape(N, dh, nh)
        k = self.linear_k(h).reshape(N, dh, nh)
        v = self.linear_v(h).reshape(N, dh, nh)

        w, ddir, tf2t = self._edge_features(g, device, h.dtype, E)

        if self.aggregate == "dst":
            # Incoming MP: target (dst) queries, source (src) keys/values.
            edge_q = q[dst]
            edge_k = k[src]
            neighbor_v = v[src]
            group = dst
        else:
            # Legacy: source queries, destination keys/values.
            edge_q = q[src]
            edge_k = k[dst]
            neighbor_v = v[dst]
            group = src

        # [E, nh] — scale only for directed variant (legacy matched unscaled logits)
        edge_attention = (edge_q * edge_k).sum(dim=1)
        if self.aggregate == "dst":
            edge_attention = edge_attention / (dh ** 0.5)

        if self.use_edge_weight_bias and hasattr(self, "edge_logit_scale"):
            edge_attention = edge_attention + self.edge_logit_scale * w.unsqueeze(-1)
        if self.use_directed_edge_bias and hasattr(self, "edge_dir_scale"):
            edge_attention = edge_attention + self.edge_dir_scale * ddir.unsqueeze(-1)
        if self.use_directed_edge_bias and hasattr(self, "edge_tf2t_scale"):
            edge_attention = edge_attention + self.edge_tf2t_scale * tf2t.unsqueeze(-1)

        attn_weights = _grouped_softmax(edge_attention, group, N)
        attn_weights = self.attn_drop(attn_weights)

        if self.use_edge_message and hasattr(self, "edge_msg_mlp"):
            e_feat = torch.stack([w, ddir], dim=-1)
            e_msg = self.edge_msg_mlp(e_feat).reshape(E, dh, nh)
            neighbor_v = neighbor_v + e_msg

        # weighted sum into group nodes
        weighted = attn_weights.unsqueeze(1) * neighbor_v  # [E, dh, nh]
        out = torch.zeros(N, dh, nh, device=device, dtype=h.dtype)
        out.index_add_(0, group, weighted)
        out = out.reshape(N, -1)
        return self.out_proj(out)


class SparseMeanAgg(nn.Module):
    """Degree-normalized mean aggregation on prior-graph edges (attention ablation).

    Same aggregate direction and optional ``[w, dir]`` value messages as SparseMHA,
    but neighbor weights are uniform ``1/degree`` instead of QK softmax.
    """

    def __init__(
        self,
        hidden_dim=80,
        num_heads=8,
        *,
        aggregate: str = "dst",
        use_edge_message: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        aggregate = str(aggregate).lower().strip()
        if aggregate not in ("src", "dst"):
            raise ValueError(f"aggregate must be 'src' or 'dst', got {aggregate!r}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.aggregate = aggregate
        self.use_edge_message = use_edge_message
        self.attn_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        if use_edge_message:
            self.edge_msg_mlp = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, g, h):
        N = h.size(0)
        nh = self.num_heads
        dh = self.hidden_dim // nh
        device = h.device
        src, dst = g.edges()
        E = src.numel()

        v = self.linear_v(h).reshape(N, dh, nh)
        if self.aggregate == "dst":
            neighbor_v = v[src]
            group = dst
        else:
            neighbor_v = v[dst]
            group = src

        if self.use_edge_message and hasattr(self, "edge_msg_mlp"):
            if "w" in g.edata:
                w = g.edata["w"].to(device=device, dtype=h.dtype).reshape(-1)
            else:
                w = torch.zeros(E, device=device, dtype=h.dtype)
            if "dir" in g.edata:
                ddir = g.edata["dir"].to(device=device, dtype=h.dtype).reshape(-1)
            else:
                ddir = torch.zeros(E, device=device, dtype=h.dtype)
            e_msg = self.edge_msg_mlp(torch.stack([w, ddir], dim=-1)).reshape(E, dh, nh)
            neighbor_v = neighbor_v + e_msg

        deg = torch.zeros(N, device=device, dtype=h.dtype)
        ones = torch.ones(E, device=device, dtype=h.dtype)
        deg.index_add_(0, group, ones)
        inv_deg = 1.0 / deg.clamp(min=1.0)
        weights = self.attn_drop(inv_deg[group])  # [E]

        weighted = weights.unsqueeze(1).unsqueeze(2) * neighbor_v  # [E, dh, nh]
        out = torch.zeros(N, dh, nh, device=device, dtype=h.dtype)
        out.index_add_(0, group, weighted)
        out = out.reshape(N, -1)
        return self.out_proj(out)


class GTLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=80,
        num_heads=8,
        use_edge_weight_bias: bool = True,
        use_directed_edge_bias: bool = False,
        *,
        aggregate: str = "src",
        use_edge_message: bool = False,
        norm_type: str = "bn",
        dropout: float = 0.0,
        use_ffn: bool = True,
        use_residual: bool = True,
        gt_agg: str = "attn",
    ):
        super().__init__()
        gt_agg = str(gt_agg).lower().strip()
        if gt_agg not in ("attn", "mean"):
            raise ValueError(f"gt_agg must be 'attn' or 'mean', got {gt_agg!r}")
        self.gt_agg = gt_agg
        self.use_ffn = bool(use_ffn)
        self.use_residual = bool(use_residual)
        if gt_agg == "mean":
            self.attention = SparseMeanAgg(
                hidden_dim,
                num_heads,
                aggregate=aggregate,
                use_edge_message=use_edge_message,
                dropout=dropout,
            )
        else:
            self.attention = SparseMHA(
                hidden_dim,
                num_heads,
                use_edge_weight_bias=use_edge_weight_bias,
                use_directed_edge_bias=use_directed_edge_bias,
                aggregate=aggregate,
                use_edge_message=use_edge_message,
                dropout=dropout,
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.norm_type = str(norm_type).lower().strip()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if self.norm_type == "ln":
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim) if self.use_ffn else None
        else:
            # Legacy attribute names (bn1/bn2) for older checkpoints.
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim) if self.use_ffn else None

        if self.use_ffn:
            if dropout and dropout > 0:
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
            else:
                # Match legacy checkpoint key layout: ffn.0 / ffn.2
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
        else:
            self.ffn = None

    def _norm1(self, x):
        if self.norm_type == "ln":
            return self.norm1(x)
        return self.bn1(x)

    def _norm2(self, x):
        if self.norm_type == "ln":
            return self.norm2(x)
        return self.bn2(x)

    def forward(self, A, h):
        h1 = self.attention(A, h)
        if self.use_residual:
            h = self._norm1(h + self.dropout(h1))
        else:
            h = self._norm1(self.dropout(h1))
        if not self.use_ffn or self.ffn is None:
            return h
        h2 = self.ffn(h)
        if self.use_residual:
            h = self._norm2(h + self.dropout(h2))
        else:
            h = self._norm2(self.dropout(h2))
        return h


class GraphTransformer(nn.Module):
    """Graph Transformer over the prior GRN graph.

    ``gt_variant``
        ``directed`` — dst-aggregation, edge-conditioned messages, LayerNorm + dropout
        (default; vectorized SparseMHA).
        ``legacy`` — original src-aggregation + BatchNorm (older ckpts / ablation).
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_heads,
        num_layers,
        use_edge_weight_attn: bool = True,
        use_directed_edge_bias: bool = False,
        *,
        triple_fuse: bool = False,
        expr_in_dim: int | None = None,
        scgpt_in_dim: int | None = None,
        skip_pos_at_gt: bool = False,
        gt_variant: str = "directed",
        dropout: float = 0.0,
        use_ffn: bool = True,
        use_residual: bool = True,
        gt_agg: str = "attn",
    ):
        super().__init__()
        self.triple_fuse = triple_fuse
        self.skip_pos_at_gt = skip_pos_at_gt
        gt_variant = str(gt_variant).lower().strip()
        if gt_variant not in ("legacy", "directed"):
            raise ValueError(f"gt_variant must be 'legacy' or 'directed', got {gt_variant!r}")
        self.gt_variant = gt_variant
        self.use_ffn = bool(use_ffn)
        self.use_residual = bool(use_residual)
        self.gt_agg = str(gt_agg).lower().strip()
        directed = gt_variant == "directed"
        aggregate = "dst" if directed else "src"
        use_edge_message = directed
        norm_type = "ln" if directed else "bn"
        drop = float(dropout) if directed else 0.0

        if triple_fuse:
            if expr_in_dim is None or scgpt_in_dim is None:
                raise ValueError("triple_fuse requires expr_in_dim and scgpt_in_dim")
            self.expr_encoder = nn.Linear(expr_in_dim, hidden_dim)
            self.scgpt_encoder = nn.Linear(scgpt_in_dim, hidden_dim)
            self.pos_linear = nn.Linear(in_dim, hidden_dim)
            self.encoder = None
        else:
            self.encoder = nn.Linear(in_dim, hidden_dim)
            self.pos_linear = None if skip_pos_at_gt else nn.Linear(in_dim, hidden_dim)
            self.expr_encoder = None
            self.scgpt_encoder = None

        self.layers = nn.ModuleList(
            [
                GTLayer(
                    hidden_dim,
                    num_heads,
                    use_edge_weight_bias=use_edge_weight_attn,
                    use_directed_edge_bias=use_directed_edge_bias,
                    aggregate=aggregate,
                    use_edge_message=use_edge_message,
                    norm_type=norm_type,
                    dropout=drop,
                    use_ffn=self.use_ffn,
                    use_residual=self.use_residual,
                    gt_agg=self.gt_agg,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        g,
        X,
        pos_enc,
        *,
        h_expr: torch.Tensor | None = None,
        h_scgpt: torch.Tensor | None = None,
    ):
        if self.triple_fuse:
            if h_expr is None or h_scgpt is None:
                raise ValueError("triple_fuse GraphTransformer requires h_expr and h_scgpt")
            h = (
                self.expr_encoder(h_expr)
                + self.scgpt_encoder(h_scgpt)
                + self.pos_linear(pos_enc)
            )
        else:
            if self.encoder is None:
                raise ValueError("encoder is required when triple_fuse is False")
            if self.skip_pos_at_gt:
                h = self.encoder(X)
            else:
                h = self.encoder(X) + self.pos_linear(pos_enc)
        for layer in self.layers:
            h = layer(g, h)
        return h
