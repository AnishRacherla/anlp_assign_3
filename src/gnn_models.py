from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    b_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = torch.zeros(b_size, x.size(-1), device=x.device, dtype=x.dtype)
    counts = torch.zeros(b_size, 1, device=x.device, dtype=x.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
    counts.scatter_add_(0, batch.unsqueeze(-1), torch.ones(batch.size(0), 1, device=x.device, dtype=x.dtype))
    return out / counts.clamp(min=1.0)

def extract_root_features(x: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
    if ptr is not None:
        root_idx = ptr[:-1]
    else:
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root_idx = torch.cat([root.new_zeros(1), root + 1], dim=0)
    return x[root_idx]

class FromScratchGCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        device = x.device
        loop_index = torch.arange(num_nodes, device=device).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)

        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, device=device).scatter_add_(0, col, torch.ones(col.size(0), device=device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x_proj = self.lin(x)
        msg = x_proj[row] * norm.unsqueeze(-1)
        out = torch.zeros(num_nodes, x_proj.size(-1), device=device).scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg)
        return out

class FromScratchSAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        device = x.device
        row, col = edge_index[0], edge_index[1]
        msg = x[row]
        deg = torch.zeros(num_nodes, 1, device=device).scatter_add_(0, col.unsqueeze(-1), torch.ones(col.size(0), 1, device=device)).clamp(min=1.0)
        neigh_sum = torch.zeros(num_nodes, x.size(-1), device=device).scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg)
        neigh_mean = neigh_sum / deg
        return self.lin_self(x) + self.lin_neigh(neigh_mean)

class FromScratchGATConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = out_channels // heads
        self.lin = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        device = x.device
        loop_index = torch.arange(num_nodes, device=device).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        row, col = edge_index[0], edge_index[1]

        z = self.lin(x).view(num_nodes, self.heads, self.head_dim)
        alpha_src = (z * self.att_src).sum(dim=-1)
        alpha_dst = (z * self.att_dst).sum(dim=-1)
        edge_scores = F.leaky_relu(alpha_src[row] + alpha_dst[col], negative_slope=0.2)

        max_val = torch.full((num_nodes, self.heads), -float('inf'), device=device)
        max_val = max_val.scatter_reduce(0, col.unsqueeze(-1).expand_as(edge_scores), edge_scores, reduce='amax', include_self=False)
        edge_exp = torch.exp((edge_scores - max_val[col]).clamp(max=20.0))
        sum_exp = torch.zeros(num_nodes, self.heads, device=device).scatter_add_(0, col.unsqueeze(-1).expand_as(edge_exp), edge_exp).clamp(min=1e-12)
        alpha = self.drop(edge_exp / sum_exp[col])

        msg = z[row] * alpha.unsqueeze(-1)
        out = torch.zeros(num_nodes, self.heads, self.head_dim, device=device)
        out.scatter_add_(0, col.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)
        return out.view(num_nodes, self.heads * self.head_dim)

class RootMLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        root_x = extract_root_features(x, batch, ptr)
        h = self.drop(F.relu(self.fc1(root_x)))
        return self.fc2(h)

class GCNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = FromScratchGCNConv(in_channels, hidden_channels)
        self.conv2 = FromScratchGCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        h = self.drop(F.relu(self.conv2(h, edge_index)))
        pooled = global_mean_pool(h, batch)
        return self.fc(pooled)

class SAGEClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = FromScratchSAGEConv(in_channels, hidden_channels)
        self.conv2 = FromScratchSAGEConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        h = self.drop(F.relu(self.conv2(h, edge_index)))
        pooled = global_mean_pool(h, batch)
        return self.fc(pooled)

class GATClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = FromScratchGATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = FromScratchGATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.drop(F.elu(self.conv1(x, edge_index)))
        h = self.drop(F.elu(self.conv2(h, edge_index)))
        pooled = global_mean_pool(h, batch)
        return self.fc(pooled)

class BiGCNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5, drop_edge: float = 0.2):
        super().__init__()
        self.td_conv1 = FromScratchGCNConv(in_channels * 2, hidden_channels)
        self.td_conv2 = FromScratchGCNConv(hidden_channels + in_channels, hidden_channels)
        self.bu_conv1 = FromScratchGCNConv(in_channels * 2, hidden_channels)
        self.bu_conv2 = FromScratchGCNConv(hidden_channels + in_channels, hidden_channels)
        self.fc = nn.Linear(2 * hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)
        self.drop_edge = drop_edge

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        rev_edge_index = edge_index[[1, 0]]
        td_ei = edge_index
        bu_ei = rev_edge_index
        if self.training and self.drop_edge > 0 and edge_index.size(1) > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.drop_edge
            td_ei = edge_index[:, mask]
            bu_ei = rev_edge_index[:, mask]

        root_x = extract_root_features(x, batch, ptr)
        root_exp = root_x[batch]

        x_enh1 = torch.cat([x, root_exp], dim=-1)
        td_h1 = self.drop(F.relu(self.td_conv1(x_enh1, td_ei)))
        bu_h1 = self.drop(F.relu(self.bu_conv1(x_enh1, bu_ei)))

        td_enh2 = torch.cat([td_h1, root_exp], dim=-1)
        bu_enh2 = torch.cat([bu_h1, root_exp], dim=-1)
        td_h2 = self.drop(F.relu(self.td_conv2(td_enh2, td_ei)))
        bu_h2 = self.drop(F.relu(self.bu_conv2(bu_enh2, bu_ei)))

        td_pool = global_mean_pool(td_h2, batch)
        bu_pool = global_mean_pool(bu_h2, batch)
        return self.fc(torch.cat([td_pool, bu_pool], dim=-1))

def build_gnn_model(model_name: str, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5) -> nn.Module:
    m = model_name.lower()
    if m == "mlp":
        return RootMLP(in_channels, hidden_channels, out_channels, dropout)
    elif m == "gcn":
        return GCNClassifier(in_channels, hidden_channels, out_channels, dropout)
    elif m == "sage":
        return SAGEClassifier(in_channels, hidden_channels, out_channels, dropout)
    elif m == "gat":
        return GATClassifier(in_channels, hidden_channels, out_channels, dropout=dropout)
    elif m == "bigcn":
        return BiGCNClassifier(in_channels, hidden_channels, out_channels, dropout)
    raise ValueError(f"Unknown GNN model: {model_name}")
