from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool

def extract_root_features(x: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
    if ptr is not None:
        root_idx = ptr[:-1]
    else:
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root_idx = torch.cat([root.new_zeros(1), root + 1], dim=0)
    return x[root_idx]

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
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin0 = nn.Linear(in_channels, hidden_channels)
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        h = self.drop(F.relu(self.conv2(h, edge_index)))
        pooled = global_mean_pool(h, batch)
        root_x = extract_root_features(x, batch, ptr)
        news = F.relu(self.lin0(root_x))
        out = self.drop(F.relu(self.lin1(torch.cat([news, pooled], dim=-1))))
        return self.lin2(out)

class SAGEClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin0 = nn.Linear(in_channels, hidden_channels)
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        h = self.drop(F.relu(self.conv2(h, edge_index)))
        pooled = global_mean_pool(h, batch)
        root_x = extract_root_features(x, batch, ptr)
        news = F.relu(self.lin0(root_x))
        out = self.drop(F.relu(self.lin1(torch.cat([news, pooled], dim=-1))))
        return self.lin2(out)

class GATClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)
        self.lin0 = nn.Linear(in_channels, hidden_channels)
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.drop(F.elu(self.conv1(x, edge_index)))
        h = self.drop(F.elu(self.conv2(h, edge_index)))
        pooled = global_mean_pool(h, batch)
        root_x = extract_root_features(x, batch, ptr)
        news = F.relu(self.lin0(root_x))
        out = self.drop(F.relu(self.lin1(torch.cat([news, pooled], dim=-1))))
        return self.lin2(out)

class BiGCNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.td_conv1 = GCNConv(in_channels, hidden_channels)
        self.td_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bu_conv1 = GCNConv(in_channels, hidden_channels)
        self.bu_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin0 = nn.Linear(in_channels, hidden_channels)
        self.lin1 = nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        rev_edge_index = edge_index[[1, 0]]
        td_h = self.drop(F.relu(self.td_conv1(x, edge_index)))
        td_h = self.drop(F.relu(self.td_conv2(td_h, edge_index)))
        td_pooled = global_mean_pool(td_h, batch)

        bu_h = self.drop(F.relu(self.bu_conv1(x, rev_edge_index)))
        bu_h = self.drop(F.relu(self.bu_conv2(bu_h, rev_edge_index)))
        bu_pooled = global_mean_pool(bu_h, batch)

        root_x = extract_root_features(x, batch, ptr)
        news = F.relu(self.lin0(root_x))
        out = self.drop(F.relu(self.lin1(torch.cat([news, td_pooled, bu_pooled], dim=-1))))
        return self.lin2(out)

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
