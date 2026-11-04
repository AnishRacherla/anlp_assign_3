from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool

class RootMLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ptr is not None:
            root_idx = ptr[:-1]
        else:
            _, root_idx = torch.unique(batch, return_inverse=False, return_counts=False)
            batch_size = int(batch.max().item()) + 1
            root_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            for i in range(batch_size):
                root_idx[i] = (batch == i).nonzero(as_tuple=True)[0][0]
        root_x = x[root_idx]
        h = self.drop(F.relu(self.fc1(root_x)))
        return self.fc2(h)

class GCNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.drop(F.relu(self.conv1(x, edge_index)))
        x = self.drop(F.relu(self.conv2(x, edge_index)))
        pooled = global_mean_pool(x, batch)
        return self.fc(pooled)

class SAGEClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.drop(F.relu(self.conv1(x, edge_index)))
        x = self.drop(F.relu(self.conv2(x, edge_index)))
        pooled = global_mean_pool(x, batch)
        return self.fc(pooled)

class GATClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.drop(F.elu(self.conv1(x, edge_index)))
        x = self.drop(F.elu(self.conv2(x, edge_index)))
        pooled = global_mean_pool(x, batch)
        return self.fc(pooled)

class BiGCNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.td_conv1 = GCNConv(in_channels, hidden_channels)
        self.td_conv2 = GCNConv(hidden_channels, hidden_channels)

        self.bu_conv1 = GCNConv(in_channels, hidden_channels)
        self.bu_conv2 = GCNConv(hidden_channels, hidden_channels)

        self.fc = nn.Linear(2 * hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        rev_edge_index = edge_index[[1, 0]]

        td_x = self.drop(F.relu(self.td_conv1(x, edge_index)))
        td_x = self.drop(F.relu(self.td_conv2(td_x, edge_index)))
        td_pooled = global_mean_pool(td_x, batch)

        bu_x = self.drop(F.relu(self.bu_conv1(x, rev_edge_index)))
        bu_x = self.drop(F.relu(self.bu_conv2(bu_x, rev_edge_index)))
        bu_pooled = global_mean_pool(bu_x, batch)

        cat_pooled = torch.cat([td_pooled, bu_pooled], dim=-1)
        return self.fc(cat_pooled)

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

