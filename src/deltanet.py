from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_ops import triton_rmsnorm
from .vocab import decode_target_sequence, NOISE_TARGET_ID

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rmsnorm(x, self.weight, self.eps)

class GatedMLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim * 2, bias=False)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(x1) * x2)

class DeltaRuleLinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 6, d_head: Optional[int] = None, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head or (d_model // n_heads)
        self.d_conv = d_conv

        total_dim = n_heads * self.d_head
        self.q_proj = nn.Linear(d_model, total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_dim, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)

        self.conv_q = nn.Conv1d(total_dim, total_dim, kernel_size=d_conv, groups=total_dim, padding=d_conv - 1)
        self.conv_k = nn.Conv1d(total_dim, total_dim, kernel_size=d_conv, groups=total_dim, padding=d_conv - 1)
        self.conv_v = nn.Conv1d(total_dim, total_dim, kernel_size=d_conv, groups=total_dim, padding=d_conv - 1)

        self.out_proj = nn.Linear(total_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        h = self.n_heads
        d_h = self.d_head

        q_raw = self.q_proj(x).transpose(1, 2)
        k_raw = self.k_proj(x).transpose(1, 2)
        v_raw = self.v_proj(x).transpose(1, 2)

        q_in = F.pad(q_raw, (self.d_conv - 1, 0))
        k_in = F.pad(k_raw, (self.d_conv - 1, 0))
        v_in = F.pad(v_raw, (self.d_conv - 1, 0))

        q_conv = F.conv1d(q_in, self.conv_q.weight, self.conv_q.bias, groups=h * d_h)[:, :, :l]
        k_conv = F.conv1d(k_in, self.conv_k.weight, self.conv_k.bias, groups=h * d_h)[:, :, :l]
        v_conv = F.conv1d(v_in, self.conv_v.weight, self.conv_v.bias, groups=h * d_h)[:, :, :l]

        q = F.silu(q_conv.transpose(1, 2)).view(b, l, h, d_h).transpose(1, 2)
        k = F.silu(k_conv.transpose(1, 2)).view(b, l, h, d_h).transpose(1, 2)
        v = F.silu(v_conv.transpose(1, 2)).view(b, l, h, d_h).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).view(b, l, h, 1).transpose(1, 2)

        k = F.normalize(k.float(), p=2, dim=-1)
        q = F.normalize(q.float(), p=2, dim=-1)
        v = v.float()
        beta = beta.float()

        B = b * h
        q_f = q.reshape(B, l, d_h)
        k_f = k.reshape(B, l, d_h)
        v_f = v.reshape(B, l, d_h)
        beta_f = beta.reshape(B, l, 1, 1)

        s = torch.zeros(B, d_h, d_h, device=x.device, dtype=torch.float32)
        out = torch.empty(B, l, d_h, device=x.device, dtype=x.dtype)
        for t in range(l):
            kt = k_f[:, t].unsqueeze(-1)
            vt = v_f[:, t].unsqueeze(-1)
            qt = q_f[:, t].unsqueeze(-1)
            bt = beta_f[:, t]

            v_pred = torch.bmm(s, kt)
            err = vt - v_pred
            s = s + bt * torch.bmm(err, kt.transpose(1, 2))
            out[:, t] = torch.bmm(s, qt).squeeze(-1)

        out = out.reshape(b, h, l, d_h).transpose(1, 2).contiguous().view(b, l, h * d_h)
        return self.out_proj(out)

class DeltaNetLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 6, d_head: Optional[int] = None, d_conv: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.block = DeltaRuleLinearAttention(d_model, n_heads=n_heads, d_head=d_head, d_conv=d_conv)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, 4 * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DeltaNet(nn.Module):
    def __init__(
        self,
        vocab_size: int = 85,
        num_classes: int = 28,
        d_model: int = 192,
        n_layers: int = 8,
        n_heads: int = 6,
        d_head: Optional[int] = None,
        d_conv: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DeltaNetLayer(d_model, n_heads=n_heads, d_head=d_head, d_conv=d_conv, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_final = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.norm_final(x))
        return logits

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, lengths: Optional[List[int]] = None) -> List[str]:
        self.eval()
        logits = self.forward(input_ids)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        results = []
        for i, row in enumerate(pred_ids):
            limit = lengths[i] if lengths is not None else len(row)
            clean_row = [tid for tid in row[1:limit] if tid != NOISE_TARGET_ID]
            results.append(decode_target_sequence(clean_row))
        return results
