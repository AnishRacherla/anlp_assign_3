import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_ops import triton_rmsnorm, triton_selective_scan
from .vocab import decode_target_sequence, NOISE_TOKEN_IDS


class HybridRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rmsnorm(x, self.weight, self.eps)


class HybridGatedMLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim * 2, bias=False)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(x1) * x2)


class HybridSelectiveSSM(nn.Module):
    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.a_log = nn.Parameter(torch.log(a_init))
        self.d = nn.Parameter(torch.ones(d_inner))

        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, l, d = u.shape
        x_proj_out = self.x_proj(u)
        delta_proj, b_proj, c_proj = torch.split(
            x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_proj)).float()
        a_mat = -torch.exp(self.a_log.float()).view(1, d, self.d_state)
        y = triton_selective_scan(u, delta, a_mat, b_proj, c_proj)
        return y + u * self.d


class HybridMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.ssm = HybridSelectiveSSM(self.d_inner, d_state=d_state, dt_rank=self.dt_rank)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = x.shape[1]
        x_branch, z_branch = self.in_proj(x).chunk(2, dim=-1)
        conv_input = F.pad(x_branch.transpose(1, 2), (self.d_conv - 1, 0))
        conv_out = F.conv1d(conv_input, self.conv1d.weight, self.conv1d.bias, groups=self.d_inner)[:, :, :l]
        x_conv = F.silu(conv_out.transpose(1, 2))
        gated = self.ssm(x_conv) * F.silu(z_branch)
        return self.drop(self.out_proj(gated))


class HybridMambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int, d_conv: int, dropout: float, d_ffn: Optional[int]):
        super().__init__()
        self.norm1 = HybridRMSNorm(d_model)
        self.block = HybridMambaBlock(d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout)
        self.norm2 = HybridRMSNorm(d_model)
        self.mlp = HybridGatedMLP(d_model, d_ffn if d_ffn is not None else 4 * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class HybridDeltaRule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_conv: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
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

    def _short_conv(self, proj: torch.Tensor, conv: nn.Conv1d, length: int) -> torch.Tensor:
        padded = F.pad(proj.transpose(1, 2), (self.d_conv - 1, 0))
        out = F.conv1d(padded, conv.weight, conv.bias, groups=self.n_heads * self.d_head)[:, :, :length]
        return F.silu(out.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        h = self.n_heads
        d_h = self.d_head

        q = self._short_conv(self.q_proj(x), self.conv_q, l).view(b, l, h, d_h).transpose(1, 2)
        k = self._short_conv(self.k_proj(x), self.conv_k, l).view(b, l, h, d_h).transpose(1, 2)
        v = self._short_conv(self.v_proj(x), self.conv_v, l).view(b, l, h, d_h).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).view(b, l, h, 1).transpose(1, 2)

        k = F.normalize(k.float(), p=2, dim=-1)
        q = F.normalize(q.float(), p=2, dim=-1)
        v = v.float()
        beta = beta.float()

        n = b * h
        q_f = q.reshape(n, l, d_h)
        k_f = k.reshape(n, l, d_h)
        v_f = v.reshape(n, l, d_h)
        beta_f = beta.reshape(n, l, 1, 1)

        s = torch.zeros(n, d_h, d_h, device=x.device, dtype=torch.float32)
        out = torch.empty(n, l, d_h, device=x.device, dtype=x.dtype)
        for t in range(l):
            kt = k_f[:, t].unsqueeze(-1)
            vt = v_f[:, t].unsqueeze(-1)
            qt = q_f[:, t].unsqueeze(-1)
            err = vt - torch.bmm(s, kt)
            s = s + beta_f[:, t] * torch.bmm(err, kt.transpose(1, 2))
            out[:, t] = torch.bmm(s, qt).squeeze(-1)

        out = out.reshape(b, h, l, d_h).transpose(1, 2).contiguous().view(b, l, h * d_h)
        return self.out_proj(out)


class HybridDeltaNetLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_conv: int, d_ffn: Optional[int]):
        super().__init__()
        self.norm1 = HybridRMSNorm(d_model)
        self.block = HybridDeltaRule(d_model, n_heads=n_heads, d_conv=d_conv)
        self.norm2 = HybridRMSNorm(d_model)
        self.mlp = HybridGatedMLP(d_model, d_ffn if d_ffn is not None else 4 * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class HybridMambaDeltaNet(nn.Module):
    def __init__(
        self,
        vocab_size: int = 85,
        num_classes: int = 28,
        d_model: int = 192,
        n_layers: int = 8,
        d_state: int = 16,
        expand: int = 2,
        n_heads: int = 4,
        d_conv: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        d_ffn: Optional[int] = None,
        d_ffn_deltanet: Optional[int] = None,
    ):
        super().__init__()
        assert n_layers % 2 == 0
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        d_ffn_d = d_ffn_deltanet if d_ffn_deltanet is not None else d_ffn

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers // 2):
            self.layers.append(HybridMambaLayer(d_model, d_state, expand, d_conv, dropout, d_ffn))
            self.layers.append(HybridDeltaNetLayer(d_model, n_heads, d_conv, d_ffn_d))

        self.norm_final = HybridRMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "dt_proj" in name:
                    continue
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm_final(x))

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, lengths: Optional[List[int]] = None) -> List[str]:
        self.eval()
        logits = self.forward(input_ids)
        pred_ids = logits[..., :27].argmax(dim=-1).cpu().tolist()
        inp_ids = input_ids.cpu().tolist()
        results = []
        for i, row in enumerate(pred_ids):
            limit = lengths[i] if lengths is not None else len(row)
            clean_row = [row[j] for j in range(1, limit) if inp_ids[i][j] not in NOISE_TOKEN_IDS]
            results.append(decode_target_sequence(clean_row))
        return results
