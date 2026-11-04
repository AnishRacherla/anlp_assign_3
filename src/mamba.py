import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import decode_target_sequence, NOISE_TARGET_ID

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class GatedMLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim * 2, bias=False)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(x1) * x2)

class SelectiveSSM(nn.Module):
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

        dt_init_std = dt_rank**-0.5
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

        delta = F.softplus(self.dt_proj(delta_proj))
        a = -torch.exp(self.a_log.float())

        delta = delta.float().unsqueeze(-1)
        a = a.view(1, 1, d, self.d_state)
        b_bar = b_proj.float().unsqueeze(2)

        delta_a = torch.exp(delta * a)
        delta_b_u = (delta * b_bar) * u.float().unsqueeze(-1)
        c_proj_f = c_proj.float()

        h = torch.zeros(b, d, self.d_state, device=u.device, dtype=torch.float32)
        ys = []
        for t in range(l):
            h = delta_a[:, t] * h + delta_b_u[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, c_proj_f[:, t])
            ys.append(y_t)

        y = torch.stack(ys, dim=1).to(u.dtype)
        y = y + u * self.d
        return y

class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_conv = d_conv
        if dt_rank is None:
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state, dt_rank=self.dt_rank)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        in_proj_out = self.in_proj(x)
        x_branch, z_branch = in_proj_out.chunk(2, dim=-1)

        conv_input = F.pad(x_branch.transpose(1, 2), (self.d_conv - 1, 0))
        conv_out = F.conv1d(
            conv_input,
            self.conv1d.weight,
            self.conv1d.bias,
            groups=self.d_inner,
        )[:, :, :l]

        x_conv = F.silu(conv_out.transpose(1, 2))
        ssm_out = self.ssm(x_conv)

        gated = ssm_out * F.silu(z_branch)
        out = self.drop(self.out_proj(gated))
        return out

class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.block = MambaBlock(d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, 4 * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Mamba(nn.Module):
    def __init__(
        self,
        vocab_size: int = 85,
        num_classes: int = 28,
        d_model: int = 192,
        n_layers: int = 8,
        d_state: int = 16,
        expand: int = 2,
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
            MambaLayer(d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout)
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
    def predict(self, input_ids: torch.Tensor) -> List[str]:
        self.eval()
        logits = self.forward(input_ids)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        results = []
        for row in pred_ids:
            clean_row = [tid for tid in row[1:] if tid != NOISE_TARGET_ID]
            results.append(decode_target_sequence(clean_row))
        return results
