import math
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

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

    def forward(self, u: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if prev_state is None:
            h = torch.zeros(b, d, self.d_state, device=u.device, dtype=torch.float32)
        else:
            h = prev_state.float()

        ys = []
        for t in range(l):
            h = delta_a[:, t] * h + delta_b_u[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, c_proj_f[:, t])
            ys.append(y_t)

        y = torch.stack(ys, dim=1).to(u.dtype)
        y = y + u * self.d
        return y, h

    def step(self, u_t: torch.Tensor, prev_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, d = u_t.shape
        x_proj_out = self.x_proj(u_t)
        delta_proj, b_proj, c_proj = torch.split(
            x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        delta = F.softplus(self.dt_proj(delta_proj)).float().unsqueeze(-1)
        a = -torch.exp(self.a_log.float()).view(1, d, self.d_state)
        b_bar = b_proj.float().unsqueeze(1)

        delta_a = torch.exp(delta * a)
        delta_b_u = (delta * b_bar) * u_t.float().unsqueeze(-1)

        h = delta_a * prev_h + delta_b_u
        y_t = torch.einsum("bdn,bn->bd", h, c_proj.float()).to(u_t.dtype)
        y_t = y_t + u_t * self.d
        return y_t, h

class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
        dropout: float = 0.1,
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

    def forward(self, x: torch.Tensor, prev_state: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, l, d = x.shape
        in_proj_out = self.in_proj(x)
        x_branch, z_branch = in_proj_out.chunk(2, dim=-1)

        if prev_state is not None and "conv" in prev_state:
            conv_input = torch.cat([prev_state["conv"], x_branch.transpose(1, 2)], dim=-1)
        else:
            conv_input = F.pad(x_branch.transpose(1, 2), (self.d_conv - 1, 0))

        conv_out = F.conv1d(
            conv_input,
            self.conv1d.weight,
            self.conv1d.bias,
            groups=self.d_inner,
        )
        new_conv_buf = conv_input[:, :, -(self.d_conv - 1):]

        x_conv = F.silu(conv_out.transpose(1, 2))
        h_prev = prev_state.get("ssm") if prev_state is not None else None
        ssm_out, new_h = self.ssm(x_conv, prev_state=h_prev)

        gated = ssm_out * F.silu(z_branch)
        out = self.drop(self.out_proj(gated))
        return out, {"conv": new_conv_buf, "ssm": new_h}

    def step(self, x_t: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        in_proj_out = self.in_proj(x_t)
        x_branch, z_branch = in_proj_out.chunk(2, dim=-1)

        conv_buf = torch.cat([state["conv"], x_branch.unsqueeze(-1)], dim=-1)
        x_conv = torch.sum(conv_buf * self.conv1d.weight.squeeze(1), dim=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = F.silu(x_conv)

        ssm_out, new_h = self.ssm.step(x_conv, state["ssm"])
        gated = ssm_out * F.silu(z_branch)
        out = self.drop(self.out_proj(gated))

        new_conv_buf = conv_buf[:, :, 1:]
        return out, {"conv": new_conv_buf, "ssm": new_h}

class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.block = MambaBlock(d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout)

    def forward(self, x: torch.Tensor, state: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out, new_state = self.block(self.norm(x), prev_state=state)
        return x + out, new_state

    def step(self, x_t: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out, new_state = self.block.step(self.norm(x_t), state)
        return x_t + out, new_state

class Mamba(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

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
            x, _ = layer(x)
        logits = self.lm_head(self.norm_f(x))
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> List[int]:
        self.eval()
        b, prompt_len = prompt_ids.shape
        x = self.token_emb(prompt_ids)
        
        layer_states = []
        for layer in self.layers:
            x, state = layer(x)
            layer_states.append(state)

        logits = self.lm_head(self.norm_f(x[:, -1]))
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = prompt_ids[0].tolist()

        for _ in range(max_new_tokens):
            if next_token.item() == eos_id:
                break
            generated.append(next_token.item())
            if len(generated) >= self.max_seq_len:
                break

            curr_x = self.token_emb(next_token)
            new_states = []
            for i, layer in enumerate(self.layers):
                curr_x, s = layer.step(curr_x.squeeze(1), layer_states[i])
                curr_x = curr_x.unsqueeze(1)
                new_states.append(s)
            layer_states = new_states

            next_logits = self.lm_head(self.norm_f(curr_x.squeeze(1)))
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        return generated
