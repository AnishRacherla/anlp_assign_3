import math
from typing import List, Optional, Tuple, Dict

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

class DeltaRuleLinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, d_head: Optional[int] = None, d_conv: int = 4):
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

    def forward(self, x: torch.Tensor, prev_state: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, l, d = x.shape
        h = self.n_heads
        d_h = self.d_head

        q_raw = self.q_proj(x).transpose(1, 2)
        k_raw = self.k_proj(x).transpose(1, 2)
        v_raw = self.v_proj(x).transpose(1, 2)

        if prev_state is not None and "conv_q" in prev_state:
            q_in = torch.cat([prev_state["conv_q"], q_raw], dim=-1)
            k_in = torch.cat([prev_state["conv_k"], k_raw], dim=-1)
            v_in = torch.cat([prev_state["conv_v"], v_raw], dim=-1)
        else:
            q_in = F.pad(q_raw, (self.d_conv - 1, 0))
            k_in = F.pad(k_raw, (self.d_conv - 1, 0))
            v_in = F.pad(v_raw, (self.d_conv - 1, 0))

        q_conv = F.conv1d(q_in, self.conv_q.weight, self.conv_q.bias, groups=h * d_h)
        k_conv = F.conv1d(k_in, self.conv_k.weight, self.conv_k.bias, groups=h * d_h)
        v_conv = F.conv1d(v_in, self.conv_v.weight, self.conv_v.bias, groups=h * d_h)

        new_conv_q = q_in[:, :, -(self.d_conv - 1):]
        new_conv_k = k_in[:, :, -(self.d_conv - 1):]
        new_conv_v = v_in[:, :, -(self.d_conv - 1):]

        q = F.silu(q_conv.transpose(1, 2)).view(b, l, h, d_h).transpose(1, 2)
        k = F.silu(k_conv.transpose(1, 2)).view(b, l, h, d_h).transpose(1, 2)
        v = F.silu(v_conv.transpose(1, 2)).view(b, l, h, d_h).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).view(b, l, h, 1).transpose(1, 2)

        k = F.normalize(k.float(), p=2, dim=-1)
        q = F.normalize(q.float(), p=2, dim=-1)
        v = v.float()
        beta = beta.float()

        if prev_state is not None and "mem" in prev_state:
            s = prev_state["mem"].float()
        else:
            s = torch.zeros(b, h, d_h, d_h, device=x.device, dtype=torch.float32)

        outputs = []
        for t in range(l):
            q_t = q[:, :, t]
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            beta_t = beta[:, :, t]

            v_pred = torch.einsum("bhij,bhj->bhi", s, k_t)
            err = v_t - v_pred
            update = torch.einsum("bhi,bhj->bhij", err, k_t)
            s = s + beta_t.unsqueeze(-1) * update

            o_t = torch.einsum("bhij,bhj->bhi", s, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2).to(x.dtype)
        out = out.transpose(1, 2).contiguous().view(b, l, h * d_h)
        out = self.out_proj(out)

        next_state = {
            "conv_q": new_conv_q,
            "conv_k": new_conv_k,
            "conv_v": new_conv_v,
            "mem": s,
        }
        return out, next_state

    def step(self, x_t: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, d = x_t.shape
        h = self.n_heads
        d_h = self.d_head

        q_raw = self.q_proj(x_t).unsqueeze(-1)
        k_raw = self.k_proj(x_t).unsqueeze(-1)
        v_raw = self.v_proj(x_t).unsqueeze(-1)

        q_in = torch.cat([state["conv_q"], q_raw], dim=-1)
        k_in = torch.cat([state["conv_k"], k_raw], dim=-1)
        v_in = torch.cat([state["conv_v"], v_raw], dim=-1)

        q_conv = torch.sum(q_in * self.conv_q.weight.squeeze(1), dim=-1)
        k_conv = torch.sum(k_in * self.conv_k.weight.squeeze(1), dim=-1)
        v_conv = torch.sum(v_in * self.conv_v.weight.squeeze(1), dim=-1)

        if self.conv_q.bias is not None:
            q_conv = q_conv + self.conv_q.bias
            k_conv = k_conv + self.conv_k.bias
            v_conv = v_conv + self.conv_v.bias

        q_t = F.normalize(F.silu(q_conv).view(b, h, d_h).float(), p=2, dim=-1)
        k_t = F.normalize(F.silu(k_conv).view(b, h, d_h).float(), p=2, dim=-1)
        v_t = F.silu(v_conv).view(b, h, d_h).float()
        beta_t = torch.sigmoid(self.beta_proj(x_t)).view(b, h, 1).float()

        s = state["mem"].float()
        v_pred = torch.einsum("bhij,bhj->bhi", s, k_t)
        err = v_t - v_pred
        update = torch.einsum("bhi,bhj->bhij", err, k_t)
        s = s + beta_t.unsqueeze(-1) * update

        o_t = torch.einsum("bhij,bhj->bhi", s, q_t).to(x_t.dtype)
        out = self.out_proj(o_t.reshape(b, h * d_h))

        next_state = {
            "conv_q": q_in[:, :, 1:],
            "conv_k": k_in[:, :, 1:],
            "conv_v": v_in[:, :, 1:],
            "mem": s,
        }
        return out, next_state

class DeltaNetBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = DeltaRuleLinearAttention(d_model, n_heads=n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, state: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attn_out, next_state = self.attn(self.norm1(x), prev_state=state)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, next_state

    def step(self, x_t: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attn_out, next_state = self.attn.step(self.norm1(x_t), state)
        x_t = x_t + attn_out
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, next_state

class DeltaNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DeltaNetBlock(d_model, n_heads=n_heads, dropout=dropout)
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
        for block in self.blocks:
            x, _ = block(x)
        logits = self.lm_head(self.norm_f(x))
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> List[int]:
        self.eval()
        b, prompt_len = prompt_ids.shape
        x = self.token_emb(prompt_ids)

        layer_states = []
        for block in self.blocks:
            x, state = block(x)
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
            for i, block in enumerate(self.blocks):
                curr_x, s = block.step(curr_x.squeeze(1), layer_states[i])
                curr_x = curr_x.unsqueeze(1)
                new_states.append(s)
            layer_states = new_states

            next_logits = self.lm_head(self.norm_f(curr_x.squeeze(1)))
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        return generated
