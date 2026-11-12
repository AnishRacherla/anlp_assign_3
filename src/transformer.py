import math
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

def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    dims = torch.arange(0, d_head, 2, dtype=torch.float32)
    freqs = 1.0 / (theta ** (dims / d_head))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_rot = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
    return x_rot.flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        causal_mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        attn = attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len=max_seq_len, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, 4 * d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 85,
        num_classes: int = 28,
        d_model: int = 192,
        n_layers: int = 8,
        n_heads: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len=max_seq_len, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_final = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)

        d_head = d_model // n_heads
        cos, sin = precompute_rope_freqs(d_head, max(max_seq_len, 4096))
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

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
        b, t = input_ids.shape
        x = self.token_emb(input_ids)
        if t > self.rope_cos.size(0):
            d_head = self.d_model // self.n_heads
            cos, sin = precompute_rope_freqs(d_head, t)
            cos = cos.to(x.device)
            sin = sin.to(x.device)
        else:
            cos = self.rope_cos[:t].to(x.device)
            sin = self.rope_sin[:t].to(x.device)

        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.head(self.norm_final(x))
        return logits

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, lengths: Optional[List[int]] = None) -> List[str]:
        self.eval()
        b, t = input_ids.shape
        if t <= self.max_seq_len:
            logits = self.forward(input_ids)
        else:
            stride = max(1, self.max_seq_len - 128)
            logits_accum = torch.zeros(b, t, self.num_classes, device=input_ids.device)
            counts = torch.zeros(b, t, 1, device=input_ids.device)
            for start in range(0, t, stride):
                end = min(start + self.max_seq_len, t)
                sub_logits = self.forward(input_ids[:, start:end])
                logits_accum[:, start:end] += sub_logits
                counts[:, start:end] += 1.0
                if end == t:
                    break
            logits = logits_accum / counts.clamp(min=1.0)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        results = []
        for i, row in enumerate(pred_ids):
            limit = lengths[i] if lengths is not None else len(row)
            clean_row = [tid for tid in row[1:limit] if tid != NOISE_TARGET_ID]
            results.append(decode_target_sequence(clean_row))
        return results
