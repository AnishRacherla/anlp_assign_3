import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=-2)
            v = torch.cat([kv_cache[1], v], dim=-2)
        new_kv = (k, v)

        seq_k = k.size(-2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if t == seq_k:
            attn = attn.masked_fill(~self.causal_mask[:, :, :t, :t], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.resid_drop(self.out_proj(out)), new_kv

class MLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, 4 * d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.ln1(x), cos, sin, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv

class Transformer(nn.Module):
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
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        d_head = d_model // n_heads
        cos, sin = precompute_rope_freqs(d_head, max_seq_len)
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
        x = self.emb_drop(self.token_emb(input_ids))
        cos = self.rope_cos[:t]
        sin = self.rope_sin[:t]
        for block in self.blocks:
            x, _ = block(x, cos, sin)
        logits = self.lm_head(self.ln_final(x))
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> List[int]:
        self.eval()
        b, prompt_len = prompt_ids.shape
        x = self.emb_drop(self.token_emb(prompt_ids))

        kv_caches = []
        cos = self.rope_cos[:prompt_len]
        sin = self.rope_sin[:prompt_len]
        for block in self.blocks:
            x, kv = block(x, cos, sin)
            kv_caches.append(kv)

        logits = self.lm_head(self.ln_final(x[:, -1]))
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = prompt_ids[0].tolist()

        curr_pos = prompt_len
        for _ in range(max_new_tokens):
            if next_token.item() == eos_id:
                break
            generated.append(next_token.item())
            if len(generated) >= self.max_seq_len:
                break

            curr_x = self.emb_drop(self.token_emb(next_token))
            cos_t = self.rope_cos[curr_pos : curr_pos + 1]
            sin_t = self.rope_sin[curr_pos : curr_pos + 1]

            new_caches = []
            for i, block in enumerate(self.blocks):
                curr_x, kv = block(curr_x, cos_t, sin_t, kv_cache=kv_caches[i])
                new_caches.append(kv)
            kv_caches = new_caches
            curr_pos += 1

            next_logits = self.lm_head(self.ln_final(curr_x.squeeze(1)))
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        return generated
