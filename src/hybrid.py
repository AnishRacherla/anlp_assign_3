from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deltanet import DeltaNetBlock, RMSNorm
from .mamba import MambaLayer

class HybridMambaDeltaNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        n_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_layers % 2 == 0
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        layers = []
        for i in range(n_layers // 2):
            layers.append(MambaLayer(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout))
            layers.append(DeltaNetBlock(d_model=d_model, n_heads=n_heads, dropout=dropout))
        self.blocks = nn.ModuleList(layers)

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
