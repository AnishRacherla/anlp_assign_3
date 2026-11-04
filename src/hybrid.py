from typing import List, Optional

import torch
import torch.nn as nn

from .deltanet import DeltaNetLayer
from .mamba import MambaLayer, RMSNorm
from .vocab import decode_target_sequence, NOISE_TARGET_ID

class HybridMambaDeltaNet(nn.Module):
    def __init__(
        self,
        vocab_size: int = 85,
        num_classes: int = 28,
        d_model: int = 192,
        n_layers: int = 8,
        d_state: int = 16,
        expand: int = 2,
        n_heads: int = 6,
        d_conv: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_layers % 2 == 0
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers // 2):
            self.layers.append(
                MambaLayer(d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout)
            )
            self.layers.append(
                DeltaNetLayer(d_model, n_heads=n_heads, d_conv=d_conv, dropout=dropout)
            )

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
