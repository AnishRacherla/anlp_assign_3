import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

from .vocab import CharTokenizer, BOS_ID, EOS_ID, SEP_ID, PAD_ID

class CipherDataset(Dataset):
    def __init__(
        self,
        cipher_path: Union[str, Path],
        cleaned_path: Optional[Union[str, Path]] = None,
        tokenizer: Optional[CharTokenizer] = None,
        max_len: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer or CharTokenizer()
        self.max_len = max_len
        self.samples = []

        cipher_path = Path(cipher_path)
        with open(cipher_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.samples.append(json.loads(line))
                if max_samples and len(self.samples) >= max_samples:
                    break

        self.has_targets = False
        if cleaned_path is not None and Path(cleaned_path).exists():
            self.has_targets = True
            targets = []
            with open(cleaned_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    targets.append(json.loads(line))
                    if max_samples and len(targets) >= max_samples:
                        break
            assert len(self.samples) == len(targets)
            for s, t in zip(self.samples, targets):
                s["text"] = t["text"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        cipher_ids = self.tokenizer.encode(item.get("ciphertext", ""))

        if not self.has_targets:
            prompt_ids = [BOS_ID] + cipher_ids + [SEP_ID]
            if len(prompt_ids) > self.max_len:
                prompt_ids = prompt_ids[:self.max_len]
            return {
                "id": item.get("id", str(idx)),
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            }

        target_ids = self.tokenizer.encode(item.get("text", ""))
        total_len = 1 + len(cipher_ids) + 1 + len(target_ids) + 1

        if total_len > self.max_len:
            budget = self.max_len - 3
            if len(cipher_ids) >= budget:
                cipher_ids = cipher_ids[:budget - 16]
                target_ids = target_ids[:16]
            else:
                target_budget = budget - len(cipher_ids)
                target_ids = target_ids[:target_budget]

        input_ids = [BOS_ID] + cipher_ids + [SEP_ID] + target_ids + [EOS_ID]
        labels = [-100] * (len(cipher_ids) + 2) + target_ids + [EOS_ID]

        return {
            "id": item.get("id", str(idx)),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def collate_train_batch(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    b_size = len(batch)

    inputs = torch.full((b_size, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((b_size, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        inp = item["input_ids"]
        lbl = item["labels"]
        inputs[i, :len(inp)] = inp
        labels[i, :len(lbl)] = lbl

    return inputs, labels

def get_dataloader(
    cipher_path: Union[str, Path],
    cleaned_path: Optional[Union[str, Path]] = None,
    tokenizer: Optional[CharTokenizer] = None,
    batch_size: int = 32,
    max_len: int = 512,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    ds = CipherDataset(
        cipher_path=cipher_path,
        cleaned_path=cleaned_path,
        tokenizer=tokenizer,
        max_len=max_len,
        max_samples=max_samples,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_train_batch if ds.has_targets else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
