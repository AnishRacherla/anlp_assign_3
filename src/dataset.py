import json
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

from .vocab import (
    CharTokenizer,
    BOS_ID,
    PAD_ID,
    NOISE_CHARS,
    NOISE_TARGET_ID,
    PAD_TARGET_ID,
    char_to_target_id,
)

class CipherDataset(Dataset):
    def __init__(
        self,
        cipher_path: Union[str, Path],
        cleaned_path: Optional[Union[str, Path]] = None,
        tokenizer: Optional[CharTokenizer] = None,
        max_len: int = 512,
        is_train: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer or CharTokenizer()
        self.max_len = max_len
        self.is_train = is_train
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
                c_str = s.get("ciphertext", "")
                p_str = t.get("text", "")
                c_clean_len = sum(1 for ch in c_str if ch not in NOISE_CHARS)
                assert c_clean_len == len(p_str), f"Length mismatch: {c_clean_len} != {len(p_str)}"
                s["text"] = p_str

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        raw_cipher = item.get("ciphertext", "")
        cipher_ids = self.tokenizer.encode(raw_cipher)

        if not self.has_targets:
            inp = [BOS_ID] + cipher_ids
            return {
                "id": item.get("id", str(idx)),
                "input_ids": torch.tensor(inp, dtype=torch.long),
                "raw_cipher": raw_cipher,
            }

        raw_plain = item.get("text", "")
        target_ids = []
        p_idx = 0
        plain_len = len(raw_plain)
        for ch in raw_cipher:
            if ch in NOISE_CHARS:
                target_ids.append(NOISE_TARGET_ID)
            else:
                if p_idx < plain_len:
                    target_ids.append(char_to_target_id(raw_plain[p_idx]))
                    p_idx += 1
                else:
                    target_ids.append(NOISE_TARGET_ID)

        input_ids = [BOS_ID] + cipher_ids
        labels = [PAD_TARGET_ID] + target_ids

        if self.is_train and self.max_len and len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]

        return {
            "id": item.get("id", str(idx)),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "raw_cipher": raw_cipher,
            "raw_plain": raw_plain,
        }

def collate_train_batch(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    b_size = len(batch)

    inputs = torch.full((b_size, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((b_size, max_len), PAD_TARGET_ID, dtype=torch.long)

    for i, item in enumerate(batch):
        inp = item["input_ids"]
        lbl = item["labels"]
        inputs[i, : len(inp)] = inp
        labels[i, : len(lbl)] = lbl

    return inputs, labels

def collate_eval_batch(batch: List[dict]):
    max_len = max(len(x["input_ids"]) for x in batch)
    b_size = len(batch)

    inputs = torch.full((b_size, max_len), PAD_ID, dtype=torch.long)
    for i, item in enumerate(batch):
        inp = item["input_ids"]
        inputs[i, : len(inp)] = inp

    ids = [x["id"] for x in batch]
    raw_ciphers = [x["raw_cipher"] for x in batch]
    raw_plains = [x.get("raw_plain", "") for x in batch]
    lengths = [len(x["input_ids"]) for x in batch]

    labels = None
    if "labels" in batch[0]:
        labels = torch.full((b_size, max_len), PAD_TARGET_ID, dtype=torch.long)
        for i, item in enumerate(batch):
            lbl = item["labels"]
            labels[i, : len(lbl)] = lbl

    return inputs, labels, ids, raw_ciphers, raw_plains, lengths

def get_dataloader(
    cipher_path: Union[str, Path],
    cleaned_path: Optional[Union[str, Path]] = None,
    tokenizer: Optional[CharTokenizer] = None,
    batch_size: int = 32,
    max_len: int = 512,
    shuffle: bool = True,
    is_train: bool = True,
    max_samples: Optional[int] = None,
) -> DataLoader:
    ds = CipherDataset(
        cipher_path=cipher_path,
        cleaned_path=cleaned_path,
        tokenizer=tokenizer,
        max_len=max_len,
        is_train=is_train,
        max_samples=max_samples,
    )
    collate_fn = collate_train_batch if is_train else collate_eval_batch
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
