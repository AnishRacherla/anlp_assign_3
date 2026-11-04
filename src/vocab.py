import json
from pathlib import Path
from typing import Dict, List, Union

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN = "<sep>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3

ALL_CHARS = " !'+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz{}"

class CharTokenizer:
    def __init__(self):
        self.pad_token = PAD_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.sep_token = SEP_TOKEN
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.sep_id = SEP_ID
        self.id_to_token: Dict[int, str] = {
            self.pad_id: self.pad_token,
            self.bos_id: self.bos_token,
            self.eos_id: self.eos_token,
            self.sep_id: self.sep_token,
        }
        for ch in ALL_CHARS:
            idx = len(self.id_to_token)
            self.id_to_token[idx] = ch
        self.token_to_id: Dict[str, int] = {v: k for k, v in self.id_to_token.items()}
        self.vocab_size = len(self.id_to_token)

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id[ch] for ch in text if ch in self.token_to_id]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        res = []
        special = {self.pad_id, self.bos_id, self.eos_id, self.sep_id}
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            if i in self.id_to_token:
                res.append(self.id_to_token[i])
        return "".join(res)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharTokenizer":
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tok.token_to_id = {v: k for k, v in tok.id_to_token.items()}
        tok.vocab_size = len(tok.id_to_token)
        return tok

