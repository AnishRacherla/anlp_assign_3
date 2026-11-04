import math
from collections import Counter
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

try:
    import editdistance
    _HAS_EDITDISTANCE = True
except ImportError:
    _HAS_EDITDISTANCE = False

def compute_levenshtein(s1: str, s2: str) -> int:
    if _HAS_EDITDISTANCE:
        return editdistance.eval(s1, s2)
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def word_levenshtein(w1: List[str], w2: List[str]) -> int:
    m, n = len(w1), len(w2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if w1[i - 1] == w2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def char_accuracy_aligned(ref: str, hyp: str) -> float:
    max_len = max(len(ref), len(hyp))
    if max_len == 0:
        return 1.0
    lev = compute_levenshtein(ref, hyp)
    return max(0.0, 1.0 - (lev / max_len))

def word_accuracy_aligned(ref: str, hyp: str) -> float:
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    max_len = max(len(ref_words), len(hyp_words))
    if max_len == 0:
        return 1.0
    w_lev = word_levenshtein(ref_words, hyp_words)
    return max(0.0, 1.0 - (w_lev / max_len))

def sequence_precision_recall_f1(ref: str, hyp: str) -> Tuple[float, float, float]:
    if not ref and not hyp:
        return 1.0, 1.0, 1.0
    if not ref or not hyp:
        return 0.0, 0.0, 0.0
    ref_chars = Counter(ref)
    hyp_chars = Counter(hyp)
    tp = sum(min(count, hyp_chars[ch]) for ch, count in ref_chars.items())
    precision = tp / len(hyp) if len(hyp) > 0 else 0.0
    recall = tp / len(ref) if len(ref) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    emb_params = sum(
        m.weight.numel() for m in model.modules() if isinstance(m, nn.Embedding)
    )
    non_emb_params = total_params - emb_params
    return total_params, non_emb_params

def measure_flops_forward(model: nn.Module, sample_input: torch.Tensor) -> int:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as mode:
            _ = model(sample_input)
            total_flops = mode.get_total_flops()
            if total_flops > 0:
                return int(total_flops)
    except Exception:
        pass
    b, l = sample_input.shape
    d = getattr(model, "d_model", 256)
    n = getattr(model, "n_layers", 6)
    v = getattr(model, "vocab_size", 85)
    return int(b * (2 * n * (12 * d * d) * l + 2 * l * d * v))

def evaluate_predictions(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    assert len(references) == len(hypotheses)
    n = len(references)
    if n == 0:
        return {}

    tot_char_acc = 0.0
    tot_word_acc = 0.0
    tot_prec = 0.0
    tot_rec = 0.0
    tot_f1 = 0.0
    tot_lev = 0.0

    for ref, hyp in zip(references, hypotheses):
        tot_char_acc += char_accuracy_aligned(ref, hyp)
        tot_word_acc += word_accuracy_aligned(ref, hyp)
        p, r, f = sequence_precision_recall_f1(ref, hyp)
        tot_prec += p
        tot_rec += r
        tot_f1 += f
        tot_lev += compute_levenshtein(ref, hyp)

    return {
        "character_accuracy": tot_char_acc / n,
        "word_accuracy": tot_word_acc / n,
        "precision": tot_prec / n,
        "recall": tot_rec / n,
        "f1": tot_f1 / n,
        "levenshtein_distance": tot_lev / n,
    }
