# Design and Hypothesis: Interleaved Mamba-DeltaNet Hybrid for Cipher Decryption

## 1. Task Structure (from data analysis, before any model training)

- Plaintext alphabet: 27 symbols (`a-z`, space). Ciphertext alphabet: 81 symbols, of which 18 punctuation symbols (`!'+,-./:;<=>?[]_{}`) are noise.
- In train and validation, `len(cipher) - len(plain)` equals the number of noise symbols in the cipher for every sample. Removing them gives a 1-to-1 character alignment with the plaintext.
- Consequence: decryption is a per-position tagging problem over the raw cipher stream, not free-form generation. Each real cipher symbol maps to one plaintext symbol, and noise symbols map to a `<noise>` class that is dropped at output.
- Noise symbols are inserted at random intervals, so the absolute index of a symbol in the raw stream is not its index in the plaintext. A model must discount noise to know where it is in the real stream.

## 2. Formulation Shared by All Four Models

- Input: `[BOS] c_1 ... c_L` (raw cipher including noise). Target at each position: the plaintext character, or `<noise>`. Loss is cross-entropy over all positions; padding is ignored.
- Output: predictions at non-noise positions in order, which are already aligned with the plaintext. No autoregressive decoding, so inference is a single forward pass.
- All models are causal, so recurrent models keep their intended form and the Transformer uses the same mask.
- Matched budget: the same embedding, depth, hidden width target and MLP size, with parameter counts reported and kept within about 10% of each other. Dropout is 0 for all.
- Training uses random crops of the sequences. Inference runs on full sequences for the recurrent models and with overlapping windows for the Transformer.

## 3. Proposed Architecture

```
[Embedding]
   -> (Mamba layer -> DeltaNet layer) x (N/2)
   -> RMSNorm -> Linear head (28-way: 27 plaintext + noise)
```

- Every layer is pre-norm with a residual connection: `x <- x + Layer(RMSNorm(x))`.
- Mamba layer: input projection, depthwise causal conv (k=4), selective SSM with input-dependent `Delta`, `B`, `C`, SiLU gate, output projection, followed by a gated MLP so the two layer types have the same feed-forward capacity.
- DeltaNet layer: short causal conv with SiLU on q, k, v, L2-normalised q and k, `beta_t = sigmoid(.)`, delta-rule state `S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T`, `o_t = S_t q_t`, then a gated MLP.
- Mamba comes first in each pair, and the first layer sees the raw noisy embeddings.

## 4. Hypotheses

**H1 (Mamba first filters noise).** The selection mechanism can set `Delta_t` near 0 at noise symbols so that the state is not advanced by them. This should let the early Mamba layer maintain a noise-free running representation of the real stream. DeltaNet, which has no per-channel decay but does have a write strength `beta_t`, can also suppress writes, but it has no continuous-time decay to carry state through unchanged.

**H2 (DeltaNet stores the cipher-to-plaintext bindings).** The delta rule overwrites the old value for a key rather than accumulating, so it suits repeated lookups of a fixed or slowly changing substitution table, and keeps the memory from saturating over long sequences.

**H3 (the order Mamba, then DeltaNet, is better than the reverse or either alone).** DeltaNet's keys and values are better formed if the noise-driven position shift has already been absorbed by the preceding Mamba layer. We expect the hybrid to be at least as accurate as the better of pure Mamba and pure DeltaNet, and to degrade more slowly as the noise rate grows.

**H4 (Transformer).** With relative positions (RoPE), noise insertion perturbs distances between real symbols. If the cipher's key depends on the real-stream position, the Transformer should be the most sensitive to the noise rate, since it has no mechanism that skips noise.

If the cipher turns out to be a position-independent substitution, H1 and H4 predict little difference in accuracy and the comparison shifts to efficiency (Section 5).

## 5. Planned Evidence

- Metrics: character accuracy, word accuracy, precision, recall, F1, Levenshtein distance, parameter count, FLOPs, all measured on the aligned output.
- Noise-rate sweep: evaluate on inputs with extra noise inserted at rates above and below the training rate.
- Noise-stripped input as an ablation: if H1 holds, the hybrid's gap to the others should shrink on stripped input.
- Length extrapolation: train on short crops and evaluate on sequences up to the maximum length.
- Layer-order ablation: Mamba then DeltaNet versus DeltaNet then Mamba, at the same parameter count.
- Efficiency: inference throughput and peak memory against sequence length, all within the 10 GB VRAM limit.

The final report will compare results against H1-H4 and state which were not supported.
