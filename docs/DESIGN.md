# Architectural Design and Hypothesis: Mamba-DeltaNet Hybrid for Neural Decryption

## 1. Overview and Problem Context
Neural cipher decryption presents a twofold challenge:
1. **Dynamic Noise and Local Entropy:** Encrypted transmissions often contain non-semantic perturbations, keystream variations, and token-level noise requiring selective filtering and adaptive forgetting.
2. **Associative Memory and Long-Range State Tracking:** Decrypting continuous sequences requires recalling previously observed cipher-plaintext alignments, maintaining a global keystream state, and retrieving associative substitutions across extended sequence contexts.

Standard attention mechanisms exhibit $\mathcal{O}(L^2)$ computational complexity, while pure linear recurrent models often struggle with selective filtering or associative recall. To address both challenges efficiently in subquadratic time ($\mathcal{O}(L)$), we propose a hybrid architecture combining **Selective State Spaces (Mamba)** and **Delta-Rule Linear Attention (DeltaNet)**.

---

## 2. Proposed Architecture: Interleaved Mamba-DeltaNet Stack

### 2.1 Layer Ordering
The hybrid backbone is configured as an **interleaved sequential stack** of alternating Mamba and DeltaNet layers:

```
[Input Embedding]
       │
       ▼
┌─────────────────────────────────┐
│ Layer 1: Mamba Block            │  <-- Local Context & Selective Filtering
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ Layer 2: DeltaNet Block         │  <-- Associative Key-Value Memory Update
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ Layer 3: Mamba Block            │  <-- Dynamic State Routing & Gating
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ Layer 4: DeltaNet Block         │  <-- Multi-Step Associative Retrieval
└─────────────────────────────────┘
       │
      ... (Repeated for N/2 stages)
       │
       ▼
[Final Normalization (RMSNorm)]
       │
       ▼
[Linear Language Model Head]
```

Each block maintains pre-normalization (RMSNorm / LayerNorm) and residual connections ($x \leftarrow x + \text{SubLayer}(\text{Norm}(x))$).

---

## 3. Theoretical Rationale and Hypothesis

### 3.1 Complementary Memory Paradigms
* **Mamba (Selective Continuous State Space):**
  Mamba operates via input-dependent discretization:
  $$\Delta_t = \text{softplus}(\text{Linear}(x_t)), \quad \bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t = \Delta_t B_t$$
  $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t$$
  The dynamic step size $\Delta_t$ allows Mamba to modulate how much information enters or decays from its hidden state $h_t \in \mathbb{R}^{D \times N}$. When encountering irrelevant noise tokens or keystream artifacts, Mamba can scale down $\Delta_t \to 0$, effectively preserving prior state without corruption. Furthermore, its initial 1D causal convolution captures local $n$-gram context essential for detecting short cipher patterns.

* **DeltaNet (Associative Matrix Memory with Delta Rule):**
  DeltaNet maintains a 2D associative matrix memory $S_t \in \mathbb{R}^{d_k \times d_v}$:
  $$S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^T$$
  $$o_t = S_t q_t$$
  Unlike plain linear attention which additively accumulates keys and values leading to capacity saturation, DeltaNet reads the existing association ($S_{t-1} k_t$), computes the prediction error $(v_t - S_{t-1} k_t)$, and updates only the residual error scaled by learning rate $\beta_t$. This provides exact associative recall and in-context key-value substitution.

### 3.2 The Interleaving Synergy
1. **Noise Filtering Before Memory Storage (Mamba $\to$ DeltaNet):**
   If raw, noisy tokens are directly fed into an associative memory, erroneous key-value bindings can accumulate. By placing Mamba first, the 1D causal convolution and input-dependent selection mechanism filter out local perturbations, presenting DeltaNet with denoised, contextualized token representations.
2. **Associative Retrieval Followed by Gated Non-Linearity (DeltaNet $\to$ Mamba):**
   DeltaNet outputs linear associations ($o_t = S_t q_t$). The subsequent Mamba block applies non-linear expansion, channel-wise SwiGLU gating, and state-space filtering, allowing the network to synthesize and verify retrieved associations before proceeding to deeper layers.
3. **Decoupled State Representation:**
   * Mamba tracks *temporal continuous dynamics* ($h_t \in \mathbb{R}^{D \times N}$).
   * DeltaNet tracks *associative key-value bindings* ($S_t \in \mathbb{R}^{d_k \times d_v}$).
   Alternating them prevents state saturation and enables the hybrid network to achieve greater expressive capacity than a homogeneous stack of either model alone at equivalent parameter counts.

---

## 4. Expected Empirical Outcomes
* **Character-Level and Word-Level Accuracy:** Expected to outperform standalone DeltaNet on high-entropy segments due to Mamba's selective gating, and outperform standalone Mamba on repetitive associative patterns due to DeltaNet's delta-rule memory.
* **Levenshtein Distance:** Expected to be lower (indicating higher sequence fidelity and fewer substitution errors) compared to both individual baselines.
* **Computational Efficiency:** Maintains strict $\mathcal{O}(L)$ time and memory scaling with sequence length, remaining within the prescribed 10GB VRAM inference budget.

