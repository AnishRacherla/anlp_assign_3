# Architectural and Implementation Choices (ANLP Assignment 3)

## 1. Overview
In accordance with Instruction 11 and Clarification 3, the codebase contains zero comments and zero docstrings in `src/`. All architectural design decisions, mathematical formulations, hyperparameter rationales, and algorithmic choices are thoroughly documented in this file.

---

## 2. Part 1: Neural Sequence Decryption

### 2.1 Vocabulary and Tokenization
- **Choice:** Unified character-level vocabulary (`CharTokenizer`) covering all distinct characters in the raw ciphertext (81 characters), all characters in the plaintext (27 characters: lowercase `a-z` and space), and 4 special tokens (`<pad>=0`, `<bos>=1`, `<eos>=2`, `<sep>=3`), totaling 85 vocabulary symbols.
- **Rationale:** 
  1. The character alphabet is completely closed across train, validation, and test splits. Character-level tokenization entirely eliminates out-of-vocabulary (`<unk>`) errors.
  2. Operating at character granularity preserves the exact sequential dynamics of the cipher stream and avoids unnatural token boundaries that BPE/subword tokenizers create on encrypted strings.
  3. Vocabulary size remains compact (85 tokens), which minimizes embedding parameters and keeps the computational footprint low.
  4. Raw data processing is strictly end-to-end: models ingest the raw ciphertext without artificial preprocessing or noise stripping, matching evaluation test set conditions.

### 2.2 Sequence-to-Sequence Modeling Formulation
- **Sequence Format:**
  $$\mathbf{x} = [\text{BOS}], c_1, c_2, \dots, c_L, [\text{SEP}], p_1, p_2, \dots, p_M, [\text{EOS}]$$
- **Target Formulation:**
  $$\mathbf{y} = \text{ignore\_index}, \dots, \text{ignore\_index}, p_1, p_2, \dots, p_M, [\text{EOS}]$$
- **Rationale:**
  1. Causal sequence-to-sequence formatting allows all three subquadratic architectures (Mamba, DeltaNet, Transformer) to use identical training and inference interfaces.
  2. Cross-entropy loss is masked (`ignore_index = -100`) across all prefix tokens ($[\text{BOS}], c_1 \dots c_L, [\text{SEP}]$), ensuring gradients are computed strictly for predicting the plaintext sequence conditioned on the cipher sequence.
  3. Autoregressive greedy decoding prompts the model with $[\text{BOS}] + \text{raw\_cipher} + [\text{SEP}]$ and samples sequentially until $[\text{EOS}]$ or the maximum sequence length is reached.

---

### 2.3 Model 1: Transformer Architecture
- **Structure:** Pre-Norm Decoder-Only Transformer.
- **Components:**
  - **Rotary Position Embeddings (RoPE):** Applied directly to query and key states to capture relative positional distances without fixed context length cutoffs.
  - **Causal Scaled Dot-Product Multi-Head Attention:** Multi-head attention masked to prevent future token leakage.
  - **Feed-Forward Network (FFN):** Standard two-layer MLP with expansion factor 4 and GELU non-linear activation.
  - **Normalization:** Pre-norm LayerNorm applied before attention and FFN blocks with residual addition.
  - **Inference KV-Cache:** During autoregressive generation, keys and values of previous tokens are cached, allowing $\mathcal{O}(1)$ step attention computation per newly generated token ($\mathcal{O}(L)$ total decoding time) instead of $\mathcal{O}(L^2)$ sequence re-evaluations.

---

### 2.4 Model 2: Mamba (Selective State Space Model)
- **Structure:** Selective State Space S6 layer as described by Gu & Dao (2023).
- **Components:**
  - **Input Expansion:** Projects input $D \to 2 \times D_{\text{in}}$ where $D_{\text{in}} = E \cdot D$ ($E=2$). Split into a convolution branch and a gating branch.
  - **1D Causal Convolution:** Kernel size $d_{\text{conv}} = 4$ capturing local $n$-gram context before entering the state space.
  - **Selection Mechanism:**
    - Step size: $\Delta_t = \text{softplus}(\text{Linear}(x_t) + \Delta_{\text{bias}})$
    - Input matrix: $B_t = \text{Linear}_N(x_t)$
    - Output matrix: $C_t = \text{Linear}_N(x_t)$
  - **Continuous-to-Discrete Discretization (ZOH):**
    $$\bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t \approx \Delta_t B_t$$
    $A$ is initialized as a structured negative diagonal matrix: $A = -\exp(A_{\text{log}})$.
  - **Recurrent State Update:**
    $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t + D x_t$$
  - **Gated Output:**
    $$\text{Output} = \text{Linear}_{\text{out}}(y \odot \text{SiLU}(\text{Gate}))$$
  - **Numerical Stability (FP32 Scan):** Discretization $\exp(\Delta_t A)$ and state recurrence updates are computed in `float32` precision even during mixed-precision training to eliminate underflow/overflow artifacts in long scan sequences.
  - **Recurrent Step State Cache:** Autoregressive generation maintains rolling state buffers $(h_t \in \mathbb{R}^{B \times D_{\text{in}} \times N})$ and 1D conv cache $(x_{\text{conv}} \in \mathbb{R}^{B \times D_{\text{in}} \times d_{\text{conv}}})$, achieving strict $\mathcal{O}(1)$ per-token generation step time.
- **Rationale for Cipher Decryption:**
  The input-dependent $\Delta_t$ gate enables the model to selectively discard noise characters ($\Delta_t \to 0$) and update its hidden state only upon encountering relevant cipher characters.

---

### 2.5 Model 3: DeltaNet Architecture
- **Structure:** Linear Attention with Fast-Weight Matrix Memory updated via the Delta Rule (Schlag et al., 2021; Sun et al., 2024).
- **Components:**
  - **Projections:** Linear projections for Query $q_t$, Key $k_t$, Value $v_t$, and dynamic learning rate $\beta_t = \text{sigmoid}(\text{Linear}(x_t))$.
  - **Normalizations:** Queries and keys are normalized ($L_2$-normalized keys $k_t / \|k_t\|_2$) ensuring mathematical contractivity and numerical stability.
  - **1D Causal Convolution:** Kernel size $d_{\text{conv}} = 4$ providing local positional awareness.
  - **Delta Rule Memory State ($S_t \in \mathbb{R}^{d_k \times d_v}$):**
    $$S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^T$$
    $$o_t = S_t q_t$$
  - **Recurrent Step State Cache:** Step-by-step decoding caches matrix memory state $S_t$ and conv buffer, evaluating each generation token in $\mathcal{O}(1)$ time with no full-sequence re-computation.
  - **FP32 Recurrence:** Delta-rule matrix state updates run in FP32 precision to prevent gradient cancellation in long recurrent chains.
- **Rationale for Cipher Decryption:**
  Unlike standard linear attention that additively saturates memory, the delta rule reads the existing memory prediction ($S_{t-1} k_t$), calculates the prediction error, and stores only the residual error. This allows exact associative recall of character-level cipher substitutions.

---

### 2.6 Model 4: Mamba-DeltaNet Hybrid
- **Structure:** Interleaved sequential arrangement of alternating Mamba and DeltaNet blocks:
  $$\text{Embedding} \to [\text{Mamba} \to \text{DeltaNet}] \times \frac{N}{2} \to \text{RMSNorm} \to \text{Head}$$
- **Full Hypothesis and Analysis:** Detailed in `docs/DESIGN.md`.
- **Unified Step Caching:** Combines Mamba S6 state cache with DeltaNet matrix memory cache to guarantee end-to-end $\mathcal{O}(1)$ step decoding.

---

### 2.7 Evaluation Metrics Implementation
- **Alignment-Based Character Accuracy:** Defined as $1 - \frac{\text{Lev}(p, \hat{p})}{\max(|p|, |\hat{p}|)}$. Unlike fragile position-by-position matching (which catastrophically scores near 0 from a single shifted character), alignment-based accuracy accurately reflects true sequence decryption fidelity.
- **Alignment-Based Word Accuracy:** Defined as $1 - \frac{\text{Lev}_{\text{word}}(p_{\text{words}}, \hat{p}_{\text{words}})}{\max(|p_{\text{words}}|, |\hat{p}_{\text{words}}|)}$, evaluating token-level edit similarity over whitespace-split word sequences.
- **Sequence Precision, Recall, and F1-Score:** Per-sample set overlap harmonic metrics computed between multi-set character counts of ground truth and decrypted hypotheses.
- **Levenshtein Distance:** Minimum edit operations (insertions, deletions, substitutions) computed via dynamic programming / `editdistance`.
- **Parameter Count:** Non-embedding and total parameter counts.
- **FLOPs:** Theoretical multiply-accumulate operations per sequence measured via `torch.utils.flop_counter.FlopCounterMode` when available, falling back to analytical per-architecture formulas.
- **Checkpoint Selection:** Best checkpoints are selected based on validation cross-entropy loss, with strict `weights_only=True` loading and loud failures if weights are missing.

---

### 2.8 WandB, Hugging Face Hub, and Distributed Execution
- **WandB Logging:** Automated logging via `src/hub.py` reading credentials from `.env` (`WANDB_KEY` or `WANDB_API_KEY`). Logs cross-entropy loss, learning rates, and validation loss per epoch.
- **Hugging Face Hub Integration:** Checkpoint publishing via `HfApi().upload_file(...)` and automated pulling via `hf_hub_download(...)`. Meets the strict requirement that evaluation scripts can fetch trained checkpoints directly from Hugging Face by default during held-out testing without requiring local weights in Git.
- **Dual-GPU Parallel Execution:** `scripts/part1/run_kaggle.sh` and `scripts/part1/run_kaggle.py` enable parallel execution across 2 GPUs (GPU 0 executes Transformer $\to$ Hybrid; GPU 1 executes Mamba $\to$ DeltaNet) for scalable model training and evaluation.

---

## 3. Part 2: Graph Neural Networks for News Propagation Analysis

### 3.1 Problem Formulation and UPFD Benchmark
- **Dataset:** GossipCop dataset from the User Preference-aware Fake News Detection (UPFD) benchmark via PyG (`torch_geometric.datasets.UPFD`).
- **Graph Structure:** Directed propagation tree trees representing the cascade of tweets and retweets following a news article. The root node represents the source news tweet, and child nodes represent user retweets.
- **Task:** Graph-level binary classification (fake news vs. real news).
- **Split Protocol:** Fixed 20% train, 10% validation, and 70% test split as defined in the benchmark.
- **Evaluation Seeds:** 5 fixed random seeds (`[42, 1337, 2024, 7, 99]`) are evaluated across every model and feature configuration.
- **Checkpoint Selection:** The model checkpoint achieving the highest validation macro $F_1$ score during training is selected for final test set evaluation.

### 3.2 Part A: Architecture Comparison (Fixed BERT-768 Node Features)
Across all models, input node features are fixed to 768-dimensional pretrained contextual BERT embeddings:
1. **Root-Only MLP (`RootMLP`):**
   - Two-layer feed-forward network reading exclusively the root node embedding (the initial news tweet), completely ignoring the propagation tree topology.
   - Serves as the crucial inductive baseline to assess whether graph structural information contributes beyond pure linguistic content of the source post.
2. **Graph Convolutional Network (`GCNClassifier`):**
   - 2-layer GCN (Kipf & Welling, 2017) using symmetric normalized neighborhood aggregation:
     $$\mathbf{h}_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} \mathbf{h}_u^{(l)} \mathbf{W}^{(l)} \right)$$
   - Followed by global mean pooling across all nodes in the propagation tree and a linear classification head.
3. **GraphSAGE (`SAGEClassifier`):**
   - 2-layer GraphSAGE (Hamilton et al., 2017) using inductive mean neighborhood aggregation:
     $$\mathbf{h}_v^{(l+1)} = \sigma \left( \mathbf{W}_1^{(l)} \mathbf{h}_v^{(l)} + \mathbf{W}_2^{(l)} \cdot \text{MEAN}_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)} \right)$$
   - Followed by global mean pooling across all cascade nodes and a classification head.
4. **Graph Attention Network (`GATClassifier`):**
   - 2-layer GAT (Veličković et al., 2018) with multi-head self-attention ($H=4$ heads) learning dynamic, data-dependent edge weights along propagation paths:
     $$\alpha_{vu} = \text{softmax}_u \left( \text{LeakyReLU} \left( \mathbf{a}^T [\mathbf{W} \mathbf{h}_v \,\|\, \mathbf{W} \mathbf{h}_u] \right) \right)$$
   - Followed by global mean pooling and a classification head.
5. **Bidirectional Graph Convolutional Network (`BiGCNClassifier`):**
   - Implements bidirectional propagation modeling (Bian et al., 2020) tailored specifically for rumor/news tree cascades:
     - **Top-Down GCN:** Operates on the original directed cascade $G$ (edges pointing from source to retweeters) to capture information spreading patterns.
     - **Bottom-Up GCN:** Operates on the reversed graph $G^T$ (edges pointing from retweeters toward parent tweets) to aggregate community feedback and skepticism.
   - **Representation Fusion:** Concatenates global mean pooled representations of the top-down tree, bottom-up tree, and the root tweet embedding:
     $$\mathbf{z} = [\mathbf{h}_{\text{root}} \,\|\, \text{GlobalMeanPool}(\mathbf{H}_{\text{TD}}) \,\|\, \text{GlobalMeanPool}(\mathbf{H}_{\text{BU}})]$$
   - Projected through an MLP classification head.

### 3.3 Part B: Feature Ablation (Fixed GraphSAGE Classifier)
Evaluates GraphSAGE across four distinct node feature representations:
1. **`profile` (10 dims):** Twitter user profile attributes (follower count, friends count, verified status, account age). Assesses whether user credibility alone informs veracity.
2. **`spacy` (300 dims):** 300-dimensional Word2Vec static lexical representations of tweet texts.
3. **`bert` (768 dims):** Contextual transformer sentence embeddings of post text.
4. **`content` (310 dims):** Combined representation fusing 10-dimensional user profile metadata with 300-dimensional linguistic features.

### 3.4 Output Formatting & Adherence
- Results are saved to `output/part2/results.json` adhering strictly to the schema of `results_template.json` (reporting `accuracy`, `precision`, `recall`, `f1`).
- When executed with `--ta`, output is written to `output/part2/results_ta.json`.
- Full seed-by-seed means and standard deviations are additionally recorded in `results/part2/gnn_metrics_detailed.json`.
- Script entry point: `scripts/part2/test_gnn.sh`.
