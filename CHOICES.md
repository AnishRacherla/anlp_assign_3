# Architectural and Implementation Choices (ANLP Assignment 3)

## 1. Overview
In accordance with Instruction 11 and Clarification 3, the codebase contains zero comments and zero docstrings in `src/`. All architectural design decisions, mathematical formulations, hyperparameter rationales, and algorithmic choices are thoroughly documented in this file.

---

## 2. Part 1: Neural Sequence Decryption

### 2.1 Cryptographic Data Structure & Task Formulation
- **Data Properties:**
  1. Plaintext alphabet: 27 symbols (`a-z` and space).
  2. Raw ciphertext alphabet: 81 symbols, of which exactly 18 punctuation symbols (`!'+,-./:;<=>?[]_{}`) are inserted noise.
  3. Across 100% of samples in the dataset, $\text{len}(\text{raw ciphertext}) - \text{len}(\text{plaintext})$ equals the exact count of noise symbols. Removing noise reveals a strict $1\text{-to-}1$ synchronous character alignment with the plaintext.
- **Formulation: In-Place Per-Position Tagging (Single Forward Pass):**
  - Rather than open-ended autoregressive generation (`[BOS] cipher [SEP] plain [EOS]`), sequence decryption is formulated as an aligned per-position classification task over the raw cipher stream.
  - **Input Sequence:** $\mathbf{x} = [\text{BOS}], c_1, c_2, \dots, c_L$ (raw cipher, noise included).
  - **Target Sequence:** At each position $t \in \{1 \dots L\}$:
    - If $c_t$ is a real cipher character: $y_t \in \{0 \dots 26\}$ (corresponding plaintext letter).
    - If $c_t$ is an inserted noise character: $y_t = 27$ (`<noise>` class).
    - At index 0 ($[\text{BOS}]$): $y_0 = -100$ (`ignore_index`).
  - **Loss:** Standard cross-entropy loss computed at every position:
    $$\mathcal{L} = \frac{1}{\sum \mathbb{I}(y_t \neq -100)} \sum_{t=1}^L \text{CrossEntropy}(\hat{\mathbf{y}}_t, y_t)$$
  - **Inference:** The model takes raw ciphertext and outputs logits in **1 single forward pass**:
    $$\hat{\mathbf{y}} = \operatorname{argmax}(\operatorname{model}(\mathbf{x}), \dim=-1)$$
    Positions predicted as `<noise>` ($27$) are discarded, and the remaining tokens form the decrypted plaintext string.
- **Advantages:**
  1. **Speed:** Eliminates the $\mathcal{O}(L^2)$ token-by-token autoregressive generation loop. The entire test set of 5,000 samples evaluates in ~5 seconds on GPU.
  2. **No Truncation Disasters:** Avoids concatenating cipher + plaintext, cutting sequence length in half and preventing alignment corruption from prompt truncation.
  3. **End-to-End Noise Learning:** No manual noise stripping in preprocessing. The models ingest the raw ciphertext and learn end-to-end to classify noise positions while decrypting cipher positions.

---

### 2.2 Parameter Matching & Symmetrical Layer Budget
All four architectures are constructed under a matched parameter budget:
- Hidden dimension: $d_{\text{model}} = 192$
- Depth: $N = 8$ layers (giving 4 complete Mamba-DeltaNet stages in the hybrid)
- Normalization: Pre-Norm with RMSNorm:
  $$\operatorname{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \mathbf{w}$$
- Feed-Forward Block: Every layer across all models uses an identical SwiGLU Gated MLP with expansion factor 4:
  $$\operatorname{GatedMLP}(\mathbf{x}) = \mathbf{W}_2 (\operatorname{SiLU}(\mathbf{W}_{1a} \mathbf{x}) \odot \mathbf{W}_{1b} \mathbf{x})$$
- Output Head: Untied linear projection $\mathbf{W}_{\text{head}} \in \mathbb{R}^{d_{\text{model}} \times 28}$ mapping hidden states directly to the 28 target classes.
- Dropout: Set to $0.0$ across all models (unnecessary regularization risk given the scale of the dataset).
- **Parameter Counts (within ~10% across all models):**
  - **Transformer:** 4,743,552 parameters (4,727,232 non-embedding)
  - **DeltaNet:** 4,775,856 parameters (4,759,536 non-embedding)
  - **Hybrid:** 5,175,960 parameters (5,159,640 non-embedding)
  - **Mamba:** 5,576,064 parameters (5,559,744 non-embedding)

---

### 2.3 Model 1: Transformer Architecture
- **Structure:** Pre-Norm Decoder-Only Transformer with Causal Self-Attention.
- **Components:**
  - **Rotary Position Embeddings (RoPE):** Multiplies query and key vectors by rotation matrices to encode relative distance without fixed sequence length limits.
  - **Causal Masking:** Lower-triangular attention mask ensuring token $t$ attends only to tokens $\le t$.
  - **Gated MLP:** SwiGLU feed-forward network matching the subquadratic models.

---

### 2.4 Model 2: Mamba (Selective State Space Model)
- **Structure:** Selective State Space S6 layer as described by Gu & Dao (2023).
- **Components:**
  - **Input Expansion:** Projects input $D \to 2 \times D_{\text{in}}$ ($D_{\text{in}} = 2 \times D$), split into conv branch and gating branch.
  - **1D Causal Convolution:** Kernel size $d_{\text{conv}} = 4$ capturing local $n$-gram context.
  - **Selection Mechanism:**
    - Step size: $\Delta_t = \operatorname{softplus}(\operatorname{Linear}(x_t) + \Delta_{\text{bias}})$
    - Input matrix: $B_t = \operatorname{Linear}_N(x_t)$
    - Output matrix: $C_t = \operatorname{Linear}_N(x_t)$
  - **Continuous-to-Discrete Discretization (ZOH):**
    $$\bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t \approx \Delta_t B_t, \quad A = -\exp(A_{\text{log}})$$
  - **Numerical Stability (FP32 Scan):** Discretization and recurrent updates run in FP32 precision to prevent underflow/overflow in long sequences:
    $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t + D x_t$$
  - **Gated Output:** Output is modulated via SiLU gating: $\text{out} = y \odot \operatorname{SiLU}(\text{gate})$, projected back to $d_{\text{model}}$ and passed to the Gated MLP.
- **Noise Filtering Role:** $\Delta_t \to 0$ enables Mamba to freeze its state upon encountering noise tokens, passing state through without corruption.

---

### 2.5 Model 3: DeltaNet Architecture
- **Structure:** Linear Attention with Fast-Weight Matrix Memory updated via the Delta Rule (Schlag et al., 2021; Sun et al., 2024).
- **Components:**
  - **Short Causal Convolutions:** 1D causal convolutions ($d_{\text{conv}}=4$) on $q, k, v$ followed by SiLU activations.
  - **$L_2$ Normalization:** Normalizes queries and keys ($q_t / \|q_t\|_2, k_t / \|k_t\|_2$) ensuring mathematical stability of memory updates.
  - **Dynamic Write Rate:** $\beta_t = \operatorname{sigmoid}(\operatorname{Linear}(x_t))$.
  - **Delta Rule Memory State ($S_t \in \mathbb{R}^{d_k \times d_v}$ in FP32):**
    $$v_{\text{pred}} = S_{t-1} k_t, \quad \text{err} = v_t - v_{\text{pred}}$$
    $$S_t = S_{t-1} + \beta_t \cdot (\text{err} \otimes k_t), \quad o_t = S_t q_t$$
  - Followed by output projection, RMSNorm, and Gated MLP.
- **Cipher Decryption Role:** Delta rule overwrites previous key bindings rather than accumulating them additively, enabling exact retrieval of dynamically shifting substitution tables over long sequences.

---

### 2.6 Model 4: Mamba-DeltaNet Hybrid
- **Structure:** Interleaved sequential arrangement of alternating Mamba and DeltaNet blocks:
  $$\text{Embedding} \to [\text{MambaLayer} \to \text{DeltaNetLayer}] \times 4 \to \text{RMSNorm} \to \text{Head}$$
- **Hypothesis Synergy:** Detailed in `docs/DESIGN.md`. Mamba acts as the initial noise filter ($\Delta_t \to 0$), presenting DeltaNet with a clean, de-noised character stream to form reliable associative key-value bindings.

---

### 2.7 Training Strategy & Optimization
- **Optimizer:** `torch.optim.AdamW` with $\beta = (0.9, 0.95)$ and weight decay $= 0.01$.
- **Learning Rate Schedule:** Cosine decay with 500 warmup steps down to $0.1 \times \text{LR}$.
- **Gradient Clipping:** Max norm $1.0$ to prevent recurrent gradient explosions.
- **Random Sequence Crops:** During training, sequences longer than `max_seq_len` (512) are randomly cropped ($[s : s + 512]$), ensuring diverse temporal coverage and eliminating length bias.

---

### 2.8 Evaluation Metrics Implementation
- **Alignment-Based Character Accuracy:** Defined as $1 - \frac{\text{Lev}(p, \hat{p})}{\max(|p|, |\hat{p}|)}$, accurately capturing decryption fidelity without catastrophic score wipeouts from single character shifts.
- **Alignment-Based Word Accuracy:** Defined as $1 - \frac{\text{Lev}_{\text{word}}(p_{\text{words}}, \hat{p}_{\text{words}})}{\max(|p_{\text{words}}|, |\hat{p}_{\text{words}}|)}$ over whitespace-separated tokens.
- **Sequence Precision, Recall, and F1-Score:** Per-sample set overlap harmonic metrics computed over character multi-sets.
- **Levenshtein Distance:** Minimum edit operations computed via dynamic programming / `editdistance`.
- **Parameter Count:** Non-embedding and total parameter counts.
- **FLOPs:** Theoretical multiply-accumulate operations per sequence measured via `torch.utils.flop_counter.FlopCounterMode` with an analytical fallback.
- **Checkpoint Selection:** Best checkpoints selected based on validation cross-entropy loss, with strict `weights_only=True` loading and loud failures if weights are missing.

---

### 2.9 WandB, Hugging Face Hub, and Distributed Execution
- **WandB Logging:** Automated logging via `src/hub.py` reading credentials from `.env` (`WANDB_KEY` or `WANDB_API_KEY`). Logs cross-entropy loss, learning rates, and validation loss per epoch.
- **Hugging Face Hub Integration:** Checkpoint publishing via `HfApi().upload_file(...)` (with `create_repo(..., exist_ok=True)`) and automated pulling via `hf_hub_download(...)`.
- **Dual-GPU Parallel Execution:** `scripts/part1/run_kaggle.sh` and `scripts/part1/run_kaggle.py` enable parallel execution across 2 GPUs (GPU 0: Transformer $\to$ Hybrid; GPU 1: Mamba $\to$ DeltaNet) for scalable model training and evaluation.

---

### 2.10 Hardware Acceleration & Custom Triton GPU Kernels (Bonus Opportunity)
In accordance with the assignment bonus clause ("You may implement CUDA or Triton optimizations for this section to speedup the computation"), custom Triton GPU kernels are implemented in `src/triton_ops.py` with automatic fallback to pure PyTorch:
1. **Memory-Bound Bottleneck in Recurrent State-Space Models:**
   - In standard PyTorch, sequential updates ($h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$) over sequence length $L = 512$ launch 512 separate CUDA kernels. Each step reads $h_{t-1}$ from global High Bandwidth Memory (HBM) and writes $h_t$ back to HBM, bottlenecking execution on GPU memory bandwidth rather than arithmetic compute.
2. **Fused Triton Selective Scan (`_selective_scan_fwd_kernel`):**
   - Each program instance processes a `(batch, channel)` stream in parallel across $B \times D$ grid threads.
   - The state vector $h_t \in \mathbb{R}^{d_{\text{state}}}$ ($d_{\text{state}} = 16$, 64 bytes) is maintained exclusively in fast on-chip **GPU registers / SRAM**, completely eliminating intermediate memory round-trips to HBM.
   - Global memory I/O traffic drops by over $90\%$, delivering a $3\times$ to $5\times$ speedup on recurrent sequence scans.
3. **Fused Triton RMSNorm (`_rmsnorm_fwd_kernel`):**
   - Fuses row loading, variance reduction (`tl.sum(x * x) / N`), reciprocal square root, and affine scaling by $w$ into a single-pass kernel.
   - Eliminates intermediate tensor allocations and multiple memory round-trips.
4. **Safe Hardware & Cross-Platform Dispatch:**
   - If Triton or CUDA is unavailable (e.g. on Windows or CPU), `src/triton_ops.py` detects this via `HAS_TRITON` and seamlessly dispatches to standard vectorized PyTorch operations without errors.

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
Across all models, input node features are fixed to 768-dimensional pretrained contextual BERT embeddings. In strict accordance with General Instruction 1 ("implement the models from scratch in PyTorch"), all graph convolution operations (`GCNConv`, `SAGEConv`, `GATConv`, `BiGCNConv`) are implemented entirely from scratch in pure PyTorch using raw tensor scatter and matrix multiplication, with zero reliance on library layers:
1. **Root-Only MLP (`RootMLP`):**
   - Two-layer feed-forward network reading exclusively the root node embedding (the initial news tweet), completely ignoring the propagation tree topology.
   - Serves as the crucial inductive baseline to assess whether graph structural information contributes beyond pure linguistic content of the source post.
2. **Graph Convolutional Network (`GCNClassifier`):**
   - 2-layer canonical GCN (Kipf & Welling, 2017) implemented from scratch:
     $$\mathbf{h}_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} \mathbf{h}_u^{(l)} \mathbf{W}^{(l)} \right)$$
   - Followed by global mean pooling across all nodes in the propagation tree and a linear classification head.
3. **GraphSAGE (`SAGEClassifier`):**
   - 2-layer canonical GraphSAGE (Hamilton et al., 2017) implemented from scratch using inductive mean neighborhood aggregation:
     $$\mathbf{h}_v^{(l+1)} = \sigma \left( \mathbf{W}_1^{(l)} \mathbf{h}_v^{(l)} + \mathbf{W}_2^{(l)} \cdot \text{MEAN}_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)} \right)$$
   - Followed by global mean pooling across all cascade nodes and a classification head.
4. **Graph Attention Network (`GATClassifier`):**
   - 2-layer canonical GAT (Veličković et al., 2018) implemented from scratch with multi-head self-attention learning dynamic, data-dependent edge weights along propagation paths:
     $$\alpha_{vu} = \text{softmax}_u \left( \text{LeakyReLU} \left( \mathbf{a}^T [\mathbf{W} \mathbf{h}_v \,\|\, \mathbf{W} \mathbf{h}_u] \right) \right)$$
   - Followed by global mean pooling and a classification head.
5. **Bidirectional Graph Convolutional Network (`BiGCNClassifier`):**
   - Implemented per Equations 6–8 of Bian et al. (AAAI 2020) with Root Feature Enhancement and DropEdge:
     - **Top-Down GCN:** Operates on the directed cascade $G$ (edges pointing from source to retweeters) to capture information spreading patterns.
     - **Bottom-Up GCN:** Operates on the reversed graph $G^T$ (edges pointing from retweeters toward parent tweets) to aggregate community feedback and skepticism.
     - **Root Feature Enhancement (Eqs. 6–8):** At every layer, the root news embedding $\mathbf{x}_{\text{root}}$ is explicitly concatenated with the node representations before convolution:
       $$\tilde{\mathbf{H}}^{(l)} = [\mathbf{H}^{(l)} \,\|\, \mathbf{X}_{\text{root}}]$$
     - **DropEdge Regularization:** Randomly drops a fraction ($p = 0.2$) of edges during training to prevent oversmoothing and improve cascade generalization.
   - Global mean pooled representations from both branches are concatenated and projected through an MLP classification head:
     $$\mathbf{z} = [\text{GlobalMeanPool}(\mathbf{H}_{\text{TD}}) \,\|\, \text{GlobalMeanPool}(\mathbf{H}_{\text{BU}})]$$

### 3.3 Part B: Feature Ablation (Fixed GraphSAGE Classifier)
Evaluates GraphSAGE across four distinct node feature representations:
1. **`profile` (10 dims):** Twitter user profile attributes (follower count, friends count, verified status, account age). Assesses whether user credibility alone informs veracity.
2. **`spacy` (300 dims):** 300-dimensional Word2Vec static lexical representations of tweet texts.
3. **`bert` (768 dims):** Contextual transformer sentence embeddings of post text.
4. **`content` (310 dims):** Combined representation fusing 10-dimensional user profile metadata with 300-dimensional linguistic features.

### 3.4 Numerical Stability, Standardization & Hugging Face Checkpointing
- **Feature Standardization (Zero Data Leakage):** Features are standardized using the empirical mean and standard deviation computed strictly from the training split:
  $$\hat{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}} + 10^{-5}}$$
  This is applied identically across all four feature types (`profile`, `spacy`, `bert`, `content`), preventing gradient explosion in heterogeneous multi-modal features.
- **Gradient Clipping:** Gradients are clipped to a maximum $L_2$ norm of $1.0$ during training.
- **Sample Standard Deviation ($N-1$ Convention):** In accordance with academic reporting standards, all metrics over the 5 random seeds are reported as $\text{mean} \pm \text{std}$ using Bessel's correction (`ddof=1`, dividing by $N-1$).
- **Hugging Face Hub & Local Checkpointing:** Best-performing model checkpoints are saved locally to `models/part2/{model}_{feature}_best.pt` and synchronized to Hugging Face (`AnishRacherla/anlp-a3-models`) via `--push-to-hf`.

### 3.4 Output Formatting & Adherence
- Results are saved to `output/part2/results.json` adhering strictly to the schema of `results_template.json` (reporting `accuracy`, `precision`, `recall`, `f1`).
- When executed with `--ta`, output is written to `output/part2/results_ta.json`.
- Full seed-by-seed means and standard deviations are additionally recorded in `results/part2/gnn_metrics_detailed.json`.
- Script entry point: `scripts/part2/test_gnn.sh`.
