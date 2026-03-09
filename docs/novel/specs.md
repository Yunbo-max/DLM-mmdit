# LSME — Latent-Steered Masked Editing: Implementation Specification

## Overview

LSME (Latent-Steered Masked Editing) enables controllable text editing in discrete masked diffusion language models by leveraging a continuous latent channel through MMDiT joint attention. Given input text and a target latent vector encoding a desired attribute (sentiment, topic, style), LSME partially masks text tokens and joint-denoises them conditioned on the target latent — the first SDEdit-style editing method for discrete DLMs.

This project addresses Gap 4 (no DLM supports latent-steered editing), Gap 6 (latent space geometry unexplored), and Gap 7 (no comprehensive DLM evaluation suite).

## Novel Contributions

1. **LSME Algorithm** — SDEdit-style text editing via partial masking + latent replacement + joint denoising. Zero architecture changes; ~100 lines of new sampling code.
2. **Latent Geometry Analysis** — First formal metrics (Semantic Smoothness Score, Monotonic Transition Score) for evaluating DLM latent space structure.
3. **DLM-Eval Suite** — 6-pillar evaluation framework covering fluency, controllability, editing quality, latent geometry, diversity, and efficiency.

---

## FROZEN SCOPE

```
Selected candidates: LSME (M1), Latent Geometry (E1), DLM-Eval Suite (E2)
Mode: combined — one paper
Method name: LSME (Latent-Steered Masked Editing)
Gaps addressed:
  - Gap 4 (C4, CC2): No DLM supports SDEdit-style editing with continuous latent steering
  - Gap 6 (C3, TD5): Latent space geometry unexplored in DLMs
  - Gap 7 (E1-E5, CC4-CC5): No comprehensive DLM evaluation suite
Out of scope: UGLC guidance framework, Seq2Seq, DualBridge, FAMO E2E, Latent reasoning
```

---

## Project Structure

```
latentDLM_mmdit/
├── models/
│   └── multimodal_mmdit.py            # Existing model (NO changes needed)
├── sample_l2t_fixed.py                # Existing L2T sampling
├── sample_lsme.py                     # NEW: LSME editing sampler
├── evaluation/
│   ├── __init__.py
│   ├── eval_suite.py                  # NEW: DLM-Eval 6-pillar runner
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── fluency.py                 # PPL, grammar
│   │   ├── controllability.py         # Classifier accuracy, attribute match
│   │   ├── edit_quality.py            # BLEU, ROUGE, BERTScore, edit distance
│   │   ├── latent_geometry.py         # NEW: SSS, MTS metrics
│   │   ├── diversity.py               # Self-BLEU, distinct-n
│   │   └── efficiency.py              # Wall-clock, NFE, tokens/sec
│   └── benchmarks/
│       ├── yelp_sentiment.py          # Yelp binary sentiment editing
│       ├── amazon_topic.py            # Amazon multi-topic editing
│       └── formality.py               # GYAFC formality transfer
├── latent_utils/
│   ├── __init__.py
│   ├── attribute_encoder.py           # NEW: Encode attribute → latent z
│   └── interpolation.py              # NEW: Latent interpolation utilities
├── scripts/
│   ├── run_lsme.py                    # Entry point: LSME editing
│   ├── run_eval.py                    # Entry point: full evaluation
│   ├── run_geometry.py                # Entry point: latent geometry analysis
│   └── run_baselines.py              # Entry point: run baseline methods
└── configs/
    ├── lsme_yelp.yaml
    ├── lsme_amazon.yaml
    └── eval.yaml
```

---

## Module Specifications

### Module 1: LSME Sampler — `sample_lsme.py`

**File:** `latentDLM_mmdit/sample_lsme.py`

**Purpose:** SDEdit-style text editing for discrete masked DLMs. Fills Gap 4 — no existing DLM supports latent-steered text editing.

**Inspired by:**
- SDEdit (image editing via noise-then-denoise)
- ReMDM ([kuleshov-group/remdm](https://github.com/kuleshov-group/remdm)) — remasking for diversity
- DiffusER ([machelreid/diffuser](https://github.com/machelreid/diffuser)) — edit-based diffusion for text
- LatentOps ([guangyliu/LatentOps](https://github.com/guangyliu/LatentOps)) — latent manipulation for style

**Novel beyond them:** ReMDM remasks but has no attribute control. DiffusER edits in continuous space. LatentOps manipulates latent but uses AR decoder. LSME combines discrete masked denoising with continuous latent steering through MMDiT joint attention.

**Algorithm:**

```
LSME-Edit(model, text_tokens, z_target, mask_ratio, steps):
  Input:
    model         — trained MultimodalMMDiT
    text_tokens   — (B, L) original text token ids
    z_target      — (B, latent_dim) target latent encoding desired attribute
    mask_ratio    — float in [0, 1], controls edit strength
    steps         — int, reverse diffusion steps (from mask_ratio to 0)

  1. PARTIAL MASKING:
     mask = Bernoulli(mask_ratio) for each position  # (B, L) boolean
     z_t = text_tokens.clone()
     z_t[mask] = MASK_TOKEN_ID

  2. COMPUTE ENTRY TIMESTEP:
     # Map mask_ratio to the MDLM noise schedule sigma
     sigma_entry = -log(1 - (1-eps) * mask_ratio)
     t_entry = mask_ratio  # linear schedule

  3. REVERSE DIFFUSION from t_entry → 0:
     ts = linspace(eps, t_entry, steps)  # partial schedule
     for i = steps-1 ... 0:
       t, tm1 = ts[i], ts[max(0, i-1)]

       # Forward pass: text tokens + TARGET latent (clean, t=0)
       text_logits, _ = model(
         text_tokens=z_t,
         latents=z_target.unsqueeze(1),     # ← TARGET latent
         text_timesteps=t.expand(B),
         latent_timesteps=zeros(B),          # latent is clean
       )

       # Standard MDLM posterior sampling (same as sample_l2t)
       text_logits[..., MASK_TOKEN_ID] = -inf
       dsigma, sigma_t = get_sigmas(t)
       _, sigma_tm1 = get_sigmas(tm1)
       move_t = 1 - exp(-sigma_t)
       move_tm1 = 1 - exp(-sigma_tm1)
       probs = softmax(text_logits) * (move_t - move_tm1)
       probs[..., MASK_TOKEN_ID] = move_tm1
       probs = probs / move_t
       z_tm1 = sample_categorical(probs)

       # Copy flag: preserve unmasked positions
       copy = (z_t != MASK_TOKEN_ID)
       z_t = copy * z_t + ~copy * z_tm1

  4. RETURN z_t  # edited text
```

**Class:** `LSMESampler`

**Input:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| text_tokens | (B, L) | int64 | Original text token ids |
| target_latent | (B, D_latent) | float32 | Target latent vector for desired attribute |
| mask_ratio | scalar | float | Edit strength: 0.0 = no edit, 1.0 = full regeneration |
| steps | scalar | int | Number of reverse diffusion steps |
| temperature | scalar | float | Sampling temperature (default 1.0) |
| mask_mode | str | string | "random", "entropy", or "suffix" |

**Output:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| edited_tokens | (B, L) | int64 | Edited text token ids |
| edit_mask | (B, L) | bool | Which positions were masked/edited |

**Mask modes:**
- `random` — Each position masked independently with probability `mask_ratio`
- `entropy` — Mask positions where model is most uncertain (highest entropy in logits)
- `suffix` — Mask the last `mask_ratio * L` positions (for continuation-style editing)

**Hyperparameters:**

| Param | Default | Description |
|-------|---------|-------------|
| mask_ratio | 0.3 | Fraction of tokens to mask (controls edit strength) |
| steps | 100 | Reverse diffusion steps from entry point |
| temperature | 1.0 | Sampling temperature |
| mask_mode | "random" | Masking strategy |

**Architecture (layer by layer):**

```
Input: text_tokens (B, L), target_latent (B, D)
  │
  ├─→ Partial masking: mask positions with probability mask_ratio
  │     → z_t (B, L) with some positions = [MASK]
  │
  ├─→ Compute entry timestep from mask_ratio
  │     → t_entry ∈ (0, 1)
  │
  └─→ Reverse diffusion loop (t_entry → 0):
        │
        ├─→ MultimodalMMDiT forward pass
        │     Input: z_t (B, L), target_latent (B, 1, D), t, 0
        │     Output: text_logits (B, L, V), latent_pred (B, 1, D)
        │     # Joint attention transfers latent → text influence
        │
        ├─→ MDLM posterior computation
        │     → transition probs (B, L, V)
        │
        ├─→ Categorical sampling
        │     → z_tm1 (B, L)
        │
        └─→ Copy flag (preserve unmasked positions)
              → z_t updated (B, L)
```

**Key design choices:**
- **Zero architecture changes:** LSME only modifies the sampling procedure. The trained MultimodalMMDiT model is used as-is. This means no retraining needed.
- **Partial noise schedule:** Unlike full L2T generation (t: 1→0), LSME enters at t_entry < 1, preserving structure from unmasked tokens.
- **Latent as clean conditioning:** The target latent is provided with t_latent=0 (clean), so the model treats it as a conditioning signal, not something to denoise.
- **Copy flag:** Already-unmasked tokens are never changed, ensuring only masked positions are edited.

**Compared to prior work:**

| Aspect | ReMDM | DiffusER | LatentOps | Diffusion-LM | **LSME** |
|--------|-------|----------|-----------|---------------|----------|
| Text representation | Discrete tokens | Discrete (edit ops) | Continuous latent | Continuous embeddings | **Discrete tokens** |
| Editing mechanism | Remasking | INSERT/DELETE/KEEP | ODE in latent space | Gradient on embeddings | **Mask + latent steer** |
| Attribute control | No | No | Yes (ODE direction) | Yes (classifier grad) | **Yes (latent replacement)** |
| Requires retraining | No | Yes (edit model) | Yes (VAE + ODE) | Yes (classifier) | **No** |
| Preserves unedited | Yes (copy flag) | Partial | N/A (full regen) | Partial | **Yes (copy flag)** |

---

### Module 2: Attribute Latent Encoder — `latent_utils/attribute_encoder.py`

**File:** `latentDLM_mmdit/latent_utils/attribute_encoder.py`

**Purpose:** Encode target attributes into latent vectors z_target for LSME conditioning.

**Approach:** Use the model's own training data to extract attribute-conditioned latent statistics.

**Class:** `AttributeLatentEncoder`

**Methods:**

```python
class AttributeLatentEncoder:
    def __init__(self, latent_dir, metadata_file):
        """
        Load pre-computed latent vectors and their metadata (attribute labels).

        Args:
            latent_dir: Directory containing .npy latent files from training data
            metadata_file: JSON/CSV mapping filenames to attributes
                           e.g., {"sample_001.npy": {"sentiment": "positive", "topic": "food"}}
        """

    def compute_attribute_centroids(self, attribute_name):
        """
        For each value of attribute_name, compute the mean latent vector.

        Returns:
            centroids: dict[str, Tensor]  e.g., {"positive": tensor(D,), "negative": tensor(D,)}
        """

    def get_target_latent(self, attribute_name, target_value):
        """
        Return the centroid latent for the target attribute value.

        Returns:
            z_target: Tensor (D,)
        """

    def interpolate(self, z_source, z_target, alpha):
        """
        Spherical linear interpolation between two latents.

        Returns:
            z_interp: Tensor (D,)
        """
```

**Input:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| latent_dir | str | path | Directory of .npy latent files |
| metadata_file | str | path | Attribute labels per sample |

**Output:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| z_target | (D,) | float32 | Target latent centroid |

**Key design choice:** Using centroids of training latents (rather than training a separate encoder) avoids any retraining. The latent space is assumed to already cluster by attributes due to joint training.

---

### Module 3: Latent Geometry Analysis — `evaluation/metrics/latent_geometry.py`

**File:** `latentDLM_mmdit/evaluation/metrics/latent_geometry.py`

**Purpose:** First formal metrics for DLM latent space structure. Fills Gap 6.

**Inspired by:**
- Cosmos ([MeshchaninovViacheslav/cosmos](https://github.com/MeshchaninovViacheslav/cosmos)) — smooth latent text diffusion
- LatentOps ([guangyliu/LatentOps](https://github.com/guangyliu/LatentOps)) — ODE-based latent traversal
- Optimus ([ChunyuanLI/Optimus](https://github.com/ChunyuanLI/Optimus)) — text VAE interpolation

**Metrics:**

#### Metric 1: Semantic Smoothness Score (SSS)

**Definition:** Given two latents z_a and z_b, interpolate N points between them, generate text at each point, and measure smoothness of the semantic trajectory.

```
SSS(z_a, z_b, N) = (1/(N-1)) * Σ_{i=1}^{N-1} sim(text_i, text_{i+1})
```

where `sim` is cosine similarity of sentence embeddings (e.g., all-MiniLM-L6-v2).

- SSS → 1.0 means adjacent interpolation points produce semantically similar texts (smooth).
- SSS → 0.0 means texts change abruptly between adjacent points (rough).

**Implementation:**

```python
def semantic_smoothness_score(model, sampler, z_a, z_b, n_points=10, n_samples=5):
    """
    Args:
        model: MultimodalMMDiT
        sampler: ReverseDiffusionSampler (L2T mode)
        z_a, z_b: (D,) latent vectors
        n_points: interpolation steps
        n_samples: samples per interpolation point (for variance reduction)

    Returns:
        sss: float in [0, 1]
        trajectory: list of (alpha, texts, embedding) tuples
    """
    alphas = torch.linspace(0, 1, n_points)
    embeddings = []

    for alpha in alphas:
        z_interp = slerp(z_a, z_b, alpha)
        texts = [sampler.sample_l2t(z_interp, ...) for _ in range(n_samples)]
        emb = sentence_encoder.encode(texts).mean(0)  # average over samples
        embeddings.append(emb)

    # Compute consecutive cosine similarities
    sims = [cosine_sim(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]
    return np.mean(sims), trajectory
```

#### Metric 2: Monotonic Transition Score (MTS)

**Definition:** For an attribute interpolation (e.g., negative → positive sentiment), measure whether a classifier's confidence changes monotonically along the interpolation path.

```
MTS(z_neg, z_pos, N) = (1/(N-1)) * Σ_{i=1}^{N-1} 1[c(text_{i+1}) >= c(text_i)]
```

where `c(text)` is classifier probability for the target attribute.

- MTS = 1.0 means classifier confidence increases monotonically (ideal).
- MTS = 0.5 means random walk (no latent structure).

**Implementation:**

```python
def monotonic_transition_score(model, sampler, z_source, z_target,
                                classifier, n_points=10, n_samples=5):
    """
    Args:
        classifier: Callable that returns P(target_attribute | text)

    Returns:
        mts: float in [0, 1]
        scores: list of classifier scores along interpolation
    """
    alphas = torch.linspace(0, 1, n_points)
    scores = []

    for alpha in alphas:
        z_interp = slerp(z_source, z_target, alpha)
        texts = [sampler.sample_l2t(z_interp, ...) for _ in range(n_samples)]
        score = np.mean([classifier(t) for t in texts])
        scores.append(score)

    # Count monotonic increases
    monotonic = sum(1 for i in range(len(scores)-1) if scores[i+1] >= scores[i])
    return monotonic / (len(scores) - 1), scores
```

#### Additional geometry metrics:

| Metric | Formula | Description |
|--------|---------|-------------|
| **Cluster Separation** | silhouette_score(latents, labels) | How well latents cluster by attribute |
| **Interpolation Fluency** | mean PPL of texts along interpolation path | Whether intermediate points produce fluent text |
| **Latent Variance Ratio** | var(between-class) / var(within-class) | Attribute-related vs. noise variance |

---

### Module 4: DLM-Eval Suite — `evaluation/eval_suite.py`

**File:** `latentDLM_mmdit/evaluation/eval_suite.py`

**Purpose:** Comprehensive 6-pillar evaluation framework. Fills Gap 7 — directly addresses CCDD reviewer criticism of "limited evaluation scope."

**The 6 Pillars:**

| Pillar | Metrics | Source |
|--------|---------|--------|
| 1. Fluency | Perplexity (GPT-2), Grammar (LanguageTool) | Standard |
| 2. Controllability | Classifier accuracy, attribute match rate | ConGenBench-inspired |
| 3. Edit Quality | BLEU, ROUGE-L, BERTScore, edit distance | EasyEdit/EditEval-inspired |
| 4. Latent Geometry | SSS, MTS, cluster separation | **Novel** |
| 5. Diversity | Self-BLEU, Distinct-1/2/3 | Standard |
| 6. Efficiency | Wall-clock time, NFE, tokens/sec | dInfer-inspired |

**Class:** `DLMEvalSuite`

```python
class DLMEvalSuite:
    def __init__(self, config):
        """
        Args:
            config: EvalConfig with:
                - fluency_model: str (GPT-2 model for PPL)
                - classifier_model: str (sentiment/topic classifier)
                - sentence_encoder: str (for BERTScore, SSS)
                - device: str
        """

    def evaluate_generation(self, generated_texts, reference_texts=None):
        """Evaluate unconditional/conditional generation."""
        return {
            "fluency": self.compute_fluency(generated_texts),
            "diversity": self.compute_diversity(generated_texts),
            "efficiency": self.timing_stats,
        }

    def evaluate_editing(self, source_texts, edited_texts, target_attribute):
        """Evaluate LSME editing quality."""
        return {
            "fluency": self.compute_fluency(edited_texts),
            "controllability": self.compute_controllability(edited_texts, target_attribute),
            "edit_quality": self.compute_edit_quality(source_texts, edited_texts),
            "diversity": self.compute_diversity(edited_texts),
            "efficiency": self.timing_stats,
        }

    def evaluate_latent_geometry(self, model, sampler, latent_pairs, labels):
        """Evaluate latent space structure."""
        return {
            "sss": self.compute_sss(model, sampler, latent_pairs),
            "mts": self.compute_mts(model, sampler, latent_pairs, labels),
            "cluster_separation": self.compute_silhouette(latent_pairs, labels),
        }

    def full_evaluation(self, model, sampler, dataset):
        """Run all 6 pillars."""
        results = {}
        results.update(self.evaluate_generation(...))
        results.update(self.evaluate_editing(...))
        results.update(self.evaluate_latent_geometry(...))
        return results
```

---

## Task Specification

### Task 1: Sentiment Editing (Yelp)

**Dataset:** Yelp Review Polarity (binary: positive/negative)
**Source:** [yelp_polarity from HuggingFace](https://huggingface.co/datasets/yelp_polarity)
**Protocol:**
1. Take 500 negative reviews from test set
2. Compute mean latent of positive training reviews → z_positive
3. Run LSME(text=negative_review, z_target=z_positive, mask_ratio=[0.1, 0.3, 0.5, 0.7])
4. Measure: sentiment classifier accuracy, BLEU vs. original, fluency (PPL)

**Success condition:** Edited text classified as positive by external classifier, while preserving content from original.

### Task 2: Topic Transfer (Amazon Reviews)

**Dataset:** Amazon Reviews (multi-domain: electronics, books, kitchen, etc.)
**Protocol:**
1. Take 500 reviews from electronics domain
2. Compute mean latent of books domain → z_books
3. Run LSME editing
4. Measure: topic classifier accuracy, content preservation

### Task 3: Formality Transfer (GYAFC)

**Dataset:** GYAFC (Grammarly's Yahoo Answers Formality Corpus)
**Protocol:**
1. Take 500 informal sentences
2. Compute mean latent of formal training sentences → z_formal
3. Run LSME editing at multiple mask ratios
4. Measure: formality classifier accuracy, meaning preservation (BERTScore)

### Task 4: Latent Interpolation Visualization

**Protocol:**
1. Pick pairs of latents from different attribute clusters
2. Interpolate 10 points between each pair
3. Generate text at each point
4. Report SSS, MTS, and qualitative examples
5. Visualize with t-SNE/UMAP colored by attribute

---

## Evaluation Specification

### Metrics

| Metric | Formula | Description | From |
|--------|---------|-------------|------|
| PPL | exp(mean NLL) via GPT-2 | Fluency | Standard |
| Attr-Acc | classifier(edited_text) == target | Control accuracy | Standard |
| BLEU | n-gram overlap (source, edited) | Content preservation | Standard |
| ROUGE-L | LCS F1 (source, edited) | Content preservation | Standard |
| BERTScore | cosine sim of BERT embeddings | Semantic preservation | Standard |
| Edit-Dist | Levenshtein(source, edited) / len | Edit magnitude | Standard |
| SSS | mean cosine-sim of adjacent interp. texts | Latent smoothness | **Novel** |
| MTS | fraction of monotonic steps | Latent attribute structure | **Novel** |
| Distinct-1/2/3 | unique n-grams / total n-grams | Diversity | Standard |
| Self-BLEU | mean BLEU between pairs of outputs | Diversity (lower=better) | Standard |
| NFE | number of model forward passes | Compute cost | Standard |
| TPS | tokens per second | Speed | dInfer |

### Baselines to run

| Baseline | Repo | How to run | What it measures |
|----------|------|------------|-----------------|
| MDLM (unconditional) | [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) | `python main.py --mode sample` | PPL baseline, no editing |
| ReMDM (remasking) | [kuleshov-group/remdm](https://github.com/kuleshov-group/remdm) | `python sample.py --eta 0.5` | Editing without attribute control |
| LatentOps | [guangyliu/LatentOps](https://github.com/guangyliu/LatentOps) | `python generate.py --direction sentiment` | Latent-space style transfer (continuous) |
| DiffusER | [machelreid/diffuser](https://github.com/machelreid/diffuser) | `python generate.py --task style_transfer` | Edit-based diffusion |
| PLANNER | [apple/ml-planner](https://github.com/apple/ml-planner) | `python sample.py --task conditional` | Latent diffusion for text |
| LD4LG | [justinlovelace/latent-diffusion-for-language](https://github.com/justinlovelace/latent-diffusion-for-language) | `python sample.py --task class_cond` | Continuous latent DLM |

### Evaluation integration plan

For each baseline:
1. Generate/edit text on the same test inputs
2. Apply the same DLM-Eval Suite metrics
3. Report in a unified table for direct comparison

---

## Flowcharts

### Overall LSME Pipeline

```
┌─────────────────────┐     ┌────────────────────┐
│  Original Text      │     │  Target Attribute   │
│  "The food was      │     │  "positive"         │
│   terrible..."      │     │                     │
└──────────┬──────────┘     └─────────┬───────────┘
           │                          │
    ┌──────▼──────┐           ┌───────▼──────────┐
    │  Tokenize   │           │  Attribute        │
    │  (B, L)     │           │  Latent Encoder   │
    └──────┬──────┘           │  → z_target (B,D) │
           │                  └───────┬───────────┘
    ┌──────▼──────────┐               │
    │  Partial Mask   │               │
    │  mask_ratio=0.3 │               │
    │  → z_t (B, L)   │               │
    └──────┬──────────┘               │
           │                          │
           └──────────┬───────────────┘
                      │
              ┌───────▼──────────────┐
              │  Reverse Diffusion   │
              │  t_entry → 0         │
              │                      │
              │  ┌─────────────────┐ │
              │  │ MultimodalMMDiT │ │
              │  │ (frozen weights)│ │
              │  │                 │ │
              │  │ text_tokens ──┐ │ │
              │  │               ├─┤ │   Joint
              │  │ target_latent─┘ │ │   Attention
              │  │                 │ │
              │  │ → text_logits   │ │
              │  └─────────────────┘ │
              │                      │
              │  MDLM posterior      │
              │  + copy flag         │
              └───────┬──────────────┘
                      │
              ┌───────▼───────┐
              │  Edited Text  │
              │  "The food    │
              │   was great!" │
              └───────────────┘
```

### Latent Geometry Analysis Pipeline

```
┌─────────────┐     ┌─────────────┐
│ z_negative  │     │ z_positive  │
│ (centroid)  │     │ (centroid)  │
└──────┬──────┘     └──────┬──────┘
       │                    │
       └────────┬───────────┘
                │
    ┌───────────▼───────────────┐
    │ SLERP interpolation       │
    │ α = [0, 0.1, ..., 1.0]   │
    │ → N latent vectors        │
    └───────────┬───────────────┘
                │
    ┌───────────▼───────────────┐
    │ For each z_i:             │
    │   L2T sample → text_i    │
    │   sentence_embed(text_i)  │
    │   classifier(text_i)      │
    └───────────┬───────────────┘
                │
    ┌───────────▼───────────────┐
    │ Compute:                  │
    │   SSS = mean(sim(i,i+1)) │
    │   MTS = frac(monotonic)   │
    │   PPL trajectory          │
    └───────────────────────────┘
```

### DLM-Eval Suite Pipeline

```
┌──────────────┐
│ Model Under  │
│ Test         │
└──────┬───────┘
       │
       ├───→ Generation mode ──→ Fluency + Diversity + Efficiency
       │
       ├───→ Editing mode ─────→ Fluency + Controllability + Edit Quality
       │                          + Diversity + Efficiency
       │
       └───→ Geometry mode ────→ SSS + MTS + Cluster Separation
                                  + Interpolation Fluency

All results → unified JSON report + LaTeX table
```

---

## Dependencies

### From existing repos (to reuse):

| Component | Source Repo | Files to adapt |
|-----------|------------|----------------|
| MDLM sampling loop | [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) | `diffusion.py` (posterior computation) |
| Remasking logic | [kuleshov-group/remdm](https://github.com/kuleshov-group/remdm) | `diffusion.py` (remask function) |
| Sentence encoder | sentence-transformers | `all-MiniLM-L6-v2` model |
| Sentiment classifier | HuggingFace | `distilbert-base-uncased-finetuned-sst-2-english` |
| GPT-2 for PPL | HuggingFace | `gpt2` / `gpt2-medium` |

### New dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| sentence-transformers | >=2.2 | Sentence embeddings for SSS |
| evaluate | >=0.4 | BLEU, ROUGE, BERTScore |
| scikit-learn | >=1.0 | Silhouette score, t-SNE |
| matplotlib | >=3.5 | Visualization |

---

## Implementation Order

1. [ ] Implement `AttributeLatentEncoder` — compute centroids from training latents
2. [ ] Implement `LSMESampler.sample_lsme()` — the core editing algorithm (~100 lines)
3. [ ] Test LSME on a single example — verify edited text changes sentiment
4. [ ] Implement `latent_geometry.py` — SSS and MTS metrics
5. [ ] Implement `eval_suite.py` — 6-pillar evaluation runner
6. [ ] Implement `fluency.py`, `controllability.py`, `edit_quality.py`, `diversity.py`, `efficiency.py`
7. [ ] Set up Yelp sentiment benchmark
8. [ ] Run LSME at mask_ratio=[0.1, 0.3, 0.5, 0.7] on Yelp
9. [ ] Run latent geometry analysis (SSS, MTS)
10. [ ] Run baselines (ReMDM, LatentOps, DiffusER) through same eval suite
11. [ ] Compile results into tables
12. [ ] Run ablations: mask_mode, steps, temperature

---

## Ablation Study Plan

| Ablation | Variable | Values | Expected Effect |
|----------|----------|--------|-----------------|
| Mask ratio | mask_ratio | 0.1, 0.3, 0.5, 0.7, 0.9 | Higher → more change, less preservation |
| Mask mode | mask_mode | random, entropy, suffix | Entropy should preserve important tokens |
| Steps | steps | 10, 50, 100, 500 | More steps → better quality, slower |
| Temperature | temperature | 0.5, 0.8, 1.0, 1.2 | Higher → more diverse, less accurate |
| Latent source | z_target | centroid, random, interpolated | Centroid should give strongest control |
| No latent (ablation) | z_target | None (zeros) | Should show no attribute change |
| Directional centroid | z_target | z_src + α·(z_pos - z_neg) | Preserves source content better than hard centroid replacement |
| SLERP scheduling | z(t) | slerp(z_src, z_tgt, α(t)) | Early steps preserve structure, late steps shift attribute |
| Nearest-neighbor z | z_target | closest real training latent in target class | Stays on Qwen manifold, more realistic than mean centroid |
