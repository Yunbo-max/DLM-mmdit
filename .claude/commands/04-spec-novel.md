---
name: 04-spec-novel
description: Clone gap-solve repos, design novel contributions solving research gaps, output specs.md (code structure, model architecture, flowcharts) and a LaTeX file showing how the method solves gaps better than prior work.
---

Design novel research contributions that solve identified research gaps, produce a detailed code spec and a contribution LaTeX paper.

## User input: $ARGUMENTS

Parse the user input to extract:
1. **Gap solutions file** — path to `/03-gap-solve` output (e.g., `docs/gap_solutions.md`). Default: `docs/gap_solutions.md`
2. **Gap analysis file** — path to `/02-research-gap` output (e.g., `docs/research_gap.tex`). Default: `docs/research_gap.tex`
3. **Output directory** — where to write specs and tex (e.g., `docs/novel/`). Default: `docs/novel/`
4. **Clone directory** — where repos are / will be cloned (e.g., `repos/`). Default: `repos/`
5. **Project name** — name for the proposed method. If not provided, generate one.

Example invocations:
- `/spec-novel` → uses all defaults
- `/spec-novel docs/gap_solutions.md docs/research_gap.tex docs/novel/ repos/ TerrainVLM`
- `/spec-novel project: MyMethod`

---

## Step 0: Pre-flight validation

Before cloning repos or designing methods, validate ALL required inputs:

1. **Gap solutions file exists?** Check `docs/gap_solutions.md`. If missing → STOP:
   > "Missing: docs/gap_solutions.md. Run `/03-gap-solve` first to generate it."
2. **Gap solutions has directions?** Scan for `### Direction` headers. If zero → STOP:
   > "No directions found in gap_solutions.md. Re-run `/03-gap-solve`."
3. **Gap solutions has GitHub links?** Scan for `github.com` URLs. If zero → STOP:
   > "No GitHub repos found in gap_solutions.md. Top 5 papers per direction require repos."
4. **Gap analysis file exists?** Check `docs/research_gap.tex`. If missing → STOP:
   > "Missing: docs/research_gap.tex. Run `/02-research-gap` first."
5. **Output directory writable?** If not → STOP:
   > "Cannot write to [output_dir]. Create it with: `mkdir -p docs/novel/`"

Only proceed to Step 1 after ALL checks pass.

### Optional: Launch `/04b-counter-think` in parallel

After pre-flight passes, suggest to the user:

> "Optional: Run `/04b-counter-think` in parallel? It challenges field assumptions and searches cross-domain for paradigm-breaking ideas — a contrarian perspective alongside the gap-filling analysis. Any selected contrarian proposals feed back into the frozen scope here."

If the user says yes, the `/04b-counter-think` skill runs as a **parallel Agent worker** alongside Steps 1-2 of this skill. Its output (`docs/novel/counter_proposals.md`) is merged into the ranking table at Step 2.1, and the user can pick from both mainstream candidates AND contrarian proposals at Step 2.2.

---

## Step 1: Read inputs and clone ALL repos

### 1a. Read the gap solutions file
Extract ALL gaps, directions, and top papers from `/03-gap-solve` output.
The gap solutions file contains **top 5 papers per direction** for each gap. Collect ALL of them.

### 1b. Read the gap analysis file
Extract the comparison matrices (methods, tasks, datasets, evaluation) from `/02-research-gap` output.

### 1c. Clone ALL repos from gap solutions

Clone EVERY repo that appears in the gap solutions (top 5 per direction × all directions × all gaps):

```bash
mkdir -p [clone_dir]
```

For EACH paper with a GitHub link across ALL gaps and directions:
- If already cloned in the clone directory, skip
- Otherwise: `git clone --depth 1 [url] [clone_dir]/[repo_name]`

Print summary:
```
Cloned repos by gap/direction:
  Gap 1 — Direction 1.1: [repo1], [repo2], [repo3], [repo4], [repo5]
  Gap 1 — Direction 1.2: [repo6], [repo7], ...
  Gap 2 — Direction 2.1: ...
  ...
Total: [N] repos cloned
```

### 1d. Read each cloned repo deeply

For EACH repo, read:
- Model architecture files → understand input/output shapes, layer types, forward pass
- Training scripts → understand training loop, losses, optimizers
- Eval scripts → understand metric computation, data loading
- Task/env files → understand how tasks are defined, how agents interact
- Config files → understand hyperparameters, model dimensions

Build a detailed technical profile per repo (beyond the summary from `/02-research-gap`):
```
Repo: [name] (Gap [X], Direction [X.Y])
- Input: [shape, type — e.g., RGB (B,3,H,W), depth (B,1,H,W), text tokens (B,L)]
- Output: [shape, type — e.g., action logits (B,A), waypoint (B,2), value (B,1)]
- Backbone: [e.g., ResNet-18, ViT-B/16, CLIP ViT-L/14]
- Key layers: [e.g., 3x TransformerBlock(d=512, h=8), LinearProj(512→256), GRU(256)]
- Loss: [e.g., CrossEntropy + 0.5*ValueLoss + 0.1*AuxLoss]
- Framework: [e.g., PyTorch + Habitat-Lab 0.3]
- Eval pipeline: [e.g., loads HM3D val episodes, runs agent, computes SR/SPL]
- Reusable components: [what can we take from this repo]
```

---

## Step 2: Learn from repos and design novel contributions PER DIRECTION (PARALLEL)

This step uses **parallel execution** — each direction is studied by an independent Agent worker. All directions run simultaneously since they have NO dependencies on each other.

### 2.0 Spawn parallel workers — one per direction:

Count the total number of directions across all gaps. For EACH direction, launch a **background Agent worker** using the Agent tool:

```
For each direction [X.Y] across all gaps:
  → Agent(
      subagent_type: "general-purpose",
      run_in_background: true,
      prompt: "
        TASK: Study 5 papers for Gap [X] Direction [X.Y]: [direction name]
        REPOS: [repo1, repo2, repo3, repo4, repo5]
        EXPECTED OUTCOME: Summary table + novel method design + scores (Novelty/Feasibility/Confidence)
        MUST DO:
          - Read each repo's core model, training, and eval code
          - Identify key innovation + key limitation per paper
          - Design a novel method that goes beyond all 5
          - Score on Novelty (0-1), Feasibility (0-1), Confidence (0-1)
        MUST NOT DO:
          - Do NOT read repos outside your assigned direction
          - Do NOT write any files — return results as text only
          - Do NOT skip any of the 5 papers
        CONTEXT: [paste gap description + direction reasoning from gap_solutions.md]
      "
    )
```

**File ownership rule:** Each Agent worker ONLY reads repos for its own direction. No two agents read the same repo simultaneously. If two directions share a repo, assign it to the first direction and provide a summary to the second.

All agents run in parallel. After ALL agents complete, merge results in Step 2.1.

### Per-direction Agent task (each agent executes this independently):

#### Loop Step A: Deep study of the 5 papers

For each of the 5 repos in this direction:
1. Read the core model code — understand the architecture end-to-end
2. Read the training pipeline — what loss, what data, what tricks
3. Read the eval pipeline — what metrics, what baselines they compare to
4. Identify the **key innovation** of each paper (the one thing that's new)
5. Identify the **key limitation** of each paper (what it can't do)

Produce a summary table for this direction:

```
## Direction [X.Y]: [direction name]
### 5 Papers Summary

| # | Paper | Key Innovation | Key Limitation | Reusable Code |
|---|-------|---------------|----------------|---------------|
| 1 | [name] | [what's novel] | [what's missing] | [files/modules] |
| 2 | [name] | [what's novel] | [what's missing] | [files/modules] |
| 3 | [name] | [what's novel] | [what's missing] | [files/modules] |
| 4 | [name] | [what's novel] | [what's missing] | [files/modules] |
| 5 | [name] | [what's novel] | [what's missing] | [files/modules] |
```

#### Loop Step B: Design a novel method for this direction

Based on the 5 papers' strengths and limitations:
1. **What gap remains** even after all 5 papers?
2. **Which innovations can be combined** that no paper has combined?
3. **What's the novel insight** — the one idea that goes beyond all 5?
4. Design the method: architecture, input/output, key modules, loss

#### Loop Step C: Score this direction's contribution

Score on 3 dimensions (0.0 to 1.0, higher is better):

**Novelty (0-1):** How new is this compared to the 5 papers?
- 0.0 = trivially combining existing ideas
- 0.5 = meaningful new combination with some novel components
- 0.8 = significant new architecture/approach not seen in any paper
- 1.0 = fundamentally new paradigm

**Implementation Feasibility (0-1):** How hard is it to build?
- 0.0 = requires new hardware or years of engineering
- 0.5 = substantial but doable in 2-3 months, reuses some existing code
- 0.8 = can build in 2-4 weeks, reuses most infrastructure from cloned repos
- 1.0 = mostly plug-and-play from existing repos, just wire together

**Theoretical Confidence (0-1):** How likely will it work?
- 0.0 = speculative, no evidence it would improve results
- 0.5 = reasonable hypothesis, supported by related work
- 0.8 = strong evidence from ablations/analysis in the 5 papers
- 1.0 = near-certain improvement, backed by solid theory + similar experiments

### 2.1 Merge parallel results into ranking table

After ALL parallel Agent workers complete, collect their outputs. If `/04b-counter-think` was run in parallel, also read `docs/novel/counter_proposals.md` and include contrarian proposals (marked as `CT-X`) alongside mainstream candidates in the ranking table.

Compile a single ranked table:

```
## Novel Contribution Candidates — Ranked

| Rank | Gap | Direction | Proposed Method | Novelty | Feasibility | Confidence | Avg | Key Idea |
|------|-----|-----------|----------------|---------|-------------|------------|-----|----------|
| 1 | Gap 2 | Dir 2.1 | [method name] | 0.85 | 0.80 | 0.75 | 0.80 | [1-line] |
| 2 | Gap 1 | Dir 1.2 | [method name] | 0.90 | 0.60 | 0.80 | 0.77 | [1-line] |
| 3 | Gap 1 | Dir 1.1 | [method name] | 0.70 | 0.90 | 0.70 | 0.77 | [1-line] |
| 4 | Gap 3 | Dir 3.1 | [method name] | 0.80 | 0.70 | 0.65 | 0.72 | [1-line] |
| 5 | Gap 2 | Dir 2.2 | [method name] | 0.60 | 0.85 | 0.70 | 0.72 | [1-line] |
| ... | | | | | | | | |

### Per-candidate detail:

#### Candidate 1: [method name] (Gap 2, Dir 2.1)
**Novelty (0.85):** [why this score — what's new vs the 5 papers]
**Feasibility (0.80):** [why — what code exists, what needs building]
**Confidence (0.75):** [why — what evidence supports it working]
**Builds on:** [Paper X] (repo: [link]) + [Paper Y] (repo: [link])
**Novel beyond them:** [the specific thing neither paper does]

#### Candidate 2: [method name] (Gap 1, Dir 1.2)
...
```

### 2.2 ASK the human to decide — THREE CATEGORIES

Present candidates in **three separate categories** so the user can pick independently from each:

```
## Novel Contribution Candidates

### Category 1: Methods (algorithmic/architectural innovations)

| Rank | Candidate | Gap | Novelty | Feasibility | Confidence | Avg | Key Idea |
|------|-----------|-----|---------|-------------|------------|-----|----------|
| M1 | [method] | ... | ... | ... | ... | ... | [1-line] |
| M2 | [method] | ... | ... | ... | ... | ... | [1-line] |
| ... |

**Honest assessment per method:** For each candidate, state clearly:
- What already exists (be honest about prior work)
- What is genuinely new (the delta beyond prior work)
- Why it matters (what capability gap it fills)

**My suggestion:** [which method and why]

---

### Category 2: Tasks & Datasets (new tasks, benchmarks, or data contributions)

| Rank | Candidate | Gap | Novelty | Feasibility | Confidence | Avg | Key Idea |
|------|-----------|-----|---------|-------------|------------|-----|----------|
| T1 | [task] | ... | ... | ... | ... | ... | [1-line] |
| T2 | [task] | ... | ... | ... | ... | ... | [1-line] |
| ... |

**My suggestion:** [which tasks and why]

---

### Category 3: Evaluation (metrics, eval frameworks, analysis)

| Rank | Candidate | Gap | Novelty | Feasibility | Confidence | Avg | Key Idea |
|------|-----------|-----|---------|-------------|------------|-----|----------|
| E1 | [eval] | ... | ... | ... | ... | ... | [1-line] |
| E2 | [eval] | ... | ... | ... | ... | ... | [1-line] |
| ... |

**My suggestion:** [which evaluation and why]

---

## Pick from each category:
- **Method:** Pick one (or none) from Category 1
- **Tasks:** Pick any from Category 2 (these define WHAT you evaluate on)
- **Evaluation:** Pick any from Category 3 (these define HOW you evaluate)

Example: "M1, T2, E1+E2" or "M1 only, use standard eval"
```

**Suggestion rules:**
- Always analyze whether candidates COMPLEMENT or CONFLICT with each other
- If two candidates share infrastructure (same backbone, same dataset), suggest combining
- If candidates are from very different domains, suggest keeping separate
- Give honest assessment per method: what already exists vs. what's genuinely new
- Combining is NOT always better — sometimes a focused paper wins

**WAIT for the user's response.** Do not proceed until they choose.

### 2.3 Freeze scope (after human selection)

Once the user picks candidates, **LOCK the scope**. The user decides one of:
- **Single method** — develop one candidate as the full contribution
- **Combined method** — merge selected candidates into one unified pipeline
- **Multiple independent methods** — develop each separately (e.g., one paper per method, or multiple contributions in one paper)

Print a frozen requirements block:

```
## FROZEN SCOPE — Do not change without user approval
Selected candidates: [list of rank numbers]
Mode: [single / combined / independent]
Method name(s): [project name(s)]
Gaps being addressed: [list]
Modules to build: [list from selected candidates]
Out of scope: [everything NOT selected]

Any scope change requires explicit user approval.
```

This prevents scope creep during Steps 3-7. If later steps reveal a need to change the method, refer back to this frozen scope and ASK the user before changing.

### 2.4 Integration step (only if mode = combined)

**Skip this step if the user chose single or independent mode.** Only run if the user wants to combine candidates.

Based on the user's chosen candidates to combine:
1. **Design how the selected candidates integrate** — shared backbone, cascaded modules, joint training, etc.
2. **Check for conflicts** — do any selected candidates contradict each other?
3. **Name the combined method** — this becomes the project
4. **Re-score the combined method** — the combination may score differently than individuals

```
## Combined Method: [Project Name]

Selected candidates: [list]
Combined scores:
  Novelty: [X] (may be higher if combination is novel, or same if just additive)
  Feasibility: [X] (may be lower due to integration complexity)
  Confidence: [X] (may be higher due to multiple reinforcing ideas)

Integration design:
  [how the selected modules connect into one pipeline]
```

**If mode = single:** Just proceed with the one selected candidate as-is.
**If mode = independent:** Proceed with each candidate separately — generate specs.md and contribution.tex for EACH method independently.

---

## Step 3: Design detailed specs for selected contributions

For EACH selected contribution that involves methods:

### 3a. Novel method design

For each selected candidate that involves methods:

1. **Identify reusable components** from cloned repos — what can we take and adapt?
2. **Identify the missing piece** — what does NO repo have?
3. **Design the novel module** with full architecture spec:

```
## Novel Module: [Name]

### Motivation
[1-2 sentences: what gap this fills, why existing approaches fail]

### Architecture
- Input: [exact tensor shapes, e.g., semantic_map (B, C, H, W), text_embed (B, D)]
- Processing:
  1. [Layer/Op]: [input shape] → [output shape] — [purpose]
  2. [Layer/Op]: [input shape] → [output shape] — [purpose]
  ...
- Output: [exact tensor shapes, e.g., energy_field (B, 1, H, W)]

### Key design choices
- [Why this layer/connection/loss over alternatives]
- [What prior paper inspires this, and what we change]

### Compared to prior work
| Aspect | [Prior Paper A] | [Prior Paper B] | Ours |
|--------|----------------|----------------|------|
| ... | ... | ... | ... |
```

Specify ALL of:
- Layer types (Linear, Conv2d, MultiHeadAttention, GRU, LayerNorm, etc.)
- Dimensions (hidden size, number of heads, kernel size, etc.)
- Activation functions (ReLU, GELU, Softmax, Sigmoid, etc.)
- Residual connections (where and why)
- Normalization (LayerNorm, BatchNorm, where)
- Dropout (rate, where)

### 3b. Novel task/environment design

For each gap that involves tasks:

1. **What existing task is closest?** (from cloned repos)
2. **What modification creates the novel task?**
3. **How does the task interface with the method?**
   - Observation space (exact dict of tensors)
   - Action space (discrete/continuous, dimensions)
   - Reward/success definition
   - Episode structure

### 3c. Novel dataset contribution

For each gap that involves datasets:

1. **What existing datasets can we combine/extend?**
2. **What annotation or processing is new?**
3. **Data pipeline spec** — how data flows from raw to model input

### 3d. Novel evaluation design

For each gap that involves evaluation:

1. **What metrics are missing?** Design new ones with mathematical definitions.
2. **What baselines should be compared?** (from cloned repos — which can we directly run?)
3. **Evaluation protocol** — exact steps to reproduce

---

## Step 4: Write specs.md

Generate `[output_dir]/specs.md` with FULL implementation specification:

```markdown
# [Project Name] — Implementation Specification

## Overview
[2-3 sentences: what this project does and which gaps it solves]

## Novel Contributions
1. [Contribution 1 — 1 sentence]
2. [Contribution 2 — 1 sentence]
3. [Contribution 3 — 1 sentence]

---

## Project Structure

```
[project_name]/
├── configs/
│   ├── default.yaml            # Default hyperparameters
│   ├── model/
│   │   ├── [model_name].yaml   # Model-specific config
│   │   └── ablation/
│   │       └── no_[module].yaml
│   └── task/
│       ├── [task_name].yaml
│       └── eval.yaml
├── models/
│   ├── __init__.py
│   ├── [module_1].py           # [what this module does]
│   │   # class [ModuleName](nn.Module):
│   │   #   Input: [shapes]
│   │   #   Output: [shapes]
│   │   #   Key layers: [list]
│   ├── [module_2].py           # [what this module does]
│   ├── backbone/
│   │   ├── __init__.py
│   │   └── [backbone].py       # [pretrained model wrapper]
│   └── heads/
│       ├── __init__.py
│       └── [head].py           # [prediction head]
├── data/
│   ├── __init__.py
│   ├── dataset.py              # [dataset class]
│   └── transforms.py           # [data augmentation/processing]
├── envs/
│   ├── __init__.py
│   ├── [env_name].py           # [environment wrapper]
│   └── tasks/
│       └── [task_name].py      # [task definition]
├── agents/
│   ├── __init__.py
│   └── [agent_name].py         # [agent policy]
├── training/
│   ├── __init__.py
│   ├── trainer.py              # [training loop]
│   └── losses.py               # [loss functions]
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py            # [eval loop]
│   ├── metrics.py              # [metric computation]
│   └── baselines/
│       └── run_[baseline].py   # [scripts to run baseline repos]
├── utils/
│   ├── __init__.py
│   └── [util].py
├── scripts/
│   ├── train.py                # Entry point: training
│   ├── eval.py                 # Entry point: evaluation
│   └── visualize.py            # Entry point: visualization
├── tests/
│   └── test_[module].py
├── requirements.txt
└── README.md
```

---

## Module Specifications

### [Module 1]: [Name]

**File:** `models/[module_1].py`

**Purpose:** [what gap this fills]

**Inspired by:** [Paper A] ([repo link]) + [Paper B] ([repo link]) — [what we take from each]

**Class:** `[ClassName](nn.Module)`

**Input:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| x | (B, C, H, W) | float32 | Semantic map |
| text | (B, D) | float32 | Text embedding |

**Output:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| energy | (B, 1, H, W) | float32 | Energy field |

**Architecture (layer by layer):**
```
Input x: (B, C, H, W)
  │
  ├─→ Conv2d(C, 64, 3, padding=1) + BatchNorm2d(64) + ReLU
  │     → (B, 64, H, W)
  │
  ├─→ Conv2d(64, 128, 3, stride=2, padding=1) + BatchNorm2d(128) + ReLU
  │     → (B, 128, H/2, W/2)
  │
  ├─→ [ResidualBlock(128)] × 2
  │     → (B, 128, H/2, W/2)
  │
  ├─→ MultiHeadAttention(d=128, heads=8, text_dim=D)
  │     # Cross-attend to text embedding
  │     → (B, 128, H/2, W/2)
  │
  ├─→ ConvTranspose2d(128, 64, 3, stride=2, padding=1) + ReLU
  │     → (B, 64, H, W)
  │
  ├─→ Conv2d(64, 1, 1) + Sigmoid
  │     → (B, 1, H, W)
  │
  └─→ Output energy: (B, 1, H, W)
```

**Residual connections:** Skip from layer 1 output to layer 5 input (channel-wise concat + 1x1 conv)

**Hyperparameters:**
| Param | Default | Description |
|-------|---------|-------------|
| hidden_dim | 128 | Hidden channel dimension |
| n_heads | 8 | Attention heads |
| n_res_blocks | 2 | Residual blocks |
| dropout | 0.1 | Dropout rate |

### [Module 2]: [Name]
... (same format)

---

## Task Specification

### Task: [Name]

**File:** `envs/tasks/[task_name].py`

**Based on:** [Existing task from repo X] — [what we modify]

**Observation space:**
| Key | Shape | Type | Source |
|-----|-------|------|--------|
| rgb | (H, W, 3) | uint8 | Sensor |
| depth | (H, W, 1) | float32 | Sensor |
| goal | (D,) | float32 | Task spec |

**Action space:**
| Action | ID | Description |
|--------|----|-------------|
| MOVE_FORWARD | 0 | 0.25m step |
| TURN_LEFT | 1 | 30° rotation |
| TURN_RIGHT | 2 | 30° rotation |
| STOP | 3 | Declare done |

**Success condition:** [exact definition]

**Reward:** [formula if RL, or N/A if zero-shot]

---

## Evaluation Specification

### Metrics

| Metric | Formula | Description | From |
|--------|---------|-------------|------|
| SR | success / total | Success rate | Standard |
| SPL | (1/N) Σ sᵢ lᵢ/max(pᵢ,lᵢ) | Success weighted by path length | Standard |
| [New metric] | [formula] | [what it measures] | **Novel** |

### Baselines to run

| Baseline | Repo | How to run | Expected output |
|----------|------|------------|-----------------|
| [Paper A] | [link] | `python eval.py --config X` | SR, SPL on HM3D val |
| [Paper B] | [link] | `python test.py --dataset Y` | SR, SPL on HM3D val |

### Integration plan
For each baseline repo, specify:
- Which eval script to use
- What config changes needed to match our evaluation protocol
- How to extract results in a common format

---

## Flowcharts

Draw ASCII flowcharts for the main pipeline:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  RGB+Depth  │────→│  Backbone    │────→│  Feature     │
│  Input      │     │  (frozen)    │     │  Map (B,D,H,W)│
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐             │
                    │  Text Goal   │─────────────┤
                    │  Encoder     │             │
                    └──────────────┘     ┌───────▼──────┐
                                        │  [Novel      │
                                        │   Module]    │
                                        │  Cross-Attn  │
                                        └───────┬──────┘
                                                │
                    ┌──────────────┐     ┌───────▼──────┐
                    │  Traversa-   │────→│  Energy      │
                    │  bility Map  │     │  Combiner    │
                    └──────────────┘     └───────┬──────┘
                                                │
                                        ┌───────▼──────┐
                                        │  FMM Planner │
                                        │  → Action    │
                                        └──────────────┘
```

Draw flowcharts for:
1. **Overall pipeline** — sensor input to action output
2. **Each novel module** — internal architecture
3. **Training pipeline** — data loading to loss computation
4. **Evaluation pipeline** — episode loop to metric aggregation

---

## Dependencies

### From existing repos (to reuse):
| Component | Source Repo | Files to copy/adapt |
|-----------|------------|---------------------|
| ... | [repo link] | `model/backbone.py`, `utils/fmm.py` |

### New dependencies:
| Package | Version | Purpose |
|---------|---------|---------|
| ... | ... | ... |

---

## Implementation Order

1. [ ] Set up project structure and configs
2. [ ] Implement [Module 1] with unit test
3. [ ] Implement [Module 2] with unit test
4. [ ] Integrate into pipeline, test end-to-end
5. [ ] Set up evaluation with baselines
6. [ ] Training loop (if applicable)
7. [ ] Run experiments and ablations
```

---

## Step 5: Write contribution LaTeX file

Generate `[output_dir]/contribution.tex` showing how the proposed method solves gaps:

```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}

\title{[Project Name]: [One-line description]}
\author{[TBD]}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
[3-4 sentences:
- What problem / gaps we address
- What our key idea is (1 sentence)
- What we achieve (results claim placeholder)
]
\end{abstract}

\section{Introduction}
% Structure:
% Para 1: Context — what is the broader problem
% Para 2: Gap — what current methods miss (cite gap analysis)
% Para 3: Our approach — high-level idea
% Para 4: Contributions list

Our contributions are:
\begin{enumerate}
  \item \textbf{[Contribution 1]:} [1 sentence — what it is + which gap it fills]
  \item \textbf{[Contribution 2]:} [1 sentence]
  \item \textbf{[Contribution 3]:} [1 sentence]
\end{enumerate}

\section{Related Work}
% For each gap, show what prior work exists and what's missing
% Use papers from /01-literature-search and /03-gap-solve

\subsection{[Topic 1]}
[Prior Paper A] does X but lacks Y. [Prior Paper B] addresses Y but not Z.
\textbf{In contrast}, our method combines X and Z through [novel module].

\subsection{[Topic 2]}
...

\section{Method}

\subsection{Overview}
% High-level pipeline description
% Reference to flowchart figure

\subsection{[Novel Module 1]}
% Detailed technical description
% Input/output specification
% Key equations:

Given input feature map $\mathbf{F} \in \mathbb{R}^{B \times C \times H \times W}$
and text embedding $\mathbf{t} \in \mathbb{R}^{B \times D}$:

\begin{equation}
  \mathbf{E} = \sigma\left( \text{Conv}_{1\times1}\left(
    \text{CrossAttn}(\mathbf{F}', \mathbf{t}) + \mathbf{F}'
  \right)\right)
\end{equation}

where $\mathbf{F}' = \text{ResBlocks}(\text{Conv}(\mathbf{F}))$.

\subsection{[Novel Module 2]}
...

\section{How Our Method Solves the Gaps}

% THIS IS THE KEY SECTION — clearly map contributions to gaps

\begin{table}[h]
\centering
\caption{Gap resolution summary}
\begin{tabular}{p{4cm}p{3cm}p{5cm}}
\toprule
\textbf{Research Gap} & \textbf{Our Component} & \textbf{How It Solves the Gap} \\
\midrule
[Gap 1 from analysis] & [Module/Contribution] & [Specific explanation] \\
[Gap 2] & [Module/Contribution] & [Specific explanation] \\
[Gap 3] & [Module/Contribution] & [Specific explanation] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Comparison with Prior Methods}

\begin{table}[h]
\centering
\caption{Feature comparison with prior work}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{[Feature 1]} & \textbf{[Feature 2]} & \textbf{[Feature 3]} & \textbf{[Feature 4]} \\
\midrule
[Prior A] & \checkmark & & \checkmark & \\
[Prior B] & & \checkmark & & \checkmark \\
[Prior C] & \checkmark & & & \\
\textbf{Ours} & \checkmark & \checkmark & \checkmark & \checkmark \\
\bottomrule
\end{tabular}
\end{table}

\section{Experimental Setup (Planned)}

\subsection{Datasets}
[Which datasets, which splits, why]

\subsection{Baselines}
[Which methods to compare, using which repos]

\subsection{Metrics}
[Standard + novel metrics with definitions]

\subsection{Implementation Details}
[Framework, hardware, hyperparameters — from specs.md]

\section{Expected Results}
% Placeholder tables for results

\begin{table}[h]
\centering
\caption{Expected results on [Dataset]}
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{SR (\%)} & \textbf{SPL (\%)} \\
\midrule
[Baseline A] & [reported] & [reported] \\
[Baseline B] & [reported] & [reported] \\
\textbf{Ours} & — & — \\
\bottomrule
\end{tabular}
\end{table}

\section{Ablation Study (Planned)}
% What ablations to run — each novel component on/off

\begin{table}[h]
\centering
\caption{Planned ablations}
\begin{tabular}{lccc}
\toprule
\textbf{Variant} & \textbf{[Module 1]} & \textbf{[Module 2]} & \textbf{Expected Effect} \\
\midrule
Full model & \checkmark & \checkmark & Best \\
w/o [Module 1] & & \checkmark & [what degrades] \\
w/o [Module 2] & \checkmark & & [what degrades] \\
Baseline only & & & Worst \\
\bottomrule
\end{tabular}
\end{table}

\end{document}
```

### Rules for the LaTeX file:
- Must compile cleanly with `pdflatex`
- Use `booktabs` for all tables
- All prior work references should cite the actual paper
- The "How Our Method Solves the Gaps" section MUST explicitly map every gap to a component
- Feature comparison table must show our method has capabilities others lack
- Equations must be precise (not pseudo-math) — they should match the specs.md architecture
- Expected results table should include reported numbers from baselines (from their papers/repos)

---

## Step 6: Verify consistency

Before finishing, cross-check:
1. **specs.md ↔ contribution.tex**: Do the module specs match the equations in the paper?
2. **specs.md ↔ gap analysis**: Does every gap have a corresponding module/contribution?
3. **specs.md ↔ cloned repos**: Are the "from existing repos" components actually available in those repos?
4. **contribution.tex ↔ gap solutions**: Do the related work sections cover the papers from `/03-gap-solve`?
5. **Evaluation plan**: Can every baseline repo actually run with the specified commands?

---

## Step 7: Summary output

Print to the user:

```
## Novel Contribution Spec Complete

**Project:** [Name]
**Gaps addressed:** X
**Novel modules:** Y
**Baselines to compare:** Z

### Outputs:
- `[output_dir]/specs.md` — Full implementation specification
- `[output_dir]/contribution.tex` — Contribution paper (compile with pdflatex)

### Novel Contributions:
1. [Contribution 1] — solves [Gap X]
2. [Contribution 2] — solves [Gap Y]
3. [Contribution 3] — solves [Gap Z]

### Implementation order:
1. [First thing to build]
2. [Second thing to build]
...

### Reusable code from existing repos:
- [component] from [repo]
- [component] from [repo]
```
