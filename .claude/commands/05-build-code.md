---
name: 05-build-code
description: Build code from specs.md, set up baselines, design ablations, prepare model/dataset download scripts, and make everything git-push ready for remote compute. Supports delta-build on existing codebases.
---

Build the implementation from specs, set up baselines and ablations, prepare download scripts, and make the repo ready to git push to remote compute.

Supports two modes:
- **Delta build** — when an existing codebase is detected or provided, only create NEW files from specs.md. Existing code is imported, not reimplemented.
- **Greenfield build** — when no existing code is found, create the entire project from scratch.

## User input: $ARGUMENTS

Parse the user input to extract:
1. **Specs file** — path to specs.md from `/04-spec-novel`. Default: `docs/novel/specs.md`
2. **Contribution tex** — path to contribution.tex. Default: `docs/novel/contribution.tex`
3. **Project directory** — where to create the project. Default: current working directory project name from specs
4. **Gap solutions file** — path to `/03-gap-solve` output. Default: `docs/gap_solutions.md`
5. **Clone directory** — where baseline repos are cloned. Default: `repos/`
6. **Existing codebase** — path to existing code directory. Default: auto-detect.
   If provided, the build only creates NEW files specified in specs.md.
   Existing modules are imported, not reimplemented.

Example invocations:
- `/build-code` → uses all defaults
- `/build-code docs/novel/specs.md`
- `/build-code docs/novel/specs.md project_dir: TerrainVLM/`
- `/build-code codebase: latentDLM_mmdit/`
- `/build-code docs/novel/specs.md codebase: latentDLM_mmdit/`

---

## Step 0: Pre-flight validation

Before writing any code, validate ALL required inputs exist and are well-formed:

1. **specs.md exists?** Check `docs/novel/specs.md`. If missing → STOP:
   > "Missing: docs/novel/specs.md. Run `/04-spec-novel` first to generate it."
2. **specs.md has required sections?** Scan for: `## Project Structure`, `## Module Specifications`, `## Evaluation Specification`. If any missing → STOP:
   > "specs.md is incomplete — missing [section]. Re-run `/04-spec-novel`."
3. **contribution.tex exists?** Check `docs/novel/contribution.tex`. If missing → STOP:
   > "Missing: docs/novel/contribution.tex. Run `/04-spec-novel` first."
4. **contribution.tex has ablation table?** Scan for `Ablation Study`. If missing → warn:
   > "Warning: No ablation table in contribution.tex. Ablation configs will be skipped."
5. **Clone directory has repos?** Check `repos/` has at least one subdirectory. If empty → warn:
   > "Warning: No cloned repos in repos/. Baseline runners may fail."
6. **Existing codebase?**
   a. If user provided `codebase:` path → verify it exists, scan it. If path does not exist → STOP:
      > "Codebase path '[path]' does not exist."
   b. If not provided → auto-detect: look for Python packages (directories with `__init__.py` or multiple `.py` files) in the project root. If found → ask user to confirm which is the codebase:
      > "Detected existing Python packages: [list]. Which is the main codebase to build on? (or 'none' for greenfield)"
   c. If none found → proceed as greenfield (original behavior). Print:
      > "No existing codebase detected. Proceeding with greenfield build."

Only proceed to Step 1 after ALL required checks pass.

---

## Step 1: Read all inputs

### 1a. Read specs.md
Extract:
- Project structure (all folders and files)
- Module specifications (classes, input/output shapes, layers, hyperparameters)
- Task/env specifications
- Evaluation specifications
- Baselines list with repos
- Dependencies
- Implementation order

### 1b. Read contribution.tex
Extract:
- Planned ablations table
- Baseline comparison table
- Expected metrics

### 1c. Read gap solutions
Extract:
- All top papers with GitHub links (these are potential baselines)
- **Tag each repo's strengths:** Which repos have good data preprocessing? Good evaluation? Good model code?

### 1d. Scan cloned repos
For each baseline repo in `repos/`:
- Check if it exists and is complete
- Read its README for setup/download instructions
- Read its eval scripts for how to run evaluation
- Read its requirements for dependencies
- **Classify reusable components** — for each repo, identify:
  - Data preprocessing code (data loaders, tokenizers, transforms, dataset classes)
  - Evaluation code (metrics, eval loops, benchmark scripts)
  - Model/method code (architecture, training loops)

Build a **reuse map** — which repos have code we can directly reuse:

```
reuse_map = {
    "repo_A": {
        "data": ["data/dataloader.py", "data/tokenizer.py"],  # reusable data code
        "eval": ["eval/metrics.py", "eval/run_eval.py"],      # reusable eval code
        "model": [],                                            # we write our own
    },
    "repo_B": {
        "data": [],
        "eval": ["evaluation/perplexity.py"],
        "model": [],
    },
    ...
}
```

### 1e. Classify specs.md modules into 3 categories

Parse specs.md's Project Structure and Module Specifications. Classify every module into exactly one of:

1. **DATA** — data loading, preprocessing, tokenization, dataset wrappers, feature extraction
2. **METHOD** — novel model architecture, training logic, the core contribution
3. **EVAL** — metrics, evaluation scripts, benchmarks, comparison runners, visualization

Build a **classification table**:

```
| specs.md file                  | Category | Reusable from repo? | Action           |
|-------------------------------|----------|---------------------|------------------|
| data/dataloader.py            | DATA     | repo_A/data/load.py | ADAPT from repo  |
| data/tokenizer.py             | DATA     | repo_A/data/tok.py  | ADAPT from repo  |
| models/novel_module.py        | METHOD   | —                   | WRITE from scratch |
| models/backbone.py            | METHOD   | —                   | WRITE from scratch |
| sample_lsme.py                | METHOD   | —                   | WRITE from scratch |
| evaluation/metrics.py         | EVAL     | repo_A/eval/met.py  | ADAPT from repo  |
| evaluation/eval_suite.py      | EVAL     | repo_B/eval/ppl.py  | ADAPT from repo  |
| scripts/run_eval.py           | EVAL     | repo_A/eval/run.py  | ADAPT from repo  |
```

**ADAPT from repo** means: copy the repo's code, modify imports/paths/formats to fit our project structure. Do NOT rewrite from scratch if a good implementation exists.

**WRITE from scratch** means: no repo has this (it's our novel contribution). Implement per specs.md.

---

## Step 1.5: Dependency Resolution & Auto-Download

Parse specs.md for ALL external dependencies — Python packages, GitHub repos, and base architectures. If any are missing, download them automatically.

### 1.5a. Parse specs.md for referenced dependencies

Scan specs.md for:
- **Python packages** — any `import xxx` or `from xxx import` or package names in the Dependencies section
- **GitHub repos** — any GitHub URLs or repo references (e.g., "based on MDLM repo", "uses MMDiT architecture from [repo]")
- **Base architectures** — model names referenced as dependencies (e.g., "uses GPT-2 for perplexity", "DiT backbone", "MMDiT")
- **HuggingFace models** — model IDs like `gpt2`, `distilbert-base-uncased`, `sentence-transformers/all-MiniLM-L6-v2`

Build a **dependency table**:

```
| Dependency           | Type      | Source                        | Status       |
|---------------------|-----------|-------------------------------|--------------|
| torch               | pip       | PyPI                          | ✓ installed  |
| omegaconf           | pip       | PyPI                          | ✗ missing    |
| mmdit_latent        | repo      | local: mmdit_latent/          | ✓ found      |
| mdlm                | repo      | github.com/user/mdlm          | ✗ missing    |
| gpt2                | hf_model  | huggingface: gpt2             | ✓ cached     |
| sentence-transformers| pip      | PyPI                          | ✗ missing    |
```

### 1.5b. Auto-resolve missing dependencies

For each **missing** dependency:

**Python packages (pip):**
```bash
pip install [package_name]  # or add to requirements.txt for remote install
```
Only install locally if needed for the build. Otherwise, just add to requirements.txt.

**GitHub repos:**
```bash
# If referenced in specs.md or gap_solutions.md with a URL
git clone [github_url] repos/[repo_name]
```

**Local codebases (like mmdit_latent):**
If specs.md references a module (e.g., `from mmdit_latent.utils import ...`) but the directory is NOT provided by the user and NOT found locally:

1. Search gap_solutions.md and literature.md for a matching GitHub URL
2. If found → `git clone [url] repos/[name]`
3. If not found → search PyPI (`pip show [name]`)
4. If still not found → ask user:
   > "specs.md references '[module_name]' but I can't find it locally or online. Please provide the path or GitHub URL."

**HuggingFace models:**
Don't download now — add to `scripts/download_models.sh` for remote download. But verify the model ID is valid:
```python
# Quick check (no download)
from huggingface_hub import model_info
model_info("[model_id]")
```

### 1.5c. Bundle or reference?

For each resolved dependency, decide:

- **BUNDLE** (copy into project dir) — if it's a small local codebase (<500KB Python) that the project modifies or extends heavily. Creates self-contained package. Add `sys.modules` aliasing in `__init__.py`.
- **REFERENCE** (keep in repos/ or as pip dep) — if it's a large repo, or only used as-is without modification. Add to requirements.txt or document in setup instructions.

Print resolution summary:
```
## Dependency Resolution
  ✓ torch (pip, already installed)
  ✓ omegaconf (pip, added to requirements.txt)
  ✓ mmdit_latent (local, BUNDLED into project)
  ✓ mdlm (cloned to repos/mdlm)
  ✓ gpt2 (HuggingFace, added to download script)
  ⚠ [dep] — not found, asked user
```

---

## Step 1.6: Codebase Inventory & Delta Analysis

**Skip this step entirely if greenfield mode (no existing codebase detected).**

If existing codebase was detected/provided:

### 1.6a. Scan existing code

Launch an Explore agent to inventory the codebase:

```
Agent(
  subagent_type: "Explore",
  prompt: "Thoroughly inventory the codebase at [codebase_path]:
    - All .py files with their classes and key functions
    - All config files (.yaml, .json)
    - Existing requirements.txt / setup.py / pyproject.toml
    - Existing evaluation scripts
    - Existing sampling/inference code
    - Existing training scripts
    - Utility modules and helper functions
    Return a structured inventory with file paths, class names, and function signatures."
)
```

### 1.6b. Map specs.md → existing code

For each module in specs.md's Project Structure, compare against the codebase inventory:

- Mark as **EXISTING** if a matching file/class already exists in the codebase
- Mark as **NEW** if no match found — this file needs to be created
- Mark as **EXTEND** if it builds on an existing file (e.g., new sampler extending existing sampler, new config extending existing configs)

Build a **delta table**:

```
| specs.md file                  | Status   | Existing file                              | Action                                    |
|-------------------------------|----------|--------------------------------------------|-------------------------------------------|
| models/multimodal_mmdit.py    | EXISTING | latentDLM_mmdit/models/multimodal_mmdit.py | SKIP — no changes                         |
| sample_lsme.py                | NEW      | —                                          | CREATE — extends sample_l2t_fixed.py      |
| evaluation/eval_suite.py      | NEW      | —                                          | CREATE                                    |
| configs/lsme_yelp.yaml        | NEW      | —                                          | CREATE — extends existing Hydra configs   |
| requirements.txt              | EXTEND   | requirements.txt                           | MERGE — add new deps only                 |
```

**Print the delta table to the user and WAIT for confirmation before proceeding.**

> "Here is the delta analysis. [N] files will be SKIPPED (already exist), [M] files will be CREATED, [K] files will be EXTENDED. Proceed? (yes/no/edit)"

### 1.6c. Extract reuse context

For each **EXTEND** module, read the existing file it extends. Pass this context to the implementation agents in Step 3 so they:
- Import from the existing module (not reimplement)
- Match coding style, naming conventions, config patterns
- Use existing utilities (e.g., `sample_categorical`, `get_sigmas`, custom loss functions)
- Follow the same argument parsing patterns, logging conventions, and checkpoint formats

Store the reuse context as a mapping:
```
reuse_context = {
    "sample_lsme.py": {
        "extends": "sample_l2t_fixed.py",
        "content": "[full file content of sample_l2t_fixed.py]",
        "imports_to_use": ["sample_categorical", "get_sigmas", ...],
        "style_notes": "Uses argparse, OmegaConf, ..."
    },
    ...
}
```

---

## Step 2: Create project structure

### If existing codebase (delta mode):

- Only create directories that DON'T already exist
- Only create `__init__.py` in NEW directories
- **NEVER overwrite or modify existing files**
- Print what was created vs what was skipped:

```
## Directory Structure (Delta)
  SKIP   latentDLM_mmdit/models/         (already exists)
  SKIP   latentDLM_mmdit/configs/        (already exists)
  CREATE latentDLM_mmdit/evaluation/     (new directory)
  CREATE latentDLM_mmdit/evaluation/__init__.py
  CREATE latentDLM_mmdit/latent_utils/   (new directory)
  CREATE latentDLM_mmdit/latent_utils/__init__.py
```

### If greenfield:

Follow the specs.md folder structure EXACTLY. Create all directories and files.

```bash
mkdir -p [project_dir]/{configs/model/ablation,configs/task,models/backbone,models/heads,data,envs/tasks,agents,training,evaluation/baselines,utils,scripts,tests}
```

Create `__init__.py` in every Python package directory.

---

## Step 3: Implement code (3 phases)

Code implementation is split into **three ordered phases**. This ordering matters because:
- Data preprocessing must be done first (methodology and evaluation both depend on it)
- Methodology is the novel contribution — written from scratch per specs.md
- Evaluation comes last (needs both data format and method outputs to be defined)

For phases 3A (Data) and 3C (Evaluation), **always scan the top GitHub repos first** for reusable code. Research repos almost always have working data loaders and eval scripts — reuse them instead of writing from scratch.

Use the **classification table** from Step 1e and the **reuse map** from Step 1d to decide what to reuse vs. write.

---

### Step 3A: Data Preprocessing (REUSE-FIRST)

**Goal:** Create all data loading, preprocessing, tokenization, and dataset wrapper code.

**Priority order:**
1. **REUSE** from top repos (adapt their data code)
2. **EXTEND** from existing codebase (if delta mode)
3. **WRITE** from scratch (only if no repo has usable data code)

#### 3A.1. Scan repos for reusable data code

For each repo in the reuse map that has data code, launch an Explore agent:

```
Agent(
  subagent_type: "Explore",
  prompt: "Read the data preprocessing code in repos/[repo_name]/[data_files].
    Extract:
    - Dataset class definitions (what datasets they load, format, splits)
    - Tokenizer / text preprocessing pipeline
    - Data transforms, augmentations, feature extraction
    - Collate functions, batch construction
    - Config / argument handling for data paths
    Return the full implementation with inline comments explaining each step."
)
```

#### 3A.2. Adapt repo data code to our project

For each DATA module in specs.md, launch an Agent:

```
Agent(
  subagent_type: "general-purpose",
  run_in_background: true,
  prompt: "Create the data preprocessing module for our project.

    REFERENCE IMPLEMENTATION (from [repo_name], adapt this — don't rewrite from scratch):
    --- [repo data file path] ---
    [paste full content of repo's data code]
    ---

    OUR SPECS (what we need):
    [paste specs.md data module spec]

    EXISTING CODEBASE (if delta mode, import from here):
    [paste relevant existing data code if any]

    RULES:
    - Start from the reference implementation and ADAPT it
    - Change imports, paths, and dataset names to match our project
    - Keep the same data loading logic — don't reinvent working code
    - If the reference uses a different tokenizer/format, adapt the pipeline
    - Add any additional preprocessing our specs.md requires that the reference doesn't have
    - Match existing codebase style if in delta mode
    - Write to [file_path]
  "
)
```

If NO repo has usable data code → write from scratch using specs.md, following standard patterns.

#### 3A.3. Data code outputs

After 3A completes, the following should exist:
- Dataset classes / data loaders
- Tokenizer or text preprocessing pipeline
- Data config files (YAML/JSON for data paths, splits, preprocessing params)
- Download or data preparation script if needed

---

### Step 3B: Methodology — Novel Method (WRITE FROM SCRATCH)

**Goal:** Implement the core novel contribution — the model architecture, training logic, and inference/sampling code.

**This is the part that makes the paper novel. NEVER reuse methodology code from repos — write it from scratch per specs.md.**

#### 3B applies to both delta and greenfield modes:

**If delta mode:** For each METHOD module from the classification table:
- **EXISTING** → SKIP. Print: `"Skipping [file] — already exists in codebase."`
- **EXTEND** → Create new file that imports from existing code (don't duplicate):

```
Agent(
  subagent_type: "general-purpose",
  run_in_background: true,
  prompt: "Create a NEW method module that extends existing code.

    EXISTING CODE (do NOT reimplement, IMPORT from it):
    --- [existing file path] ---
    [paste full content of existing file from reuse_context]
    ---

    NEW MODULE TO CREATE:
    [paste specs.md module spec]

    RULES:
    - Import classes/functions from existing code, don't duplicate them
    - Match the existing coding style (indentation, naming, docstring format)
    - Match existing config patterns (Hydra/OmegaConf if used)
    - Use existing utilities (e.g., sample_categorical, get_sigmas, custom losses)
    - Write to [NEW file path], NEVER modify existing files
    - Include shape comments on every tensor operation
    - Write a docstring for each class with Input/Output specification
    - Add assert statements for input shape validation
  "
)
```

- **NEW** → Write from scratch per specs.md

**If greenfield:** Identify module dependency graph. Build in parallel where possible:

```
Dependency analysis:
  Module A (backbone wrapper) ← no deps → Agent 1 (background)
  Module B (novel module 1)   ← depends on A → wait for Agent 1, then Agent 2
  Module C (novel module 2)   ← no deps on B → Agent 3 (background, parallel with Agent 2)
  Module D (combiner)         ← depends on B + C → wait for both, then Agent 4
```

**File ownership rule:** Each Agent writes ONLY its assigned files. No two agents write to the same file.

#### Rules for methodology implementation:
- Match the **exact layer types, dimensions, and shapes** from specs.md
- Include shape comments on every tensor operation: `# (B, C, H, W) -> (B, D)`
- Write a docstring for each class with Input/Output specification
- Add `assert` statements for input shape validation in `forward()`
- Follow the residual connections and normalization exactly as specified
- Use the hyperparameters from specs.md as defaults, but make them configurable

#### Per method module:
```python
"""
[Module Name] — [purpose]

Solves gap: [which research gap]
Inspired by: [Paper A] + [Paper B]
"""

import torch
import torch.nn as nn

class [ClassName](nn.Module):
    """
    [Description]

    Input:
        x: (B, C, H, W) — [description]
        text: (B, D) — [description]

    Output:
        energy: (B, 1, H, W) — [description]
    """

    def __init__(self, ...):
        super().__init__()
        # Build layers exactly as specs.md

    def forward(self, x, text):
        # Shape assertions
        assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"
        B, C, H, W = x.shape

        # Layer-by-layer with shape comments
        x = self.conv1(x)  # (B, C, H, W) -> (B, 64, H, W)
        ...

        return energy  # (B, 1, H, W)
```

#### 3B also creates (NEW files only in delta mode):
- **Config files** (YAML) with all hyperparameters from specs.md
- **Loss functions** in `training/losses.py`
- **Training script** (`scripts/train.py`) with argparse
- **Inference/sampling script** (e.g., `scripts/sample.py`)

---

### Step 3C: Evaluation (REUSE-FIRST)

**Goal:** Create all evaluation metrics, benchmark scripts, eval runners, and comparison tools.

**Priority order** (same as 3A):
1. **REUSE** from top repos (adapt their eval code)
2. **EXTEND** from existing codebase (if delta mode)
3. **WRITE** from scratch (only for novel metrics that no repo has)

#### 3C.1. Scan repos for reusable evaluation code

For each repo in the reuse map that has eval code, launch an Explore agent:

```
Agent(
  subagent_type: "Explore",
  prompt: "Read the evaluation code in repos/[repo_name]/[eval_files].
    Extract:
    - Metric computation functions (what metrics, how computed)
    - Eval loop / runner scripts (how they iterate over test data)
    - Benchmark setup (datasets, splits, preprocessing for eval)
    - Output format (JSON, CSV, printed tables)
    - Any external model dependencies for eval (e.g., GPT-2 for perplexity, classifiers for accuracy)
    Return the full implementation with inline comments."
)
```

#### 3C.2. Classify each eval module

For each EVAL module in specs.md:
- **Standard metric** (perplexity, BLEU, ROUGE, accuracy, etc.) → almost certainly exists in a repo → **REUSE**
- **Novel metric** (our contribution, e.g., Semantic Smoothness Score) → **WRITE from scratch**
- **Benchmark wrapper** (loading test data, running eval loop) → likely exists in repo → **REUSE**
- **Comparison script** (running baselines + ours, producing tables) → **WRITE** (project-specific)

#### 3C.3. Adapt repo eval code to our project

For each standard eval module, launch an Agent:

```
Agent(
  subagent_type: "general-purpose",
  run_in_background: true,
  prompt: "Create the evaluation module for our project.

    REFERENCE IMPLEMENTATION (from [repo_name], adapt this):
    --- [repo eval file path] ---
    [paste full content of repo's eval code]
    ---

    OUR SPECS (what we need):
    [paste specs.md eval module spec]

    OUR DATA FORMAT (from Step 3A):
    [describe the data format our pipeline produces]

    OUR METHOD OUTPUT FORMAT (from Step 3B):
    [describe what our method outputs — e.g., edited texts, generated samples]

    RULES:
    - Start from the reference implementation and ADAPT it
    - Change imports, paths, and model names to match our project
    - Keep working metric implementations — don't reimplement standard metrics
    - Add novel metrics from specs.md that the reference doesn't have
    - Ensure eval script accepts our method's output format
    - Write to [file_path]
  "
)
```

For **novel metrics** (our contribution) → write from scratch per specs.md. These are unique to our paper and won't exist in any repo.

#### 3C.4. Eval code outputs

After 3C completes, the following should exist:
- Metric computation modules (standard + novel)
- Eval runner scripts (run evaluation on method outputs)
- Benchmark setup scripts (per dataset/task)
- Comparison / table generation scripts
- Eval config files (YAML)

---

### Step 3 summary

Print a build summary after all 3 phases complete:

```
## Code Implementation Summary

### Phase 3A — Data Preprocessing:
  REUSED from repo_A: data/dataloader.py, data/tokenizer.py
  WRITTEN from scratch: data/custom_transform.py
  [N] data files total

### Phase 3B — Methodology (Novel):
  SKIPPED (existing): models/backbone.py
  EXTENDED (from existing): sample_novel.py (extends sample_base.py)
  WRITTEN from scratch: models/novel_module.py, training/loss.py
  [M] method files total

### Phase 3C — Evaluation:
  REUSED from repo_A: evaluation/perplexity.py, evaluation/bleu.py
  REUSED from repo_B: evaluation/classifier_accuracy.py
  WRITTEN from scratch: evaluation/novel_metric.py (our SSS metric)
  [K] eval files total

Total: [N+M+K] files created
```

---

## Step 4: Set up baselines (PARALLEL — all baselines are independent)

All baseline runners are independent of each other. Launch **parallel Agent workers**, one per baseline:

```
For each baseline [A, B, C, ...]:
  → Agent(
      subagent_type: "general-purpose",
      run_in_background: true,
      prompt: "Create baseline runner for [Baseline X].
               Repo: [repo_path]. Read its eval scripts and README.
               Write to evaluation/baselines/run_[x].py.
               Follow the template below."
    )
```

**File ownership:** Each agent writes ONLY `evaluation/baselines/run_[its_baseline].py`. After all agents complete, create the unified `evaluation/compare_all.py`.

### 4a. Create baseline runner script

Create `evaluation/baselines/run_[baseline_name].py` for each baseline:

```python
"""
Baseline runner: [Paper Name]
Repo: [GitHub URL]
"""

import subprocess
import sys
import os

# Path to cloned baseline repo
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "../../../repos/[repo_name]")

def setup():
    """Install baseline dependencies."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    os.path.join(BASELINE_DIR, "requirements.txt")])

def run_eval(dataset_path, output_path, split="val"):
    """
    Run baseline evaluation.

    Args:
        dataset_path: Path to dataset
        output_path: Where to save results
        split: Dataset split (val/test)
    """
    cmd = [
        sys.executable,
        os.path.join(BASELINE_DIR, "[eval_script].py"),
        "--dataset", dataset_path,
        "--output", output_path,
        "--split", split,
        # Add baseline-specific flags from their README
    ]
    subprocess.run(cmd, check=True)

def parse_results(output_path):
    """Parse baseline results into common format."""
    # Read output and extract metrics
    results = {}
    # ... parse based on baseline output format
    return {
        "SR": results.get("success_rate", None),
        "SPL": results.get("spl", None),
        # Map to common metric names
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="results/[baseline_name]/")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    setup()
    run_eval(args.dataset, args.output, args.split)
    results = parse_results(args.output)
    print(results)
```

### 4b. Create unified comparison script

Create `evaluation/compare_all.py`:

```python
"""
Run all baselines and our method, produce comparison table.
"""

import json
from pathlib import Path

METHODS = {
    "ours": {"runner": "scripts/eval.py", "args": [...]},
    "[baseline_1]": {"runner": "evaluation/baselines/run_[b1].py", "args": [...]},
    "[baseline_2]": {"runner": "evaluation/baselines/run_[b2].py", "args": [...]},
}

def run_all(dataset_path, output_dir):
    results = {}
    for name, config in METHODS.items():
        # Run each method
        # Collect results in common format
        pass

    # Print comparison table
    # Save as JSON and LaTeX table
```

---

## Step 5: Design ablation study

From contribution.tex ablation table, create ablation configs and runner:

### 5a. Create ablation configs

**If existing codebase uses Hydra/OmegaConf:** Create ablation configs that extend existing configs using Hydra's `defaults:` mechanism rather than standalone YAML files. Reference the existing base config path.

**If greenfield or no Hydra:** Create standalone ablation configs.

For each ablation variant in the planned ablation table:

```yaml
# configs/model/ablation/no_[module_name].yaml
# Ablation: remove [Module Name]
# Expected effect: [what should degrade]

defaults:
  - ../[base_model].yaml

model:
  use_[module_name]: false
```

### 5b. Create ablation runner

Create `scripts/run_ablations.py`:

```python
"""
Run all ablation experiments.
Reads ablation configs from configs/model/ablation/
"""

import os
import subprocess
import sys
from pathlib import Path

ABLATION_DIR = Path("configs/model/ablation")
RESULTS_DIR = Path("results/ablations")

def get_ablation_configs():
    """Find all ablation config files."""
    return sorted(ABLATION_DIR.glob("*.yaml"))

def run_ablation(config_path, dataset_path, output_dir):
    """Run one ablation experiment."""
    name = config_path.stem  # e.g., "no_cross_attention"
    cmd = [
        sys.executable, "scripts/eval.py",
        "--config", str(config_path),
        "--dataset", dataset_path,
        "--output", str(output_dir / name),
    ]
    subprocess.run(cmd, check=True)

def run_all_ablations(dataset_path):
    """Run full model + all ablations."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run full model
    run_ablation(Path("configs/model/[base].yaml"), dataset_path, RESULTS_DIR)

    # 2. Run each ablation
    for config in get_ablation_configs():
        run_ablation(config, dataset_path, RESULTS_DIR)

    # 3. Collect and compare results
    # Print ablation table

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run_all_ablations(args.dataset)
```

---

## Step 6: Prepare download scripts (for remote compute)

**If existing codebase:** Before creating scripts, check for existing download/setup scripts and config files:
- If `requirements.txt` exists → **MERGE** new dependencies into it (append only new deps, don't overwrite or remove existing ones)
- If `setup_env.sh` or similar exists → **EXTEND** it with new setup steps rather than replacing
- If download scripts exist → check if they already cover needed models/datasets before creating new ones

**If greenfield:** Create all scripts from scratch.

### 6a. Model download script

Create `scripts/download_models.sh`:

```bash
#!/bin/bash
# Download all required models for [Project Name]
# Run this on remote compute before training/eval

set -e

MODEL_DIR="${MODEL_DIR:-models/pretrained}"
mkdir -p "$MODEL_DIR"

echo "=== Downloading models ==="

# --- Foundation models ---

# [Model 1: e.g., CLIP]
if [ ! -d "$MODEL_DIR/[model_name]" ]; then
    echo "Downloading [Model Name]..."
    # Use huggingface-cli if HuggingFace model
    huggingface-cli download [org/model_name] --local-dir "$MODEL_DIR/[model_name]"
    # OR use wget/curl for direct links
    # wget -P "$MODEL_DIR/" [direct_url]
    echo "[Model Name] downloaded."
else
    echo "[Model Name] already exists, skipping."
fi

# [Model 2: e.g., VLM like LLaVA]
if [ ! -d "$MODEL_DIR/[vlm_name]" ]; then
    echo "Downloading [VLM Name]..."
    huggingface-cli download [org/vlm_name] --local-dir "$MODEL_DIR/[vlm_name]"
    echo "[VLM Name] downloaded."
else
    echo "[VLM Name] already exists, skipping."
fi

# [Model 3: e.g., depth estimator]
# ... same pattern

# --- Baseline model checkpoints ---

# [Baseline A checkpoint]
if [ ! -f "$MODEL_DIR/baselines/[baseline_a].pth" ]; then
    echo "Downloading [Baseline A] checkpoint..."
    mkdir -p "$MODEL_DIR/baselines"
    wget -O "$MODEL_DIR/baselines/[baseline_a].pth" "[checkpoint_url]"
    # OR gdown for Google Drive links
    # gdown "[gdrive_id]" -O "$MODEL_DIR/baselines/[baseline_a].pth"
fi

echo ""
echo "=== All models downloaded ==="
echo "Models directory: $MODEL_DIR"
du -sh "$MODEL_DIR"/*
```

### 6b. Dataset download script

Create `scripts/download_datasets.sh`:

```bash
#!/bin/bash
# Download all required datasets for [Project Name]
# Run this on remote compute before training/eval

set -e

DATA_DIR="${DATA_DIR:-data/datasets}"
mkdir -p "$DATA_DIR"

echo "=== Downloading datasets ==="

# --- [Dataset 1: e.g., HM3D] ---
if [ ! -d "$DATA_DIR/[dataset_name]" ]; then
    echo "Downloading [Dataset Name]..."
    # For Habitat datasets:
    # python -m habitat_sim.utils.datasets_download --uids [dataset_uid] --data-path "$DATA_DIR"
    # For direct download:
    # wget -P "$DATA_DIR/" [url]
    # For torrent/academic download:
    echo "NOTE: [Dataset Name] requires manual download from [url]"
    echo "  1. Register at [registration_url]"
    echo "  2. Download [files] to $DATA_DIR/[dataset_name]/"
    echo "  3. Extract: tar -xzf [file].tar.gz -C $DATA_DIR/[dataset_name]/"
else
    echo "[Dataset Name] already exists, skipping."
fi

# --- [Dataset 2] ---
# ... same pattern

# --- Episode datasets (task-specific splits) ---
if [ ! -d "$DATA_DIR/episodes" ]; then
    echo "Downloading episode datasets..."
    mkdir -p "$DATA_DIR/episodes"
    # wget or git clone episode files
fi

echo ""
echo "=== All datasets downloaded ==="
echo "Data directory: $DATA_DIR"
du -sh "$DATA_DIR"/*
```

### 6c. Environment setup script

Create `scripts/setup_env.sh` (or extend existing if in delta mode):

```bash
#!/bin/bash
# Set up the compute environment for [Project Name]
# Run this ONCE on remote compute

set -e

echo "=== Setting up environment ==="

# 1. Create conda env
conda create -n [project_name] python=3.10 -y
conda activate [project_name]

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Install habitat-sim (if needed)
# conda install habitat-sim -c conda-forge -c aihabitat

# 5. Install baseline dependencies
for req in repos/*/requirements.txt; do
    echo "Installing deps from $req"
    pip install -r "$req" || echo "WARNING: Some deps from $req failed"
done

# 6. Download models
bash scripts/download_models.sh

# 7. Download datasets
bash scripts/download_datasets.sh

# 8. Verify installation
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
# import project modules
# import [project_name]
# print('Project modules OK')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate [project_name]"
echo "Run eval: python scripts/eval.py --config configs/default.yaml"
```

---

## Step 7: Write experiments.tex

Generate `[output_dir]/experiments.tex` — a single LaTeX file containing the full experimental design, setup, hyperparameters, result tables (to be filled), and ablation study.

This file is the experiment companion to `contribution.tex` from `/04-spec-novel`. It covers everything needed to run and report experiments.

```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage[margin=1in]{geometry}

\newcommand{\tbd}[1]{\textcolor{red}{#1}}  % Mark to-be-filled cells
\newcommand{\best}[1]{\textbf{#1}}          % Bold best result

\title{Experiments: [Project Name]}
\author{[TBD]}
\date{\today}

\begin{document}
\maketitle

% ============================================================
\section{Experimental Design}
% ============================================================

\subsection{Research Questions}

We design experiments to answer the following questions:

\begin{enumerate}
  \item \textbf{RQ1 (Overall):} Does [Project Name] outperform existing methods on [task]?
  \item \textbf{RQ2 (Component):} How much does each novel component contribute? (ablation)
  \item \textbf{RQ3 (Generalization):} Does [Project Name] generalize across [datasets/settings]?
  \item \textbf{RQ4 (Efficiency):} What is the computational overhead of [novel components]?
  % Add more RQs as needed per the gaps being addressed
\end{enumerate}

\subsection{Experimental Protocol}

For each experiment:
\begin{itemize}
  \item Run [N] episodes / seeds per configuration
  \item Report mean $\pm$ std across runs
  \item Use identical hardware for all methods (see Section~\ref{sec:hardware})
  \item Use official evaluation code from baseline repos where available
\end{itemize}

% ============================================================
\section{Setup}
\label{sec:setup}
% ============================================================

\subsection{Datasets}

\begin{table}[h]
\centering
\caption{Datasets used in experiments}
\begin{tabular}{llrrrl}
\toprule
\textbf{Dataset} & \textbf{Type} & \textbf{Scenes} & \textbf{Episodes} & \textbf{Split} & \textbf{Source} \\
\midrule
% Fill from specs.md dataset section
[Dataset 1] & [indoor/outdoor] & [N] & [N] & val & \href{[url]}{[link]} \\
[Dataset 2] & [type] & [N] & [N] & val & \href{[url]}{[link]} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Baselines}

\begin{table}[h]
\centering
\caption{Baseline methods}
\begin{tabular}{llll}
\toprule
\textbf{Method} & \textbf{Venue} & \textbf{Key Idea} & \textbf{Code} \\
\midrule
% Fill from baseline list
[Baseline A] & [Venue Year] & [1-line] & \href{[github_url]}{[org/repo]} \\
[Baseline B] & [Venue Year] & [1-line] & \href{[github_url]}{[org/repo]} \\
[Baseline C] & [Venue Year] & [1-line] & \href{[github_url]}{[org/repo]} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Evaluation Metrics}

\begin{table}[h]
\centering
\caption{Evaluation metrics}
\begin{tabular}{lll}
\toprule
\textbf{Metric} & \textbf{Formula} & \textbf{Higher/Lower Better} \\
\midrule
Success Rate (SR) & $\frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[d_i < \tau]$ & Higher \\
SPL & $\frac{1}{N}\sum_{i=1}^{N} s_i \frac{l_i}{\max(p_i, l_i)}$ & Higher \\
% Add novel metrics from specs.md
[Novel Metric] & $[formula]$ & [Higher/Lower] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Hardware}
\label{sec:hardware}

\begin{table}[h]
\centering
\caption{Hardware configuration}
\begin{tabular}{ll}
\toprule
\textbf{Component} & \textbf{Specification} \\
\midrule
GPU & \tbd{[e.g., NVIDIA A100 80GB]} \\
CPU & \tbd{[e.g., AMD EPYC 7763]} \\
RAM & \tbd{[e.g., 256GB]} \\
Storage & \tbd{[e.g., 2TB NVMe SSD]} \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{Hyperparameters}
\label{sec:hyperparams}
% ============================================================

\subsection{Model Hyperparameters}

\begin{table}[h]
\centering
\caption{Model architecture hyperparameters}
\begin{tabular}{llr}
\toprule
\textbf{Module} & \textbf{Parameter} & \textbf{Value} \\
\midrule
% Fill from specs.md module hyperparameters
\multirow{3}{*}{[Module 1]} & Hidden dimension & [value] \\
& Attention heads & [value] \\
& Residual blocks & [value] \\
\midrule
\multirow{2}{*}{[Module 2]} & [param] & [value] \\
& [param] & [value] \\
\midrule
\multirow{2}{*}{Backbone} & Model & [e.g., ViT-B/16] \\
& Frozen & [Yes/No] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Training Hyperparameters}

\begin{table}[h]
\centering
\caption{Training hyperparameters (if applicable)}
\begin{tabular}{lr}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Optimizer & [e.g., AdamW] \\
Learning rate & [e.g., 1e-4] \\
LR schedule & [e.g., cosine decay] \\
Warmup steps & [value] \\
Batch size & [value] \\
Training steps/epochs & [value] \\
Weight decay & [value] \\
Dropout & [value] \\
Gradient clipping & [value] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Inference Hyperparameters}

\begin{table}[h]
\centering
\caption{Inference/planning hyperparameters}
\begin{tabular}{lr}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
% Fill from specs.md
[e.g., VLM temperature] & [value] \\
[e.g., Planning frequency] & [every N steps] \\
[e.g., Map resolution] & [value] \\
[e.g., Max episode steps] & [value] \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{Main Results (RQ1)}
\label{sec:results}
% ============================================================

\subsection{[Dataset 1] Results}

\begin{table}[h]
\centering
\caption{Results on [Dataset 1] [split]. \best{Bold} = best. \tbd{Red} = to be filled.}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{SR $\uparrow$} & \textbf{SPL $\uparrow$} & \textbf{[Metric 3] $\uparrow$} & \textbf{[Metric 4] $\downarrow$} \\
\midrule
% Baselines — fill reported numbers from their papers
[Baseline A] & [reported] & [reported] & [reported] & [reported] \\
[Baseline B] & [reported] & [reported] & [reported] & [reported] \\
[Baseline C] & [reported] & [reported] & [reported] & [reported] \\
\midrule
\textbf{Ours} & \tbd{—} & \tbd{—} & \tbd{—} & \tbd{—} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{[Dataset 2] Results}

\begin{table}[h]
\centering
\caption{Results on [Dataset 2] [split].}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{SR $\uparrow$} & \textbf{SPL $\uparrow$} & \textbf{[Metric 3]} & \textbf{[Metric 4]} \\
\midrule
[Baseline A] & [reported] & [reported] & — & — \\
[Baseline B] & [reported] & [reported] & — & — \\
\midrule
\textbf{Ours} & \tbd{—} & \tbd{—} & \tbd{—} & \tbd{—} \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{Ablation Study (RQ2)}
\label{sec:ablation}
% ============================================================

\subsection{Component Ablation}

\begin{table}[h]
\centering
\caption{Ablation study on [Dataset 1] [split]. Each row removes one component.}
\begin{tabular}{lccccc}
\toprule
\textbf{Variant} & \textbf{[Module 1]} & \textbf{[Module 2]} & \textbf{SR $\uparrow$} & \textbf{SPL $\uparrow$} & \textbf{$\Delta$ SR} \\
\midrule
Full model & \checkmark & \checkmark & \tbd{—} & \tbd{—} & — \\
w/o [Module 1] & & \checkmark & \tbd{—} & \tbd{—} & \tbd{—} \\
w/o [Module 2] & \checkmark & & \tbd{—} & \tbd{—} & \tbd{—} \\
w/o both (baseline) & & & \tbd{—} & \tbd{—} & \tbd{—} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Hyperparameter Sensitivity}

\begin{table}[h]
\centering
\caption{Sensitivity to [key hyperparameter, e.g., hidden dimension].}
\begin{tabular}{lcc}
\toprule
\textbf{[Param] value} & \textbf{SR $\uparrow$} & \textbf{SPL $\uparrow$} \\
\midrule
[value 1] & \tbd{—} & \tbd{—} \\
[value 2] & \tbd{—} & \tbd{—} \\
[value 3 (default)] & \tbd{—} & \tbd{—} \\
[value 4] & \tbd{—} & \tbd{—} \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{Generalization (RQ3)}
\label{sec:generalization}
% ============================================================

\begin{table}[h]
\centering
\caption{Cross-dataset generalization. Train on [X], test on [Y].}
\begin{tabular}{llcc}
\toprule
\textbf{Method} & \textbf{Test Set} & \textbf{SR $\uparrow$} & \textbf{SPL $\uparrow$} \\
\midrule
[Baseline A] & [Dataset Y] & [reported] & [reported] \\
\textbf{Ours} & [Dataset Y] & \tbd{—} & \tbd{—} \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{Efficiency Analysis (RQ4)}
\label{sec:efficiency}
% ============================================================

\begin{table}[h]
\centering
\caption{Computational cost comparison.}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Params (M)} & \textbf{FLOPs (G)} & \textbf{Inference (ms/step)} & \textbf{GPU Mem (GB)} \\
\midrule
[Baseline A] & [value] & [value] & [value] & [value] \\
[Baseline B] & [value] & [value] & [value] & [value] \\
\midrule
\textbf{Ours} & \tbd{—} & \tbd{—} & \tbd{—} & \tbd{—} \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{Run Commands Reference}
\label{sec:commands}
% ============================================================

All experiments can be reproduced with:

\begin{verbatim}
# Setup
bash scripts/setup_env.sh

# Main results
python scripts/eval.py --config configs/default.yaml \
    --dataset data/datasets/[name] --output results/main/

# Baselines
python evaluation/compare_all.py --dataset data/datasets/[name] \
    --output results/baselines/

# Ablations
python scripts/run_ablations.py --dataset data/datasets/[name] \
    --output results/ablations/

# Efficiency profiling
python scripts/eval.py --config configs/default.yaml \
    --dataset data/datasets/[name] --profile --output results/efficiency/
\end{verbatim}

\end{document}
```

### Rules for experiments.tex:
- Must compile cleanly with `pdflatex`
- Use `booktabs` for all tables
- Use `\tbd{—}` (red) for all cells that need to be filled after running experiments
- Use `\best{}` (bold) for best results once filled
- Include reported baseline numbers from their papers (not TBD)
- Every table must have a caption
- Metrics must have $\uparrow$ or $\downarrow$ to indicate direction
- The Run Commands section must match the actual scripts created in this skill
- Hyperparameter tables must match configs/ YAML files exactly
- Ablation table rows must match ablation configs in configs/model/ablation/

---

## Step 8: Create README.md and prepare for git push

### 8a. Create README.md — ALWAYS create for the project directory

**ALWAYS** create a `README.md` inside the project output directory (e.g., `lsme/README.md`), even in delta mode. This README documents what was built, how to use it, and what it depends on. If a top-level repo README already exists, do NOT touch it — create the README inside the new project subdirectory.

The README must be **auto-generated from the actual build** — fill in real file paths, real module names, real config paths, and real commands based on what was actually created in Steps 2-7. Do NOT use placeholders.

```markdown
# [Project Name]

[One-line description from specs.md]

## Overview

[2-3 sentence summary of what this project does, what method it implements, and what problem it solves. Pull from specs.md and contribution.tex.]

## Installation

```bash
# Create environment
conda create -n [project_name] python=3.10 -y && conda activate [project_name]

# Install dependencies
pip install -r requirements.txt

# Download models (run on compute node)
bash scripts/download_models.sh

# Download datasets
bash scripts/download_datasets.sh
```

OR full setup in one command:
```bash
bash scripts/setup_env.sh
```

## Project Structure

```
[project_dir]/
├── [actual tree generated from the build — show every file created]
├── [group by: data/, models/, evaluation/, scripts/, configs/]
└── README.md
```

### Code Organization (3 phases)

| Phase | Directory | Source | Description |
|-------|-----------|--------|-------------|
| Data Preprocessing | `data/` or `[data_module]/` | Adapted from [repo_name] | [what it does] |
| Methodology (Novel) | `models/`, `[main_module].py` | Written from scratch | [the core contribution] |
| Evaluation | `evaluation/` | Adapted from [repo_name] | [metrics and benchmarks] |

## Quick Start

### Train
```bash
[actual training command with real config paths]
```

### Run method (inference/editing/sampling)
```bash
[actual inference command — e.g., python -m lsme.scripts.run_lsme --checkpoint ... --config ...]
```

### Evaluate
```bash
[actual eval command — e.g., python -m lsme.scripts.run_eval --results_file ...]
```

### Run baselines
```bash
[actual baseline comparison command]
```

### Run ablations
```bash
[actual ablation command]
```

## Dependencies

### Bundled (included in this directory)
| Module | Source | Why bundled |
|--------|--------|-------------|
| [e.g., mmdit_latent/] | [original location or GitHub URL] | [core dependency, modified/extended] |

### External (pip install)
| Package | Version | Purpose |
|---------|---------|---------|
| [e.g., torch] | >=2.0 | [deep learning framework] |
| [e.g., transformers] | >=4.30 | [HuggingFace models for eval] |

### Models (download separately)
| Model | Size | Source | Purpose |
|-------|------|--------|---------|
| [e.g., gpt2] | [size] | HuggingFace | [perplexity evaluation] |
| [e.g., distilbert-sst2] | [size] | HuggingFace | [sentiment classification] |

## Datasets

| Dataset | Task | Size | Download |
|---------|------|------|----------|
| [e.g., Yelp Polarity] | [sentiment editing] | [size] | `datasets.load_dataset("yelp_polarity")` |
| [e.g., Amazon Reviews] | [topic transfer] | [size] | `datasets.load_dataset("amazon_polarity")` |

## Baselines

| Method | Paper | Repo | Runner |
|--------|-------|------|--------|
| [Baseline A] | [venue year] | [GitHub URL] | `[actual runner script path]` |
| [Baseline B] | [venue year] | [GitHub URL] | `[actual runner script path]` |

## Ablation Study

| Variant | Config | What changes |
|---------|--------|--------------|
| Full model | `[actual config path]` | All components |
| w/o [Component] | `[actual ablation config path]` | Removes [component] |

## Configs

All configs are in `[configs_dir]/`:
| Config | Purpose |
|--------|---------|
| [e.g., lsme_yelp.yaml] | [Yelp sentiment editing experiment] |
| [e.g., lsme_amazon.yaml] | [Amazon topic transfer experiment] |
| [e.g., eval.yaml] | [Evaluation suite settings] |

## Hardware Requirements

- GPU: [actual estimate based on model size — e.g., 1x A100 40GB]
- RAM: [estimate]
- Disk: [total for models + datasets + checkpoints]

## Citation

If you use this code, please cite:
```bibtex
@article{[author][year][keyword],
  title={[paper title from contribution.tex]},
  author={[authors]},
  year={[year]},
  journal={[venue]}
}
```

## Reused Code Attribution

This project adapts code from the following repositories:
| Component | Source Repo | License | What was adapted |
|-----------|------------|---------|------------------|
| [e.g., Data loading] | [repo_name (GitHub URL)] | [MIT/Apache/etc.] | [data preprocessing pipeline] |
| [e.g., Perplexity eval] | [repo_name (GitHub URL)] | [license] | [GPT-2 perplexity computation] |
```

### README Rules:
- **No placeholders** — every path, command, and module name must be real (from the actual build)
- **Attribution required** — if code was adapted from repos (Step 3A/3C), list them in "Reused Code Attribution"
- **Self-contained instructions** — a new person should be able to reproduce everything from just this README
- **Matches experiments.tex** — the run commands here must match the commands in experiments.tex Section "Run Commands Reference"
- Write the README as the LAST file, after everything else is built, so all paths are known

### 8b. Create .gitignore (greenfield) or verify existing (delta mode)

**If delta mode and .gitignore exists:** Verify it covers model/dataset/result exclusions. If not, suggest additions.

**If greenfield or no .gitignore:**

```
# Models (download via script, don't commit)
models/pretrained/
*.pth
*.pt
*.bin
*.safetensors
*.onnx

# Datasets (download via script, don't commit)
data/datasets/

# Results (generated, don't commit)
results/

# Cloned baseline repos (download via script)
repos/

# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/

# Environment
.env
*.log
wandb/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

### 8c. Handle requirements.txt

**If delta mode and requirements.txt exists:** MERGE new dependencies — read the existing file, identify which deps from specs.md are not yet listed, append only the new ones. Print what was added:
> "Merged requirements.txt: added [dep1], [dep2], [dep3] (existing deps untouched)"

**If greenfield:** Create from specs.md dependencies + what the code actually imports:

```
torch>=2.0
torchvision>=0.15
numpy
scipy
scikit-image
pyyaml
tqdm
huggingface-hub
transformers
# Add project-specific deps from specs.md
```

### 8d. Pre-commit quality gate

Run ALL quality checks before telling the user to push. If ANY required check fails, fix the issue before proceeding.

**Required checks (must ALL pass):**

1. **Import check:** For delta mode, use the existing package name: `python -c "from [existing_package].models import *"`. For greenfield: `python -c "from [project_name].models import *"`. Also verify new modules import correctly from existing code.
2. **Unit tests:** `python -m pytest tests/ -x` — must pass (if tests exist)
3. **No syntax errors:** `python -m py_compile [every NEW .py file]` — must all pass
4. **No secrets:** Scan all NEW files for API keys, tokens, passwords (patterns: `sk-`, `AKIA`, `password=`, `token=`, `.env`)
5. **No large files:** `find . -size +50M -not -path './.git/*' -not -path './repos/*' -not -path './models/pretrained/*' -not -path './data/datasets/*'`
6. **.gitignore works:** `git status` should NOT show model/dataset/result files

**Delta-mode additional checks:**

7. **No overwrites:** Verify that NO existing files were modified during the build (compare timestamps or use git diff)
8. **Import correctness:** Verify new code correctly imports from existing codebase (no import errors, no circular imports)

**Recommended checks (warn if fail):**

9. **Type hints:** `python -m mypy models/ --ignore-missing-imports` (if mypy installed)
10. **Scripts executable:** `chmod +x scripts/*.sh`
11. **README complete:** Verify all download commands and paths are filled in
12. **Config consistency:** YAML configs match module default hyperparameters

Print quality gate summary:
```
## Quality Gate: [PASS / FAIL]
  ✓ Import check passed
  ✓ 5/5 unit tests passed
  ✓ No syntax errors (12 files)
  ✓ No secrets found
  ✓ No large files
  ✓ .gitignore working
  ✓ No existing files modified (delta mode)
  ✓ New imports resolve correctly (delta mode)
  ⚠ mypy: 3 type warnings (non-blocking)
```

---

## Step 9: Summary output

### If delta mode:

```
## Build Complete

**Project:** [name]
**Mode:** Delta build on existing codebase: [codebase_path]

### Phase 3A — Data Preprocessing:
  REUSED from [repo]: [list of adapted data files]
  WRITTEN: [list of new data files, if any]

### Phase 3B — Methodology (Novel):
  SKIPPED (existing): [list of existing files]
  EXTENDED: [list of files that extend existing code]
  WRITTEN: [list of new method files]

### Phase 3C — Evaluation:
  REUSED from [repo]: [list of adapted eval files]
  WRITTEN: [list of novel metric files]

### Dependencies resolved:
  BUNDLED: [list of bundled codebases]
  INSTALLED: [list of pip packages added]
  CLONED: [list of repos cloned]

### Extended (merged):
- [X] requirements.txt (+N new deps)
- [X] [new config files]

### README: [project_dir]/README.md

### Quality Gate: PASS
  [summary from Step 8d]

### Next steps:
```bash
git add [list of NEW files only]
git commit -m "Add [feature]: [N] new modules for [project name]"
git push
```

### On remote compute:
```bash
git pull
pip install -r requirements.txt  # picks up new deps
python [new_eval_script] --config [new_config]
```
```

### If greenfield:

```
## Build Complete — Ready to Git Push

**Project:** [name]
**Location:** [project_dir]
**Mode:** Greenfield build

### Phase 3A — Data Preprocessing:
  REUSED from [repo]: [list]
  WRITTEN: [list]

### Phase 3B — Methodology (Novel):
  WRITTEN: [list — all new in greenfield]

### Phase 3C — Evaluation:
  REUSED from [repo]: [list]
  WRITTEN: [list]

### Also created:
- [X] [N] baseline runners
- [X] [N] ablation configs
- [X] experiments.tex
- [X] Download scripts (models, datasets, env setup)
- [X] README.md ([project_dir]/README.md)
- [X] .gitignore
- [X] requirements.txt

### Dependencies resolved:
  INSTALLED: [list of pip packages]
  CLONED: [list of repos]

### Git push:
```bash
cd [project_dir]
git init
git add -A
git commit -m "Initial implementation: [project name]"
git remote add origin [your_repo_url]
git push -u origin main
```

### On remote compute:
```bash
git clone [your_repo_url]
cd [project_name]
bash scripts/setup_env.sh   # sets up everything
# See [project_dir]/README.md for all run commands
```

### Disk space needed:
- Models: ~[X] GB
- Datasets: ~[X] GB
- Total: ~[X] GB
```
