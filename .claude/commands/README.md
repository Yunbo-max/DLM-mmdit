# Paper Pipeline — Claude Code Skills

An 8-step autonomous research pipeline for Claude Code (+ 1 optional contrarian track). Goes from a single seed paper to a submitted, reviewed, and improved conference paper.

**Key features:** Pre-flight validation, multi-agent parallel execution, frozen scope control, experiment ID traceability, pre-commit quality gates, actionable error messages, optional counter-think for assumption-challenging innovation.

## Pipeline Overview

```
  Seed Paper
      │
      ▼
┌─────────────────────┐
│ 01-literature-search │  Find top 10 related papers with GitHub links
│  [pre-flight]        │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 02-research-gap      │  Clone repos, analyze code, find gaps
│  [pre-flight]        │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 03-gap-solve         │  Reason per gap, cross-domain top 5 per direction
│  [pre-flight]        │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 04-spec-novel        │◄──────────────────────────────────┐
│  [pre-flight]        │  Parallel agents per direction     │
│  [parallel agents]   │  Score → Rank → Human decides     │
│  [frozen scope]      │  Freeze scope → Synthesize        │
│                      │                                    │
│  ┌─────────────────┐ │  OPTIONAL parallel track:          │
│  │ 04b-counter-    │ │  Challenge assumptions,            │
│  │ think [optional]│ │  cross-domain search,              │
│  │ [parallel]      │ │  first-principles proposals        │
│  └────────┬────────┘ │  → merges into ranking table       │
│           └──────────│──────────────────┐                 │
└──────────┬──────────┘                  │                 │
           │◄────────────────────────────┘                 │
           ▼                                               │
┌─────────────────────┐                                    │
│ 05-build-code        │◄─────────────────────┐            │
│  [pre-flight]        │  Parallel modules +   │            │
│  [parallel agents]   │  baseline runners     │            │
│  [quality gate]      │                       │            │
└──────────┬──────────┘                       │            │
           ▼                                  │            │
┌─────────────────────┐                       │            │
│ 06-run-experiments   │  [pre-flight]        │            │
│  [parallel groups]   │  Parallel baselines,  │            │
│  [experiment IDs]    │  ablations by group   │            │
└──────────┬──────────┘                       │            │
           ▼                                  │            │
┌─────────────────────┐                       │            │
│ 07-write-paper       │  [pre-flight]        │            │
│  [parallel sections] │  4 section agents    │            │
│  Assemble from tex   │  → merge main.tex    │            │
└──────────┬──────────┘                       │            │
           ▼                                  │            │
┌─────────────────────┐                       │            │
│ 08-review-loop       │  [pre-flight]        │            │
│  Submit + parse      │                      │            │
│  feedback            │                      │            │
└──────────┬──────────┘                       │            │
           │                                  │            │
     ┌─────┼─────┐                            │            │
     ▼     ▼     ▼                            │            │
   ┌───┐ ┌───┐ ┌───┐                         │            │
   │ C │ │ B │ │ A │                         │            │
   └─┬─┘ └─┬─┘ └─┬─┘                         │            │
     │     │     │                            │            │
     │     │     └── Rejected (< 5/10) ───────┼────────────┘
     │     │         redesign method           │
     │     │                                   │
     │     └──── Borderline (5-7/10) ──────────┘
     │               improve experiments
     │
     └────── Accepted (> 7/10) ──► Conference Submission
```

## Quick Start

### Full pipeline from a seed paper:

```bash
# Step 1: Find related work
/01-literature-search https://arxiv.org/abs/XXXX.XXXXX direction: your exploration angle

# Step 2: Analyze gaps
/02-research-gap

# Step 3: Find solutions to gaps
/03-gap-solve

# Step 4: Design novel method
/04-spec-novel
# Optional: run in parallel with 04 for contrarian ideas
# /04b-counter-think frustration: "your frustration here"

# Step 5: Build the code
/05-build-code

# Step 6: Run experiments
/06-run-experiments

# Step 7: Write the paper
/07-write-paper CVPR

# Step 8: Review and improve
/08-review-loop
```

### Or run individual skills:

```bash
# Topic-based literature search (no seed paper)
/01-literature-search VLM navigation, semantic mapping 2024

# Check experiment results and update tex
/06-run-experiments update

# Write for a specific conference
/07-write-paper ICLR 2026

# Parse existing review feedback
/08-review-loop parse https://paperreview.ai/result/xxx
```

## Skills Reference

### 01-literature-search

**Two modes:**
- **Mode A (topic-based):** `/01-literature-search VLM navigation, terrain planning 2024`
- **Mode B (paper-seeded):** `/01-literature-search https://arxiv.org/abs/XXXX.XXXXX`

Mode B reads the paper, asks for your exploration direction, then searches across ML conferences, robotics venues, and preprints.

**Pre-flight:** Validates seed paper URL is reachable, direction is provided, output path exists.

**OpenReview reviews:** For papers on OpenReview (ICLR, NeurIPS, etc.), also fetches reviewer scores, key weaknesses, and suggested improvements. Reviewer criticisms = validated gaps for later steps.

**Output:** `docs/literature.md` — Seed paper analysis + top 10 with code + top 10 without code + reviewer feedback

---

### 02-research-gap

**Input:** `docs/literature.md` from step 01

**Pre-flight:** Validates literature.md exists and has paper links.

Analyzes **both** Top 10 lists from step 01: clones repos for code papers, reads PDFs for no-code papers. Builds comparison matrices across all 20 papers:
- Methods (architecture, representation, planning, foundation model)
- Tasks (ObjectNav, VLN, indoor/outdoor, multi-agent)
- Datasets (HM3D, MP3D, etc.)
- Evaluation (metrics, baselines, real-robot)

**Output:** `docs/research_gap.tex` — Gap matrices + identified gaps per dimension

---

### 03-gap-solve

**Input:** `docs/research_gap.tex` from step 02

**Pre-flight:** Validates research_gap.tex exists and has identified gaps.

For each gap:
1. Reasons about the core problem and sub-problems
2. Thinks laterally — what other fields solve similar problems?
3. Generates 3-5 solution directions with cross-domain search angles
4. Searches all venues + **OpenReview reviewer comments** for papers per direction
   - Reviewer weaknesses = validated gaps
   - Rejected papers = learn what NOT to do
   - "Missing baseline" complaints = methods the community expects
5. Selects top 5 papers per direction (GitHub links required)

**Output:** `docs/gap_solutions.md` — Gaps + directions + top 5 papers each

---

### 04-spec-novel

**Input:** `docs/gap_solutions.md` + `docs/research_gap.tex`

Clones ALL repos from gap solutions (top 5 per direction x all directions x all gaps). For EACH direction, **parallel Agent workers** deeply study 5 papers and design a novel method.

**Pipeline:**
1. **Pre-flight validation** — checks gap_solutions.md + research_gap.tex exist and have content
2. **Clone ALL repos** — every paper from every direction across all gaps
3. **Parallel study** — 1 Agent per direction, each reads 5 repos independently
4. **Score each direction** — Novelty (0-1), Feasibility (0-1), Confidence (0-1)
5. **Rank all candidates** — single table sorted by average score
6. **Suggest + human decides** — gives recommendation with reasoning, user picks:
   - **Single** — one candidate as the full method
   - **Combined** — merge candidates into unified pipeline (only if they complement)
   - **Independent** — develop each separately
7. **Freeze scope** — selected candidates + mode locked, no changes without user approval
8. **Integrate** — if combined mode, design how candidates connect; otherwise skip
9. **Write specs.md + contribution.tex**

- **`docs/novel/specs.md`** — Full implementation spec:
  - Folder structure with every Python file
  - Module architecture (input/output shapes, layer-by-layer, residual connections)
  - ASCII flowcharts (pipeline, modules, training, eval)
  - Task/env spec (observation/action spaces)
  - Evaluation plan with baseline run commands
  - Implementation order checklist

- **`docs/novel/contribution.tex`** — Contribution paper:
  - Contributions mapped to gaps
  - Method equations matching specs
  - Feature comparison table
  - Planned ablations

---

### 04b-counter-think (optional)

**Input:** `docs/research_gap.tex` + `docs/literature.md` (+ optionally `docs/gap_solutions.md`)

**Optional parallel track** for `/04-spec-novel`. Instead of filling gaps from papers, challenges the field's fundamental assumptions and searches cross-domain for paradigm-breaking ideas.

**Why:** The standard pipeline (read papers → find gaps → fill gaps) produces solid incremental work. But Transformer didn't come from "RNN's gap" — it came from questioning "do we need recurrence at all?" This skill provides that second perspective.

**Pipeline:**
1. **Ask the human's frustration** — what personally annoys you about this field? (most valuable input)
2. **Extract assumptions** (parallel: 3 agents) — architecture, problem formulation, evaluation assumptions
3. **Cross-domain search** (parallel: 1 agent per assumption) — neuroscience, physics, biology, economics, control theory
4. **Generate contrarian proposals** — combine frustration + weak assumptions + cross-domain concepts
5. **Score differently** — Paradigm Shift (not Novelty), Testability, Surprise Factor
6. **Present two paths** — gap-filling (safe) vs assumption-challenging (risky but memorable)
7. **Human decides** — selected contrarian proposals feed into 04-spec-novel's frozen scope

**Output:** `docs/novel/counter_proposals.md` — Assumption registry + cross-domain insights + contrarian proposals

```bash
# Run standalone
/04b-counter-think

# Run with your frustration as seed
/04b-counter-think frustration: "why does every method need an explicit map?"
```

---

### 05-build-code

**Input:** `docs/novel/specs.md` + `docs/novel/contribution.tex`

**Pre-flight:** Validates specs.md has required sections (Project Structure, Module Specifications, Evaluation).

**Parallel:** Independent modules + baseline runners built by separate Agent workers.

Builds everything needed to run on remote compute:
- Core model modules (from specs)
- Baseline runner scripts
- Ablation configs
- `scripts/download_models.sh` — Download LLMs/VLMs/checkpoints
- `scripts/download_datasets.sh` — Download datasets
- `scripts/setup_env.sh` — Full environment setup
- `experiments.tex` — Experiment design, hyperparameters, result tables (TBD cells in red)
- `README.md` — Quick start for remote compute
- `.gitignore` — Excludes models/data/results

**Quality gate:** Import check, unit tests, syntax validation, secret scan — must ALL pass before push.

**Ready to:** `git push` and run on remote compute

---

### 06-run-experiments

**Pre-flight:** Validates experiments.tex and code exist.

**Two execution modes:**
- `/06-run-experiments auto` — Run all experiments automatically, update plan + tex after each
- `/06-run-experiments manual` — Generate dated plan with commands, human runs them, then update

**Parallel (auto mode):** Independent experiments run simultaneously in groups (baselines → main → ablations).

**Experiment IDs:** Every experiment gets a unique ID (`EXP-B01`, `EXP-M01`, `EXP-A01`). Results stored at `results/[EXP-ID]/` for reliable auto-matching.

Creates a dated experiment plan (`docs/novel/experiment_plan.md`) tracking:
- ID, date, command, status, result per experiment
- Progress summary (completed / pending / failed)

**Update command** (for manual mode):
`/06-run-experiments update results/EXP-B01/results.json` — auto-matches by ID, updates plan + experiments.tex

---

### 07-write-paper

**Input:** ALL LaTeX/md files from steps 01-06 (assembles, does NOT write from scratch)

**Pre-flight:** Validates ALL 5 required source files exist (contribution.tex, experiments.tex, specs.md, literature.md, research_gap.tex). Lists every missing file with which command to run.

```bash
/07-write-paper CVPR
/07-write-paper ICLR 2026
```

**Parallel:** 4 Agent workers write sections simultaneously (intro+related, method, experiments, discussion), each to a separate file. Then assembled into main.tex.

Downloads conference LaTeX template, then **copies and reformats** content from:
- `contribution.tex` → title, contributions, method + equations, related work
- `experiments.tex` → all tables (setup, results, ablations, efficiency)
- `experiment_plan.md` → filled result values
- `specs.md` → architecture details
- `literature.md` → citations for references.bib
- `research_gap.tex` → introduction motivation

Each section has a clear source map — nothing is generated from scratch. Checks page budget.

---

### 08-review-loop

**Pre-flight:** Validates paper/main.tex exists and compiles, checks for style files and references.bib.

Submit to paperreview.ai, parse feedback, classify into **3 cases**:

| Case | Score | Problem | Loop back to | What changes |
|------|-------|---------|-------------|-------------|
| **A — Rejected** | < 5/10 | Method not novel / flawed | `/04-spec-novel` | Redesign architecture, new specs + contribution.tex |
| **B — Borderline** | 5-7/10 | Weak experiments | `/05-build-code` | Add baselines, ablations, datasets, re-run experiments |
| **C — Accepted** | > 7/10 | Minor writing issues | `/07-write-paper` | Fix typos, clarify text, add citations |

Each case cascades through the remaining pipeline steps and resubmits.
Tracks score progression across rounds in `paper/reviews/progression.md`.

## Pipeline Features

### Pre-flight Validation (all steps)

Every skill runs a **Step 0: Pre-flight validation** before doing any work. Checks that all required input files exist and are well-formed. If anything is missing, the skill STOPS with an actionable error message telling you exactly which command to run:

> "Missing: docs/research_gap.tex. Run `/02-research-gap` first to generate it."

### Parallel Execution (steps 04-07)

Steps 04-07 use **multi-agent parallel execution** to speed up independent tasks. Each parallel worker gets exclusive file ownership to avoid conflicts.

| Step | What runs in parallel | Workers |
|------|----------------------|---------|
| **04-spec-novel** | Per-direction paper study + method design | 1 agent per direction |
| **04b-counter-think** (optional) | Assumption extraction + cross-domain search | 3 assumption agents + 1 per LOW-certainty assumption |
| **05-build-code** | Independent module implementation + baseline runners | 1 agent per module/baseline |
| **06-run-experiments** | Independent experiments (baselines, ablations) | 1 agent per experiment group |
| **07-write-paper** | Paper sections (intro, method, experiments, discussion) | 4 section agents |

Workers use structured prompts: TASK / EXPECTED OUTCOME / MUST DO / MUST NOT DO / CONTEXT.

### Frozen Scope (step 04)

After the user picks which directions to develop, scope is **frozen** — no changes without explicit user approval. Prevents scope creep during implementation.

### Experiment IDs (step 06)

Every experiment gets a unique ID (`EXP-B01`, `EXP-M01`, `EXP-A01`, etc.). Result files reference their ID for reliable auto-matching when updating tables.

### Quality Gate (step 05)

Before git push, runs: import checks, unit tests, syntax validation, secret scanning, large file detection. Must ALL pass before proceeding.

## File Map

All intermediate outputs with default paths:

```
project/
├── docs/
│   ├── literature.md          ← 01-literature-search
│   ├── research_gap.tex       ← 02-research-gap
│   ├── gap_solutions.md       ← 03-gap-solve
│   └── novel/
│       ├── specs.md           ← 04-spec-novel
│       ├── contribution.tex   ← 04-spec-novel
│       ├── counter_proposals.md ← 04b-counter-think (optional)
│       └── experiments.tex    ← 05-build-code
├── repos/                     ← 02-research-gap (cloned repos)
├── results/                   ← 06-run-experiments
├── paper/
│   ├── main.tex               ← 07-write-paper
│   ├── references.bib         ← 07-write-paper
│   ├── figures/               ← 07-write-paper (placeholders)
│   ├── supplementary/         ← 07-write-paper
│   └── reviews/
│       ├── round1.md          ← 08-review-loop
│       └── round1_changes.md  ← 08-review-loop
├── configs/                   ← 05-build-code
├── models/                    ← 05-build-code
├── scripts/
│   ├── download_models.sh     ← 05-build-code
│   ├── download_datasets.sh   ← 05-build-code
│   ├── setup_env.sh           ← 05-build-code
│   ├── run_remote.sh          ← 06-run-experiments
│   └── run_ablations.py       ← 05-build-code
└── evaluation/
    ├── baselines/             ← 05-build-code
    └── compare_all.py         ← 05-build-code
```

## Requirements

- [Claude Code](https://claude.com/claude-code) CLI
- Git
- Python 3.10+
- (Optional) GPU for local experiments
- (Optional) API keys for LLM/VLM inference

## Installation

Copy the `.claude/commands/` directory into your project:

```bash
mkdir -p .claude/commands
cp path/to/01-literature-search.md .claude/commands/
cp path/to/02-research-gap.md .claude/commands/
# ... etc
```

Or clone this repo and symlink:

```bash
ln -s path/to/paper-pipeline/.claude/commands .claude/commands
```

Then use any skill with `/01-literature-search`, `/02-research-gap`, etc. in Claude Code.
