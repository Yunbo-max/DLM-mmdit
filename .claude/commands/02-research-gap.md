---
name: 02-research-gap
description: Clone GitHub repos from literature search results, read and analyze all codebases, and produce a LaTeX file identifying research gaps across methods, tasks, datasets, and evaluation.
---

Analyze repos from a literature search and produce a research gap analysis as a .tex file.

## User input: $ARGUMENTS

Parse the user input to extract:
1. **Literature file** — path to the literature search output (e.g., `docs/literature.md`). If not provided, default to `docs/literature.md`
2. **Output .tex file** — path for the LaTeX output (e.g., `docs/research_gap.tex`). If not provided, default to `docs/research_gap.tex`
3. **Clone directory** — where to clone repos (e.g., `repos/`). If not provided, default to `repos/`

Example invocations:
- `/research-gap` → reads `docs/literature.md`, clones to `repos/`, outputs `docs/research_gap.tex`
- `/research-gap docs/my_papers.md`
- `/research-gap docs/literature.md docs/gap_analysis.tex repos/cloned/`

---

## Step 0: Pre-flight validation

Before cloning repos or analyzing code, validate ALL required inputs:

1. **Literature file exists?** Check `docs/literature.md` (or user-specified path). If missing → STOP:
   > "Missing: docs/literature.md. Run `/01-literature-search` first to generate it."
2. **Literature file has GitHub links?** Scan for `github.com` URLs. If zero found → STOP:
   > "No GitHub links found in [file]. Re-run `/01-literature-search` — top 10 papers require GitHub repos."
3. **Clone directory writable?** Check parent exists. If not → STOP:
   > "Clone directory parent does not exist. Create it with: `mkdir -p repos/`"
4. **Disk space?** Warn if many repos: "About to clone [N] repos. Ensure sufficient disk space (~[N*500]MB estimated)."

Only proceed to Step 1 after ALL checks pass.

---

## Step 1: Parse the literature file

Read the literature file (from `/01-literature-search` Mode B output). Extract BOTH top 10 lists:

### 1a. Top 10 with Code
Extract the **Top 10 Most Related Papers (with Code)** table. For each paper, collect:
- Paper title
- GitHub URL (the `[org/repo](url)` link)
- Venue + year
- "Why Related" description

### 1b. Top 10 without Code
Extract the **Top 10 Most Related Papers (without Code)** table. For each paper, collect:
- Paper title
- Paper URL (arXiv, DOI, or conference page)
- Venue + year
- "Key Method/Result" description
- "Why Related" description

If the literature file uses Mode A format, extract ALL papers — those with GitHub links go to the code list, those without go to the no-code list.

If no literature file exists, STOP and tell the user to run `/01-literature-search` first.

---

## Step 2a: Clone repos (code papers)

```bash
mkdir -p [clone_directory]
```

For EACH GitHub repo from Step 1a:
- Clone with shallow depth to save space: `git clone --depth 1 [github_url] [clone_directory]/[repo_name]`
- If clone fails (private repo, deleted, etc.), log the failure and skip
- Print a summary: how many repos cloned successfully vs failed

---

## Step 2b: Read no-code papers from PDF/abstract

For EACH paper from Step 1b (no GitHub repo), use **WebFetch** to read the paper's arXiv page, PDF, or conference page. Extract:

1. **Abstract** — full abstract text
2. **Method description** — architecture, approach, key components (from the paper text)
3. **Reported results** — exact numbers from their experiments (datasets, metrics, scores)
4. **Baselines they compare to** — which other methods they benchmark against
5. **Datasets used** — which datasets, which splits
6. **Limitations / future work** — what they acknowledge is missing

Build the same structured summary as code papers, but mark the source:

```
## [Paper Title] ([Venue] [Year]) — FROM PDF (no code)
- **Methods:** [architecture, representation, planning approach — from paper text]
- **Tasks:** [task type, single/multi-agent, indoor/outdoor]
- **Datasets:** [dataset names, simulators]
- **Evaluation:** [metrics, reported results, baselines compared]
- **Key results:** [SR=X%, SPL=Y% on Dataset Z — exact numbers from paper]
- **Limitations noted:** [from paper's future work / discussion section]
```

**Depth difference:** Code papers (Step 3) get layer-by-layer architecture analysis from reading the actual code. No-code papers get method-level understanding from the paper text. Both go into the same gap matrices, but code papers have richer detail.

---

## Step 3: Read and analyze each repo (code papers)

For EACH cloned repo, systematically read and extract:

### 3a. Methods
- Read `README.md` for method overview
- Find the main model/architecture files (look for `model/`, `models/`, `network/`, `agent/`, `core/`)
- Read the core model files to understand:
  - What architecture is used? (transformer, CNN, GNN, RL policy, etc.)
  - What representation? (point cloud, voxel grid, semantic map, scene graph, BEV, etc.)
  - What planning approach? (learned policy, classical planner, LLM/VLM reasoning, frontier-based, etc.)
  - What loss functions / training objectives?
  - Any novel components or modules?

### 3b. Tasks
- Read config files, README, and eval scripts to identify:
  - What task(s) does this solve? (ObjectNav, PointNav, VLN, exploration, etc.)
  - Single-agent or multi-agent?
  - Indoor or outdoor?
  - Sim-only or sim-to-real?

### 3c. Datasets
- Search for dataset loading code (`data/`, `dataset/`, `dataloader`)
- Read configs for dataset names
- Identify:
  - Which datasets/simulators? (HM3D, MP3D, Gibson, Habitat, AI2-THOR, RoboTHOR, CARLA, etc.)
  - Which splits? (train/val/test, minival, etc.)
  - Any custom data or augmentation?

### 3d. Evaluation
- Find eval scripts (`eval.py`, `evaluate.py`, `test.py`, `scripts/eval*`)
- Read to identify:
  - What metrics? (Success Rate, SPL, SoftSPL, DTS, exploration coverage, etc.)
  - What baselines are compared against?
  - Any real-robot evaluation?
  - Any ablation structure?

### 3e. Dependencies and Integration
- Read `requirements.txt`, `setup.py`, `pyproject.toml`, `environment.yml`
- What frameworks? (Habitat, Isaac, ROS, PyTorch, JAX, etc.)
- What foundation models used? (CLIP, GPT-4, LLaVA, SAM, DINO, etc.)

Build a structured summary per repo:

```
## [Paper Title] ([Venue] [Year])
- **Methods:** [architecture, representation, planning approach, novel components]
- **Tasks:** [task type, single/multi-agent, indoor/outdoor]
- **Datasets:** [dataset names, simulators]
- **Evaluation:** [metrics, baselines, real-robot?]
- **Dependencies:** [frameworks, foundation models]
- **Limitations noted in code/README:** [any TODOs, known issues, missing features]
```

---

## Step 4: Cross-paper gap analysis

After analyzing ALL papers (both code and no-code), compare them systematically. Include ALL 20 papers in the gap matrices — mark no-code papers with `†` so later steps know which ones can be cloned vs only referenced.

### 4a. Methods gap matrix
Create a comparison table:

| Paper | Architecture | Representation | Planning | Foundation Model | Training |
|-------|-------------|----------------|----------|-----------------|----------|
| ... | ... | ... | ... | ... | ... |

Identify:
- What method combinations has NO paper tried?
- What components from paper A could benefit paper B?
- Are there standard techniques from adjacent fields missing entirely?

### 4b. Tasks gap matrix

| Paper | ObjectNav | VLN | Exploration | Multi-Agent | Outdoor | Real-Robot |
|-------|-----------|-----|-------------|-------------|---------|------------|
| ... | x | | x | | | |

Identify:
- Which task combinations are unexplored?
- Is there a task that only 1 paper addresses?
- What task settings are completely missing?

### 4c. Datasets gap matrix

| Paper | HM3D | MP3D | Gibson | AI2-THOR | CARLA | Real-World | Custom |
|-------|------|------|--------|----------|-------|------------|--------|
| ... | x | x | | | | | |

Identify:
- Which datasets are over-represented vs under-represented?
- Are results comparable across papers? (same splits, same metrics?)
- Missing dataset types (outdoor, multi-floor, dynamic environments, etc.)

### 4d. Evaluation gap matrix

| Paper | SR | SPL | SoftSPL | DTS | Coverage | Efficiency | Real-Robot |
|-------|-----|-----|---------|-----|----------|------------|------------|
| ... | x | x | | | | | |

Identify:
- Inconsistent metrics across papers
- Missing evaluation dimensions (robustness, generalization, compute cost, etc.)
- Papers that don't compare against each other when they should

---

## Step 5: Write LaTeX output

Generate a `.tex` file with the following structure:

```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}

\title{Research Gap Analysis: [Seed Paper Topic]}
\author{Auto-generated from literature analysis}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
[2-3 sentence summary: We analyzed N papers with public code related to [seed paper / topic].
We identify gaps across methods, tasks, datasets, and evaluation that represent
opportunities for future research.]
\end{abstract}

\section{Papers Analyzed}
[Table of all papers: title, venue, year, GitHub link or "PDF only†"]
% Mark no-code papers with † throughout all tables

\section{Methods Analysis}
\subsection{Comparison}
[Methods gap matrix as LaTeX table]
\subsection{Identified Gaps}
[Numbered list of method-level research gaps]

\section{Tasks Analysis}
\subsection{Comparison}
[Tasks gap matrix]
\subsection{Identified Gaps}
[Numbered list of task-level gaps]

\section{Datasets Analysis}
\subsection{Comparison}
[Datasets gap matrix]
\subsection{Identified Gaps}
[Numbered list of dataset-level gaps]

\section{Evaluation Analysis}
\subsection{Comparison}
[Evaluation gap matrix]
\subsection{Identified Gaps}
[Numbered list of evaluation-level gaps]

\section{Cross-Cutting Research Gaps}
[The most important gaps that span multiple dimensions — e.g., "No paper combines
VLM-based planning with outdoor terrain datasets using multi-agent evaluation"]

\section{Recommended Research Directions}
[Top 5 actionable research directions based on the gaps found, each with:
- What to do
- Which existing repos to build on
- Expected contribution]

\end{document}
```

### Rules for the LaTeX file:
- All tables must use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`)
- GitHub links must be clickable `\href{url}{org/repo}`
- Paper references should include arXiv links where available
- Gaps must be SPECIFIC and ACTIONABLE (not generic like "more research needed")
- Each gap should reference which papers it relates to
- The file must compile cleanly with `pdflatex`

---

## Step 6: Summary output

After writing the .tex file, print a summary to the user:

```
## Research Gap Analysis Complete

**Repos cloned:** X/Y successful (code papers)
**Papers read from PDF:** X (no-code papers)
**Total papers analyzed:** X
**Output:** [path to .tex file]

### Top 5 Research Gaps:
1. [gap 1]
2. [gap 2]
3. [gap 3]
4. [gap 4]
5. [gap 5]

### Quick start:
cd [clone_directory] && pdflatex [output.tex]
```
