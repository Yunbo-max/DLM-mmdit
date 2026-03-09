---
name: 03-gap-solve
description: For each research gap from /02-research-gap, reason about solution directions, search cross-domain papers that address the gap, and output top 5 papers per direction with GitHub links.
---

Take research gaps from the gap analysis and find cross-domain papers that solve each gap.

## User input: $ARGUMENTS

Parse the user input to extract:
1. **Gap analysis file** — path to the .tex file from `/02-research-gap` (e.g., `docs/research_gap.tex`). If not provided, default to `docs/research_gap.tex`
2. **Output file** — path for the output (e.g., `docs/gap_solutions.md`). If not provided, default to `docs/gap_solutions.md`
3. **Specific gaps** — optionally the user can specify which gaps to focus on (e.g., "gap 1, gap 3"). If not provided, process ALL gaps.

Example invocations:
- `/gap-solve` → reads `docs/research_gap.tex`, outputs `docs/gap_solutions.md`
- `/gap-solve docs/research_gap.tex docs/solutions.md`
- `/gap-solve docs/research_gap.tex gap 1, gap 3`

---

## Step 0: Pre-flight validation

Before reasoning about gaps, validate ALL required inputs:

1. **Gap analysis file exists?** Check `docs/research_gap.tex` (or user-specified path). If missing → STOP:
   > "Missing: docs/research_gap.tex. Run `/02-research-gap` first to generate it."
2. **Gap analysis has gaps?** Scan for `\subsection{Identified Gaps}` sections. If zero found → STOP:
   > "No identified gaps found in [file]. The gap analysis may be incomplete — re-run `/02-research-gap`."
3. **Output path writable?** Check parent directory exists. If not → STOP:
   > "Output directory does not exist. Create it with: `mkdir -p docs/`"

Only proceed to Step 1 after ALL checks pass.

---

## Step 1: Extract research gaps

Read the gap analysis .tex file from `/02-research-gap`. Extract ALL identified gaps from these sections:
- `\section{Methods Analysis}` → `\subsection{Identified Gaps}`
- `\section{Tasks Analysis}` → `\subsection{Identified Gaps}`
- `\section{Datasets Analysis}` → `\subsection{Identified Gaps}`
- `\section{Evaluation Analysis}` → `\subsection{Identified Gaps}`
- `\section{Cross-Cutting Research Gaps}`

Build a structured list:

```
Gap 1: [description] (type: methods)
Gap 2: [description] (type: tasks)
Gap 3: [description] (type: datasets)
...
```

If the .tex file doesn't exist, STOP and tell the user to run `/02-research-gap` first.

---

## Step 2: Reason about solution directions (CRITICAL STEP)

For EACH research gap, reason deeply about HOW it could be solved. Do NOT just search the gap keywords directly — think about what approaches from OTHER fields address the underlying problem.

### Reasoning process per gap:

1. **Decompose the gap** — What is the core problem? Break it into sub-problems.
   - Example gap: "No paper achieves real-time VLM reasoning for navigation"
   - Core problem: VLM inference is slow
   - Sub-problems: model size, inference optimization, when to call the VLM

2. **Think laterally** — What OTHER fields have solved similar problems?
   - Latent reasoning / speculative decoding (NLP efficiency)
   - Cached inference / KV-cache reuse (LLM systems)
   - When-to-think / adaptive compute (RL, cognitive science)
   - Model distillation / pruning (ML optimization)
   - Asynchronous perception-action (robotics)

3. **Generate 3-5 solution directions** — Each direction is a concrete research angle with cross-domain inspiration:

```
Gap: "No paper achieves real-time VLM reasoning for navigation"

Direction 1: Latent reasoning — compress VLM chain-of-thought into latent space
  Search domains: NLP efficiency, latent diffusion, cognitive architectures

Direction 2: Adaptive compute — only invoke VLM at decision points, not every step
  Search domains: active perception, attention mechanisms, option framework in RL

Direction 3: Distilled navigation VLM — train small task-specific model from large VLM
  Search domains: knowledge distillation, model compression, TinyML

Direction 4: Asynchronous VLM — run VLM in background, act on cached reasoning
  Search domains: anytime algorithms, async robotics, real-time systems

Direction 5: Structured caching — reuse VLM outputs for similar spatial queries
  Search domains: spatial databases, KV-cache sharing, retrieval-augmented generation
```

### Reasoning examples for common gap types:

| Gap Type | Lateral Thinking Examples |
|----------|--------------------------|
| **Efficiency** | Latent reasoning, speculative decoding, model distillation, pruning, quantization, early exit, adaptive compute, caching |
| **Generalization** | Domain adaptation, sim-to-real transfer, meta-learning, foundation model fine-tuning, data augmentation, curriculum learning |
| **Multi-agent** | Game theory, auction mechanisms, consensus algorithms, swarm intelligence, communication protocols, role assignment |
| **Outdoor/terrain** | Self-driving perception, geological sensing, agricultural robotics, drone mapping, military path planning |
| **Evaluation** | Benchmark design, standardized metrics, reproducibility frameworks, statistical testing, human evaluation protocols |
| **Dataset** | Synthetic data generation, procedural environments, data mixing, cross-domain benchmarks, annotation tools |
| **Robustness** | Adversarial training, uncertainty estimation, Bayesian methods, ensemble methods, failure detection |
| **Real-robot transfer** | Sim-to-real, domain randomization, digital twins, hardware-in-the-loop, compliant control |

Present ALL reasoning to the user before searching:

```
## Gap Reasoning Summary

### Gap 1: [description]
**Core problem:** ...
**Sub-problems:** ...
**Solution directions:**
1. [direction] — search in [domains]
2. [direction] — search in [domains]
3. ...
```

---

## Step 3: Search for papers per direction

For EACH direction of EACH gap, search using the same venues as `/01-literature-search`:

### Search strategy per direction:
- Use the direction keywords + domain-specific terms
- Search across ALL venues (ML conferences + robotics + preprints)
- Prioritize papers that:
  1. Have a public GitHub repo (REQUIRED for top 5)
  2. Are from a DIFFERENT domain than the original gap (cross-pollination)
  3. Are recent (prefer last 2 years)

### Venues to search (same as /01-literature-search):

**ML conferences:** ICLR, CVPR, ICCV, ECCV, NeurIPS
**Robotics:** ICRA, CoRL, RSS, IROS, RA-L
**Preprints:** arXiv, HuggingFace Papers
**Systems/efficiency:** MLSys, OSDI, SOSP (for efficiency gaps)
**NLP:** ACL, EMNLP, NAACL (for reasoning/language gaps)
**GitHub:** search for implementations

### OpenReview reviewer comments (critical source for gap validation + solution hints)

Reviewer comments on OpenReview (ICLR, NeurIPS, EMNLP, etc.) are one of the most valuable sources for gap-solving because:
- **Reviewers identify real weaknesses** — these are validated gaps, not speculative ones
- **Reviewers suggest experiments** — "you should compare to X" or "try this on dataset Y"
- **Reviewers point to missing work** — "you should cite/compare [paper Z]"
- **Rejected papers** — papers that tried to solve a similar gap but failed reveal what NOT to do

For EACH direction, also search:

| What | How to search | Why valuable |
|------|--------------|-------------|
| **Reviews of papers in this direction** | `site:openreview.net [direction keywords] [venue] [year]` | See what reviewers think is weak about current approaches |
| **Rejected papers on this topic** | `site:openreview.net [gap keywords] reject` | Learn from failed attempts — what went wrong? |
| **Author rebuttals** | Read the full OpenReview thread for top papers | Authors often propose future directions in rebuttals |
| **"Missing baselines" complaints** | Look for reviewer comments mentioning "missing comparison" or "should compare to" | Reveals which methods the community considers important |

For each paper found via OpenReview reviews, add to the collection:
- The paper being reviewed (if relevant to the direction)
- Any paper CITED by reviewers ("you should compare to [X]")
- The specific reviewer criticism that connects to your gap

In the output, mark papers discovered through reviewer comments:

```
| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 3 | [Paper X] | ICLR | 2025 | ... | [repo](url) | Reviewer of [Paper Y] said "this outperforms the approach in [gap area]" |
```

### For each paper found, collect:
- Title
- Venue + year
- arXiv link
- GitHub repo link (search thoroughly)
- 1-line summary of how it addresses the direction
- What domain it comes from (to show cross-pollination)

---

## Step 4: Rank and select top 5 per direction

For EACH direction, rank papers by:
1. **Relevance** to solving the gap (most important)
2. **Has GitHub repo** (REQUIRED — no repo = not in top 5)
3. **Cross-domain value** (prefer papers from different fields)
4. **Recency** (prefer newer)
5. **Impact** (venue quality, citations if known)

Select the **top 5 papers** per direction. If fewer than 5 have GitHub repos, output however many qualify.

---

## Step 5: Output

Write the output markdown file:

```markdown
# Research Gap Solutions

Generated from: [gap analysis .tex file]
Date: [today]

---

## Gap 1: [gap description]
**Type:** [methods / tasks / datasets / evaluation / cross-cutting]
**Core problem:** [from reasoning step]

### Direction 1.1: [direction name]
**Reasoning:** [why this direction could solve the gap]
**Search domains:** [which fields searched]

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | ... | ... | ... | ... | [org/repo](url) | ... |
| 2 | ... | ... | ... | ... | [org/repo](url) | ... |
| 3 | ... | ... | ... | ... | [org/repo](url) | ... |
| 4 | ... | ... | ... | ... | [org/repo](url) | ... |
| 5 | ... | ... | ... | ... | [org/repo](url) | ... |

### Direction 1.2: [direction name]
...

---

## Gap 2: [gap description]
...

---

## Summary: All Solution Papers

Total unique papers found: X
Total with GitHub repos: Y

| Gap | Direction | Top Paper | GitHub | Domain |
|-----|-----------|-----------|--------|--------|
| 1 | 1.1 | [best paper] | [link] | [domain] |
| 1 | 1.2 | [best paper] | [link] | [domain] |
| 2 | 2.1 | [best paper] | [link] | [domain] |
| ... | ... | ... | ... | ... |
```

### Output rules:
- **Top 5 papers per direction MUST have GitHub links** — no exceptions
- **"How It Helps"** column must be specific to the gap (not generic summary)
- Each gap must have at least 2 directions
- Each direction must have at least 3 papers (ideally 5)
- The Summary table at the end gives a quick overview of the best paper per direction
- Papers CAN appear in multiple directions/gaps if relevant
