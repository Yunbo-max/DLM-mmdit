---
name: 07-write-paper
description: Download conference LaTeX template, then WRITE the paper with critical thinking using all pipeline outputs as input. Checks venue page limits, writes math proofs, creates pipeline figures, and manages main/appendix split.
---

Write a full conference paper with critical thinking and analytical depth, using all pipeline outputs (01–06) as input material. This is NOT a copy-paste assembler — it is a WRITER that synthesizes, reasons about, and presents the research at the level expected by top ML venues (NeurIPS, ICML, ICLR, ACL, EMNLP).

## CRITICAL PRINCIPLE

**This skill is a WRITER, not an assembler.** The pipeline outputs provide the raw material:

| Input File | From | Provides |
|-----------|------|----------|
| `literature.md` | `/01-literature-search` | Related work papers, field context |
| `research_gap.tex` | `/02-research-gap` | Identified gaps, motivation |
| `gap_solutions.md` | `/03-gap-solve` | Cross-domain solutions, design rationale |
| `contribution.tex` | `/04-spec-novel` | Method description, equations, contributions |
| `specs.md` | `/04-spec-novel` | Architecture details, module specs, flowcharts |
| `experiments.tex` | `/05-build-code` | Tables, metrics, hyperparameters, setup |
| `experiment_plan.md` | `/06-run-experiments` | Filled results, analysis |

You READ these files as input, then WRITE each section with:
- **Critical thinking** — why does this matter? what are the tradeoffs? what would a reviewer question?
- **Analytical depth** — don't just state results, explain WHY the method works
- **Mathematical rigor** — proper definitions, theorems, proofs where applicable
- **Narrative flow** — each section builds on the previous, telling a coherent story
- **Honest assessment** — acknowledge limitations, discuss failure cases

---

## Step 0: Pre-flight validation

### 0a. Validate ALL source files exist

| # | File | Required? | Check | If missing |
|---|------|----------|-------|------------|
| 1 | `docs/novel/contribution.tex` | REQUIRED | Has `\section{Method}` | → Run `/04-spec-novel` |
| 2 | `docs/novel/experiments.tex` | REQUIRED | Has `\begin{table}` | → Run `/05-build-code` |
| 3 | `docs/novel/specs.md` | REQUIRED | Has `## Module Specifications` | → Run `/04-spec-novel` |
| 4 | `docs/literature.md` | REQUIRED | Has `## Top 10` or paper tables | → Run `/01-literature-search` |
| 5 | `docs/research_gap.tex` | REQUIRED | Has `\section{Cross-Cutting}` | → Run `/02-research-gap` |
| 6 | `docs/gap_solutions.md` | optional | Has `## Gap` sections | Warning only |
| 7 | `docs/novel/experiment_plan.md` | optional | Has filled results | Warning: result cells will stay TBD |

For EACH required file that is missing, print the error and STOP:
> "Cannot write paper. Missing [N] required files:
> 1. docs/novel/contribution.tex → Run `/04-spec-novel`
> Run these skills first, then re-run `/07-write-paper`."

### 0b. Read ALL source files

Read every file listed above. These are your INPUT MATERIAL — the facts, equations, tables, and references you will draw from. But the WRITING is yours to do.

---

## User input: $ARGUMENTS

Parse the user input to extract:
1. **Conference** — target venue. REQUIRED. Examples: "NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "CVPR", "AAAI"
2. **Output directory** — where to write the paper. Default: `paper/`
3. **Paper title** — optional override. If not given, use title from contribution.tex

Example invocations:
- `/07-write-paper NeurIPS`
- `/07-write-paper ICLR 2026`
- `/07-write-paper NeurIPS paper/ "My Paper Title"`

---

## Step 1: Check conference requirements from the venue website

### 1a. Search for the official call for papers

Use WebSearch to find the EXACT requirements:
```
Search: "[Conference] [year] call for papers submission guidelines"
Search: "[Conference] [year] style file LaTeX template"
```

### 1b. Extract these requirements

| Requirement | Example |
|------------|---------|
| **Max pages (main body)** | NeurIPS: 10 pages, ICML: 8, ICLR: 10, ACL: 8 long / 4 short |
| **Page limit includes/excludes references?** | Most ML: excludes. ACL: includes for short. |
| **Appendix policy** | NeurIPS: unlimited supplementary. ICML: appendix after refs. |
| **Review type** | Double-blind (NeurIPS, ICLR, ICML) vs. single-blind |
| **Required sections** | NeurIPS: Broader Impact. ICML: Reproducibility. |
| **Citation style** | natbib, numbered, author-year — check the .bst file |
| **Figure/table placement** | Some venues prefer top-of-page `[t]` only |
| **Abstract word limit** | NeurIPS: no strict limit. ACL: 200 words. |

### 1c. Download the official LaTeX template

```bash
mkdir -p [output_dir]
cd [output_dir]

# For GitHub-hosted templates:
git clone --depth 1 [template_repo_url] template/

# For zip downloads:
wget [template_url] -O template.zip && unzip template.zip -d template/
```

### 1d. Set up paper directory

```
[output_dir]/
├── template/              # Original template (reference only)
├── main.tex               # Our paper
├── references.bib         # Bibliography
├── sections/              # One file per section
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── method.tex
│   ├── experiments.tex
│   ├── conclusion.tex
│   └── appendix.tex
├── figures/               # All figures
│   ├── pipeline.pdf       # Method overview (or .tikz)
│   └── ...
└── supplementary/         # If venue supports separate supplementary
    └── supp.tex
```

Copy style files:
```bash
cp template/*.sty template/*.cls template/*.bst [output_dir]/
```

### 1e. Record the page budget

```
VENUE: [Conference]
MAX_PAGES_MAIN: [N]
REFS_EXCLUDED: [yes/no]
APPENDIX_POLICY: [unlimited supplementary / appendix after refs / none]
DOUBLE_BLIND: [yes/no]
CITATION_STYLE: [natbib/numbered/author-year]
```

This budget DETERMINES what goes in main vs. appendix (Step 6).

---

## Step 2: Generate references.bib FIRST

Build the bibliography before writing any sections, so all agents know the citation keys.

### 2a. Collect all paper references

From ALL input files, extract every paper mentioned:
- `literature.md` — top-10 papers with venues, years, authors
- `research_gap.tex` — papers cited in gap analysis
- `gap_solutions.md` — cross-domain papers
- `contribution.tex` — papers compared against
- `experiments.tex` — baseline papers

### 2b. Look up proper BibTeX entries

For each paper:
1. Search `dblp.org` for the exact BibTeX entry (preferred — canonical venue names)
2. If not on DBLP, search `scholar.google.com` for BibTeX
3. If arXiv-only, use arXiv metadata

### 2c. Write references.bib

Citation key convention: `[firstauthor_lastname][year][keyword]`
- e.g., `sahoo2024mdlm`, `shi2024simplified`, `lou2024discrete`
- If collision, add a distinguishing keyword

```bibtex
@inproceedings{sahoo2024mdlm,
  title     = {Simple and Effective Masked Diffusion Language Models},
  author    = {Sahoo, Subham Sekhar and Arriola, Marianne and ...},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
}
```

### 2d. Create a citation key reference list

Print a mapping for the writing agents:
```
MDLM → \cite{sahoo2024mdlm}
ReMDM → \cite{shi2024remdm}
LatentOps → \cite{liu2022latentops}
...
```

---

## Step 3: Write each section (PARALLEL where possible)

Write sections in parallel using Agent workers. Each agent gets:
- The relevant source files
- The citation key reference list
- The conference requirements (page budget, citation style)
- Instructions for critical thinking and analytical writing

### Parallel writing groups:

```
Group 1 (parallel — independent sections):

  → Agent A: Introduction
      Sources: research_gap.tex, literature.md, gap_solutions.md, contribution.tex
      Output: sections/introduction.tex

  → Agent B: Related Work
      Sources: literature.md, research_gap.tex, gap_solutions.md, contribution.tex
      Output: sections/related_work.tex

  → Agent C: Methodology + Math + Pipeline Figure
      Sources: contribution.tex, specs.md, gap_solutions.md
      Output: sections/method.tex

  → Agent D: Experiments (setup + results + ablation)
      Sources: experiments.tex, experiment_plan.md
      Output: sections/experiments.tex

  → Agent E: Limitations + Conclusion
      Sources: contribution.tex, research_gap.tex, experiments.tex, specs.md
      Output: sections/conclusion.tex

Group 2 (sequential — needs all sections):
  → Abstract (needs to reference results from Agent D)
  → Assemble main.tex from section files
  → Page budget check → main/appendix split (Step 6)
```

### Shared conventions for ALL agents:

- Citation: use `\cite{key}` or `\citep{key}` / `\citet{key}` depending on venue style
- Figure labels: `fig:[name]` (e.g., `fig:pipeline`, `fig:geometry`)
- Table labels: `tab:[name]` (e.g., `tab:main_results`, `tab:ablation_mr`)
- Equation labels: `eq:[name]` (e.g., `eq:sss`, `eq:mts`, `eq:lsme_objective`)
- Write in present tense for method, past tense for experiments
- Every claim MUST have evidence (citation, table reference, or equation)
- Every figure/table MUST be referenced in text

---

### 3a. Introduction (Agent A) — ~1.5 pages

**Sources:** `research_gap.tex` (01, 02), `literature.md` (01), `gap_solutions.md` (03), `contribution.tex` (04)

Write with this structure (5 paragraphs):

**Paragraph 1: Context + Excitement** (4-5 sentences)
- What is the broad problem area?
- Why is it important NOW? (recent progress, new capabilities)
- Cite 3-4 foundational works from literature.md
- Write with energy — this sets the tone for the whole paper

**Paragraph 2: The Gap** (4-5 sentences)
- What do existing methods fail to do?
- Be SPECIFIC — cite papers that have each limitation
- Use research_gap.tex as the source, but rewrite as a narrative
- Critical thinking: frame the gap as a fundamental issue, not just "nobody tried it yet"

**Paragraph 3: Our Approach** (3-4 sentences)
- "In this work, we propose [Method], which..."
- State the key insight in ONE clear sentence
- Explain WHY this approach addresses the gap (not just WHAT it does)
- Use gap_solutions.md to connect to cross-domain inspiration

**Paragraph 4: Contributions** (numbered list, 3-4 items)
- Copy from contribution.tex, but polish to be crisp and specific
- Each contribution should be VERIFIABLE — "We achieve X on Y" not "We propose a novel method"
- Include both methodological AND empirical contributions

**Paragraph 5: Results Preview** (2-3 sentences)
- Brief mention of key results: "On [benchmark], our method achieves [X%], improving over [baseline] by [Y%]"
- If results are still TBD, write: "Experiments on [benchmarks] demonstrate [expected outcome]"

**Figure 1 (Teaser):** A high-level overview figure showing the problem and solution at a glance. Include a `\begin{figure}[t]` placeholder with a detailed description comment for figure creation.

---

### 3b. Related Work (Agent B) — ~1-1.5 pages

**Sources:** `literature.md` (01), `research_gap.tex` (02), `gap_solutions.md` (03), `contribution.tex` (04)

**Critical thinking approach:**
- Do NOT write a survey. Write a POSITIONING argument.
- Each subsection should end by contrasting prior work with YOUR method.
- Group papers by theme, not chronologically.

Structure into 3-4 subsections based on the domains from literature.md:

For each subsection:
```
\subsection{[Theme Name]}

[Overview paragraph: 3-4 sentences covering the main approaches in this area.
Cite 4-6 key papers. Group by approach type.]

[Limitation paragraph: 2-3 sentences identifying what these methods
lack or where they fall short. Source: research_gap.tex per-domain gaps.]

[Contrast sentence: "In contrast, our method [specific difference]
because [specific reason]." Source: contribution.tex comparison.]
```

Rules:
- Every cited paper MUST be in references.bib — use `\cite{key}` or `\citet{key}`
- Cover ALL major related areas (show breadth) but be concise
- End EACH subsection with positioning against your work
- Do NOT just list papers — analyze and group them by approach

---

### 3c. Methodology (Agent C) — ~2.5 pages (FULL version, trimmed in Step 6)

**Sources:** `contribution.tex` (04), `specs.md` (04), `gap_solutions.md` (03)

**CRITICAL: Write the methodology with FULL mathematical detail first.** Include all proofs, derivations, and theoretical justification. In Step 6, excess content will be moved to appendix based on page budget.

#### Structure:

**3c-i. Problem Formulation** (~0.5 page)
```
\subsection{Problem Formulation}

Define the problem FORMALLY:
- Input space, output space, objective
- Notation table (define ALL symbols used in the paper)
- Mathematical setup (e.g., diffusion process, masking, latent space)

Source: contribution.tex problem statement, specs.md overview
```

**3c-ii. Method Overview** (~0.25 page)
- High-level pipeline description in prose (3-4 sentences)
- Reference the pipeline figure (Figure 2)
- List the key components and how they connect

**3c-iii. Pipeline Figure (Figure 2)** — REQUIRED

Generate a publication-quality pipeline figure using the OpenAI Image API. The figure prompt should be a **professional summary of the methodology**, NOT a childish list of boxes and colors. The image model is intelligent — describe WHAT the method does and it will create an appropriate academic diagram.

**Writing the prompt:**

Write a methodology summary to `paper/figures/pipeline_prompt.txt` that covers:
1. The paper title and venue context (e.g., "for a NeurIPS paper on [Method Name]")
2. A concise explanation of what the method does, step by step
3. The key architectural insight (e.g., joint attention between two modalities)
4. A concrete example showing input → processing → output
5. What the figure should emphasize (e.g., "clearly show the two parallel streams and cross-attention")
6. One line on style: "Clean, flat, professional academic figure. White background, no decorative elements."

**Example prompt** (adapt to YOUR method):

```
Academic pipeline figure for a NeurIPS paper on [Method Name], a method
that [one-sentence description of what it does].

[Method Name] works in N steps. Given [input description]:

Step 1 — [Name]: [2-3 sentences describing what happens, with concrete
example tokens/values]

Step 2 — [Name]: [2-3 sentences describing the transformation]

Step 3 — [Name]: [2-3 sentences describing the core computation,
emphasizing the novel architectural component]

The figure should clearly show [key structural elements to emphasize,
e.g., parallel streams, cross-attention, conditioning pathway].

Style: Clean, flat, professional academic figure suitable for a top-tier
ML venue. White background, no decorative elements. Ensure generous
margins so nothing is cropped.
```

**DO NOT** write prompts like "blue box with rounded corners 3pt" or "arrow from box A to box B". The model understands methodology — describe the METHOD, not the visual layout.

**Generating the figure:**

```bash
# Requires OPENAI_API_KEY in .claude/commands/.env
python scripts/generate_figure.py \
    --prompt_file paper/figures/pipeline_prompt.txt \
    --output paper/figures/pipeline.png \
    --model gpt-4o \
    --size 1536x1024

# View the result and iterate if needed (re-run with refined prompt)
```

**Include in LaTeX:**
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/pipeline.png}
\caption{Overview of [Method Name]. [Describe the pipeline flow
and highlight what each stage does.]}
\label{fig:pipeline}
\end{figure*}
```

**If no API key:** Write the methodology summary as a LaTeX comment so the user can paste it into ChatGPT web interface manually.

**Iteration:** Generate 2-3 versions with slight prompt variations, pick the best one. Minor text typos from the image model are acceptable — the caption provides the precise description.

**3c-iv. Core Modules** (~1-1.5 pages)

For EACH key module from specs.md, write a subsection:

```
\subsection{[Module Name]}

\paragraph{Motivation.} Why is this component needed? What problem does it solve?
(2-3 sentences, critical thinking — connect to the gap from Introduction)

\paragraph{Formulation.} Mathematical description:
- Define input/output formally
- State the key equations with labels: \begin{equation} ... \label{eq:xxx} \end{equation}
- Explain each variable and design choice

\paragraph{Analysis.} Why does this formulation work?
- Theoretical justification (if applicable)
- Connection to prior work (cite relevant papers)
- Intuition for practitioners
```

**3c-v. Mathematical Proofs and Derivations** (~0.5-1 page)

Write FULL proofs for any theoretical claims:
- Convergence properties
- Correctness of the diffusion schedule with partial masking
- Relationship between mask ratio and entry timestep
- SSS/MTS metric properties (e.g., bounds, expected values under random baseline)

Use proper theorem environments:
```latex
\begin{theorem}[Semantic Smoothness Bound]
\label{thm:sss_bound}
For a Lipschitz-continuous decoder $D$ with constant $L$ and a SLERP path
of $N$ points with angular separation $\theta / (N-1)$, the SSS satisfies:
\begin{equation}
\text{SSS} \geq 1 - L \cdot \frac{\theta}{N-1}
\end{equation}
\end{theorem}

\begin{proof}
[Full proof here]
\end{proof}
```

**NOTE:** These proofs will be moved to appendix in Step 6 if the page budget is tight. But write them FULLY now.

**3c-vi. Training and Inference** (~0.25 page)
- Loss function with equation
- Training procedure summary
- Inference/editing algorithm (pseudocode in `\begin{algorithm}` environment)

---

### 3d. Experiments (Agent D) — ~2-2.5 pages

**Sources:** `experiments.tex` (05), `experiment_plan.md` (06)

#### 3d-i. Experimental Setup (~0.75 page)

Write these as `\paragraph{}` blocks:

**Datasets.** Describe each dataset in 2-3 sentences. Include:
- What task it evaluates (sentiment, topic, formality)
- Size, split used, number of test samples
- Why this dataset was chosen (what aspect of the method it tests)
- Source: experiments.tex dataset table, rewrite as analytical prose

**Baselines.** Group baselines into categories and explain WHY each was selected:
- Category 1: [type] — tests whether [aspect]
- Category 2: [type] — tests whether [aspect]
- Source: experiments.tex baseline table

**Metrics.** Describe the 6-pillar DLM-Eval Suite. For novel metrics (SSS, MTS), give the full definition with equation reference from the methodology section. For standard metrics, 1 sentence each.

**Implementation Details.** Key hyperparameters in prose (not a dump of all values). Mention what matters: model size, training data, key LSME parameters.

#### 3d-ii. Main Results (~0.75 page)

**Critical thinking — don't just state numbers, ANALYZE them:**

For each dataset/table from experiments.tex:
1. **State the headline result:** "On Yelp, LSME achieves [X%] Attr-Acc, outperforming the best baseline [name] by [+Y%]."
2. **Explain WHY:** "This improvement stems from [specific mechanism]. The latent conditioning directly steers generation toward [target], whereas [baseline] relies on [weaker mechanism]."
3. **Discuss tradeoffs:** "At mask ratio $r=0.3$, LSME preserves [X%] BLEU while achieving [Y%] accuracy. Increasing to $r=0.7$ improves accuracy to [Z%] but reduces BLEU to [W%], reflecting the fundamental accuracy-preservation tradeoff."
4. **Cross-dataset patterns:** "The improvement is consistent across all three tasks, with the largest gain on [dataset] ([+X%]) where [explanation]."

Copy result tables from experiments.tex. If cells still have `\tbd{---}`, keep them with a TODO comment.

#### 3d-iii. Ablation Study (~0.75 page)

**Critical thinking — each ablation should answer a QUESTION:**

For each ablation:
1. **State the question:** "Does entropy-based masking outperform random masking?"
2. **Present the evidence:** Reference the ablation table.
3. **Interpret the result:** "Entropy masking achieves [X%] vs. random's [Y%], confirming that targeting uncertain tokens [explanation]."
4. **Connect to the method design:** "This validates our design choice of [component] in Section [X]."

#### 3d-iv. Latent Geometry Analysis (~0.5 page)

Present SSS/MTS results with interpretation:
- What do the numbers mean concretely?
- Compare against expected random baseline values
- Discuss what this reveals about the learned latent space

#### 3d-v. Efficiency Analysis (~0.25 page)

Brief comparison table + 1 paragraph interpreting the cost-benefit tradeoff.

---

### 3e. Limitations + Conclusion (Agent E) — ~0.75 page

**Sources:** `contribution.tex` (04), `research_gap.tex` (02), `experiments.tex` (05), `specs.md` (04)

#### Limitations (~0.3 page)

**Be honest and specific — reviewers respect intellectual honesty:**

Write 3-4 specific limitations:
1. **Methodological:** What assumptions does the method make? When might it fail?
2. **Empirical:** What tasks/settings were NOT tested? What about scale?
3. **Computational:** Any efficiency concerns at scale?
4. **Data:** Dataset biases, limited domains tested

Source: specs.md limitations + your own critical analysis of what experiments DON'T show.

#### Conclusion (~0.3 page)

**Paragraph 1: Summary** (4-5 sentences)
- Restate the problem and approach (don't just repeat the abstract)
- Summarize key contributions
- State the main empirical result

**Paragraph 2: Future Work** (3-4 sentences)
- Connect to remaining gaps from research_gap.tex
- Suggest 2-3 concrete next steps
- End on an optimistic but grounded note

---

### 3f. Abstract (written LAST, after all sections) — ~200 words

Structure as 4-5 sentences:

```
Sentence 1: [PROBLEM] — What challenge exists? (from Introduction ¶2)
Sentence 2: [APPROACH] — What is our key idea? (from Introduction ¶3)
Sentence 3: [METHOD] — How does it work at a high level? (from Method overview)
Sentence 4: [RESULTS] — What do we achieve? Use concrete numbers. (from Experiments)
Sentence 5: [IMPACT] — Why does this matter? Code release? (from Conclusion)
```

---

## Step 4: Assemble main.tex

### 4a. Create main.tex using \input{} for each section

```latex
\documentclass{[conference_style]}
\usepackage{...}  % From template

% Custom commands
\newcommand{\method}{[Method Name]}
\newcommand{\best}[1]{\textbf{#1}}

\title{[Paper Title]}
\author{Anonymous}  % For double-blind venues

\begin{document}
\maketitle

\input{sections/abstract}
\input{sections/introduction}
\input{sections/related_work}
\input{sections/method}
\input{sections/experiments}
\input{sections/conclusion}

% References
\bibliographystyle{[venue_style]}
\bibliography{references}

% Appendix (if venue supports inline appendix)
\appendix
\input{sections/appendix}

\end{document}
```

### 4b. Verify all cross-references

Check that:
- Every `\cite{key}` has a matching entry in references.bib
- Every `\ref{label}` has a matching `\label{label}`
- Every figure/table is referenced in text
- No orphan figures (discussed but not shown) or orphan tables (shown but not discussed)

---

## Step 5: Compile and count pages

```bash
cd [output_dir]
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Count pages:
- Main body pages (before references): [X]
- Reference pages: [Y]
- Total: [X + Y]

Compare against venue limit from Step 1e.

---

## Step 6: Main / Appendix split based on page budget

This is the CRITICAL step that determines the final paper structure.

### 6a. If OVER page limit:

Prioritize cutting in this order (least important first):

1. **Move full proofs to appendix** — keep theorem statements in main, move proofs to appendix with "Proof in Appendix [X]"
2. **Move detailed derivations to appendix** — keep key equations in main, move step-by-step derivations
3. **Compress Related Work** — merge subsections, reduce per-paper descriptions
4. **Move additional ablations to appendix** — keep the 2 most important ablations in main
5. **Move efficiency analysis to appendix** — keep 1-sentence summary in main
6. **Compress Experimental Setup** — move full hyperparameter tables to appendix, keep 1-paragraph prose summary

For each item moved, add a forward reference:
```latex
% In main text:
\begin{theorem}[SSS Bound] ... \end{theorem}
See Appendix~\ref{app:proof_sss} for the full proof.

% In appendix:
\section{Proof of Theorem~\ref{thm:sss_bound}}
\label{app:proof_sss}
[Full proof]
```

### 6b. If UNDER page limit:

Add content in this order (most valuable first):

1. **Expand qualitative analysis** — add example outputs with discussion
2. **Add more ablation results** — from experiments.tex remaining ablation tables
3. **Expand method intuition** — add more explanation of WHY design choices were made
4. **Add a Discussion section** — broader implications, connections to other work

### 6c. Write the appendix

Create `sections/appendix.tex`:

```latex
\section{Full Proofs}
\label{app:proofs}
% Proofs moved from Section 3

\section{Additional Implementation Details}
\label{app:impl}
% Full hyperparameter tables from experiments.tex
% Training details, optimizer settings

\section{Additional Experimental Results}
\label{app:results}
% Ablation tables that didn't fit in main
% Per-category breakdowns
% Full latent geometry analysis

\section{Qualitative Examples}
\label{app:qualitative}
% Example inputs and outputs
% Success cases and failure cases with analysis
```

For venues that require SEPARATE supplementary (NeurIPS, CVPR):
Create `supplementary/supp.tex` with the same content.

### 6d. Re-compile and verify page count

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Verify: main body ≤ MAX_PAGES_MAIN from Step 1e.

---

## Step 7: Citation and reference quality check

### 7a. Verify citation style matches venue

- **natbib with author-year** (ICLR, some NeurIPS): `\citet{sahoo2024mdlm}` for "Sahoo et al. (2024)", `\citep{sahoo2024mdlm}` for "(Sahoo et al., 2024)"
- **Numbered** (CVPR, ICML, ACL): `\cite{sahoo2024mdlm}` → "[1]"
- Check the venue's .bst file to determine which style is expected

### 7b. Check for in-text citation correctness

- Never write "Sahoo et al. [2024]" by hand — always use `\citet{}` or `\cite{}`
- Never hardcode "[1]" — always use `\cite{}`
- Use `\citet{}` when the author is the subject: "As \citet{sahoo2024mdlm} showed, ..."
- Use `\citep{}` for parenthetical: "Recent work on masked diffusion \citep{sahoo2024mdlm, shi2024remdm} has shown ..."

### 7c. Verify all BibTeX entries compile

```bash
grep "Warning--" main.blg  # Check for bibtex warnings
grep "undefined" main.log | grep "citation"  # Check for missing citations
```

Fix any undefined citations or bibliography warnings.

---

## Step 8: Final quality check

### 8a. LaTeX quality

- No overfull/underfull hbox warnings (adjust text or use `\sloppy` locally)
- All figures render correctly
- Tables fit within column width
- No compilation errors

### 8b. Content quality (NeurIPS-level checklist)

| Check | Status |
|-------|--------|
| Abstract ≤ 250 words, contains concrete results? | |
| Introduction clearly states the gap and contribution? | |
| Related work positions our method, not just surveys? | |
| Method has formal problem statement + all variables defined? | |
| Pipeline figure clearly shows the full method? | |
| All equations are numbered and referenced? | |
| Experiments answer the stated research questions? | |
| Every claim has evidence (number, citation, or proof)? | |
| Ablation validates each design choice? | |
| Limitations are honest and specific? | |
| All `\tbd{---}` cells are documented with TODO comments? | |

### 8c. Reviewer-perspective check

Ask yourself what a critical reviewer would question:
- "Why not compare against [X]?" → Address in Related Work or Limitations
- "What about [edge case]?" → Address in Discussion or Limitations
- "Is the improvement statistically significant?" → Include std dev where possible
- "Why this architecture and not [simpler alternative]?" → Address in Method + Ablation

---

## Step 9: Summary output

```
## Paper Draft Complete

**Conference:** [venue] [year]
**Template:** [source]
**Title:** [paper title]
**Page count:** [X] / [limit] pages (main body)

### Output files:
- [output_dir]/main.tex — Full paper (assembled from sections/)
- [output_dir]/sections/ — Individual section files
- [output_dir]/references.bib — [N] references
- [output_dir]/sections/appendix.tex — Appendix with proofs + extra results
- [output_dir]/figures/ — Figure files/placeholders

### Section status:
| Section | Pages | Status | Notes |
|---------|-------|--------|-------|
| Abstract | 0.25 | Written | [word count] words |
| Introduction | [X] | Written | [N] citations |
| Related Work | [X] | Written | [N] citations, [M] subsections |
| Method | [X] | Written | [N] equations, pipeline figure, [M] theorems |
| Experiments | [X] | Written/Partial | [N] tables, [M] cells TBD |
| Limitations | [X] | Written | |
| Conclusion | [X] | Written | |
| Appendix | [X] | Written | Proofs, extra ablations, details |

### Main/appendix split:
- **Main body:** [describe what's in main]
- **Appendix:** [describe what was moved to appendix]
- **Reason:** Page budget [X]/[limit] — moved [items] to appendix

### TODOs before submission:
1. [ ] Create pipeline figure (see Figure 2 description in method.tex)
2. [ ] Fill `\tbd{---}` cells (run `/06-run-experiments update`)
3. [ ] Proofread for grammar and clarity
4. [ ] Verify page count after filling results
5. [ ] Generate PDF: `cd [output_dir] && pdflatex main && bibtex main && pdflatex main && pdflatex main`

### Citation statistics:
- Total references: [N]
- Venue breakdown: [X] ICML, [Y] NeurIPS, [Z] arXiv, ...
```
