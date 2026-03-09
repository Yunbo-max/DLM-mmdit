---
name: 01-literature-search
description: Search academic venues (ICLR, CVPR, ICCV, ECCV, NeurIPS, ICRA, CoRL, RSS, arXiv, HuggingFace) for recent papers on given topics, find GitHub repos, and compile into a literature review markdown file. Supports paper-seeded mode for exploring related work from a seed paper.
---

Search for recent papers on user-specified topics and update a literature review markdown file.

## Two Modes

This skill operates in **two modes** depending on user input:

### Mode A: Topic-based search (original)
Triggered when user provides keyword topics directly.

### Mode B: Paper-seeded exploration (new)
Triggered when user provides a **paper URL** (arXiv, OpenReview, conference page, PDF link) or the conversation contains a seed paper. The skill will:
1. Read and analyze the seed paper's contributions
2. Ask the user for the **exploration direction** (if not already provided)
3. Cross-pollinate search across related domains based on the paper's ideas + user direction

---

## User input: $ARGUMENTS

Parse the user input to detect the mode:

### Detecting Mode B (paper-seeded):
If the input contains any of: an arXiv URL, OpenReview URL, Semantic Scholar URL, conference paper URL, PDF link, or the user says "this paper" / "seed paper" / provides a paper title with venue — use **Mode B**.

### Detecting Mode A (topic-based):
If the input is comma-separated keywords with no paper URL — use **Mode A**.

### Mode A parsing:
1. **Topics** — comma-separated keywords (e.g., "VLM navigation, terrain planning, multi-agent")
2. **Year** — if a 4-digit year is found (e.g., "2025"), search from that year through the current year (e.g., "2024" means search 2024, 2025, 2026); if no year given, default to current year only
3. **Output file** — if a file path is given, use it; otherwise default to `docs/literature.md`

### Mode B parsing:
1. **Paper URL/reference** — the seed paper to analyze
2. **Direction** — the exploration angle (e.g., "how to make this work for outdoor navigation", "extend to multi-agent", "replace the backbone with a VLM"). If not provided, ASK the user.
3. **Year** — same as Mode A; default to current year
4. **Output file** — same as Mode A; default to `docs/literature.md`

Example invocations:
- `/literature-search VLM navigation, object goal navigation, semantic mapping`
- `/literature-search embodied AI, multi-agent coordination 2025`
- `/literature-search SLAM, visual odometry, depth estimation 2024 docs/slam-papers.md`
- `/literature-search https://arxiv.org/abs/2312.03275` → will read paper, then ask for direction
- `/literature-search https://arxiv.org/abs/2312.03275 direction: extend to outdoor terrain-aware navigation 2024`

---

## Mode B Steps (paper-seeded exploration)

If Mode B is detected, execute these steps BEFORE Step 1.

### Step B1: Read and analyze the seed paper

Use **WebFetch** to read the paper URL. Extract:

1. **Title** and authors
2. **Core contributions** — list 3-5 key technical contributions (be specific: architecture, loss function, representation, algorithm)
3. **Key methods** — what techniques/components does the paper use? (e.g., "frontier-based exploration", "cosine similarity scoring", "fast marching method", "scene graphs", "energy-based models")
4. **Limitations** — what does the paper NOT do or acknowledge as future work?
5. **Domain** — what field is this paper in? (e.g., indoor ObjectNav, outdoor autonomous driving, manipulation)

Present this analysis to the user as a summary:

```
## Seed Paper Analysis: [Title]

**Contributions:**
1. ...
2. ...

**Key Methods:** [method1], [method2], ...

**Limitations / Future Work:** ...

**Domain:** ...
```

### Step B2: Get exploration direction from user

If the user has NOT already provided a direction, ASK them:

> "I've analyzed the paper. What direction do you want to explore? For example:
> - How to extend [method X] to [new domain]?
> - Alternative approaches to [contribution Y]?
> - How to combine this with [other technique]?
> - What papers improve upon [specific component]?"

Wait for the user's response before proceeding.

### Step B3: Generate cross-domain search topics from seed paper + direction

Based on the seed paper's contributions + the user's exploration direction, generate **search topics** that span MULTIPLE related domains. The goal is cross-pollination — finding ideas from adjacent fields.

For each seed paper contribution, brainstorm which OTHER fields solve similar problems differently:

| Seed paper concept | Cross-domain search angles |
|-------------------|---------------------------|
| Frontier scoring with VLM | Energy-based exploration, information-gain exploration, curiosity-driven RL, active perception |
| Semantic map building | Scene graphs, neural radiance fields for mapping, BEV perception, spatial memory networks |
| Object goal navigation | Visual search in surveillance, target-driven grasping, drone search-and-rescue |
| Multi-agent coordination | Swarm robotics, game-theoretic planning, auction-based task allocation |

Generate **6-12 cross-domain topic keywords** that combine:
- The seed paper's methods applied to the user's target direction
- Alternative methods from adjacent fields that solve the same sub-problems
- Papers that cite or are cited by the seed paper (search `cites:[arxiv_id]` or `cited by [paper title]`)

These generated topics then flow into **Step 1** as if the user had typed them directly.

### Step B4: Search for citing and related papers

In addition to the venue searches in Step 2, also run:
- `[seed paper title] cited by` — papers that build on this work
- `[seed paper title] related work` — papers in the same neighborhood
- `semantic scholar [seed paper title]` — for the citation graph
- For each author: `[author name] [year] [direction topic]` — what else the authors published

Merge these results into the main paper collection.

---

## Step 0: Pre-flight validation

Before burning tokens on search, validate inputs:

1. **Mode B — seed paper reachable?** Try to fetch the paper URL. If 404 or unreachable → STOP with actionable error:
   > "Cannot reach [URL]. Check the link or provide a different paper URL."
2. **Mode B — direction provided?** If not, ask (Step B2). Do NOT proceed to search without a direction.
3. **Output path writable?** Check parent directory exists: `ls -d $(dirname [output_file])`. If not → STOP:
   > "Output directory does not exist. Create it with: `mkdir -p docs/`"
4. **Existing output?** If output file already exists, warn user: "Found existing [file]. New papers will be appended, not overwritten."

Only proceed to Step 1 after ALL checks pass.

---

## Step 1: Expand topics into rich search queries

For EACH user topic keyword, generate 3-5 expanded search variations to maximize recall. For example:

| User keyword | Expanded queries |
|-------------|-----------------|
| "VLM navigation" | "vision-language model navigation", "VLM embodied navigation", "large vision model robot navigation", "LVLM spatial reasoning navigation" |
| "terrain planning" | "terrain-aware navigation", "traversability estimation", "off-road autonomous navigation", "costmap generation terrain" |
| "multi-agent" | "multi-agent navigation", "multi-robot exploration", "cooperative embodied agents", "decentralized multi-agent planning" |

Use these expanded queries in all searches below.

## Step 2: Search venues (use WebSearch)

Search ALL of the following venues for EACH expanded topic query + year:

### Top ML conferences
| Venue | How to search |
|-------|--------------|
| **ICLR** | `site:openreview.net ICLR [year] [topic]` — check oral/spotlight/poster |
| **CVPR** | `CVPR [year] [topic] accepted` |
| **ICCV** | `ICCV [year] [topic] accepted` (odd years only: 2023, 2025) |
| **ECCV** | `ECCV [year] [topic] accepted` (even years only: 2024, 2026) |
| **NeurIPS** | `NeurIPS [year] [topic]` |

### Robotics venues
| Venue | How to search |
|-------|--------------|
| **ICRA** | `ICRA [year] [topic]` |
| **CoRL** | `CoRL [year] [topic]` |
| **RSS** | `RSS [year] [topic]` |
| **IROS** | `IROS [year] [topic]` |
| **RA-L** | `IEEE RA-L [year] [topic]` |

### Preprint & trending
| Source | How to search |
|--------|--------------|
| **HuggingFace Papers** | `site:huggingface.co/papers [topic] [year]` |
| **arXiv** | `arxiv [topic] [year]` in cs.RO, cs.CV, cs.AI |
| **GitHub** | `awesome [topic]` for awesome-lists |

### OpenReview — reviewer comments (critical for understanding field weaknesses)

For papers found on OpenReview (ICLR, NeurIPS, EMNLP, etc.), also fetch **reviewer comments and author responses**. These reveal what the community thinks is actually weak or missing.

| What to search | How |
|---------------|-----|
| **Reviews of found papers** | `site:openreview.net [paper title] reviews` |
| **Topic forums** | `site:openreview.net [topic] [venue] [year]` |
| **Rejected papers** | `site:openreview.net [topic] [venue] [year] reject` — rejected papers often identify real problems but with flawed solutions |

For each paper found on OpenReview, extract:
- **Reviewer scores** (e.g., 6/5/7)
- **Key weaknesses raised** — what reviewers think is missing or flawed (1-2 lines)
- **Suggested improvements** — what reviewers wish the paper had done
- **Author rebuttals** — how authors responded to criticism

Add a `Reviews` column to the paper tables for papers that have OpenReview reviews:

```
| Paper | Venue | Year | ... | Reviews Summary |
|-------|-------|------|-----|-----------------|
| ... | ICLR | 2025 | ... | Scores: 6/5/7. Weakness: "no real-robot eval, limited datasets" |
```

This review data is especially valuable for `/02-research-gap` (reviewer weaknesses = validated gaps) and `/04b-counter-think` (reviewer assumptions = attack surfaces).

## Step 3: For each paper found, collect

- Title
- Venue and year
- arXiv link (if available)
- GitHub repo link: search `github.com [paper name]` or `github.com [arxiv id]`
- 1-line summary (what it does + key result)

## Step 4: GitHub repo status

For EVERY paper:
- If GitHub repo exists: include the URL as `[org/repo](url)`
- If no public code: mark with `:x:`

## Step 5: Output

### Mode A output:

Create or update the output markdown file with this structure:

```markdown
# Literature Review: [topics]

## Sources
| Source | Coverage |
|--------|----------|
| ... | ... |

## 1. [Topic Category 1]
| Paper | Venue | Year | Link | GitHub | Summary |
|-------|-------|------|------|--------|---------|
| ... | ... | ... | ... | ... | ... |

## 2. [Topic Category 2]
...

## N-2. GitHub Awesome-Lists
| Repository | Stars | Link |
|-----------|-------|------|

## N-1. Surveys
| Title | Year | Link |
|-------|------|------|

## N. Key Trends
1. ...
```

### Mode B output:

For paper-seeded exploration, the output follows this structure — analysis first, then other papers, then top 10:

```markdown
# Related Work: [Seed Paper Title]

## Seed Paper Analysis
**[Title]** | [Venue] [Year] | [arXiv link]

**Contributions:**
1. ...
2. ...
3. ...

**Key Methods:** [method1], [method2], ...

**Limitations / Future Work:** ...

**Exploration Direction:** [user's direction]

---

## Other Related Papers

Papers found during search that are relevant but do NOT have a public GitHub repo. Listed for reference only.

| Paper | Venue | Year | Link | Summary |
|-------|-------|------|------|---------|
| ... | ... | ... | ... | ... |

---

## Top 10 Most Related Papers (with Code)

Ranked by relevance to the seed paper + exploration direction.
**STRICT RULE: Every paper here MUST have a working GitHub link. No GitHub repo = not in this list.**

| Rank | Paper | Venue | Year | Link | GitHub | Why Related |
|------|-------|-------|------|------|--------|-------------|
| 1 | ... | ... | ... | ... | [org/repo](url) | 1-line connection to seed paper |
| 2 | ... | ... | ... | ... | [org/repo](url) | ... |
| ... | ... | ... | ... | ... | ... | ... |
| 10 | ... | ... | ... | ... | [org/repo](url) | ... |

---

## Top 10 Most Related Papers (without Code)

Ranked by relevance. These papers have NO public GitHub repo but are important for understanding the field. `/02-research-gap` will analyze these from their PDF/abstract.

| Rank | Paper | Venue | Year | Link | Key Method/Result | Why Related |
|------|-------|-------|------|------|-------------------|-------------|
| 1 | ... | ... | ... | [arXiv/doi](url) | 1-line method + main result | 1-line connection to seed paper |
| 2 | ... | ... | ... | [arXiv/doi](url) | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |
| 10 | ... | ... | ... | [arXiv/doi](url) | ... | ... |
```

### Rules for Mode B:
- **Two Top 10 lists**: one with code (for cloning in 02), one without code (for PDF reading in 02)
- Papers with GitHub repos go in the "with Code" list ONLY — no overlap between the two lists
- If fewer than 10 papers qualify for either list, output however many qualify
- Search thoroughly for repos before putting a paper in the "without Code" list: `github.com [paper title]`, `github.com [arxiv id]`, check the paper's PDF/abstract page for code links
- The **"Why Related"** column must explain the specific connection (not generic)
- The **"Key Method/Result"** column in the no-code list should capture what the paper contributes (since we can't read their code)
- **"Other Related Papers"** section goes BEFORE both top 10 lists — overflow papers that didn't make either top 10
- No cross-domain sections, no surveys section, no research gaps section — keep it clean: analysis → others → top 10 with code → top 10 without code

### Rules for Mode A (unchanged):
- Group papers into logical categories based on the user's topics
- Each topic keyword should map to at least one section
- Add "GitHub Awesome-Lists", "Surveys", and "Key Trends" as final sections
- Each paper entry MUST have all required columns per the template above

## Step 6: If the output file already exists

- Read the existing file first
- ADD new papers to existing sections (do not remove old entries)
- If a paper already exists, skip it (no duplicates)
- Update the Key Trends section if new trends emerge
