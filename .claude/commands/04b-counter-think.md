---
name: 04b-counter-think
description: "Optional parallel track for 04-spec-novel. Challenges field assumptions, searches cross-domain, generates first-principles proposals that go beyond gap-filling."
---

**Optional skill — run in parallel with `/04-spec-novel` to generate contrarian research directions alongside conventional gap-filling.**

The standard pipeline (01→02→03→04) reads papers → finds gaps → fills gaps. This produces solid incremental work. But real breakthroughs come from questioning assumptions, not filling holes. This skill provides that second perspective.

## User input: $ARGUMENTS

Parse the user input to extract:
1. **Gap analysis file** — path to `docs/research_gap.tex`. Default: `docs/research_gap.tex`
2. **Gap solutions file** — path to `docs/gap_solutions.md`. Default: `docs/gap_solutions.md`
3. **Literature file** — path to `docs/literature.md`. Default: `docs/literature.md`
4. **Output file** — Default: `docs/novel/counter_proposals.md`
5. **User's frustration** — optional: what personally frustrates the user about the field (the most valuable input)

Example invocations:
- `/04b-counter-think` → uses all defaults
- `/04b-counter-think frustration: "every VLN method needs a map but humans don't build explicit maps"`
- `/04b-counter-think frustration: "why do we need separate perception and planning modules?"`

---

## Step 0: Pre-flight validation

1. **research_gap.tex exists?** If missing → STOP:
   > "Missing: docs/research_gap.tex. Run `/02-research-gap` first."
2. **gap_solutions.md exists?** If missing → warn (not fatal — this skill can run with just the gap analysis):
   > "Warning: docs/gap_solutions.md not found. Running without solution context. This is fine — we're questioning the paradigm, not the solutions."
3. **literature.md exists?** If missing → warn (not fatal).

---

## Step 1: Derive frustrations from agent analysis

Instead of asking the user, have the assumption-extraction agents (Step 2) also identify the most likely frustrations a researcher working in this field would have, based on:
- Recurring reviewer complaints across papers
- Obvious limitations no paper acknowledges
- Patterns of avoidance (things the field systematically doesn't measure or address)

The agents derive the **seed insights** for Step 4 from the assumption analysis itself.

---

## Step 2: Extract field assumptions (PARALLEL)

Read `docs/research_gap.tex` and `docs/literature.md`. Extract every implicit assumption the field makes.

Launch **3 parallel Agent workers**, each questioning a different layer:

```
Agent 1 — Architecture assumptions:
  TASK: Read research_gap.tex and literature.md. List every architectural
        assumption the field makes. For each, state the assumption clearly
        and ask "is this actually necessary?"
  EXAMPLES:
    - "Assumption: Navigation needs an explicit map representation"
      → Question: Do humans build explicit grid maps? No. What if we don't?
    - "Assumption: Visual features need a pretrained backbone"
      → Question: What if raw pixels + simple conv is enough with the right objective?
    - "Assumption: Planning and perception must be separate modules"
      → Question: What if a single model does both end-to-end?
  MUST DO: Find at least 8 assumptions. Be specific — cite which papers make each assumption.
  MUST NOT DO: Do not suggest solutions yet. Just list and question.

Agent 2 — Problem formulation assumptions:
  TASK: Read research_gap.tex and literature.md. List every assumption about
        HOW THE PROBLEM IS DEFINED. Question whether the problem itself is right.
  EXAMPLES:
    - "Assumption: Navigation success = reaching the goal point"
      → Question: Is reaching a point the right metric? What about reaching a region?
    - "Assumption: The agent acts alone in a static environment"
      → Question: Real environments have other agents, moving objects, changing layouts
    - "Assumption: Instructions are given in natural language text"
      → Question: What if instructions are gestures, sketches, or demonstrations?
  MUST DO: Find at least 6 assumptions. Question the task definition, not just the solution.
  MUST NOT DO: Do not confuse method assumptions with problem assumptions.

Agent 3 — Evaluation assumptions:
  TASK: Read research_gap.tex and literature.md. List every assumption about
        HOW WE MEASURE SUCCESS. Question whether we're measuring the right thing.
  EXAMPLES:
    - "Assumption: SR and SPL are the right metrics"
      → Question: SPL penalizes exploration. But exploration is how humans navigate new places.
    - "Assumption: Simulated environments are valid testbeds"
      → Question: Sim-to-real gap is huge. Are we optimizing for the simulator?
    - "Assumption: Comparing to published baselines is fair"
      → Question: Different codebases, different hardware, different random seeds
  MUST DO: Find at least 5 assumptions. Be honest about what the field's metrics actually reward.
  MUST NOT DO: Do not just list metrics. Question why those metrics exist.
```

After all 3 agents complete, merge into a single **Assumption Registry**:

```markdown
## Assumption Registry

### Architecture Assumptions
| # | Assumption | Papers that make it | Question | Certainty it's necessary |
|---|-----------|--------------------|---------|-----------------------|
| A1 | Navigation needs explicit maps | [Paper A], [Paper B], ... | Do humans build grid maps? | LOW — maybe not |
| A2 | ... | ... | ... | HIGH / MEDIUM / LOW |

### Problem Formulation Assumptions
| # | Assumption | Papers | Question | Certainty |
|---|-----------|--------|----------|-----------|
| P1 | ... | ... | ... | ... |

### Evaluation Assumptions
| # | Assumption | Papers | Question | Certainty |
|---|-----------|--------|----------|-----------|
| E1 | ... | ... | ... | ... |
```

Mark assumptions with **LOW certainty** (= most likely to be wrong/unnecessary) — these are the attack surface for innovation.

---

## Step 3: Cross-domain search (PARALLEL)

For each LOW-certainty assumption, search OUTSIDE the field for how other domains solve the analogous problem. The key insight: **every problem in AI has been solved differently in another field.**

Launch **parallel Agent workers** — one per LOW-certainty assumption:

```
For each LOW-certainty assumption [AX / PX / EX]:

Agent(
  subagent_type: "general-purpose",
  run_in_background: true,
  prompt: "
    TASK: Find cross-domain solutions to challenge assumption: '[assumption text]'

    SEARCH THESE DOMAINS (not AI/ML papers):
    1. Neuroscience / cognitive science — how do biological systems handle this?
    2. Physics / mathematics — is there a principled formulation?
    3. Economics / game theory — how do strategic agents solve this?
    4. Ecology / biology — how do animals/organisms solve this?
    5. Engineering / control theory — what's the classical solution?
    6. Philosophy / epistemology — what are the foundational questions?

    For each domain where you find a relevant idea:
    - What is the concept? (1-2 sentences)
    - How does it challenge the assumption? (1 sentence)
    - Has anyone in AI tried this? If yes, what happened? If no, why not?
    - Concrete translation: what would this look like as a method? (2-3 sentences)

    MUST DO: Search at least 3 different domains. Find ACTUAL concepts, not vague analogies.
    MUST NOT DO: Do not search ML/AI papers. The whole point is to look OUTSIDE the field.

    Return results as structured text.
  "
)
```

Merge all cross-domain findings into a **Cross-Domain Insight Map**:

```markdown
## Cross-Domain Insight Map

### Assumption A1: "Navigation needs explicit maps"

| Domain | Concept | Challenge to assumption | AI translation |
|--------|---------|----------------------|----------------|
| Neuroscience | Place cells + grid cells (O'Keefe & Moser) | Brains don't build grid maps — they have distributed spatial representations | Replace explicit map with learned latent spatial memory |
| Ecology | Ant pheromone trails | No individual ant has a map — emergent navigation from local signals | Stigmergic navigation: leave traces, follow gradients |
| Control Theory | Model Predictive Control | No global map — replan from local state at every step | Receding horizon planner with visual observations only |

### Assumption P1: ...
...
```

---

## Step 4: Generate contrarian proposals

Now combine three sources:
1. **User's frustration** from Step 1 (the seed insight)
2. **LOW-certainty assumptions** from Step 2 (attack surfaces)
3. **Cross-domain concepts** from Step 3 (alternative paradigms)

For each promising combination, generate a **Contrarian Proposal**:

```markdown
## Contrarian Proposals

### Proposal CT-1: [Name — should sound provocative]
**Kills assumption:** [A1] — "[assumption text]"
**Inspired by:** [Domain: concept name]
**User frustration connection:** [how this relates to what the user said, if applicable]

**The idea in one sentence:**
[What if we... — should be a clear, concrete, testable claim]

**Why the field hasn't tried this:**
[Honest assessment — is it because it's bad, or because of inertia/fashion?]

**What it would look like:**
[3-5 sentences: concrete method description. Not vague — specific enough to implement.]

**Risk level:** [HIGH / MEDIUM / LOW]
- HIGH = might not work at all, but if it does, it's a paradigm shift
- MEDIUM = likely works differently, may or may not beat SOTA
- LOW = safe bet, clearly works, modest improvement

**Evidence it could work:**
[Any evidence from the cross-domain search, from failed papers, or from first principles]

**Minimum viable experiment:**
[The SMALLEST experiment that would tell you if this idea has legs. Not a full paper — a weekend hack.]

---

### Proposal CT-2: [Name]
...
```

Generate **at least 3 proposals**, spanning:
- At least 1 HIGH-risk (paradigm shift potential)
- At least 1 MEDIUM-risk (concrete alternative approach)
- At least 1 that connects to the user's frustration (if provided)

---

## Step 5: Score and compare with mainstream

Score each proposal on a DIFFERENT scale than 04-spec-novel (because these are not gap-fillers):

| Dimension | 0.0 | 0.5 | 1.0 |
|-----------|-----|-----|-----|
| **Paradigm Shift** | Incremental improvement to existing paradigm | New approach within existing framework | Challenges a fundamental assumption |
| **Testability** | Would need years to validate | Needs full system to test | Can test core claim with a weekend experiment |
| **Surprise Factor** | Reviewers would say "obvious" | Reviewers would say "interesting" | Reviewers would say "wait, that actually works?" |

```markdown
## Contrarian Proposals — Ranked

| Rank | Proposal | Kills Assumption | Paradigm Shift | Testability | Surprise | Risk |
|------|----------|-----------------|---------------|-------------|----------|------|
| 1 | CT-1: [name] | A1 | 0.9 | 0.7 | 0.8 | HIGH |
| 2 | CT-3: [name] | P2 | 0.6 | 0.9 | 0.7 | MEDIUM |
| 3 | CT-2: [name] | A3 | 0.8 | 0.4 | 0.9 | HIGH |

### vs. Mainstream candidates (from /04-spec-novel, if available):

| Source | Top Candidate | Novelty | Risk | Type |
|--------|--------------|---------|------|------|
| 04-spec-novel | [gap-filling method] | 0.85 | LOW | Incremental — fills gap, extends existing paradigm |
| 04b-counter-think | CT-1: [name] | 0.90 | HIGH | Contrarian — challenges assumption, new paradigm |
```

---

## Step 6: Present to human — honest tradeoffs

```
## Two Paths Forward

### Path A: Gap-filling (from /04-spec-novel)
Your best candidates fill identified gaps in the literature.
- **Pros:** Reviewers understand it. Baselines exist. Clearer path to publication.
- **Cons:** Incremental. Another paper in the same paradigm. Forgettable in 2 years.
- **Best for:** Getting a paper accepted. Building your publication record.

### Path B: Assumption-challenging (from /04b-counter-think)
Your contrarian proposals challenge what the field takes for granted.
- **Pros:** If it works, it's memorable. Opens a new research direction. Fun to work on.
- **Cons:** Higher risk. Reviewers may not get it. May need more experiments to convince.
- **Best for:** Making a real contribution. Doing research you're proud of.

### Path C: Combine both
Use the gap-filling method as your main contribution (safe),
and add the contrarian idea as a secondary contribution or ablation (exciting).
This hedges your bets.

### My honest recommendation:
[Based on the specific proposals and the user's stated frustration,
give a specific recommendation with reasoning. Be honest about which
path is riskier and which is safer.]

Which path? Or pick specific proposals by ID
(e.g., "CT-1 + candidate 3 from 04-spec-novel").
```

**WAIT for the user's response.**

If the user picks any contrarian proposal → feed it into 04-spec-novel's frozen scope as a selected candidate. It then flows through the normal pipeline (05→06→07→08).

---

## Step 7: Write output

Save `docs/novel/counter_proposals.md`:

```markdown
# Counter-Think Analysis

**Date:** [today]
**Field:** [from research_gap.tex]
**User frustration:** [what they said, or "not provided"]

## Assumption Registry
[Full table from Step 2]

## Cross-Domain Insight Map
[Full table from Step 3]

## Contrarian Proposals
[All proposals from Step 4 with scores]

## Comparison with Mainstream
[Table from Step 5]

## Human Decision
**Path chosen:** [A / B / C]
**Selected proposals:** [list]
**Reasoning:** [what the user said]
```

---

## Step 8: Integration with 04-spec-novel

If the user selected any contrarian proposal:

1. **Create a method spec** for the contrarian idea — same format as 04-spec-novel's candidate designs (architecture, input/output, modules)
2. **Add to the frozen scope** in 04-spec-novel — the contrarian proposal becomes another selected candidate
3. **Flag it as contrarian** in specs.md and contribution.tex — the paper should frame it as "challenging assumption X" rather than "filling gap Y"

The contrarian candidate then flows through the normal pipeline:
```
04b-counter-think → feeds into 04-spec-novel frozen scope
  ↓
05-build-code → builds the contrarian method
  ↓
06-run-experiments → runs experiments (the "minimum viable experiment" first!)
  ↓
07-write-paper → frames it as assumption-challenging, not gap-filling
  ↓
08-review-loop → if rejected, the contrarian framing may need stronger evidence
```

---

## Summary output

```
## Counter-Think Analysis Complete

**Assumptions found:** [N] ([M] marked LOW certainty)
**Cross-domain concepts found:** [N] across [M] domains
**Contrarian proposals generated:** [N]

### Top contrarian proposal:
CT-[X]: [name]
  Challenges: [assumption]
  Inspired by: [domain: concept]
  Risk: [HIGH/MEDIUM/LOW]
  Minimum experiment: [description]

### Human decision: [Path A / B / C]
### Selected for development: [list]

### Output:
- docs/novel/counter_proposals.md

### Next:
[If proposals selected → they're in 04-spec-novel's frozen scope → run /05-build-code]
[If no proposals selected → proceed with /04-spec-novel's mainstream candidates]
```
