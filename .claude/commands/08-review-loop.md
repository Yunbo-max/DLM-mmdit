---
name: 08-review-loop
description: Submit paper to paperreview.ai, parse feedback into 3 cases — (A) rejected → redesign method at 04, (B) borderline → improve experiments at 05, (C) accepted → minor writing fixes at 07.
---

Submit paper to paperreview.ai, parse feedback, classify into one of 3 cases, and loop back to the appropriate pipeline step.

## User input: $ARGUMENTS

Parse the user input to extract:
1. **Paper PDF or tex** — path to the paper. Default: `paper/main.tex` (compiles to PDF first)
2. **Mode:**
   - `"submit"` — compile and submit to paperreview.ai
   - `"parse [url]"` — parse feedback from a review URL or file
   - `"full"` — submit → parse → classify → loop
3. **Review URL or file** — if user already has feedback

Example invocations:
- `/08-review-loop` → full loop
- `/08-review-loop submit`
- `/08-review-loop parse https://paperreview.ai/result/xxx`
- `/08-review-loop parse paper/reviews/round1.md`

---

## Step 0: Pre-flight validation

Before compiling or submitting, validate:

1. **Paper exists?** Check `paper/main.tex`. If missing → STOP:
   > "Missing: paper/main.tex. Run `/07-write-paper [venue]` first to generate it."
2. **Paper compiles?** Quick check: scan for `\begin{document}` and `\end{document}`. If missing → STOP:
   > "paper/main.tex appears malformed. Re-run `/07-write-paper`."
3. **Style files exist?** Check for `.sty` or `.cls` files in `paper/`. If missing → warn:
   > "Warning: No style files found. Paper may not compile. Re-run `/07-write-paper` to download template."
4. **references.bib exists?** Check `paper/references.bib`. If missing → warn:
   > "Warning: No references.bib. Citations will be unresolved."
5. **Previous reviews?** Scan `paper/reviews/` for existing round files to auto-detect round number.

Only proceed to Step 1 after required checks pass.

---

## Step 1: Compile and submit

### 1a. Compile paper
```bash
cd paper/
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Verify `paper/main.pdf` exists.

### 1b. Submit to paperreview.ai

Try in order:
1. **API** (if available): `curl -X POST https://paperreview.ai/api/review -F "paper=@paper/main.pdf"`
2. **CLI** (if available): `paperreview submit paper/main.pdf`
3. **Manual**: Tell user to upload at https://paperreview.ai/ and provide the result URL

---

## Step 2: Parse reviewer feedback

Fetch and structure the review:

```markdown
# Review Round [N] — [Date]

## Scores
| Criterion | Score | Max |
|-----------|-------|-----|
| Novelty | X | 10 |
| Soundness | X | 10 |
| Clarity | X | 10 |
| Significance | X | 10 |
| **Overall** | **X** | **10** |

## Strengths
1. ...

## Weaknesses
W1: ...
W2: ...
W3: ...

## Questions
Q1: ...

## Per-Section Feedback
### Abstract: ...
### Introduction: ...
### Related Work: ...
### Method: ...
### Experiments: ...
### Writing: ...
```

Save to `paper/reviews/round[N].md`.

---

## Step 3: Classify into one of 3 cases

Based on the overall score and weakness patterns, classify:

### Case A: REJECTED (Overall < 5/10)
**Trigger:** Low novelty score, fundamental method flaws, "incremental contribution", "not suitable for venue"
**Action:** Loop back to `/04-spec-novel` → redesign the method

### Case B: BORDERLINE (Overall 5-7/10)
**Trigger:** Weak experiments, missing baselines/ablations, "not convincing results", "need more evaluation"
**Action:** Loop back to `/05-build-code` → improve code/experiments

### Case C: ACCEPTED (Overall > 7/10)
**Trigger:** Minor writing issues, small clarifications, "polish the presentation"
**Action:** Stay at `/07-write-paper` → minor writing fixes only

Present the classification to the user:

```
## Review Classification: CASE [A/B/C]

Overall score: X/10
Decision: [REJECTED / BORDERLINE / ACCEPTED]

Key weaknesses driving this classification:
1. [W_id]: [description] → [why this maps to case A/B/C]
2. [W_id]: [description]

Recommended action: Loop back to /[04/05/07]
```

---

## Step 4A: REJECTED — Redesign method (loop to 04)

When the method itself needs fundamental changes:

### 4A.1. Analyze what's wrong
From weaknesses, identify:
- Is the core idea not novel enough?
- Is the approach fundamentally flawed?
- Is the problem formulation wrong?
- Are we solving the wrong gap?

### 4A.2. Update gap analysis
- Re-read `docs/gap_solutions.md` — are there stronger directions we missed?
- Search for new papers reviewers mentioned: `/01-literature-search [cited paper]`
- Update `docs/research_gap.tex` if the gaps were wrong

### 4A.3. Re-run `/04-spec-novel`
- Redesign the method addressing reviewer concerns
- Generate new `specs.md` with revised architecture
- Generate new `contribution.tex` with stronger novelty claims
- Keep what worked, change what was criticized

### 4A.4. Then cascade through the pipeline:
```
/04-spec-novel  → new specs.md + contribution.tex
     ↓
/05-build-code  → rebuild code with new architecture
     ↓
/06-run-experiments  → re-run experiments
     ↓
/07-write-paper  → reassemble paper from new files
     ↓
/08-review-loop  → resubmit
```

---

## Step 4B: BORDERLINE — Improve experiments (loop to 05)

When the method is okay but experiments need strengthening:

### 4B.1. Identify what's missing
From weaknesses, identify:
- Missing baselines? → Add them
- Missing datasets? → Add them
- Missing ablations? → Add them
- Results not statistically significant? → More seeds
- Missing qualitative analysis? → Add examples
- Missing efficiency comparison? → Add profiling

### 4B.2. Re-run `/05-build-code`
- Add new baseline runner scripts
- Add new ablation configs
- Add new dataset download commands
- Update `experiments.tex` with new tables/rows

### 4B.3. Re-run `/06-run-experiments`
- Run new experiments (auto or manual)
- Update plan with new experiment dates
- Fill new result cells in experiments.tex

### 4B.4. Re-run `/07-write-paper`
- Reassemble paper with new experiment results
- Add discussion of new results
- Address reviewer questions in the text

### 4B.5. Resubmit
```
/05-build-code  → add baselines/ablations/datasets
     ↓
/06-run-experiments  → run new experiments
     ↓
/07-write-paper  → reassemble with new results
     ↓
/08-review-loop  → resubmit
```

---

## Step 4C: ACCEPTED — Minor writing fixes (stay at 07)

When the paper is fundamentally sound, just needs polish:

### 4C.1. Apply minor fixes directly
- Fix typos, grammar, unclear sentences
- Add missing citations to `references.bib`
- Clarify method description where reviewers were confused
- Add a sentence or two addressing reviewer questions
- Fix figure captions, table formatting
- Expand limitations section if requested

### 4C.2. Do NOT change:
- Method architecture
- Experiment setup
- Core contributions

### 4C.3. Recompile and verify
```bash
cd paper/
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### 4C.4. Optionally resubmit for score verification
```
/07-write-paper  → minor edits only
     ↓
/08-review-loop  → resubmit (optional, for verification)
```

---

## Step 5: Track changes and score progression

### 5a. Save change log

Create `paper/reviews/round[N]_changes.md`:
```markdown
# Changes — Round [N]
**Case:** [A/B/C]
**Action:** [Looped back to /04 | /05 | /07]

## Weaknesses addressed:
### W1: [description]
Action: [what changed]
Files: [list]

### W2: [description]
Action: [what changed]
Files: [list]

## Skills re-run:
- [list of skills re-run and what changed]
```

### 5b. Track score progression

Update `paper/reviews/progression.md`:
```markdown
# Review Score Progression

| Round | Date | Overall | Novelty | Soundness | Clarity | Case | Action |
|-------|------|---------|---------|-----------|---------|------|--------|
| 1 | [date] | X/10 | X | X | X | [A/B/C] | [action] |
| 2 | [date] | X/10 | X | X | X | [A/B/C] | [action] |
| Δ | | +X | +X | +X | +X | | |
```

---

## Step 6: Summary

```
## Review Loop — Round [N]

**Score:** X/10
**Case:** [A — Rejected | B — Borderline | C — Accepted]

### Action taken:
[A] Looped back to /04-spec-novel → redesigned [module/approach]
[B] Looped back to /05-build-code → added [N baselines, N ablations]
[C] Fixed [N] minor writing issues in /07-write-paper

### Score progression:
Round 1: X/10 → Round 2: X/10 (ΔX)

### Files modified:
- [list]

### Next:
[If Case A/B: resubmit after changes]
[If Case C: ready for conference submission]
```
