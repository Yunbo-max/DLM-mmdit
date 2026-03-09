---
name: 06-run-experiments
description: Parse experiments.tex for experiment plan, run automatically or provide manual commands, track progress, and fill \tbd cells with real results.
---

Parse experiments.tex to extract the experiment plan (IDs, tables, commands), then run experiments or support manual runs, and update `\tbd{---}` cells with real results.

## User input: $ARGUMENTS

Parse the user input to detect the MODE:

### Mode detection:

1. **"auto"** → Extract plan from experiments.tex + run all experiments automatically + update `\tbd{---}` cells after each run
2. **"manual"** → Extract plan from experiments.tex + print copy-paste commands grouped by day + provide update command
3. **"update [path]"** → Parse results from a completed experiment and fill `\tbd{---}` cells in experiments.tex
4. **"update"** (no path) → Scan entire `results/` for new result files and batch-update experiments.tex
5. **No arguments** → ASK the user: automatic or manual?

### Also parse:
- **Experiments tex** — path to experiments.tex. Default: `docs/novel/experiments.tex`
- **Project directory** — where the code lives. Default: detect from experiments.tex path
- **Results directory** — where results are stored. Default: `results/`

Example invocations:
- `/06-run-experiments` → asks auto or manual
- `/06-run-experiments auto` → run everything automatically
- `/06-run-experiments manual` → print grouped commands for human
- `/06-run-experiments update results/EXP-M01/output.json` → parse result, fill tex cells
- `/06-run-experiments update` → scan results/ for all new results, batch-update

---

## Step 0: Pre-flight — Parse experiments.tex

Before anything else, READ experiments.tex and extract its structure:

### 0a. Validate experiments.tex exists

Check `docs/novel/experiments.tex`. If missing → STOP:
> "Missing: docs/novel/experiments.tex. Run `/05-build-code` first."

### 0b. Parse the Experiment Index table

Find `\label{tab:exp_index}` and parse the longtable. This is the master registry:

```
EXP-B01--B06 | Baseline   | 6 baselines on Yelp        | Table~\ref{tab:yelp_results}
EXP-M01      | Main       | LSME r=0.3 Yelp            | Table~\ref{tab:yelp_results}
EXP-A01--A05 | Ablation   | Mask ratio sweep            | Table~\ref{tab:ablation_mask_ratio}
EXP-G01--G03 | Geometry   | SSS/MTS interpolation       | Table~\ref{tab:geometry}
EXP-E01--E03 | Efficiency | LSME + baseline profiling   | Table~\ref{tab:efficiency}
```

Build a mapping: **EXP-ID → (table label, row name, column names)**

For example:
- `EXP-B01` → `tab:yelp_results`, row="MDLM (uncond.)", columns=[Attr-Acc, PPL, BLEU, BERTScore, Distinct-2]
- `EXP-M01` → `tab:yelp_results`, row="LSME ($r$=0.3)", columns=[Attr-Acc, PPL, BLEU, BERTScore, Distinct-2]
- `EXP-A01` → `tab:ablation_mask_ratio`, row="$r = 0.1$", columns=[Attr-Acc, PPL, BLEU, BERTScore]

### 0c. Count `\tbd{---}` cells

Count how many `\tbd{---}` markers exist. This is the total number of cells to fill.
Report: "Found [N] experiments across [M] tables with [K] cells to fill."

### 0d. Parse Run Commands Reference

Find `\section{Run Commands Reference}` and extract:
- Environment variable setup (CKPT, LATENT_DIR, META, input files)
- LSME editing commands
- Evaluation commands
- Geometry commands
- Ablation sweep commands

These become the actual commands in the plan.

### 0e. Parse the 7-day schedule

Find `\subsection{Experimental Protocol}` and extract the day groupings:
- Day 1: Baselines (10 runs, parallelizable)
- Day 2: Main LSME on Yelp + Amazon (5 runs)
- Day 3: Main LSME on GYAFC + full eval (5 runs)
- Day 4: Ablations — mask ratio, mask mode (8 runs)
- Day 5: Ablations — steps, temperature, latent source (13 runs)
- Day 6: Geometry analysis (4 runs)
- Day 7: Efficiency profiling (3 runs)

### 0f. Validate code exists

Check that the project's Python modules exist:
- Check for `scripts/run_lsme.py` or equivalent entry points
- Check for `scripts/run_eval.py`
- Check for `scripts/run_baselines.py`
- Check for `scripts/run_geometry.py`

If entry points are missing, WARN but don't stop — commands may need adjustment.

### 0g. Check for existing results

Scan `results/` for any EXP-ID directories that already have results:
```
results/EXP-B01/output.json → already done
results/EXP-M01/output.json → already done
```

Report: "[X] experiments already have results. [Y] remaining."

---

## Step 1: Ask execution mode

If mode not specified, ask the user:

> "How do you want to run experiments?
> - **Auto**: I'll run each experiment, collect results, and fill `\tbd{---}` cells automatically
> - **Manual**: I'll print commands grouped by day. You run them, then use `/06-run-experiments update` to fill results"

---

## Step 2: Build experiment plan from experiments.tex

The plan is NOT invented from scratch — it is EXTRACTED from experiments.tex.

### 2a. Build the master plan table

From the experiment index (Step 0b), run commands (Step 0d), and schedule (Step 0e), construct:

| ID | Day | Category | Description | Table | Row | Command | Status |
|----|-----|----------|-------------|-------|-----|---------|--------|
| EXP-B01 | 1 | Baseline | MDLM on Yelp | tab:yelp_results | MDLM (uncond.) | `python -m lsme.scripts.run_baselines ...` | pending |
| EXP-B02 | 1 | Baseline | ReMDM on Yelp | tab:yelp_results | ReMDM | `python -m lsme.scripts.run_baselines ...` | pending |
| ... | | | | | | | |
| EXP-M01 | 2 | Main | LSME r=0.3 Yelp | tab:yelp_results | LSME ($r$=0.3) | `python -m lsme.scripts.run_lsme --mask_ratio 0.3 ...` | pending |
| ... | | | | | | | |
| EXP-A01 | 4 | Ablation | Mask ratio r=0.1 | tab:ablation_mask_ratio | $r = 0.1$ | `python -m lsme.scripts.run_lsme --mask_ratio 0.1 ...` | pending |

Mark experiments that already have results (Step 0g) as `done`.

### 2b. Generate concrete commands

For each experiment, construct the full command from the Run Commands Reference:

**Baselines (EXP-B series):**
```bash
python -m lsme.scripts.run_baselines \
    --method [mdlm|remdm|latentops|diffuser|planner|ld4lg] \
    --input_file $YELP_INPUT \
    --output_dir results/EXP-B01/ \
    --device cuda
```

**Main LSME (EXP-M series):**
```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR --metadata_file $META \
    --attribute [sentiment|topic|formality] --target_value [positive|books|formal] \
    --mask_ratio [0.3|0.5|0.7] --steps 100 --temperature 1.0 \
    --mask_mode random \
    --input_file [YELP_INPUT|AMAZON_INPUT|GYAFC_INPUT] \
    --output_file results/EXP-M0X/output.json \
    --batch_size 32 --device cuda
```

**Ablations (EXP-A series):**
Sweep one parameter at a time, fixing others to defaults from `\label{tab:lsme_hparams}`:
```bash
# Mask ratio sweep (EXP-A01 to A05)
for r in 0.1 0.3 0.5 0.7 0.9; do
    python -m lsme.scripts.run_lsme \
        --mask_ratio $r --steps 100 --temperature 1.0 --mask_mode random \
        --output_file results/ablation_mr/r${r}/output.json ...
done

# Mask mode sweep (EXP-A06 to A08)
for mode in random entropy suffix; do
    python -m lsme.scripts.run_lsme \
        --mask_ratio 0.3 --mask_mode $mode ...
done

# Steps sweep (EXP-A09 to A12)
for s in 10 50 100 500; do
    python -m lsme.scripts.run_lsme \
        --mask_ratio 0.3 --steps $s ...
done

# Temperature sweep (EXP-A13 to A16)
for t in 0.5 0.8 1.0 1.2; do
    python -m lsme.scripts.run_lsme \
        --mask_ratio 0.3 --temperature $t ...
done

# Latent source sweep (EXP-A17 to A21)
for src in zeros random centroid nn directional; do
    python -m lsme.scripts.run_lsme \
        --mask_ratio 0.3 --latent_source $src ...
done
```

**Geometry (EXP-G series):**
```bash
python -m lsme.scripts.run_geometry \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR --metadata_file $META \
    --attribute [sentiment|topic|formality] \
    --source_value [negative|electronics|informal] \
    --target_value [positive|books|formal] \
    --n_pairs 10 --n_points 10 --n_samples 5 \
    --output_dir results/EXP-G0X/ --device cuda
```

**Efficiency (EXP-E series):**
```bash
python -m lsme.scripts.run_lsme \
    --mask_ratio 0.3 --steps [50|100] --profile \
    --batch_size 32 --device cuda \
    --output_file results/EXP-E0X/output.json
```

**Evaluation (run after each editing experiment):**
```bash
python -m lsme.scripts.run_eval \
    --results_file results/EXP-XXX/output.json \
    --output_dir results/EXP-XXX/eval/ \
    --target_attribute [POSITIVE|BOOKS|FORMAL] \
    --device cuda
```

### 2c. Identify reusable experiments

Some experiments share outputs (noted in experiments.tex protocol):
- EXP-M01 (LSME r=0.3 Yelp) = EXP-A02 (mask ratio r=0.3 ablation baseline)
- EXP-M01 (LSME r=0.3 100 steps) = EXP-A11 (steps=100 ablation baseline)

Mark these as `reuse: EXP-M01` — do NOT re-run, just copy the result.

### 2d. Save the plan

Write `docs/novel/experiment_plan.md` with the full schedule, commands, and status.

---

## Step 3 (Auto mode): Run experiments by dependency group

### Dependency groups (from experiments.tex protocol):

```
Group 1 (Day 1): Baselines — EXP-B01 to EXP-B10
  All independent, can run in parallel.
  → Launch background agents, one per baseline.

Group 2 (Day 2-3): Main LSME — EXP-M01 to EXP-M07
  Independent per dataset. Requires checkpoint.
  → Launch background agents, one per (dataset, mask_ratio) pair.

Group 3 (Day 4-5): Ablations — EXP-A01 to EXP-A21
  Independent of each other. Some reuse main results.
  → Skip reused experiments. Launch agents for new ones.

Group 4 (Day 6): Geometry — EXP-G01 to EXP-G04
  Independent per attribute pair. Requires checkpoint.
  → Launch background agents, one per attribute pair.

Group 5 (Day 7): Efficiency — EXP-E01 to EXP-E03
  Sequential (profiling sensitive to other GPU load).
  → Run one at a time.
```

### For each group:

#### 3a. Pre-run check
```bash
# Verify checkpoint exists
ls $CKPT
# Verify latent directory
ls $LATENT_DIR
# Verify input data
ls $YELP_INPUT $AMAZON_INPUT $GYAFC_INPUT
```

If missing → tell user what to download/prepare and skip group.

#### 3b. Launch parallel workers

For each experiment in the group:
```
Agent(
    subagent_type: "general-purpose",
    run_in_background: true,
    prompt: "
      Run this experiment:
      ID: EXP-M01
      Command: python -m lsme.scripts.run_lsme [full args from 2b]

      Then evaluate:
      Command: python -m lsme.scripts.run_eval --results_file results/EXP-M01/output.json ...

      Return a JSON with these metrics:
      {attr_acc, ppl, bleu, bertscore, distinct_2, edit_dist, self_bleu, tps}
      (include whichever metrics were computed)
    "
)
```

#### 3c. After each experiment completes — fill `\tbd{---}` cells

1. **Parse the result** JSON from `results/EXP-XXX/eval/results.json`

2. **Map result to table cells** using the EXP-ID → (table, row, columns) mapping from Step 0b:
   - EXP-M01 → `tab:yelp_results`, row="LSME ($r$=0.3)", fill columns:
     - Attr-Acc ↑: `0.823` → `82.3`
     - PPL ↓: `45.2` → `45.2`
     - BLEU ↑: `0.312` → `31.2`
     - BERTScore ↑: `0.891` → `89.1`
     - Distinct-2 ↑: `0.734` → `73.4`

3. **Edit experiments.tex:**
   - Find the row in the correct table (match row text + `\tbd{---}`)
   - Replace each `\tbd{---}` in that row with the formatted number
   - Format: round to 1 decimal place, percentage where appropriate

   Example — before:
   ```latex
   \textbf{LSME} ($r$=0.3) & \tbd{---} & \tbd{---} & \tbd{---} & \tbd{---} & \tbd{---} \\
   ```
   After:
   ```latex
   \textbf{LSME} ($r$=0.3) & 82.3 & 45.2 & 31.2 & 89.1 & 73.4 \\
   ```

4. **Bold best-in-column** after ALL rows in a table are filled:
   - For ↑ metrics: `\best{82.3}` if 82.3 is the highest in that column
   - For ↓ metrics: `\best{45.2}` if 45.2 is the lowest in that column
   - Only apply `\best{}` when ALL rows in the table have real values (no remaining `\tbd{---}`)

5. **Update experiment_plan.md:**
   - Change Status: `pending` → `done`
   - Fill Result: `Acc=82.3, PPL=45.2`

6. **Print progress:**
   ```
   EXP-M01 LSME r=0.3 Yelp — DONE
   Attr-Acc: 82.3% | PPL: 45.2 | BLEU: 31.2 | BERTScore: 89.1 | Distinct-2: 73.4
   Progress: [X]/48 complete, [K] \tbd cells remaining
   Next group: [group name]
   ```

#### 3d. Handle failures
- Mark as `failed` in plan
- Log error in Result column
- Leave `\tbd{---}` cells unchanged in experiments.tex
- Skip to next experiment
- Do NOT retry — human can retry with `/06-run-experiments update`

#### 3e. Handle reused experiments
For experiments marked as reuse (Step 2c):
- Copy results from the source experiment: `cp -r results/EXP-M01/ results/EXP-A02/`
- Fill the corresponding `\tbd{---}` cells in the ablation table
- Mark as `done (reused from EXP-M01)` in plan

#### 3f. After all groups complete
Print final summary and verify `\tbd{---}` count is zero.

---

## Step 4 (Manual mode): Print grouped commands

### 4a. Print environment setup

```bash
# Environment setup — run once before all experiments
export CKPT=checkpoints/mmdit_latent
export LATENT_DIR=data/latents
export META=data/latents/metadata.json
export YELP_INPUT=data/yelp_negatives.txt
export AMAZON_INPUT=data/amazon_electronics.txt
export GYAFC_INPUT=data/gyafc_informal.txt
```

### 4b. Print commands grouped by day

For each day group from the schedule:

```
## Day 1: Baselines (EXP-B01 to EXP-B10)
Can run in parallel if multiple GPUs available.

# EXP-B01: MDLM on Yelp
python -m lsme.scripts.run_baselines --method mdlm --input_file $YELP_INPUT --output_dir results/EXP-B01/ --device cuda

# EXP-B02: ReMDM on Yelp
python -m lsme.scripts.run_baselines --method remdm --input_file $YELP_INPUT --output_dir results/EXP-B02/ --device cuda

...

# After Day 1 runs complete, update results:
/06-run-experiments update

---

## Day 2: Main LSME on Yelp + Amazon (EXP-M01 to EXP-M05)

# EXP-M01: LSME r=0.3 Yelp
python -m lsme.scripts.run_lsme --checkpoint_path $CKPT --latent_dir $LATENT_DIR --metadata_file $META \
    --attribute sentiment --target_value positive --mask_ratio 0.3 --steps 100 --temperature 1.0 \
    --mask_mode random --input_file $YELP_INPUT --output_file results/EXP-M01/output.json \
    --batch_size 32 --device cuda

# Evaluate:
python -m lsme.scripts.run_eval --results_file results/EXP-M01/output.json --output_dir results/EXP-M01/eval/ \
    --target_attribute POSITIVE --device cuda

...

# After Day 2 runs complete:
/06-run-experiments update

---

## Day 4: Ablations — mask ratio + mask mode (EXP-A01 to EXP-A08)
Note: EXP-A02 (r=0.3) reuses EXP-M01 results — skip if already done.

# EXP-A01: Mask ratio r=0.1
python -m lsme.scripts.run_lsme --mask_ratio 0.1 --steps 100 --temperature 1.0 --mask_mode random \
    --input_file $YELP_INPUT --output_file results/EXP-A01/output.json ...

...
```

### 4c. Print reuse instructions

```
## Reusable experiments (do NOT re-run):
- EXP-A02 = EXP-M01 (LSME r=0.3, 100 steps)
- EXP-A07 = EXP-M01 (random mask mode at r=0.3)
- EXP-A11 = EXP-M01 (100 steps at r=0.3)
- EXP-A15 = EXP-M01 (temperature=1.0 at r=0.3)
- EXP-A19 = EXP-M01 (centroid latent source at r=0.3)

To copy results:
cp -r results/EXP-M01/ results/EXP-A02/
...then run /06-run-experiments update
```

---

## Step 5: Update mode — fill `\tbd{---}` cells from results

This runs when user calls `/06-run-experiments update [optional path]`.

### 5a. Find result files

**If path provided:** Parse that specific file.
**If no path:** Scan for all result files:
```
results/EXP-*/output.json
results/EXP-*/eval/results.json
results/ablation_*/*/output.json
```

### 5b. Identify which EXP-ID each result belongs to

Match by directory name:
- `results/EXP-M01/output.json` → EXP-M01
- `results/EXP-B03/eval/results.json` → EXP-B03
- `results/ablation_mr/r0.1/output.json` → EXP-A01 (match by sweep parameter)

If ambiguous, ask user to confirm the mapping.

### 5c. Parse metrics from each result

**JSON format** (preferred):
```json
{
    "attr_acc": 0.823,
    "ppl_mean": 45.2,
    "bleu": 0.312,
    "bertscore_f1": 0.891,
    "distinct_2": 0.734,
    "edit_distance": 0.456,
    "self_bleu": 0.321,
    "sss": 0.87,
    "mts": 0.73,
    "tps": 142.5,
    "nfe": 100,
    "gpu_mem_gb": 12.3,
    "params_m": 124.5
}
```

**Log format** (fallback): search for metric patterns like `Attr-Acc: 82.3%`, `PPL: 45.2`.

Validate: flag NaN, zeros, or out-of-range values (e.g., PPL > 10000, Acc > 1.0).

### 5d. Map metrics to table columns

Each table has specific columns. Map metric keys to column positions:

**tab:yelp_results / tab:amazon_results / tab:gyafc_results:**
| Column | Metric Key | Format |
|--------|-----------|--------|
| Attr-Acc ↑ | attr_acc | × 100, 1 decimal |
| PPL ↓ | ppl_mean | 1 decimal |
| BLEU ↑ | bleu | × 100, 1 decimal |
| BERTScore ↑ | bertscore_f1 | × 100, 1 decimal |
| Distinct-2 ↑ / Edit-Dist ↓ | distinct_2 / edit_distance | × 100, 1 decimal |

**tab:ablation_mask_ratio / tab:ablation_mask_mode / tab:ablation_latent_source:**
| Column | Metric Key | Format |
|--------|-----------|--------|
| Attr-Acc ↑ | attr_acc | × 100, 1 decimal |
| PPL ↓ | ppl_mean | 1 decimal |
| BLEU ↑ | bleu | × 100, 1 decimal |
| BERTScore ↑ | bertscore_f1 | × 100, 1 decimal |

**tab:ablation_steps:**
| Column | Metric Key | Format |
|--------|-----------|--------|
| Attr-Acc ↑ | attr_acc | × 100, 1 decimal |
| PPL ↓ | ppl_mean | 1 decimal |
| BLEU ↑ | bleu | × 100, 1 decimal |
| TPS ↑ | tps | 1 decimal |

**tab:ablation_temperature:**
| Column | Metric Key | Format |
|--------|-----------|--------|
| Attr-Acc ↑ | attr_acc | × 100, 1 decimal |
| PPL ↓ | ppl_mean | 1 decimal |
| Distinct-2 ↑ | distinct_2 | × 100, 1 decimal |
| Self-BLEU ↓ | self_bleu | × 100, 1 decimal |

**tab:geometry:**
| Column | Metric Key | Format |
|--------|-----------|--------|
| SSS ↑ | sss | 2 decimals |
| MTS ↑ | mts | 2 decimals |
| Interp. PPL ↓ | interp_ppl | 1 decimal |

**tab:efficiency:**
| Column | Metric Key | Format |
|--------|-----------|--------|
| Params (M) | params_m | 1 decimal |
| NFE | nfe | integer |
| TPS ↑ | tps | 1 decimal |
| GPU Mem (GB) | gpu_mem_gb | 1 decimal |

### 5e. Fill `\tbd{---}` cells in experiments.tex

For each result:

1. Open experiments.tex
2. Find the target table (by `\label{tab:xxx}`)
3. Find the target row (by matching the row text pattern)
4. Replace `\tbd{---}` with the formatted number, left to right matching column order
5. Save

**Row matching patterns:**
- Baselines: `MDLM`, `ReMDM`, `LatentOps`, `DiffusER`, `PLANNER`, `LD4LG`
- Main LSME: `LSME.*r.*=.*0.3`, `LSME.*r.*=.*0.5`, `LSME.*r.*=.*0.7`
- Mask ratio: `r = 0.1`, `r = 0.3`, etc.
- Mask mode: `Random`, `Entropy`, `Suffix`
- Steps: `^10 &`, `^50 &`, `^100 &`, `^500 &`
- Temperature: `T = 0.5`, `T = 0.8`, `T = 1.0`, `T = 1.2`
- Latent source: `No latent`, `Random latent`, `Centroid`, `Nearest neighbor`, `Directional`
- Geometry: `Negative.*Positive`, `Electronics.*Books`, `Informal.*Formal`, `Silhouette`, `Variance`
- Efficiency: `MDLM`, `ReMDM`, `LatentOps`, `DiffusER`, `LSME.*(100`, `LSME.*(50`

**Important:** Only replace `\tbd{---}`, never overwrite existing numbers.

### 5f. Apply `\best{}` formatting

After filling cells, check if ALL rows in a table now have real values (no remaining `\tbd{---}`).
If so, for each column:
- ↑ metric: wrap the highest value with `\best{}`
- ↓ metric: wrap the lowest value with `\best{}`

### 5g. Update experiment_plan.md

For each updated experiment:
- Change Status: `pending` → `done`
- Fill Result column with key metrics
- Update progress counts

### 5h. Print update summary

```
## Results Updated

### Filled:
| EXP-ID | Table | Row | Metrics |
|--------|-------|-----|---------|
| EXP-M01 | yelp_results | LSME r=0.3 | Acc=82.3, PPL=45.2, BLEU=31.2 |
| EXP-M02 | yelp_results | LSME r=0.5 | Acc=87.1, PPL=52.3, BLEU=24.8 |

### \tbd cells remaining: [K] / [total]
### Experiments done: [X] / 48

### Tables fully filled:
- [x] tab:yelp_results (all rows filled, \best{} applied)
- [ ] tab:amazon_results (2/4 rows filled)
- [ ] tab:ablation_mask_ratio (0/5 rows filled)

### Next experiments to run:
| EXP-ID | Description | Command |
|--------|-------------|---------|
| EXP-B01 | MDLM on Yelp | `python -m lsme.scripts.run_baselines ...` |

### Files updated:
- docs/novel/experiments.tex
- docs/novel/experiment_plan.md
```

---

## Step 6: Summary (after any mode)

```
## Experiment Status

**Experiments tex:** docs/novel/experiments.tex
**Plan:** docs/novel/experiment_plan.md

### Progress: [X] / 48 experiments ([K] \tbd cells remaining)

| Day | Group | Done | Total | Key Results |
|-----|-------|------|-------|-------------|
| 1 | Baselines (B01-B10) | [n]/10 | 10 | Best baseline Acc: XX.X% |
| 2-3 | Main LSME (M01-M07) | [n]/7 | 7 | LSME r=0.3 Acc: XX.X% |
| 4-5 | Ablations (A01-A21) | [n]/21 | 21 | Best mask mode: entropy |
| 6 | Geometry (G01-G04) | [n]/4 | 4 | SSS: X.XX, MTS: X.XX |
| 7 | Efficiency (E01-E03) | [n]/3 | 3 | LSME TPS: XXX |

### Key findings (auto-generated from results):
- LSME vs best baseline: Acc [XX.X vs XX.X] (+X.X%), BLEU [XX.X vs XX.X]
- Best mask ratio: r=X.X (Acc=XX.X%, BLEU=XX.X%)
- Latent steering effect: centroid vs zeros [+X.X% Acc]
- Latent smoothness: SSS=X.XX, MTS=X.XX

### Tables with all \best{} applied:
- [list of fully completed tables]

### Next:
Run `/06-run-experiments update` after completing more experiments.
```
