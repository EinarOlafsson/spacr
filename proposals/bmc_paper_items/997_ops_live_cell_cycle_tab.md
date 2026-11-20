# Item spec 997: OPS Live "Cell cycle" tab

- **Lifecycle:** future (alpha). The tab sits behind "Show alpha features" AND the OPS Live switch (item 971). Registered as `ALPHA_FEATURES[997]`. Off by default.
- **Number:** 997 was the next free number on 2026-11-20. Recheck `features/feature_list/` and the claims files before landing.
- **Rule:** a settings-file rerun gives IDENTICAL preview output (same sample, same stages, probabilities and flags, same DNA fit, same galleries; bit-for-bit on the CPU path). Every documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly (written against 53e91b5b17).
- **Paper-critical:** no (a quick look at a cell-cycle model and the G1/S/G2 fit before a run; the run itself is items 990 / 995).
- **Related:** 971 (OPS Live preview), 990 (cell-cycle option), 995 (field-centred DNA fit), 996 (Deep-learning tab).

## BUILD GATE
**Implementation starts only when both items 990 and 995 are "Ready" in `proposals/READINESS.md`** (row "Per-cell cell-cycle phase classifier" and its successors). Before then this file is the only deliverable.

## 1. What it does
On the Objects & phenotype output of the preview workspace, the tab runs, on a small seeded sample:
1. the chosen cell-cycle model from item 990 (`ops_cell_cycle`: `DNA` or `DNA+MT`; the mitotic model of 856/990), and
2. the item 995 field-centred DNA-content fit (G1/S/G2 mixture).

It shows what a run would produce for that sample and runs the self-checks, so a user can judge the model before the run. Preview workspace only.

## 2. Dependency and integration
- `_STEP_NEEDS["cycle"] = ("objects",)` (closure pulls `align`).
- A new step key `"cycle"` is added to `spacr.ops_live` only through `_steps_for(alpha)`, which returns `_STEPS` unchanged when the alpha flag is off. Module-level tuples stay as they are (off-switch byte identity). If 996 lands first, both keys share the one function.
- `_run_cycle(ctx, cancelled=None)` reads `_result(ctx, "objects", cancelled)` (the measured cell table and crops), runs on `_preview_settings(...)`, caches under `_signature(settings, "cycle")`.
- If `ops_cell_cycle` is `None` the tab shows the "no model chosen" message and scores nothing; the DNA histogram and fit still run when the DNA channel exists, since 995 does not need the model.
- Workspace layout: `<workspace>/cycle/` holds `sample.parquet`, `stages.parquet`, `dna_fit.json`, `checks.json`, `gallery/`. Nothing else is written.

## 3. Settings (alpha)
The model and fit settings are 990's and 995's own; the tab adds only these (prefix `ops_live_cc_`):
| Key | Values | Default | Meaning |
|---|---|---|---|
| `ops_live_cc_sample` | int | `2000` | Cells scored (seeded draw, stratified by field so every field is represented). |
| `ops_live_cc_gallery` | int | `12` | Crops per class gallery. |
| `ops_live_cc_gallery_order` | `'probability'` \| `'random'` | `'probability'` | Highest-probability crops first, or a seeded random draw (the honest view). |
| `ops_live_cc_seed` | int | `0` | Sample draw, gallery draw and every random stream. |

Read, not duplicated, from the form: `ops_cell_cycle` (`None`/`DNA`/`DNA+MT`), `ops_cell_cycle_model_family`, `ops_cell_cycle_tabular`, `ops_cell_cycle_base_model`, `ops_cell_cycle_dna_channel`, `ops_cell_cycle_mt_channel`, `ops_cell_cycle_seed` (990) and `cell_cycle_field_centre`, `cell_cycle_field_min_cells`, `cell_cycle_fit_group`, `cell_cycle_mitotic_column` (995).
Preview forcing: `ops_cell_cycle_finetune='none'` and `ops_cell_cycle_label_queue=0` (the preview never fine-tunes and never queues a labelling job). `cell_cycle_field_centre` is used as set on the form.

## 4. UI layout
New page `_CyclePage` in `spacr/qt/widgets/ops_live_preview.py`, same structure as the other pages, registered only when the alpha flag is on.
- Top: blurb and a one-line status (model, input DNA / DNA+MT, sample size, seed).
- **Class galleries:** one row per class: interphase, G2/prophase flag, prometaphase, metaphase, ana/telo/cyto, telophase daughters, artefact. Each shows up to `ops_live_cc_gallery` crops with the called probability under the crop and the per-stage flag (`auto` / `unresolved`; `verified` only when a labels file is set). Fixed display range per channel; no per-crop stretch. A class with no calls shows "No cells called."
- **DNA-content histogram:** log2 DNA content (field-centred when `cell_cycle_field_centre`), with the fitted G1, S and G2 components overlaid, the G2/G1 ratio and the fit R squared in the legend; sub-G1 and above-4N shaded grey.
- **Self-checks table:** rows below, each with value, threshold, pass/fail.
- **Resolvable-stage flags:** one line per stage: `resolved` or `collapsed into <parent>` (the 990 collapse rule).

**Self-checks shown (990 section 5, computed by the same functions)**
- 4N share of mitotic calls (S1: prometaphase + metaphase with DNA above the G1/G2 midpoint);
- 2N daughter pairs (S2);
- per-field stability of the mitotic share (S4, overdispersion across fields with at least 50 cells);
- G1/S/G2 fit quality (S6: R squared and G2/G1 within 1.6-2.4);
- the resolvable-stage flags from the collapse rule.

S3 (shuffled-label control) and S5 (knockout enrichment) are not run in the preview: S3 needs hundreds of permutations and S5 needs the full library. The table lists them as "Not run in the preview".

## 5. Functions reused (no duplicated helpers)
- `spacr.ops_cell_cycle` (990): the predict function, the parent map and collapse, and the self-check functions S1, S2, S4, S6. The tab calls them on the sample and does not reimplement any rule.
- `spacr.ops_mitotic`: `mitotic_features`, `_well_robust_z` (called through 990).
- `spacr.measure`: `_fit_dna_content`, `_gate_dna_content`, and the 995 field-centring and external-mitotic path in `_phases_by_measurements`. The histogram overlay reads the fit dict these functions return.
- `spacr.gene_measurement_compare.g1_mode` (through 995).
- `spacr.ops_live`: `_context`, `_result`, `_signature`, `_workspace`, `_preview_settings`, `_check`, `_write_crop`.
- Qt: `LivePreviewContract`, `PREVIEW_*` strings, the panel's worker thread, and the preview's existing matplotlib palette (navy/teal/blue, 60 % alpha).

## 6. GPU / CPU
CPU by default (tabular family). The `cnn` / `transformer` families run only if 990 is Ready for them and a GPU is usable; otherwise the tab uses the tabular learner and says so, as 990 does. Tests run with `CUDA_VISIBLE_DEVICES=''`, `pytest -n 4` at most.

## 7. Tests (CPU, synthetic, small)
- **T1 off switch:** with the alpha flag off, `_STEPS`, `_STEP_NEEDS`, tab order and every existing step's output are byte-identical to nightly; the `ops_live_cc_*` keys are absent from the schema, help and panel. With `ops_cell_cycle=None` a run's output is also unchanged.
- **T2 workspace only:** run the tab on the synthetic OPS fixture under a temporary `$HOME`; the set of files created or modified outside `<workspace>/cycle/` (and the existing workspace caches) must be empty; the run's `dst_root` is untouched; no labelling job is registered on Home.
- **T3 smoke:** a synthetic cell table with planted 2N/4N content, condensed 4N "mitoses" and 2N daughter pairs: every class gallery is filled where classes exist, S1 and S2 pass, the histogram fit recovers G2/G1 within 0.05 of 2.0; with planted labels shuffled, the checks fail and the stages show `collapsed into <parent>`.
- **T4 identical rerun:** the same settings file run twice (CPU, same seed): `stages.parquet`, `dna_fit.json`, `checks.json` and the gallery file order are byte-identical.
- **T5 centring:** fields with planted log2 offsets: with `cell_cycle_field_centre=True` the CV of the fit is smaller than without, as in 995.
- **T6 sample:** the sample is stratified by field and seeded; changing the seed changes it; `ops_live_cc_sample` larger than the cells available uses all cells and says so.
- **T7 offscreen Qt:** the page builds, shows the "Not run yet" text, fills after a worker run, and each empty class shows "No cells called."
- **T8 no model:** `ops_cell_cycle=None` shows the message and still draws the DNA fit.
- Ratchets before handover: raw-reader, nested-function, documented-callable and hash-comment.

## 8. Acceptance (makes the tab Ready)
- **A1:** on a Funk plate A well A1 sample, the tab's stages equal the stages the full 990 run gives for the same cells (identical).
- **A2:** the tab's DNA fit equals the 995 reference (`cc_measure_fc.py`) for the same field-centred data (identical labels; G2/G1 1.945-1.955).
- **A3:** the settings-file rerun is identical (T4).

## 9. Strings for Codex
All translations go to Codex via `/mnt/firecuda2/codex/scratch-docs/codex_inbox/tr_997_task.md` (sources with SHA-256, 9 locales). No translations are written in this item. Every user-facing string the tab adds, literally:

**Tab title**
- `Cell cycle`

**Blurb**
- `Score a sample of cells with the chosen cell-cycle model, fit the DNA content into G1, S and G2, and run the self-checks that tell you which stages can be trusted.`

**Labels and headings**
- `Model`
- `Input`
- `Sample size`
- `Seed`
- `Cell classes`
- `Interphase`
- `G2 / prophase (flag)`
- `Prometaphase`
- `Metaphase`
- `Anaphase / telophase / cytokinesis`
- `Telophase daughters`
- `Artefact`
- `Probability {value}`
- `DNA content`
- `log2 DNA content`
- `Cells`
- `Fitted G1`
- `Fitted S`
- `Fitted G2`
- `G2/G1 ratio`
- `Fit R squared`
- `Sub-G1`
- `Above 4N`
- `Self-checks`
- `Check`
- `Value`
- `Threshold`
- `Result`
- `Pass`
- `Fail`
- `Not run in the preview`
- `4N share of mitotic calls`
- `2N daughter pairs`
- `Stability across fields`
- `G1/S/G2 fit quality`
- `Resolvable stages`
- `Stage flag`
- `auto`
- `verified`
- `unresolved`
- `resolved`
- `collapsed into {parent}`

**Tooltips**
- `The cell-cycle model chosen in the settings: DNA only, or DNA plus microtubules.`
- `The cells shown here are a seeded sample. The same seed always draws the same cells.`
- `Crops are ordered by the model's probability, highest first. Choose random order for an unbiased look.`
- `G2 / prophase is a flag, not a mitotic stage: it marks cells with 4N DNA that are about to enter mitosis.`
- `Fraction of prometaphase and metaphase calls whose DNA content is 4N. Real mitotic cells have replicated DNA, so this should be high.`
- `Fraction of anaphase / telophase calls that have a 2N partner right next to them. Daughter cells come in pairs.`
- `How evenly the mitotic share is spread over fields. A large spread means the model reacts to the field, not to the cells.`
- `How well the DNA-content histogram is explained by G1, S and G2, and whether G2 sits at about twice G1.`
- `A stage that fails its check is merged into its parent stage, so that the tables stay honest.`
- `Not run in the preview: this check needs many permutations or the whole library. It runs in the full run.`

**Messages**
- `No cell-cycle model is chosen. Set "ops_cell_cycle" to DNA or DNA+MT. The DNA fit is shown below.`
- `Run the Objects & phenotype tab first: this tab scores its cells.`
- `No DNA channel found. Set the DNA channel in the settings.`
- `Not enough cells ({n}) to fit G1, S and G2.`
- `No cells called.`
- `The image model needs crops or a GPU; using the table model instead.`
- `The preview never fine-tunes the model and never creates a labelling job.`
- `Using all {n} cells: fewer than the sample size.`
- `Alpha: this tab is under development.`

**Settings help (the four `ops_live_cc_*` keys, in table order)**
- `How many cells the Cell cycle tab scores.`
- `How many crops to show per class.`
- `Show the highest-probability crops first, or a random draw.`
- `Random seed for the sample, the galleries and the checks.`

## 10. Effort
M: step + page M; galleries and histogram S; tests S. Blocked by the build gate (990 and 995 Ready), not by code.
