# Item spec 990: OPS cell-cycle option (stage, probability and verification flag per cell)

- **Lifecycle:** future (alpha). The option sits behind "Show alpha features" (off by default), like every new feature.
  It is registered as `ALPHA_FEATURES[990]`.
- **Number:** 990 was the next free number on 2026-11-19 (`features/feature_list/` reaches 988, and 989 is the
  seed-variance item). Recheck the claims files before landing.
- **Design:** the maintainer's decision of 2026-11-19 (RUN_PLAN, entry "~17:05 Cell-cycle option becomes a spaCR
  feature"). Transformer arm: **SubCell only** (maintainer, 2026-11-19). DINO4Cells and OpenPhenom may be tested later.
- **Rule:**
  - A settings-file rerun gives **identical** output: the same rows, the same stages, probabilities and flags
    bit-for-bit on the CPU path, and the same checkpoint hashes. This is acceptance test T1.
  - Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes. The BMC paper's cell-cycle-resolved OPS analyses use it.
- **Readiness:** Under development. Per `proposals/READINESS.md` it becomes Ready only after
  acceptance A1–A4 below.

## 1. What it does

For every cell of an OPS phenotype run, the option returns:

- **a stage**, one of:
  - G1, S, G2 (interphase, from the DNA-content fit);
  - g2_prophase (a flag class, never counted as mitotic);
  - prometaphase, metaphase, ana_telo_cyto (anaphase + telophase/cytokinesis, daughters included);
  - artefact;
- **a probability** for the called stage;
- **a per-stage flag:** `auto` (the model's call, self-checks passed), `verified` (the stage passed a check against
  the user's own labels) or `unresolved` (the stage failed its self-checks and collapsed into its parent class).

It starts from a shipped base model, fine-tunes it per screen (plates are the cross-validation groups) and runs
automatic self-checks. An optional labelling job (about 100 cells) on the Home screen turns stages from `auto` into
`verified`.

## 2. Settings (OPS settings, all alpha, all in `ALPHA_FEATURES[990]['settings']`)

| Setting | Values | Default | Meaning |
|---|---|---|---|
| `ops_cell_cycle` | `None` \| `'DNA'` \| `'DNA+MT'` | `None` | Off, DNA-stain model, or DNA + microtubule model. `None` leaves the run byte-identical to today's. |
| `ops_cell_cycle_model_family` | `'tabular'` \| `'cnn'` \| `'transformer'` | `'tabular'` | The model family. `tabular` runs on the cell tables. `cnn` and `transformer` need crops (`ph_crops`) or streamed fields. |
| `ops_cell_cycle_tabular` | `'hgb'` \| `'xgboost'` | `'hgb'` | The tabular learner: HistGradientBoosting (current) or XGBoost. When `cnn`/`transformer` is chosen but crops cannot be made or streamed, the run **falls back to this learner** and says so in the report. |
| `ops_cell_cycle_transformer` | `'subcell'` | `'subcell'` | The SubCell ViT (Lundberg lab, Human Protein Atlas immunofluorescence), frozen embedding + head, or fine-tuned (next row). Channel variant, weights, licence and size: **pending verification** (separate agent). |
| `ops_cell_cycle_finetune_backbone` | bool | `False` | Transformer and CNN only: fine-tune the backbone too, not just the head. |
| `ops_cell_cycle_base_model` | path \| `''` | `''` (= the shipped base model for the chosen input) | The base checkpoint to start from. |
| `ops_cell_cycle_finetune` | `'screen'` \| `'plate'` \| `'none'` | `'screen'` | Fine-tune once per screen (pooled plates) or per plate. Per well is not offered: a well has too few mitotic cells (plate A: about 25–70 labelled mitotic cells per well). `'screen'` is the default by maintainer decision (2026-11-19). The per-screen vs per-plate test was skipped, and the maintainer will test it later; `'plate'` stays available but unvalidated. |
| `ops_cell_cycle_dna_channel` | channel name | `nucleus_channel` | The DNA stain. |
| `ops_cell_cycle_mt_channel` | channel name \| `None` | `None` | The microtubule (tubulin) stain. Required for `'DNA+MT'`. |
| `ops_cell_cycle_arrest_genes` | list \| `'default'` \| `None` | `'default'` | Knockouts with a known mitotic arrest, used in self-check S5 when they are in the library (default list in §5). |
| `ops_cell_cycle_labels` | path \| `''` | `''` | User labels: written by the Home-screen job, or given by hand. |
| `ops_cell_cycle_label_queue` | int | `0` (off) | Size of the labelling job to create (100 recommended). |
| `ops_cell_cycle_seed` | int | `0` | Every random stream: folds, shuffles, model seed, queue draw. |

- **Name (approved by the maintainer 2026-11-19): `ops_cell_cycle`.** Measure already has a `cell_cycle` setting (bool, `get_measure_crop_settings`). The OPS option
  therefore takes the `ops_` prefix, like every OPS setting, so that one key never has two meanings.
- **Relation to existing settings:** `ops_mitotic_source` (item 890) gains a value `'cell_cycle'`, which takes M
  from this option. `ops_mitotic_model` (item 856) stays as the binary path.

## 3. Module placement (reuse first)

- **New module `spacr/ops_cell_cycle.py`**, a thin layer that calls existing code:
  - Features: `spacr.ops_mitotic.mitotic_features(columns, dna=..., tubulin=None|mt)` and `_well_robust_z`, so the
    normalisation is the 856 one (robust z per well against all cells of the well).
  - Tabular learners:
    - `spacr.ops_mitotic._model` gains a `'xgboost'` kind through `spacr.ml._xgboost_cpu_safe`.
    - The multiclass form follows `spacr.measure._phases_by_xgboost`.
    - Grouped splits: `spacr.classifier_evaluation.grouped_split`.
  - DNA-content fit: `spacr.measure._fit_dna_content` + `_gate_dna_content` (Dean-Jett-Fox mixture by expectation
    maximisation, G2 held at 1.6–2.4 × G1), fitted per plate on cells called interphase, after field-mode centring
    (the `h_phase.field_modes` step, ported).
  - CNN: `spacr.deep_spacr.train_test_model` / `apply_model`, the path `spacr.measure._phases_by_torch` already uses,
    on `ph_crops` shards (`ops_engine._ph_crops`) or crops streamed by `spacr.crop_loader`.
    - **Gap to close:** `train_test_model` reads 8-bit RGB PNG folders with per-image percentile normalisation.
      This item adds a raw multi-channel uint16 crop dataset with one fixed scale per plate × channel (the grid's
      `ccg_cnn.py` is the reference), so DNA-only (1 plane) and DNA+MT (2+ planes) run without an 8-bit step.
  - Transformer: `spacr.embeddings` (`subcell` / `subcell_rybg` backbones are registered there; the channel
    mapping for DNA / DNA+MT is pending verification) + a logistic or multilayer-perceptron head.
    Fine-tuning reuses `deep_spacr` training with the SubCell backbone.
  - Labelling queue: `spacr.active_learning.build_queue` / `rank_by_uncertainty`, `queue_rows` for the Annotate
    screen.
  - Checkpoint I/O: `spacr.checkpoint` (atomic, signature-checked).
- **OPS engine:** a new phase `ph_cell_cycle`, run after `ph_tables` (and `ph_crops` for image families) and before
  `aggregate`. `_ph_aggregate` reads `cell_cycle_stage` for `ops_profile_cell_cycle` (item 890) and
  `ops_profile_cell_state` (item 914).
- **GUI:** `spacr/qt/widgets/home.py` job list + `spacr.qt.bridge.registry()` (§8).

## 4. Training, fine-tuning and the base model

### 4.1 Base models (shipped)

- Two variants per family: `DNA` and `DNA+MT`.
- Training data:
  - Funk et al. 2022 plate A: the 1,467 labelled cells of the 2026-11-19 validation, plus the enriched
    prophase/anaphase round when it lands.
  - PERISCOPE CP186A (Ramezani et al. 2025): labels from the Home-screen job run once by the maintainer, about 100
    cells, plus a top-up of the rare classes.
  - CP186A has no microtubule stain, so the **DNA+MT base is plate A only** until a tubulin screen joins.
    The card says so.
- Classes: the 4-class scheme (maintainer, 2026-11-19): interphase, g2_prophase (flag), prometaphase, metaphase,
  ana_telo_cyto, artefact.
  - The 7-class scheme (prophase and anaphase separate) is used only if the enriched round gives F1 ≥ 0.7 for every
    stage (rule of 2026-11-19).
- Cross-validation: field-grouped folds, 5 repeats × 5 folds, within each screen. Plate-held-out when ≥ 2 plates.
- Weights: shipped through the spaCR model registry (Hugging Face), never inside the wheel.
  - Each checkpoint has a model card: data, label counts, per-class F1 with 95 % field-bootstrap confidence intervals
    (CIs), the shuffled-label control and the self-check results.
  - Upload needs the maintainer's OK (weights keep their own approval).

### 4.2 Fine-tuning per screen

1. Predict the screen's cells with the base model and run the self-checks (§5) with the base model as is.
2. With user labels (`ops_cell_cycle_labels`), fine-tune:
   - tabular: refit on base labels + screen labels, with screen labels up-weighted ×5 (fixed);
   - CNN and transformer: head only, or the backbone too with `ops_cell_cycle_finetune_backbone`.
   - Groups: plates (`'screen'`, one fine-tune for all plates, leave-one-plate-out CV when ≥ 2 plates, else
     field-grouped 5-fold) or one fine-tune per plate (`'plate'`).
3. Without user labels:
   - there is no fine-tuning;
   - the robust z per well is the only adaptation;
   - the flags stay `auto` or `unresolved`.

### 4.3 G1 / S / G2

- Only cells called interphase (and the g2_prophase flag cells, which are kept out of the fit) are fitted, per plate.
- The DNA value is the field-centred log2 integrated nuclear DNA intensity.
- A plate with < 30 usable nuclei uses the pooled fit (the existing `_phases_by_measurements` rule).
- The fit R² and the G2/G1 ratio are reported. A plate whose G2/G1 ratio lies outside 1.6–2.4 has S and G2
  collapsed to `S/G2`, flagged `unresolved`.

## 5. Self-checks and collapse rule (run automatically, per plate, every run)

The pass criteria were fixed on 2026-11-19. The reference values come from plate A (transfer card 20261119); the
numbers are recomputed from the base model card at build time.

| ID | Check | Stage(s) judged | Pass |
|---|---|---|---|
| S1 | 4N in mitosis: share of prometaphase + metaphase calls with DNA above the G1/G2 midpoint | prometaphase, metaphase | ≥ 0.70, and ≥ the base card's value − 0.15 |
| S2 | 2N daughter pairs: share of 2N ana_telo_cyto calls with a 2N ana_telo_cyto partner within 2.5 nuclear diameters (\|Δ log2 DNA\| ≤ 0.3) | ana_telo_cyto | above the 97.5th percentile of 500 within-field call permutations |
| S3 | Shuffled-label control: the same model class trained on permuted labels | all | the real model's held-out macro F1 is above the shuffled model's 97.5th percentile (500 permutations, CPU tabular; 20 for image families) — needs labels; without labels the check is skipped and reported |
| S4 | Field and plate stability of the mitotic share: overdispersion φ = observed / binomial variance across fields (fields with ≥ 50 cells); share per plate | all mitotic stages | φ ≤ 2 × the base card's φ, and no plate outside [0.5×, 2×] the screen median |
| S5 | Knockout enrichment (only when ≥ 100 cells of the arrest set are in the library): mitotic share of the arrest-gene cells vs non-targeting (NT) cells | prometaphase, metaphase (arrest) | one-sided permutation p < 0.05 against 500 random gene sets with matched cell counts |
| S6 | G1/S/G2 fit: R² ≥ 0.9 and G2/G1 within 1.6–2.4 | G1, S, G2 | both |

- **Default arrest set (S5):** KIF11, TPX2, CENPE, KIF18A, NUF2, NDC80, SPC24, SPC25, CKAP5, CDC20, PLK1, AURKA,
  ANAPC1, ANAPC2, ANAPC4, ANAPC5, ANAPC10, ANAPC11, CDC16, CDC23, CDC27.
- **Collapse rule:** a stage that fails its check collapses into its parent, and its flag becomes `unresolved`:

  ```text
  prometaphase | metaphase | ana_telo_cyto  ->  mitotic
  mitotic                                   ->  G2/M
  g2_prophase                               ->  G2
  G1 | S | G2                               ->  interphase
  ```

  - A parent that itself fails collapses further.
  - Collapses are written per plate. A screen-level stage is collapsed when any plate collapses it, so tables stay
    comparable.
  - If S1 and S2 fail but S4 passes, the stages collapse to `mitotic` and the mitotic share is kept.
  - If S4 also fails, mitotic collapses to `G2/M`.
- **Verified:** a stage is `verified` when the user's labels for it reach F1 ≥ 0.7 with the lower 95 % CI bound
  above the shuffled-control's 97.5th percentile. The labels are scored out of fold: never the labels the model was
  fine-tuned on in that fold.

## 6. Outputs

- `measurements.db:ops_cell_cycle` (and Parquet next to it in the spaCR store, item 974), one row per cell. Columns:
  - `plate, well, site, label`;
  - `cell_cycle_stage`, `cell_cycle_probability` (the probability of the called stage, after collapse = the sum
    over its children), `cell_cycle_flag` (`auto|verified|unresolved`);
  - `p_<class>` for each model class, `dna_log2_fc`, `dna_phase_posterior_G1|S|G2`, `model_id`.
- `ops_cell_cycle_checks`: one row per plate × check. Columns: value, control, threshold, pass, and collapse applied.
- `ops_cell_cycle_report.html`, which includes:
  - per-class F1 with field-bootstrap CIs and the shuffled-control row;
  - the DNA histograms with the fit;
  - the 4N/2N panels and stability maps;
  - the knockout panel.
  - Every number has its control row, and every abbreviation is spelled out once.
- Per-cell outputs never overwrite `ops_mitotic` columns.

## 7. Checkpoint format

- One directory per checkpoint: `cell_cycle_<family>_<input>_<hash12>/`, containing:
  - `model.joblib` for tabular; `model.pt` + `head.pt` for CNN and transformer;
  - `card.json`: format `spacr-ops-cell-cycle-1`; classes; parent map; features or channel map; normalisation;
    the training data list with file SHA-256s; label counts; CV metrics; self-check thresholds; seed; spaCR version;
    and the SubCell weight SHA-256 when used;
  - `SHA256SUMS`.
- Loading checks `format` and every SHA-256 before unpickling. joblib/torch files are loaded only from the registry
  or from paths the user names, with a warning as in `load_mitotic_classifier`.
- The fine-tuned screen checkpoint records its base checkpoint hash, so the chain is traceable.

## 8. Home-screen labelling job

- With `ops_cell_cycle_label_queue = N`, the run registers an **external job** on Home
  (`spacr.qt.bridge._track_external_job`). The job is titled "Cell cycle: label N cells" and is listed with the other
  runs, in the queued/recent panels.
- The job does not block the run: the run finishes with `auto`/`unresolved` flags.
- **Queue:** N cells drawn by `active_learning.rank_by_uncertainty` (margin), stratified so that every stage has at
  least ⌈N/10⌉ candidates and every plate is represented. The draw is seeded by `ops_cell_cycle_seed`.
  The crops are the raw `ph_crops` crops with one fixed display range per channel (no per-crop stretch).
- **Labelling:** clicking the job opens Annotate on `queue_rows(queue)` with the stage classes as keys. Labels are
  written to the screen's `ops_cell_cycle_labels` file (append-only CSV with user and time).
- **Automatic retrain:** when the job is marked done, spaCR fine-tunes (§4.2), reruns §5, sets `verified` on the
  stages that pass, and rewrites the `ops_cell_cycle` tables and report.
  - A new run-journal entry records the label file hash.
  - The settings file gains the label path, so a rerun reproduces the verified state (T1).
- A second job can be queued for the stages that are still `unresolved`.

## 9. Tests (CPU, `CUDA_VISIBLE_DEVICES=''`, `pytest -n 4` max)

- **T1 reproducibility:** run the same settings file twice on the synthetic OPS fixture (and on Funk A1 test data,
  item for the HF dataset). `ops_cell_cycle` tables, checks and checkpoint hashes must be byte-identical
  (tabular path). For CNN/transformer: identical on CPU with deterministic torch; on GPU, identical stages and
  probabilities within 1e-6 is reported, not required.
- **T2 `None`:** with `ops_cell_cycle=None` the run output is byte-identical to the run without the key.
- **T3 synthetic stages:**
  - synthetic nuclei with planted 2N/4N content, condensed 4N "mitoses" and 2N daughter pairs;
  - S1/S2 pass;
  - with the planted labels shuffled, S3 fails and the stages collapse to the parents in §5.
- **T4 collapse:** a unit test of the parent map and of plate → screen collapse.
- **T5 fallback:** `cnn` without crops → tabular fallback, with a report line.
- **T6 knockout:** a fixture with an arrest set at 3× the mitotic share passes S5; at 1× it fails.
- **T7 Home job:** registry entry created/finished headless (offscreen Qt), and the label file round-trips.
- **T8 checkpoint:** a SHA-256 mismatch refuses to load, and the format marker is checked.
- Ratchets before handover: raw-reader, nested-function, documented-callable and hash-comment.

## 10. Acceptance (makes it Ready)

- **A1:** on plate A, a spaCR settings run with `ops_cell_cycle='DNA+MT'`, tabular HGB, gives stages identical to
  the reference script calls (`cellcycle_validation_1119` 4-class model on the same features, same seed).
- **A2:** the self-checks reproduce the transfer card's reference rows on plate A (identical numbers, same seeds).
- **A3:** CP186A with `'DNA'`: the self-check table matches the transfer test (card 20261119_cellcycle_transfer_cp186a).
- **A4:** the model-grid card (20261119_cellcycle_model_grid, SIGNED 2026-11-19) picks the default family.
  - Its terms: fixed settings for every model, no tuning; the 4-class scheme now and the 7-class scheme after the enriched labels; the granularity test skipped.
  - Until the grid reports, the default is tabular HGB.

## 11. Effort

L:

- tabular + checks + outputs: M;
- CNN/transformer arms: M;
- Home job: S–M.
