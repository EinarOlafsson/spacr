# Item spec 999: Measure cell-cycle label for every cell (the OPS cell-cycle work, reused)

- **Lifecycle:** future (alpha), behind "Show alpha features" (off by default), like every new feature. Register it as `ALPHA_FEATURES[999]`. The default-on rule below applies only where alpha features are shown.
- **Number:** 999 was the next free number on 2026-11-20. Recheck `features/feature_list/` and the claims files before landing.
- **Rule (maintainer):**
  - It is the same cell-cycle work as OPS: the 990 model plus 995's field-centred DNA fit. Reuse the 990 and 995 code. No second implementation.
  - A settings-file rerun gives identical output (same labels, probabilities and flags, byte for byte on the CPU path).
  - Each documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** no. It makes the OPS cell-cycle method available to every Measure user.
- **BUILD GATE:** do not start before **990 and 995 are both Ready** in `proposals/READINESS.md`. Until then this spec only waits.
- **Readiness:** Under development until acceptance A1-A3 below pass.

## 1. What it does

Measure gives every cell a **cell-cycle label**, available whenever the user has a DNA stain, or a DNA stain plus a microtubule (tubulin) stain.

- **Default:**
  - **OFF** in Measure.
  - **ON** when a nucleus mask channel is set (`nucleus_channel` is not `None`).
  - The user can always override the default either way.
- **DNA channel:** defaults to Measure's nucleus channel. "Measure specifies what the nucleus channel is, so use whatever is in that." The setting is stored as Measure stores it (zero-indexed) and shown **1-indexed in the GUI**, exactly as Measure shows `nucleus_channel` today. No new channel conversion is written: use the one Measure already has. (At build time, confirm which helper Measure's panel uses for the +1 display and call it.)
- **Microtubules are optional.**
  - With a microtubule channel, the DNA+MT model runs. The spindle and tubulin features feed the mitotic and stage classifier, so mitotic calls improve. In the cell-cycle model grid, all channels beat DNA only: HistGradientBoosting (HGB) macro F1 0.85 against 0.76 (`reanalysis/cellcycle_model_grid/tables/grid_table.csv`). The macro F1 is the unweighted mean of the per-class F1 values.
  - Without one, the DNA-only model runs. G1/S/G2 are the same in both cases (they come from the DNA-content fit); only the mitotic sub-stages differ.
- **Same method as OPS, nothing new:**
  - Stages, probabilities, flags, self-checks, collapse rule and checkpoints: item 990 sections 1, 4, 5 and 7.
  - G1/S/G2 from the field-centred DNA fit: item 995 (`cell_cycle_field_centre`, `cell_cycle_fit_group`, `cell_cycle_mitotic_column`).
  - This item adds only the Measure settings, the default rule, the output tables and the GUI strings.

## 2. Screens and the amendment of 2026-11-20

The 990 amendment applies unchanged. It was not in the 990 text on this base (53e91b5b17), so this section is the binding statement until it lands there.

- The **mitotic stages** (prometaphase, metaphase, ana_telo_cyto) are trained on the **Funk plates only**.
- **G1/S/G2** come from 995's field-centred DNA fit on **every screen**.
- On a screen with too few mitotic cells for the mitotic self-checks, the mitotic stages are called **`unresolved`** and collapse as in 990 section 5. G1/S/G2 are still reported.
- A run never hides this: the Measure report says which stages are resolved and which are `unresolved`, per plate.

## 3. Settings (Measure; alpha group)

| Setting | Values | Default | Meaning |
|---|---|---|---|
| `measure_cell_cycle` | `None` (automatic) \| `'DNA'` \| `'DNA+MT'` \| `'off'` | `None` = automatic (rule below) | Automatic, DNA-stain model, DNA + microtubule model, or off. Same model values as OPS `ops_cell_cycle`. |
| `measure_cell_cycle_dna_channel` | channel (1-indexed in the GUI) \| `None` | `None` = `nucleus_channel` | The DNA stain. Blank takes whatever `nucleus_channel` holds. |
| `measure_cell_cycle_mt_channel` | channel \| `None` | `None` | The microtubule stain. Required for `'DNA+MT'`. With `None` the DNA-only model runs. |
| `measure_cell_cycle_base_model`, `_finetune`, `_labels`, `_seed` | as 990 | as 990 | Aliases of `ops_cell_cycle_base_model`, `_finetune`, `_labels` and `_seed`: same meaning and defaults. The model family is tabular (990's default) until 990's grid card picks another. |

- **How the automatic default works (`measure_cell_cycle = None`):**
  - `nucleus_channel` is set: the label runs. It is `'DNA+MT'` when `measure_cell_cycle_mt_channel` is set, else `'DNA'`.
  - `nucleus_channel` is `None`: the label does not run, and the run is byte-identical to today's.
  - An explicit `'DNA'` or `'DNA+MT'` runs the label whether or not a nucleus channel exists. It then needs `measure_cell_cycle_dna_channel`.
  - An explicit `'off'` disables it even when a nucleus channel is set.
- **One implementation:** the Measure keys are resolved to the 990 settings by a single function, and the 990 module runs. The Measure code path contains no model, fit or check code of its own.
- **Same semantics as OPS where they overlap:** the field-centring keys of 995 (`cell_cycle_field_centre`, `cell_cycle_field_min_cells`, `cell_cycle_fit_group`, `cell_cycle_mitotic_column`) are read by both. The label turns field centring ON (it is the adopted path for G1/S/G2). A user who sets `cell_cycle_field_centre=False` explicitly gets the old fit, and the report says so.

### 3.1 Relation to the existing `cell_cycle` bool (migration rule)

Measure already has `cell_cycle` (bool, default False) with `cell_cycle_method`, `cell_cycle_channel`, `cell_cycle_gates`, `cell_cycle_mitotic_ratio`, `cell_cycle_fucci_channels`, `cell_cycle_labels`, `cell_cycle_model` and `cell_cycle_epochs`. It writes `measurements.db:cell_cycle` (one row per nucleus, call in `cell_cycle_phase`) and `cell_cycle_well`.

- **`cell_cycle` stays, unchanged.** It is the legacy per-nucleus phase call, and a run that uses it (and nothing new) is byte-identical to today's.
- **`measure_cell_cycle` is the new label.** The two never share a key. This is the same reason 990 took the `ops_` prefix.
- **Both on:**
  - Both run, and write to their own tables.
  - The report says that the legacy call and the new label are independent, and shows the agreement matrix.
  - `cell_cycle_channel`, when set, is also the default DNA channel of the new label, ahead of `nucleus_channel` (an explicit user choice beats the generic one).
- **Migration of old files:**
  - An old file with `cell_cycle: true` and no `measure_cell_cycle` keeps its legacy behaviour and gets **no** new label (see the schema row, section 4).
  - The GUI offers a one-click "Use the new cell-cycle label" that sets `measure_cell_cycle` and leaves `cell_cycle` as it was.
  - `cell_cycle` is not removed in this item. A later item may deprecate it once the new label is Ready and in use.

## 4. Schema: the default change needs a row (rule 41)

Turning the label ON by default when a nucleus channel is set changes what an existing settings file produces, because older files have a nucleus channel and no `measure_cell_cycle` key. Rule 41: a default change needs a schema row.

- **Add:**
  - `SETTINGS_SCHEMA_VERSION` = the next free schema number at landing (master, 2026-11-20: the schema number is NOT the item number; 945-948 are taken by 987b, 978, 967 and X). It must be above every existing key.
  - One `DEFAULT_CHANGES[<that schema number>]` row holding the **old** default: `{'measure_cell_cycle': 'off'}`.
- **Effect:** a file stamped below 999, or with no stamp, is folded to `measure_cell_cycle = 'off'`. It reproduces exactly what it produced before, with a logged notice naming the key and how to take today's default instead (the existing `_fold_older_defaults` message). A new file is stamped 999 and takes the automatic default.
- **Explicit values** in an old file are kept as written.
- **Document the row** in the docstring of `_fold_older_defaults` (the list of version notes) and in `docs/source/reproducibility_manifests.rst`, like 931-945.
- **Register the keys** in the settings type table, the help text, the settings panel (alpha group) and the help index, as for any new setting. `'off'` is a documented value of `measure_cell_cycle`.
- The `None` (automatic) default and the stored `'off'` of old files are different values on purpose. Do not collapse them.

## 5. Outputs

- A **per-cell label column** in the Measure tables and the spaCR store (item 974 Parquet next to `measurements.db`), written to a new table `measure_cell_cycle` (one row per cell; plate, well, field, label keys as in the cell table) and joined onto the cell table as columns when the user asks for the joined view. Per-cell columns:
  - `cell_cycle_stage`: `G1`, `S`, `G2`, `g2_prophase`, `prometaphase`, `metaphase`, `ana_telo_cyto`, `artefact`, or `unresolved` where the self-checks fail. Collapsed parents (`mitotic`, `G2/M`, `S/G2`, `interphase`) appear as in 990 section 5.
  - `cell_cycle_probability`: the probability of the called stage (after collapse, the sum over its children).
  - `cell_cycle_flag`: `auto`, `verified` or `unresolved`.
  - `cell_cycle_resolved_interphase` and `cell_cycle_resolved_mitotic`: booleans, the **resolvable flags** (false when that stage family failed its checks on the cell's plate).
  - `p_<class>` per class, `dna_log2_fc`, `dna_phase_posterior_G1|S|G2`, `field_offset_log2` (995), `model_id`.
- `measure_cell_cycle_checks` (one row per plate and check) and `measure_cell_cycle_report.html`: the 990 report, with the Measure channel names. It states the DNA channel (1-indexed), whether microtubules were used, and which model ran.
- Nothing already written by Measure changes: no existing column or table is touched.

## 6. Tests (CPU, `CUDA_VISIBLE_DEVICES=''`, `pytest -n 2` to 4 max)

- **M1 off-switch byte identity:** with `measure_cell_cycle='off'` (and with automatic and no nucleus channel) Measure's output is byte-identical to a run without the key, on the synthetic Measure fixture.
- **M2 default-on rule:** the automatic default runs the label when `nucleus_channel` is set (`'DNA'`, or `'DNA+MT'` with an MT channel) and does not run it when `nucleus_channel` is `None`. An explicit value overrides in both directions.
- **M3 old settings files reproduce:** a file stamped below 999 (and an unstamped file) with a nucleus channel produces the same output as on the base commit, and logs the notice. A new file takes the new default.
- **M4 DNA only vs DNA+MT:**
  - With a microtubule channel the spindle and tubulin features are used.
  - Without one the DNA-only model runs and G1/S/G2 are equal in both.
  - On the plate A subset the DNA+MT mitotic macro F1 is above the DNA-only value (reference: grid 0.85 against 0.76).
- **M5 identity with OPS:** on the same cells and settings, Measure and OPS give identical labels, probabilities and flags (one implementation).
- **M6 settings-file rerun identical:** run twice from the saved settings file, with tables, checks and checkpoint hashes byte-identical.
- **M7 channel display:** the GUI shows the DNA channel 1-indexed and stores it as Measure stores `nucleus_channel`; blank follows `nucleus_channel`.
- **M8 unresolved:** a low-mitosis fixture gives `unresolved` mitotic stages, still resolved G1/S/G2, and the resolvable flags match.
- **M9 legacy bool:** `cell_cycle=True` alone is byte-identical to today's; both on writes both tables.
- **Ratchets before handover:** raw-reader, nested-function, documented-callable and hash-comment.

## 7. Acceptance (makes it Ready)

- **A1:** on Funk plate A, a Measure settings run (nucleus channel set, MT channel set) gives labels identical to the OPS run of 990/995 on the same cells.
- **A2:** Measure on a DNA-only dataset gives the 990 DNA-only calls on it, with the checks reported.
- **A3:** M1-M9 pass, and the default-on rule is shown with and without a nucleus channel.

## 8. Strings for Codex

The user-facing strings are listed literally here. At implementation they go to Codex as `/mnt/firecuda2/codex/scratch-docs/codex_inbox/tr_999_task.md` (sources with SHA-256, 9 locales). **This item writes no translations.**

- Setting labels: `Cell-cycle label`, `Cell-cycle DNA channel`, `Cell-cycle microtubule channel`, `Cell-cycle base model`, `Cell-cycle fine-tuning`, `Cell-cycle labels file`, `Cell-cycle seed`.
- Values: `Off`, `DNA`, `DNA + microtubules`, `Automatic`.
- Help, `measure_cell_cycle`: "(str or None) - Give every cell a cell-cycle label: G1, S, G2 and the mitotic stages. DNA uses the DNA stain; DNA+MT also uses a microtubule stain, which improves the mitotic calls. Blank (automatic) turns it on when a nucleus channel is set and off otherwise. Off turns it off. Default automatic."
- Help, `measure_cell_cycle_dna_channel`: "(int or None) - The DNA stain (DAPI or Hoechst) the label is read from, counted from 1 as in the channel lists. Blank uses the nucleus channel. Default blank."
- Help, `measure_cell_cycle_mt_channel`: "(int or None) - The microtubule (tubulin) stain, counted from 1. Optional. With it the spindle and tubulin features improve the mitotic calls; without it the DNA-only model runs. Default blank."
- Help, base model, fine-tuning, labels, seed: the 990 help strings, with "cell-cycle label" in place of "OPS cell-cycle option".
- Report and table text: `Cell-cycle label`, `Stage`, `Probability`, `Flag`, `auto`, `verified`, `unresolved`, `Resolved (interphase)`, `Resolved (mitotic)`, `DNA channel`, `Microtubule channel`, `Model: DNA only`, `Model: DNA + microtubules`.
- Log and notice text:
  - "Cell-cycle label: on (nucleus channel {n} is set). Turn it off with measure_cell_cycle = Off."
  - "Cell-cycle label: off (no nucleus channel is set)."
  - "This settings file was written before the cell-cycle label became the default; it keeps the label off. Set measure_cell_cycle to Automatic to take today's default."
  - "Mitotic stages are unresolved on this screen (too few mitotic cells for the self-checks); G1, S and G2 are reported."
- GUI: `Use the new cell-cycle label` (button), `The legacy cell-cycle call and the new label run independently.` (note).
- Settings-panel group: the existing alpha group heading, no new heading.

## 9. Effort

S-M: the settings resolution, schema row, tables and GUI, on top of 990 and 995.
