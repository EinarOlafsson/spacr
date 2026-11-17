# Item 971: OPS Live preview, one tab per pipeline step

- **Number:** 971 (master session, 2026-11-17). Built in lane `claude-971-1117`.
- **Maintainer's words (2026-11-17):** "the OPS live preview should allow the user to test out each step in the ops pipeline: aligning/stitching; spot detection and genotyping; object detection and phenotyping; merging/aggregation; clustering. so the test data needs to have phenotype and genotype test data. and there should be several tabs that the user can go between in the live preview for ops. the live preview should allow the user to test all settings in the module."
- **Readiness:** a GUI preview. It writes nothing into the user's output folder, so the default OPS pipeline output is unchanged. No settings schema bump. Alpha (`ALPHA_FEATURES[971]`), hidden unless Preferences > Show alpha features is on.
- **What exists on nightly:**
  - The OPS module is a page fold of Mask Generation: an `AppScreen` with `app_key="ops"`, built by `map_barcodes.build_settings_screen`. It has no Live switch.
  - The shared Live machinery in `app_screen.py`: the `preview_controls` table puts a Live switch (`_preview_switch`) on a preview card, `_on_preview_switch` shows the card and seeds it once from the form (`_prime_preview` -> `panel.apply_settings(model.collect())`), and `_show_the_src_live` re-reads the source. Each committed setting goes through `SettingsWidgets._setting_committed(key)` -> `AppScreen._observe_settings_commit(key)`.
  - The shared live-view contract `spacr/qt/widgets/preview_contract.py` (`LivePreviewContract`: run guard, cancellation token, busy/idle buttons, wording).
  - The engine's step functions, which the tabs call (listed per tab below).
  - The OPS end-to-end test data (`ops_testdata_funk2022_A1`, core and full tiers on Hugging Face), with stored step outputs under `reference/`.

## Design

### Where it sits
- An `"ops"` row in `preview_controls`, and an `"ops"` arm in `_build_runtime_panel` that builds the card the same way the Motility and Timelapse arms do: figures, preview card, console in one splitter.
- The Live switch rides on the card title (the existing `AiToggleLabel`), with object name `OpsLiveSwitch`, gated by `ALPHA_FEATURES[971]`. With alpha off the switch is hidden and the card is never shown.
- The card body is a `QTabWidget` with five tabs. Every tab has the same geometry: a header line (step, sample, status, seconds), one picture area of a fixed 3:1 aspect, and one table area of fixed height. Pictures are drawn with matplotlib Agg in the worker thread in the active theme's palette (navy/teal/blue on the theme's own surface), so dark and light both read.

### Tabs, samples, spaCR functions and timing targets (CPU, two threads)

| Tab | Shows | Sample | spaCR functions called | Target |
|---|---|---|---|---|
| 1 Align & stitch | cycle-1 DAPI tile mosaic with tile frames and overlaps; cycle-to-cycle shifts of one SBS crop (before/after overlay of cycle 1 vs the last cycle, shift table); phenotype field footprints and a phenotype DAPI overlay placed on the SBS mosaic | a 2 x 2 block of SBS tiles (`ops_sites`) and the phenotype fields over it; one SBS crop (default 400 px) | `run_ops(phases=["stitch", "phenotype"])` into the preview workspace (stitch, seeded or anchored placement); `ops_cycles.align_field` with the decode's registration options | <= 5 s |
| 2 Spots & genotyping | detected spots on the SBS crop over DAPI, coloured by strength; the base calls of the strongest reads; per-cell barcode calls with reads, vote fraction and quality | the same SBS crop, every cycle | SBS nuclei: `_cellpose_model` on the crop's cycle-1 DAPI (cached until a nucleus setting changes); calls: `ops_engine._decode_field` on the crop, with the task built from the settings exactly as `_decode` builds it (rule (a) when the preset turns it on); `ops_sbs.call_reads` for the per-read letters | <= 10 s (first nuclei pass up to about 20 s, then cached) |
| 3 Objects & phenotype | nuclei and cell outlines on one phenotype crop (DNA + cell stain); a few per-cell features as a table and a histogram | one phenotype crop (default 512 field px) | `_ph_read_field`, `_ph_segment_field` with the `ops_ph_*` models, `ops_sample.measure_phenotype_field` with the measure options the `ph_measure` task carries; a hook (`OBJECT_LAYERS`) for item 970's pathogen and organelle objects | <= 10 s |
| 4 Merge & aggregation | phenotype cells placed on the SBS nuclei, matched pairs drawn, matched share; the per-gene aggregate table | the aggregation set: the two phenotype fields over the 2 x 2 block (about 4,500 cells) | `run_ops(phases=["merge"])` then `run_ops(phases=["aggregate"])` in the preview workspace | <= 8 s |
| 5 Clustering | gene embedding coloured by cluster, cluster sizes, members of each cluster | the genes of tab 4 with at least `ops_profile_min_cells` cells | `ops_clusters.embed_profiles`, `gene_graph`, `leiden_partitions` at the first resolution and seed of the settings | <= 5 s |

No step is reimplemented. Where the engine takes a task dict rather than settings, the preview builds the dict with the engine's own setting readers (`_base_channels`, `_setting_number`, `_spot_detector`, `_registration_choice`, `_cell_rule`, `decode_rule`, `_ph_channels`, `_ph_profile_features`, `_ph_granularity_subsample`).

### Which setting refreshes which tab
- The tab of a setting is read from its **category in the settings registry** (`spacr.settings` categories of `"ops"`), never from a key list in the preview, so item 969's reorganisation into subcategories flows through.
- Rules, as data: a category whose name speaks of input or performance belongs to every tab; align/stitch/placement -> 1; decod/genotyp/spot/barcode/read -> 2; segment/object/measure/phenotyp/flat -> 3; merge/aggregat -> 4; cluster/profile/embedding/module/score -> 5. A broad category ("OPS decoding", "OPS phenotype" today) is split by the setting's stem (`ops_registration_*` -> 1, `ops_merge_*`/`ops_aggregate_*` -> 4, `ops_cluster_*`/`ops_profile_*`/`ops_tvn_*` -> 5). Subcategories from item 969 win over stems.
- A committed change invalidates the caches of its tab and of every later tab. Only the tab on view refreshes, after a 600 ms debounce; a later tab refreshes when it is shown. Nothing runs while Live is off.

### Off the GUI thread, debounced, cancellable
- One worker per refresh (`QThread`), as Mask, Timelapse and Motility do, with the shared contract's run guard and cancellation token. A new request supersedes the one in flight: its result is dropped by token. Cancel drops the pass; nothing it made is kept.
- Upstream outputs are cached in the preview workspace, keyed by the settings that change them: the stitch/placement database, the SBS nuclei of the crop, the phenotype crop's masks, the merged/aggregated tables. A later tab reuses them.

### Data
- The preview works in its own workspace, `~/.cache/spacr/ops_live/<hash of the inputs>/`, never the user's `dst_root`.
- With no `genotype_source` set, it uses the OPS end-to-end test data: the full tier when it is in the example cache, else the core tier (whose samples lie inside its region of interest), and offers to download the core tier when neither is there. The shipped settings of the test data fill what the form leaves empty (paths, plate, `ops_ph_placement_seed`, `ops_design`); the form's own values win for every other setting.
- Tabs 4 and 5 need more cells than a live crop holds. They run on the aggregation set: for the test data, the stored SBS nuclei, calls and phenotype cells of `reference/`; for a user's data, the same tables from the user's own finished run (`dst_root/measurements.db` and the phenotype cell tables), read only. The tab says which inputs it used.

### Test data assessment (to be filled from the build)
- Phenotype and SBS for overlapping sites: yes (SBS tiles 321, 322, 331, 332, all 11 cycles; phenotype fields 1255 and 1264 over them, 1280 empty). The placement seed places the fields without whole-well anchors.
- Merge and aggregation: about 4,100 merged cells, 3,200 with a guide, 881 genes, median 3 cells per gene (230 genes with 5 or more). Enough to exercise every setting; too few cells per gene for a stable profile.
- Clustering: runs on the 519 genes with 3 or more cells; the clusters are not biologically meaningful at this depth. Scoring against CORUM/KEGG/STRING needs `ops_cluster_reference_dir`, which the archive does not hold.
- The report proposes an archive extension (not published; publishing needs the maintainer).

## Tests (Qt offscreen, CPU)
1. The OPS screen has the Live switch; it is hidden with alpha off.
2. Each tab renders on a tiny synthetic fixture (synthetic arrays through the same render code; the compute calls stubbed at their spaCR entry points).
3. A committed change to a setting of tab k refreshes tab k only and marks later tabs stale; earlier tabs untouched.
4. Cancel drops the pass in flight; its result is never drawn.
5. Nothing runs while Live is off.
6. The tab of every OPS setting is found from its registry category (no setting without a tab).

## Proof
- Each tab timed on the real test data (CPU, two threads), in REPORT.md with a glossary.
- A default OPS run on the test fixture (stitch, seeded phenotype, merge, aggregate) on this branch is byte-identical to nightly (table hashes).
- Offscreen screenshots of every tab in both themes in `BMCspacr/reanalysis/item971_proof/`.
- Gate groups items, ops, settings, i18n, ratchets, deck, rerun2.

## Tutorials that should show the Live preview
Listed in the report; nothing recorded in this item.
