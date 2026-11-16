# Item 960: OPS live view, a parameter grid search on representative tiles

- **Number:** 960 (reserved by master on 2026-11-16).
- **Maintainer decision (2026-11-16):** "our live view needs to fill these functions as well". The reference text (Brieflow, Nat Commun 2026, doi 10.1038/s41467-026-73643-7, Methods, SBS module) reads: "A grid search utility evaluates combinations of detection parameters on representative tiles, reporting spot count, mapping rate, and cell assignment rate for each combination, enabling users to identify optimal settings before full-scale processing."
- **When:** build after the genotyping freeze (maintainer decision 2026-11-16), so the grid shows the frozen decoder.
- **Gap:** spaCR's live preview covers segmentation only. OPS decode thresholds are tuned inside the run (auto mode), with no per-combination table the user can inspect before a full run.
- **Comparator:** Brieflow `workflow/lib/sbs/automated_parameter_search.py` (`automated_parameter_search(aligned_images, mask, barcodes, df_barcode_library, …, param_grid, metric_fn, fixed_params)`) uses joblib over `itertools.product` and reports metrics per combination.

## Spec
1. **Tile picker.** In the OPS live view, choose N representative tiles (default 8). Modes: random stratified across wells (seeded), well edge vs centre, and user-picked. The selection and seed are saved in the settings file.
2. **Parameter grid.** Any subset of the spot-detection and decode settings, each with a list of values. Examples: spot peak threshold, peak width, read-quality floor (`ops_single_read_quality`), pixel quality, `ops_min_fraction`, `ops_read_mismatches` / `ops_prefix_max_distance`, and `ops_read_mode` (item 952). The grid is defined in the GUI or as lists in the settings file (one shared parser with the CLI).
3. **Metrics per combination,** for each tile and pooled:
   - spot count (raw peaks and reads after the quality filter);
   - read mapping rate;
   - cell assignment rate (% of nuclei or cells with one gene);
   - decoy FDR (calibrated if item 949 calibration is on);
   - barcode-shuffle call rate (item 949, 3 seeds);
   - time per tile.
   Brieflow reports only the first three; spaCR adds the two error rates so users never pick a setting on yield alone.
4. **Output.**
   - A sortable table plus a yield-vs-shuffle-rate plot, showing the current default and the Pareto front.
   - A one-click "apply to settings", which writes the chosen values into the run settings and records the grid run ID.
   - Every grid run saved as CSV next to the settings file. Rerunning the same settings file gives identical output.
5. **Reuse:**
   - The engine path is the same as a full run: the per-field evidence cache (`ops_store_field_cache`) and the decode functions.
   - No second implementation of decoding.
   - Parallel over combinations within the CPU budget of the shared resource fitter.
6. **Alpha:**
   - Off by default and registered in ALPHA_FEATURES.
   - No schema bump: the new settings default to off/empty.
   - 9-locale tooltips and help, arrival contracts and catalogue records in the same commit.

## Proof check (pre-set)
- **Identity:** a 1×1 grid at the default settings reproduces the full-run numbers for those tiles exactly (spot count, mapping, assignment, decoy FDR, shuffle rate).
- **Comparator reproduction:** Brieflow's `automated_parameter_search` rerun by us on the same 8 tiles and grid gives spot count, mapping rate and cell-assignment rate. These are shown next to spaCR's grid and next to Brieflow's published Fig. 2c / Supplementary Fig. 3 values, where they exist, as computed AND published with source; otherwise "n/a in paper".
- **Generality:** the proof runs on Funk plate A (8 tiles across A1–A3) and PERISCOPE CP186A (8 held-out sites).
- **Control:** the shuffled-library control at every grid point (3 seeds). A single-seed value lies within the 500-permutation 95 % range at the default point.
- **Speed:** the time per combination per tile is reported against Brieflow's on the same machine and tiles.
