# Item spec: host-nucleus effector translocation against parasite load: nucleus-territory cells without a cell stain, vacuoles per cell, raw and corrected nuclear effector signal, and its controls

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related:
  - `ITEM_vacuole_replication_titration.md` (vacuole grouping by gap, `vacuole_gap_px`, host assignment by majority overlap): build that first or together; this item reuses its vacuole table and settings and does not define them again;
  - 815 (replication report, done), 829 (parasites per vacuole from shape, open);
  - ops_engine `expand_labels` nucleus → cell (OPS spatial path) and `spacr.qt.mask_engine.dilate_objects` (curation): same operation, not reachable from Make Masks / Measure.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Why
HRMAn (Fisch et al. 2019) ships a public Protein_translocation plate: HFF cells, 18 h, 20×; d0 nuclei, d1 SAG1 (parasites), d2 HA-tagged effector; **no cell stain**. Pru TgIST-HA in rows B–C, RH GRA16-HA in D–E, RH GRA24-HA in F–G; columns 02–11, 16 fields per well; no uninfected wells. HRMAn's published readout is total nuclear HA against vacuoles per cell (0–10), normalised to 0-vacuole cells; at 10 vs 0 vacuoles: TgIST 1.88, GRA16 1.82, GRA24 1.40 (`Result.pzfx` in the dataset).

The analysis is not yet in spaCR. spaCR segments nuclei and parasites (Cellpose-SAM, Make Masks with `cell_channel=None`) and measures them (`measure_crop`), but it cannot:
1. make cell masks when there is no cell stain; `measure_crop` also rejects the `_None.npy` arrays Make Masks writes in that case;
2. count vacuoles per host cell (it counts pathogen objects per cell);
3. compute the nuclear effector signal with parasite pixels excluded and background subtracted. HA-tagged GRA16/GRA24/TgIST also fills the parasites and vacuole, so a vacuole lying over the nucleus in 2D adds signal that is not nuclear; HRMAn's raw total does not correct this;
4. produce the vacuole-binned curves normalised to 0-vacuole cells, with the controls that make them interpretable.

## Reference implementation
- **Masks:** spaCR Make Masks, ArrayScan, `nucleus_channel` 0, `pathogen_channel` 1, `cell_channel` None, `nucleus_diameter` 37, `pathogen_diameter` 15, `merge_pathogens` False, `pathogen_background` 200, `pathogen_signal_to_noise` 20, `pathogen_cellprob_threshold` −1 (settings saved with the run).
- **`cell_territories.py`:** drop nuclei < `MIN_NUC` 250 px (parasite nuclei in the DNA channel); grow each nucleus by up to 40 px without overlap (`skimage.segmentation.expand_labels`, a distance-capped Voronoi territory); write `[d0, d1, d2, cell, nucleus, pathogen]` (cell_mask_dim 3, nucleus 4, pathogen 5).
- **Measure:** `measure_crop` with `uninfected=True`, `cell_min_size` 300, `nucleus_min_size` 250, `pathogen_min_size` 15.
- **`translocation_objects.py`** (per host cell, one nucleus each):
  - parasites < `MIN_PAR` 15 px dropped; `near_par` = parasite mask dilated by `EXCL_PX` 2;
  - vacuoles: `vacuoles_from_parasites(par, gap=VAC_GAP 4)`, host by majority overlap → `n_vacuoles`; parasites by majority → `n_parasites`;
  - `ha_total_nucleus` = sum of raw d2 over the nucleus (HRMAn `Total_FluorescenceInt_Nucleus` equivalent, no background subtraction);
  - `ha_total_nucleus_excl`, `ha_mean_nucleus_excl` over nucleus pixels not in `near_par`; `frac_nucleus_excluded`;
  - `ha_mean_cyto` over territory − nucleus − `near_par`;
  - `dna_mean_nucleus`, `nucleus_area` (negative-control readouts);
  - `bg_ha` = median d2 over pixels outside every territory and `near_par` (5th percentile of the field when fewer than 1,000 such pixels).
- **`summary_pt.py`:**
  - corrected readout `ha_corr_total = (ha_mean_nucleus_excl − bg_ha) × nucleus_area`, `ha_corr_mean`;
  - curves by k = vacuoles per cell 0–10 (`KMAX`), ratio to k = 0 of the same effector, for readouts hrman_equiv, corrected, corrected_mean, hrman_equiv_by_parasites, dna_control, nucleus_area, cyto_ha, frac_nucleus_excluded, and spaCR's own `nucleus_channel_2_integrated_intensity` binned by pathogen objects;
  - per-well ratios (k mean / k = 0 mean in the same well, ≥ 3 cells in the bin) → mean ± SEM over wells;
  - null threshold: 99th percentile of `ha_corr_mean` in k = 0 cells per effector → % HA-positive nuclei per k and per well (infected vs uninfected);
  - shuffle null: k permuted among cells within each field, 200 times, `np.random.default_rng(0)` consumed effector by effector, hrman_equiv then corrected;
  - agreement with HRMAn over k = 1–10: Pearson r, Spearman ρ, mean |Δ ratio|, ratio at k = 10;
  - HRMAn reference parsed from `Result.pzfx` (`hrman_reference`).
- **Outputs:** `counts_per_effector.csv`, `curves_all_readouts.csv`, `per_well_ratios.csv`, `per_well_summary.csv`, `null_thresholds.csv`, `pct_ha_positive_nuclei.csv`, `pct_ha_positive_per_well.csv`, `shuffle_null.csv`, `agreement_with_hrman.csv`, `hrman_reference.csv`, `spacr_infection_report_per_well.csv`; per-cell `cells_translocation.csv`.

## What to build
1. **Cells from nuclei when there is no cell stain** (Make Masks, alpha):
   - `cell_from_nucleus_expand_px` (0 = off; 40 here): with `cell_channel` None, grow each nucleus by up to this many px without overlap (Euclidean, `expand_labels`, the same operation as `dilate_objects` / the OPS spatial path; share one helper) and save it as the cell mask;
   - `cell_from_nucleus_min_nucleus_size` (250): nuclei below this are removed before growing;
   - merged arrays then carry a cell plane and the standard name, so `measure_crop` reads them without a workaround; the cell table gets `cell_source = 'nucleus_territory'` and the run report says the cell outlines are territories, not cells.
2. **Vacuoles per host cell:** reuse `vacuole_from_parasites`, `vacuole_gap_px` (4), `vacuole_min_size`, `host_assignment` ('majority_overlap') from `ITEM_vacuole_replication_titration.md`; add per-cell `n_vacuoles` and `n_pathogens` (majority overlap) to the cell table. Here `pathogen_min_size` 15.
3. **Nuclear translocation readout** in `spacr.host_pathogen` (alpha), `nuclear_translocation(settings)`, run after `measure_crop`:
   - `translocation` (bool, False);
   - `translocation_channel` (2: the effector) and `translocation_control_channel` (0: DNA);
   - `translocation_exclude_px` (2): nucleus pixels within this distance of a parasite pixel are excluded;
   - `translocation_background` ('outside_cells_median' | 'none'), `translocation_background_min_px` (1000), `translocation_background_fallback_percentile` (5);
   - `translocation_min_nucleus_size` (250);
   - `translocation_k_column` ('n_vacuoles' | 'n_pathogens'), `translocation_kmax` (10);
   - `translocation_groups` (path to the plate map, or a row → group mapping; here rows B–C TgIST, D–E GRA16, F–G GRA24);
   - `translocation_min_cells_per_bin` (3, per-well ratios);
   - per-cell table `translocation_cells` (database table and CSV) with every column of `cells_translocation.csv`;
   - two readouts, always both: **raw** (`ha_total_nucleus`, the HRMAn-equivalent) and **corrected** (`(mean over kept nucleus pixels − background) × nucleus area`), each normalised to k = 0 cells of the same group.
4. **Controls** (always written, plain labels "control"):
   - 0-vacuole reference (ratio 1);
   - null threshold `translocation_null_percentile` (99) of the corrected mean in k = 0 cells per group → % positive nuclei per k and per well (infected vs uninfected), with the threshold table;
   - within-field shuffle null: `translocation_shuffle_n` (200), `translocation_seed` (0), consumed group by group, raw then corrected (as the reference, so the null is identical);
   - DNA-stain control (mean DNA in the nucleus vs k), nucleus area vs k, and the fraction of each nucleus excluded vs k;
   - `translocation_reference` (optional path to a reference curve CSV, e.g. HRMAn's `Result.pzfx` parsed to `effector, k, total, ratio`) → agreement table (Pearson r, Spearman ρ, mean |Δ ratio|, ratio at kmax). The HRMAn values are data in that CSV, never code.
5. **Outputs** (`<src>/translocation/`): the CSVs listed under Reference implementation with the same columns (`pct_ha_positive_*` named after the channel), plus one figure: ratio vs k per group, raw and corrected, bars + jittered wells at ~60 % opacity, shuffle-null band, the reference curve dashed (spaCR palette).

## Validation data and acceptance
- **Data:** HRMAn Protein_translocation plate (Dryad; Fisch et al. 2019). Pilot: 3 fields (`B02_3`, `D05_3`, `F08_3`), masks made on CPU.
- **Acceptance:** identical output to the reference outputs:
  - From the pilot merged arrays: territory masks identical to `cell_territories.py`; `translocation_cells` identical to the reference `cells_translocation.csv` (floats ≤ 1e-9); every pilot summary table identical, including the shuffle null with seed 0.
  - From the saved pilot settings end to end on CPU (same torch version and thread count as the pilot): nucleus and parasite masks identical to the pilot; downstream identical.
  - The reference curve parsed from `Result.pzfx` identical to the reference `hrman_reference.csv` (TgIST 1.88, GRA16 1.82, GRA24 1.40 at k = 10).
  - **Full plate:** masks within the GPU self-diff floor; the reference is the reference implementation's full-plate output on the same masks, and spaCR must match it identically. Scientific checks, not pass/fail for the port: the raw readout's per-group ratio at k = 10 is reported beside HRMAn's; the shuffle null stays ~1; the DNA control stays flat.
- **Synthetic test** (unit test, CPU): nuclei with a planted effector signal rising with vacuole count, plus a bright vacuole lying over some nuclei: the raw readout rises with k and is inflated where the vacuole overlaps; the corrected readout recovers the planted slope; with no planted signal the corrected ratio stays at 1 and the shuffle null covers it.

## Literature
- HRMAn: Fisch et al. 2019 (eLife); HRMAn 2.0: Fisch et al. 2021 (Cell Microbiol).
- Effectors: TgIST (Gay et al. 2016, J Exp Med; Olias et al. 2016, Cell Host Microbe); GRA16 (Bougdour et al. 2013, Cell Host Microbe); GRA24 (Braun et al. 2013, J Exp Med); MYR1-dependent export (Franco et al. 2016, mBio).
- Nucleus-seeded cell territories: CellProfiler "Distance-N" secondary objects (Carpenter et al. 2006, Genome Biol; Stirling et al. 2021, BMC Bioinformatics).
- Segmentation: Cellpose-SAM (Pachitariu et al. 2025, bioRxiv).

## Effort
- **Effort:** M (territories S; translocation readout and controls M; depends on the vacuole item).
