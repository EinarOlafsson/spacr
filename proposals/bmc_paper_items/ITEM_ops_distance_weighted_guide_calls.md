# Item 988: neighbour-bleed discount, read distance and real-mask ownership in the guide call (Funk full-barcode and PERISCOPE spacer-prefix)

- **Number:** 988, confirmed free and reserved by master on 2026-11-18.
- **Status:** SIGNED OFF by the maintainer on 2026-11-18 (decisions below). Step 0 (post-hoc proof check) approved to run now; results in section 3c. Not implemented in spaCR yet.
- **Maintainer decisions, 2026-11-18:**
  1. **N1 (neighbour-bleed discount) is the MAIN arm.** Distance (A3-A5) is secondary; real masks in Funk (M1/M2) are the third arm.
  2. Step 0 runs now. The full acceptance run waits for the 967 default decision.
  3. PERISCOPE control row C0 = recipe_simple + 967 at the decided k.
  4. Approved compute: one Funk A1 re-decode (about 40 min, field caches kept so all Funk arms are then post hoc), about 16 min of GPU for the M1 objects rerun (via gpu_turn), and tuning L and b as extra axes of the existing decoy tuner with "off" always in the grid.
  5. PERISCOPE distance: the **nucleus-mask distance** (distance from the read to the nucleus of its own cell, 0 inside; fallback for the 1 % of reads whose nearest nucleus is another cell's: distance to the own cell's centroid minus the site's median nucleus radius, flagged). Chosen because the masks exist and the nucleus is what the maintainer's idea refers to.
  6. Multi-guide cells: a QC table (0/1/2/3/≥ 4 guides plus the share explained by neighbour bleed) and `ops_keep_multi_guide_cells` (alpha, off). The pooled doubles-vs-singles test has its own card: `experiment_designs/20261118_multi_guide_doubles_vs_singles.md` (card only, no run).
  7. Item number 988: confirmed and reserved by master (2026-11-18).
- **Maintainer decisions, second round (2026-11-18):**
  1. The doubles-vs-singles card is signed off; it runs after 988's rule is decided, on spaCR-measured profiles (PERISCOPE Well1, Funk plate A).
  2. Number 988 confirmed and reserved by master.
  3. **PERISCOPE spot threshold:** the acceptance grid adds read-threshold values **20 and 30** to the existing {40, 60, 80, 100, 130, 160}. The k0 tuner chose 40, the lowest value in the grid, so the optimum may lie below it.
  4. **Working point for every method = the knee** of its yield-vs-decoy-FDR curve, found by one fixed geometric rule chosen on the tune half and applied alike to spaCR, PERISCOPE and Funk/Brieflow. Every table also reports each arm at decoy FDR 0.1 % and 1 % (0.1 and 1 per 100 calls). The pass criterion (4.6) judges arms both at their knees and at matched FDR.
  5. **Decoy set:** a separate spec (`ITEM_ops_decoy_matching_and_genotype_qc.md`, number to be assigned) proposes distance-matched decoys as the default; its proof is pending. The 988 acceptance uses whichever decoy set is the spaCR default at run time, records which one, and is **re-scored if the default changes** (the read tables are kept, so re-scoring is post hoc).
- **Input from a sibling analysis (`/mnt/wd4tb/scratch_rules_1118/`, 2026-11-18), used in the acceptance design:** PERISCOPE 1-mismatch substitutions are strongly biased by channel crosstalk (A→C 85 %, G→T 60 %, T→C 52 %, C→A 48 %); Funk cycle 7 has 2-3× the mismatches of other cycles; decoy nearest-guide distances are 88/12 % at distance 2/3 for Funk (guides 93/7) and 91/9 % for PERISCOPE (guides 96/4). Consequences: (a) reads repaired by a 1-mismatch correction should not carry full weight when the substitution is not one of the crosstalk-typical ones; this is out of scope for 988 but every 988 table reports exact and corrected reads separately; (b) the decoy FDR is a fair proxy for the error of a near-miss call, because decoys sit at almost the same Hamming distance to the guides as guides do to each other; (c) in the Funk edge/position check (4.5), add cycle-7 mismatch rate per field as a covariate.
- **Requested by the maintainer (2026-11-18):** "use a read's distance to the nucleus as a quality metric when calling a cell's guide". Example: a bright read close to the nucleus that clearly matches one guide, and a dimmer read farther away that matches a decoy or another guide; the cell should still be called by the close, high-quality read.
- **Order (hard rule, confirmed 2026-11-18):** the acceptance run starts only **after item 967's default is decided** (`ops_prefix_minority_min_reads`, k grid 0/2/3/4 on CP186A Well1; k4 is still running). Otherwise a gain from distance and a gain from ignoring minority reads cannot be told apart. The control row in section 6 uses whatever 967 setting becomes the default.
- **Readiness:** under development. Only proof checks until the spaCR build passes acceptance (8-step workflow). No paper numbers from scripts.

Abbreviations: FDR = false discovery rate (here: decoy calls per 100 real calls, from the decoy library); CI = 95 % confidence interval (field bootstrap, 1,000 resamples); px = pixels; SBS = sequencing by synthesis (the barcode imaging cycles); OPS = optical pooled screen; PERISCOPE = the Broad pooled Cell Painting screen (CP186A plate) and its spacer-prefix read method; NT = non-targeting guide.

## 1. Current code (verified 2026-11-18 on nightly, read only)

**Funk full-barcode path**
- `spacr/ops_sbs.py:1046-1083` `nearest_boundaries`: each nucleus is treated as an equal-area disc. Gap = distance from the read to the centroid minus the disc radius (negative = inside the disc). The two nearest nuclei are kept.
- `spacr/ops_decode.py:522-526`: a read counts when gap ≤ `ops_footprint` (10 px). A read is "ambiguous" when the runner-up nucleus is within `ops_tie` (1 px) of the first.
- `spacr/ops_sbs.py:1148-1150`: reads that match no library barcode are dropped before the vote. Decoy-matched reads do vote.
- `spacr/ops_sbs.py:1160-1186`: vote weight = min(1, quality) + `weight_floor` (0.05). The top guide needs ≥ 0.6 of the vote and no tie. A one-read call also needs quality ≥ `single_read_quality` (auto-tuned).
- `spacr/ops_decode.py:405-433, 543-586`: pixel fallback for nuclei no read named.
- `spacr/ops_decode.py:1090-1158` (`choose_well_thresholds`) and `1290-1340` (`tune_thresholds`): the well's fields are split 50/50 by seed; on the tuning half every grid setting is scored; a setting passes when the upper CI of the decoy FDR ≤ 0.1 per 100; the passing setting with most real calls wins; the report half only reports. `tune_thresholds` takes any list of axes, so a new axis is a small change.
- The gap is computed per read in the field evidence (`evidence["gap"]`) but **is never used as a weight or tie-breaker, and is not saved** in any output file.

**PERISCOPE spacer-prefix path**
- `spacr/ops_prefix.py:1592-1594`: a read goes to the cell label under its peak pixel. No distance is computed.
- `spacr/ops_prefix.py:614-678` `recipe_simple_quality`: a port of the Broad recipe; only cells with exactly one barcode are called ("perfect"/"great").
- `spacr/ops_prefix.py:681-701, 758-810`, item 967 (alpha, default 0 = off): a barcode with fewer than k reads is ignored when another barcode in the cell has strictly more reads. Ties still block.
- The PERISCOPE tuner (`ops_prefix.py` around 1570-1610) uses the same `tune_thresholds` with axes read mode, read threshold, minimum read quality and maximum distance.

## 2. Proposal

### 2a. Read weight (one formula, both paths)
For every matched read (exact or corrected, guide or decoy):

  w = q_term × d_term

- `q_term` = min(1, quality) + 0.05 (today's Funk weight), or 1 in the "distance only" arm.
- `d_term` = a decay in the read's distance g (px) to its own nucleus:
  - g = 0 inside the nucleus; outside, the distance to the nucleus edge.
  - **Funk:** g = max(0, disc gap) from `nearest_boundaries` (no new geometry). Optional later: the true mask edge (distance transform of the nucleus label), only if the disc version passes.
  - **PERISCOPE:** g = distance from the read to the nucleus mask of the same cell (distance transform of `{site}_nuclei.npy`; nucleus label ids match cell ids in 97.6 % of nucleus pixels on site 12). When the cell has no nucleus: g = distance to the cell centroid minus the equal-area nucleus radius from the well median (and flag it).
  - Shape: `d_term = 1` for g ≤ g0, then `exp(-(g - g0) / L)`, floored at `d_min` (default 0.05) so a far read never counts as zero.
  - g0 = the "interior" allowance (default 0 px); L = decay length in px.
- Why a soft decay and not a hard cut: Funk already has a hard cut (`ops_footprint` = 10 px), and PERISCOPE reads lie anywhere inside the cell. A soft weight only changes who wins inside a cell; it does not change which reads are owned.

### 2b. How it is used
- **Funk:** the weighted vote replaces the current weight when the setting is on. The 0.6 share rule, the tie rule and the single-read floor stay. The single-read floor is applied to `q_term × d_term` of that read (so a lone far read needs higher quality). Pixel fallback is unchanged (it already uses `pixel_radius`).
- **PERISCOPE:** a new value `weighted_majority` of the existing `ops_prefix_quality_method`:
  - for each barcode in a cell, W = sum of w over its matched reads;
  - call the top barcode when W_top / W_total ≥ `ops_prefix_weighted_min_share` (default 0.6, same as Funk) and W_top > W_second (strict);
  - decoy-top cells are kept as decoy calls (they feed the decoy FDR), as today.
- **Relation to 967:** 967 is the special case w = 1 for all reads, with "ignore a barcode with fewer than k reads when another has more". The weighted majority with `q_term = d_term = 1` is a count-share rule, related to but not the same as 967 (967 is a count floor on the minority, not a share). The weighted rule generalises 967 in spirit: a minority barcode loses when its reads carry little weight, whether because they are few, dim or far. So:
  - the two are not combined in the same run; the weighted rule is an alternative to 967;
  - the table also shows "weighted majority, quality and distance off" (count majority) so the effect of switching rule family is separated from the effect of distance.

### 2c. New settings (alpha, all OFF by default; `ALPHA_FEATURES[988]`)
| Setting | Type | Default | Meaning |
|---|---|---|---|
| `ops_read_distance_weight` | str | `"off"` | `"off"`, `"decay"` (q × d) or `"distance_only"` (d only, for the control arm). Funk path. |
| `ops_read_distance_decay_px` | float | 4.0 | L, decay length in px (used only when on). |
| `ops_read_distance_interior_px` | float | 0.0 | g0, distance with full weight. |
| `ops_read_distance_min_weight` | float | 0.05 | d_min. |
| `ops_prefix_quality_method` (existing, `QUALITY_METHODS` in `ops_prefix.py:57`) | str | `"spacr"` (unchanged; the 967 CP186A runs set `"recipe_simple"`) | add a third value `"weighted_majority"`. |
| `ops_prefix_distance_weight` | str | `"off"` | same three values, PERISCOPE path. |
| `ops_prefix_distance_decay_px` / `_interior_px` / `_min_weight` | float | 8.0 / 0.0 / 0.05 | as above, in SBS px of the CP186A frame. |
| `ops_prefix_weighted_min_share` | float | 0.6 | share for `weighted_majority`. |

- Defaults keep today's output byte-identical. No `SETTINGS_SCHEMA_VERSION` bump and no `DEFAULT_CHANGES` entry now; both only if a later sign-off changes a default.
- Each call table gets `d_weight_top` (summed distance weight of the winning guide) and `g_median_top` (median g of its reads); the read table gets `nuc_gap_px`.
- Settings go in the existing OPS decoding category, after the vote keys. Catalogue records and 9-locale entries ship in the same commit as the code.

### 2d. Auto-tuning of the decay length
- **Yes, as one extra axis**, never free: `L ∈ {off, 2, 4, 8, 16} px` for Funk (reads lie within 10 px of the disc) and `{off, 4, 8, 16, 32} px` for PERISCOPE (median read is 11 px from its nucleus) added to the existing decoy tuner grid (Funk: single-read floor × pixel quality × L; PERISCOPE: its four axes × L). `off` must be in the grid, so the tuner can only pick distance when it gives more real calls under the same FDR bound on the tuning half.
- g0 and d_min stay fixed (0 px, 0.05): more axes would overfit a 50/50 field split.
- The grid grows 5×. Funk tuning is cheap (counts per field per setting); PERISCOPE tuning reruns the cell rule per setting (minutes, the reads are already in memory).

## 3. Feasibility check (read-only, 2026-11-18)
Scripts: `feas.py`, `feas2.py`, `mult_periscope.py`, `mult_funk.py` (kept with this spec under `/mnt/wd4tb/scratch_specs_1118/feasibility/`, with their JSON outputs). `python -I`, `nice 10`, ≤ 9 GB address-space cap; read only.

**Do the read tables already hold the read-to-nucleus distance?**
- **Funk (LaC024A A1 production):** **no.** Outputs are per-cell only (`calls_A1.parquet`: 976,209 calls, columns such as `n_guides_seen`, `vote_frac`, `q_best`; `measurements.ops_barcodes.parquet`). The per-read gap exists only in memory during decoding (`evidence["gap"]`). Any per-read Funk number needs one re-decode.
- **PERISCOPE (CP186A Well1, 967 k0 outputs):** **no**, but it can be computed. `reads_CP186A_W1.parquet` (1,158,917 reads) has `y`, `x`, `cell`, `quality`, `prefix`, `status`, `level`. Nucleus masks for all 86 sites exist in the production seg; nucleus ids match cell ids (the nearest nucleus belongs to the read's own cell for 98.97 % of matched reads).

**How often does a cell have a second barcode?**
- **Funk A1:** 14.3 % of read-called cells (primary source, 913,289) saw ≥ 2 guides among their matched reads and were still called; their median vote share is 0.725. Cells blocked by a second guide are not in the calls file, so the blocked share needs the re-decode.
- **PERISCOPE:** 35.6 % of cells with an exact read (246,145) have ≥ 2 distinct exact barcodes. In 54,255 of them one barcode clearly leads; there the runner-up is a single read in 71.9 % of cells and is a decoy in only 0.69 % (so the maintainer's "far decoy" case is rare; the competitor is almost always another real guide).

**Is the second read farther or dimmer? (PERISCOPE, one random read of each barcode per cell, 54,255 cells)**
| Measure | Top barcode read | Runner-up read |
|---|---|---|
| Median distance to own nucleus (px; 0 = inside) | 11 | 14 |
| Inside the nucleus | 17.5 % | 11.8 % |
| Median read quality | 0.261 | 0.258 |
- Runner-up farther in 56.2 % of cells, closer in 41.7 %, tied 2.1 %: net +14.4 points (CI 13.6 to 15.3).
- Runner-up dimmer in 50.7 % of cells: **no quality difference** (a coin flip).
- Farther and dimmer at once: 28.4 %.

**Does distance flag bad reads by itself? (all 674,995 exact reads in cells)**
| Distance to nucleus | Reads | Decoy reads per 1,000 | Median quality |
|---|---|---|---|
| inside | 110,637 | 3.41 | 0.244 |
| 1-3 px | 52,354 | 3.57 | 0.245 |
| 4-6 px | 60,622 | 3.45 | 0.245 |
| 7-10 px | 88,177 | 3.20 | 0.246 |
| 11-20 px | 194,085 | 3.30 | 0.250 |
| 21-40 px | 148,375 | 2.94 | 0.261 |
| > 40 px | 20,745 | 1.64 | 0.289 |
- Decoy reads are **not** more common far from the nucleus, and quality does not fall with distance. By cell-centre distance (radius-scaled) the decoy share rises only slightly toward the cell edge (0.29-0.31 % in the inner half, 0.38 % at 1.0-1.5 radii).
- Distance to the image border is unrelated to distance to the nucleus (Spearman -0.002), so field-edge position is unlikely to explain a distance effect; section 4.5 still tests it on calls.

**Per-cell guide multiplicity (maintainer request, 2026-11-18)**

Share of all cells (PERISCOPE: 535,390 cells in the label masks, 86 sites; Funk: 31,810 owned nuclei in a sample of 8 A1 fields, 150-157) by the number of distinct **library guides** among their matched reads. Shuffled = read-to-cell assignment permuted within each field, mean of 500 permutations (each cell keeps its read count). Decoy-matched barcodes are counted separately.

*PERISCOPE CP186A Well1 (967 k0 = production rule; per-read quality floor = the tuner's choice, 0.0; read threshold 40)*
| Counting | Guides: 0 | 1 | 2 | 3 | ≥ 4 | Shuffled ≥ 2 (95 % range) | Cells with ≥ 1 decoy barcode |
|---|---|---|---|---|---|---|---|
| (i) all matched reads | 54.18 % | 29.58 % | 12.06 % | 3.33 % | 0.85 % | 27.23 % (27.22-27.23) | 0.37 % |
| (ii) guides with ≥ 2 reads each | 77.63 % | 18.24 % | 3.63 % | 0.46 % | 0.05 % | 0.008 % (0.006-0.011) | 0.02 % |

*Funk LaC024A A1, 8-field sample (production code and thresholds: read threshold 315, footprint 10 px, 1-mismatch reads kept only when they pass the production runner-up check; quality floor 0.06 = the tuned single-read floor; sample genotyped 74.9 % vs 76.2 % for the whole well)*
| Counting | Guides: 0 | 1 | 2 | 3 | ≥ 4 | Shuffled ≥ 2 (95 % range) | Cells with ≥ 1 decoy barcode |
|---|---|---|---|---|---|---|---|
| (i) all matched reads, quality ≥ 0.06 | 24.24 % | 61.51 % | 12.97 % | 1.24 % | 0.04 % | 47.7 % (47.6-47.9)* | 0.21 % |
| (ii) guides with ≥ 2 reads each, quality ≥ 0.06 | 59.89 % | 39.50 % | 0.60 % | 0.01 % | 0.00 % | 0.000 % (0.000-0.003)* | 0.01 % |

\* The Funk shuffle column comes from the first pass (`mult_funk.py`), which kept 1-mismatch reads without the runner-up check; that pass inflated decoy barcodes (4.5 % of cells in row i) but barely changed the guide counts (13.7 % vs 12.97 % with two guides). The checked pass is `step0/step0_funk.py`.

- **Reading the shuffle:** with single reads counted, a random field-wide shuffle gives *more* doubles than observed (reads are strongly clustered by cell). With ≥ 2 reads per guide, random placement explains almost none of the observed doubles (PERISCOPE 3.63 % + 0.46 % + 0.05 % observed vs 0.008 % shuffled; Funk 0.61 % vs 0.000 %). A field-wide shuffle is the wrong null for bleed between touching cells, so a local test was added:
- **Neighbour bleed (PERISCOPE):** in 53,999 two-guide cells with a clear top guide, the runner-up guide is the top guide of a **touching cell in 60.0 %** of cells; null (runner-ups swapped between test cells of the same field, 200 permutations) 0.32 % (0.27-0.37). So most PERISCOPE doubles are reads from the neighbour. Funk (step 0): 48.8 % of 2,551 such cells, null 0.07 % (0-0.20), neighbours = nuclei whose discs come within 10 px.
- **Funk, top vs runner-up guide (2,816 two-guide cells, one random read each):** runner-up farther in 50.7 %, dimmer in 51.4 %; median gap 1.56 vs 1.86 px. **No distance or quality difference in Funk** (reads lie within 10 px of a nucleus by design).
- **How the Funk sample was made:** spaCR's own decode (production commit 4792e9d, `run_unit.py` via a wrapper) on sites 150-157 of A1 into a scratch destination (`/mnt/wd4tb/scratch_specs_1118/funk_sample/`, copied `measurements.db` and placement), raw tiles read from the NAS (A1 is no longer staged), manual thresholds = production's tuned ones, `ops_store_field_cache` on; 406 s wall, 4.1 GB. Nothing in production was touched.

**What this means:** distance carries a real but small signal about *which barcode belongs to the cell* (runner-ups sit a little farther out, and 60 % of them are the touching neighbour's guide), and no signal about *read errors* (decoy rate and quality are flat). In Funk, distance shows no signal at all in the sample. The useful lever is therefore most likely the **neighbour-bleed discount (arm N1)**, with distance as a weaker proxy for it; the expected gain is in multi-guide cells, mostly in PERISCOPE; a cheap post-hoc proof check (step 0 below) should come before any tuned run.

**Proposed step 0 (minutes, read only, after sign-off):** on the k0 reads, (a) apply A1/A3/A4 post hoc at the chosen threshold, with the 3-seed shuffle and decoy counts as in `item967_alpha/post.py`; (b) the Funk neighbour test (is the runner-up the top guide of the nearest other nucleus?) on the 8-field sample caches, and a whole-well Funk count of cells blocked by a second guide (needs `ops_store_reads` or field caches for all 333 fields: one re-decode, about 40 min); If (a) shows no gain over C0 and 967 k3, stop and report. For Funk, the sample already shows no distance signal: the Funk distance arms (A3-A5) run only if the maintainer wants them after step 0; M1 (real masks) and N1 stay.

## 3b. Multi-guide cells as possible double knockouts

**Proposal (alpha, default off):** `ops_keep_multi_guide_cells` (bool, default False). When on, a cell with ≥ 2 high-quality guides (each with ≥ `ops_multi_guide_min_reads` reads, default 2, each read passing the quality floor, and neither guide flagged as neighbour bleed by N1) is kept as a labelled class `multi_guide` with both guides listed, instead of being dropped. It never enters single-guide gene profiles; it is a separate table.

**Telling real multiple integration from read bleed:**
- **Neighbour bleed:** the second guide is the top guide of a touching cell (N1 test). On PERISCOPE this explains 60 % of clear-majority two-guide cells (section 3), so most doubles there are bleed, not double infection.
- **Random placement (whole-field shuffle):** explains almost none of the doubles that have ≥ 2 reads per guide (0.008 % of cells vs 4.14 % observed on PERISCOPE).
- **Read counts and quality per guide:** a real second integrant should give a read count and quality like the first (both expressed in the same cell); bleed gives one or two reads. Report the ratio of reads and median quality, second vs first.
- **Distance:** a real second guide's reads should sit at the same distance from the nucleus as the first; bleed reads sit toward the shared border with the neighbour.
- **Expected rate:** at a low infection rate (multiplicity of infection about 0.3, as usual for pooled screens) Poisson gives about 14 % of infected cells with ≥ 2 integrants before selection; the observed rate after removing bleed is compared with this.

**Power for pair-level analysis (honest):**
- PERISCOPE CP186A Well1: 3.63 % of cells (about 19,400) have exactly two guides with ≥ 2 reads each; if 60 % are bleed, about 7,800 candidate true doubles remain. The library has 82,677 guides (about 20,000 genes), so there are about 3.4 billion guide pairs and about 200 million gene pairs. Almost every pair is seen at most once. **No pair-level analysis is possible.**
- Funk LaC024A: the library has 20,445 guides (about 5,000 genes, about 12.5 million gene pairs); in the 8-field sample 0.60 % of nuclei have two guides with ≥ 2 reads each, which is about 7,700 cells per well, about half of them neighbour bleed; the distance and quality of the two guides do not differ (section 3). Again almost every pair is seen at most once: **no pair-level analysis.**
- What is possible: (a) a pooled test of "doubles vs matched singles" (do doubles look like the sum of their two single-gene profiles?) on all doubles at once; (b) tests restricted to pre-defined gene sets (for example pairs within one protein complex), where pairs are pooled. Both are exploratory and need their own design card.
- **Recommendation:** ship `ops_keep_multi_guide_cells` only as a labelled QC table (how many doubles, how many explained by bleed) unless the maintainer wants the pooled test; do not promise double-knockout biology from these screens.

## 3c. Step 0 result (post-hoc proof check, run 2026-11-18 after sign-off)

Scripts and outputs: `/mnt/wd4tb/scratch_specs_1118/step0/` (`step0_periscope.py`, `step0_periscope_arms.csv`, `step0_periscope_meta.json`, `step0_funk.py`, `step0_funk.json`). `python -I`, `nice 10`, `run_capped 8G`, 4 threads, RAM-guard line while running. Proof check only, not a spaCR result.

**PERISCOPE CP186A Well1**
- Input: the 967 k0 read table at the tuner's chosen read threshold (40, raw mode, exact reads only), so the read set is fixed and only the cell rule changes.
- Tune/report split: the 967 tuner's own split (43 + 43 fields).
- Report half: 254,766 cells. CIs come from a field bootstrap (1,000 resamples). Δ is paired by field.
- Decoy FDR is per 100 real calls (k = 4 decoys per guide).
- Shuffle rate is the mean of 3 shuffled-library seeds, as in `item967_alpha/post.py`. The 500-permutation version comes in the acceptance run.
- **Validation:** the post-hoc rules reproduce the pipeline exactly: C0 gives 157,768 real and 825 decoy calls; 967 k3 gives 206,979 and 861.
- Every weighted rule uses the vote-share rule (top ≥ 0.6 of the weight, no tie). The N1 discount applies only to a strict-minority barcode that is the top barcode of a touching cell.

| Arm (report half) | Genotyped % (CI) | Decoy FDR per 100 (CI) | Shuffle call rate, % of cells | Δ vs 967 k3, points (CI) |
|---|---|---|---|---|
| C0 k0: recipe_simple, production (control) | 28.54 (24.25-32.43) | 0.149 (0.118-0.192) | 0.569 | -7.59 (-10.11 to -5.23) |
| 967 k2 | 34.67 | 0.125 | 0.571 | -1.45 (-1.99 to -0.93) |
| **967 k3 (reference for "matched")** | 36.11 (29.82-42.35) | 0.120 (0.093-0.160) | 0.571 | 0 |
| 967 k4 | 36.57 | 0.119 | 0.571 | +0.47 (0.28 to 0.67) |
| A1 count majority | 35.18 | 0.123 | 0.571 | -0.93 (-1.28 to -0.61) |
| A2 quality only | 36.58 | 0.122 | 0.572 | +0.46 (0.18 to 0.72) |
| N1 b = 0.25 (main arm, counts) | 36.35 | 0.120 | 0.571 | +0.24 (0.08 to 0.43) |
| N1 b = 0 (counts) | 36.51 | 0.119 | 0.571 | +0.40 (0.19 to 0.63) |
| N1 b = 0 + quality | 38.44 (31.60-45.21) | 0.116 (0.091-0.153) | 0.572 | +2.32 (1.81 to 2.85) |
| A3 distance only, L = 4 px | 39.04 | **0.134** (0.109-0.171) | 0.572 | +2.92 (2.25 to 3.56) |
| A4 quality × distance, L = 4 px | 39.58 | 0.126 (0.100-0.161) | 0.572 | +3.46 (2.70 to 4.22) |
| A4 quality × distance, L = 16 px | 38.20 | 0.119 | 0.572 | +2.08 (1.67 to 2.49) |
| N1 b = 0.25 + A4, L = 8 px (chosen on the tune half) | **40.00** (32.80-47.17) | 0.122 (0.096-0.157) | 0.572 | **+3.88 (2.96 to 4.77)** |
| N1 b = 0.25 + A4, L = 16 px | 39.35 (32.32-46.32) | 0.116 (0.091-0.152) | 0.572 | +3.24 (2.50 to 3.97) |

- **Selection rule:** most real calls on the tune half, with tune-half FDR ≤ 967 k3's (0.088). It chose N1 b = 0.25 + A4 at L = 8 px. On the report half that arm gives +3.88 points over 967 k3 and +11.5 over C0. Its FDR is 0.122 vs 0.120, a difference well inside the CIs; its upper CI (0.157) is below k3's (0.160).
- **At strictly equal or lower FDR,** N1 b = 0.25 + A4 at L = 16 px gives +3.24 points (FDR 0.116 vs 0.120).
- **N1 by itself gives little over 967 k3:** +0.24 to +0.40 points. 967 already drops most stray single reads. N1 adds clearly when reads are weighted by quality (+2.3 points) and when combined with distance.
- **Distance gains more than the feasibility check suggested,** but distance alone (A3) raises the decoy FDR (0.134). It needs the quality term.
- **The shuffle call rate does not move** (0.569 % to 0.572 % of cells), so the gain is not chance barcodes.

**Funk LaC024A A1, 8-field sample** (31,810 nuclei; primary read vote only, no pixel fallback; no tune/report split; 24-25 decoy calls, so no CI)
| Arm | Genotyped % | Decoy FDR per 100 |
|---|---|---|
| C0, quality vote (control, production rule re-implemented) | 70.00 | 0.027 |
| N1 b = 0.5 | 70.78 | 0.027 |
| N1 b = 0.25 | 70.95 | 0.028 |
| N1 b = 0 | 71.01 | 0.028 |
- About +1 point for Funk with a flat FDR.
- Neighbour bleed is common in Funk too: 48.8 % of 2,551 two-guide cells, against a null of 0.07 %.
- The Funk distance arms were not run, because the sample shows no distance signal.

**Step 0 verdict:**
- The signed plan is worth running. The gains are real at equal FDR, and the shuffle rate is unchanged. They are largest for N1 + quality + distance in PERISCOPE.
- Caveats:
  1. The read threshold is fixed at the k0 choice; the acceptance run tunes it jointly with L and b.
  2. The edge/position confounder checks (4.5) and the 500-permutation controls are not done yet.
  3. PERISCOPE has one well.
  4. The Funk numbers come from an 8-field sample with very few decoys.
- **Acceptance grid** (signed): b ∈ {1, 0.5, 0.25, 0} × L ∈ {off, 4, 8, 16, 32} px × quality term on/off × the PERISCOPE read threshold, which now also includes **20 and 30** (decision 2026-11-18), inside the decoy tuner, after the 967 default is decided. C0 = recipe_simple + 967 at the decided k. Note that the weighted rule replaces 967 and is not stacked on it.

## 4. Design card

### 4.1 Question and pre-set answer
- **Question:** at a matched decoy FDR, does weighting each read by its distance to the nucleus (with or without quality) genotype more cells than the current default rule, on fields the tuner never saw?
- **Read-out:** % genotyped on the report half at matched decoy FDR, with CI; and for PERISCOPE also the shuffled-barcode call rate.

### 4.2 Data
- **Funk:** LaC024A well A1, production freeze 1116 (`/mnt/wd4tb/production_freeze_1116/out/funk/20200202_6W-LaC024A/`; 1,279,317 nuclei, 333 fields; current report half 76.37 % genotyped at 0.060 decoy FDR per 100). Per-read gaps are not saved, so A1 is re-decoded once with the new code (evidence kept in memory as today). Tune/report split: the production seed, so the report fields are the same as the control.
- **PERISCOPE:** CP186A Well1, 86 sites (the 967 alpha set; 535,390 cells; report half 43 fields, 280,624 cells). Nucleus masks from the production seg (`/mnt/wd4tb/production_freeze_1116/out/periscope/CP186A_Well1/seg/labels/{site}_nuclei.npy`), cell masks as used by 967.
- Same data, same n, same library, same decoys and same tuner for every arm (fair-comparison rule).

### 4.3 Arms (each tuned by the same decoy tuner on the same tuning half)
| Arm | Funk | PERISCOPE |
|---|---|---|
| C0 control: current default | quality vote (today) | recipe_simple + 967 at its decided default |
| C1 967 grid reference | — | 967 k = 2, 3, 4 (existing outputs, for context only) |
| A1 count majority | — | weighted_majority, q = d = 1 |
| A2 quality only | (= C0) | weighted_majority, q only |
| A3 distance only, no quality | d only | weighted_majority, d only |
| A4 quality × distance (main arm) | q × d | weighted_majority, q × d |
| A5 main arm, L fixed at 4 / 8 px (not tuned) | yes | yes |
| N1 neighbour-bleed discount (new, from the feasibility check) | a minority guide that is the top guide of a touching nucleus gets weight × `b` | same, with touching cells from the cell mask |
| M1 real masks, no distance weight | reads owned by the nucleus mask expanded by `e` px (no equal-area disc) | — (PERISCOPE already uses the real cell mask) |
| M2 real masks + A4 weight | M1 ownership, g from the true nucleus edge | — |

- A5 shows whether the gain needs tuning; A3 shows whether distance carries signal by itself.
- **N1 (neighbour-bleed discount):** added because the check in section 3 found that in 60 % of PERISCOPE two-guide cells the runner-up guide is the top guide of a touching cell (null 0.32 %). It targets the likely mechanism directly instead of through distance. `b` ∈ {1 (off), 0.5, 0.25, 0} as one tuner axis. A4 + N1 is also run. Setting: `ops_read_neighbour_discount` (float, default 1.0 = off; alpha), shared by both paths.
- **M1/M2 (Funk real-mask arms, maintainer 2026-11-18):** the disc rule gives a read to the nearest equal-area disc edge. M1 instead gives it to the nucleus whose real mask, expanded by `e` px, contains it (e ∈ {3, 6, 10} px as one tuner axis; 10 px matches today's `ops_footprint`); reads in two expanded masks go to the nearer true edge, with the same 1 px tie rule. Where a cell mask exists in the sequencing frame it replaces the expanded nucleus. Compared with C0 (disc) at matched decoy FDR on the report half, same pass criterion (4.6).
  - **What is needed:** Funk production keeps only centroid, area and box per nucleus (`ops_objects` table), not masks. M1 needs the A1 objects phase rerun with `ops_objects_keep_windows` = true (production took 954 s wall, 2.4 GB GPU), into a scratch destination. There is no cell mask in the sequencing frame today (cells are segmented in the phenotype frame, `ops_ph_cell_*`); mapping them into the sequencing frame is a separate step and is proposed only if M1 passes.
  - New alpha setting: `ops_read_owner_geometry` (`"disc"` default = today, `"mask"`), and `ops_read_owner_expand_px` (float, default 10.0).
- Context from the existing 967 outputs (report half, 43 fields, 280,624 cells; tuner-chosen settings): k0 28.54 % genotyped, decoy FDR 0.149 per 100 (CI 0.117-0.188); k2 34.67 % / 0.125; k3 36.11 % / 0.120 (CI 0.094-0.157). Whole-well numbers quoted for 967 (29.47 % / 0.131 at k0, 38.66 % / 0.104 at k3) are over all 86 sites and are not the report-half numbers.

### 4.4 Measures (every row)
- % genotyped on the report half (real calls / all cells or nuclei in report fields) with field-bootstrap CI.
- Decoy FDR per 100 on the report half with CI (k = 4 decoys per guide for PERISCOPE; production k for Funk).
- **PERISCOPE, mandatory:** shuffled-barcode call rate (mean of 3 seeds as in `item967_alpha/post.py`, and the 500-permutation mean with CI for the final rows, permutation-control rule) next to the decoy FDR.
- Δ genotyped vs C0 (paired by field, bootstrap CI), cells rescued (called by the arm, not by C0), cells lost, and the guide-agreement rate of cells called by both.
- For rescued cells: share whose winning read set is inside or within 3 px of the nucleus, and the share where C0's blocking barcode was a decoy.
- Time per field vs C0.
- Reference/control row and spelled-out abbreviations in every table.

### 4.5 Edge and field-position check (confounder)
- Distance to the nucleus could stand in for "near the field edge" (cropped nuclei, worse focus, registration drift), or for crowding.
- Checks, on the report half:
  1. Stratify Δ genotyped by distance of the cell to the field border (quartiles) and by field-level reads per nucleus. The gain must be present (CI above 0) in the inner two border quartiles.
  2. A logistic model of "rescued by the arm" on g (or the neighbour flag for N1), border distance, local nucleus density and read quality; report the g coefficient with and without the other terms. It must not shrink to CI-including-0 once border distance and density are added.
  3. A within-cell permutation control: shuffle g among the reads of each cell (500 times); the arm must beat the permuted mean (CI excluding the permuted 95 % range).

### 4.6 Pre-set pass criterion (judges spaCR only)
**Working point (decision 2026-11-18):** every arm, and every comparator (PERISCOPE, Funk/Brieflow), is read at two working points.
- **(a) Its knee.** The knee is the point of the yield-vs-decoy-FDR curve farthest from the straight line joining the curve's ends. The curve is genotyped % against decoy FDR per 100, over the tuner grid's Pareto front. The geometric rule and its axis scaling are fixed once on the tune half and then applied unchanged to every method.
- **(b) A matched FDR** (the C0 FDR, as before).
- Tables also give each arm at decoy FDR 0.1 % and 1 %, interpolated along its front. A point is marked "not reached" when the front does not get there.

Arms are judged in the signed order: **N1 first (main)**, then the distance arms (A4, or A5 if A4 fails to beat it), then M1/M2 (Funk). Each judged arm passes on a dataset when all hold (for N1, "A4" below reads "N1", and the confounder model in 4.5 uses "the minority guide is a neighbour's top guide" in place of g):
1. On the report half, genotyped % is higher than C0, with a paired-field CI of Δ excluding 0. This must hold **both at the arms' knees and at matched FDR**. At the knee, the arm's knee FDR must also be no higher than C0's knee FDR (upper CI). If the two working points disagree, report both and ask.
2. Decoy FDR is equal or lower than C0 (upper CI of A4 ≤ upper CI of C0, and ≤ 0.1 per 100 target).
3. PERISCOPE: shuffled-barcode call rate not higher than C0 (difference CI includes 0 or is below).
4. The edge/position checks in 4.5 pass.
5. Cells called by both C0 and A4 agree on the guide in ≥ 99.5 %.
- Passing on both Funk A1 and CP186A Well1 → propose a default change (separate sign-off, then schema bump + `DEFAULT_CHANGES`). Passing on one → keep alpha, report the other as a null.
- If A3 (distance only) beats A4, report it and ask; quality should not be dropped without a second well.

### 4.7 Compute estimate
- **Build** (spaCR, alpha, both paths + tests): about one working day for an agent; CPU only.
- **Funk A1 re-decode:** production A1 decode took 2,204 s wall (37 min), 6.9 GB peak. One decode with the L axis in the grid covers C0, A3, A4 and A5 (counts per setting are computed together): about 45-60 min, ≤ 8 GB RAM, 1 CPU-heavy slot.
- **PERISCOPE Well1:** the 967 alpha decodes took 8,571-11,291 s (2.4-3.1 h) each at 14 GB. One run with the L axis and the cell-rule arms: about 3-4 h, 14-16 GB, 1 CPU-heavy slot. A cheap proof check first: apply the rules post hoc to the k0 read table at the chosen threshold (minutes), before the full tuned run.
- **Post-analysis** (bootstrap, 500 permutations, edge model): about 30 min CPU.
- Total ≈ 5 h of one CPU-heavy slot, no GPU. Fits the paper-session cap (2 CPU-heavy slots).

### 4.8 Risks and limits
- The disc approximation (Funk) puts reads in irregular or elongated nuclei at the wrong g; the true-mask version is a follow-up only if the disc version passes.
- In PERISCOPE, reads sit mostly in the cytoplasm (spacer-prefix amplicons are cytoplasmic), so "close to the nucleus" may mean little biologically; the feasibility numbers in section 3 decide whether the test is worth its compute.
- A soft weight cannot rescue cells where the competing barcode has many reads at the same distance; those stay blocked, as they should.
- Results are proof checks until a spaCR settings-file rerun gives identical numbers.

## 5. Open points for the maintainer
1. **Sign the design card** (sections 4.1-4.8), or change it. Nothing runs before sign-off.
2. **Order:** confirm that this waits for 967's default decision (k4 is still running), and that C0 for PERISCOPE is "recipe_simple + 967 at its decided k".
3. **Funk re-decode:** per-read gaps are not saved anywhere, so Funk A1 needs one re-decode (about 1 h CPU). OK, or PERISCOPE only first?
4. **PERISCOPE distance:** nucleus-mask distance (proposed, masks exist) or cell-centroid distance (no nucleus needed, works where nuclei are missing)?
5. **Tuning:** add the decay length L as one tuner axis {off, 2, 4, 8, 16} px (proposed), or keep L fixed?
6. **Rule family for PERISCOPE:** the weighted majority is an alternative to 967, not stacked on it. OK, or also test "967 then distance weight"?
7. **Proof check first:** apply the rules post hoc to the existing k0 read table (minutes) before the full tuned 3-4 h run. OK?
8. **Is it worth running?** Distance does not mark read errors (decoy rate and quality are flat), only weakly marks the competing barcode in PERISCOPE (runner-up 14 vs 11 px), and shows nothing in Funk. The neighbour-bleed signal is strong (60 % vs 0.32 %). Proposed: make N1 (neighbour discount) the main arm, keep distance as a secondary arm, run step 0 (minutes) and decide on the full run from it.
10. **Funk real-mask arm (M1/M2):** needs the A1 objects phase rerun with label windows kept (about 16 min, 2.4 GB GPU, via gpu_turn) into scratch. OK?
11. **Multi-guide cells:** ship `ops_keep_multi_guide_cells` as a labelled QC table only (proposed), or also design the pooled "doubles vs singles" test? Pair-level double-knockout analysis is not powered on either screen.
9. **Item number 988:** settled; master confirmed and reserved it on 2026-11-18.
