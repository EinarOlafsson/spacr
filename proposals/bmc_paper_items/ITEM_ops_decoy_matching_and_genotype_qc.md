# Item 989: Distance-matched decoys as the default, plus a genotyping QC figure set produced by default

**Item 989** (assigned by master 2026-11-18). Schema: DEFAULT_CHANGES row **947** (946 is taken by 978); if 947 is taken by build time, the builder takes the next free number.


- **Number:** to be assigned by master. **Schema:** the same number. It must be above every existing `DEFAULT_CHANGES` key; this checkout has `SETTINGS_SCHEMA_VERSION = 944`, so check nightly first.
- **Maintainer decision (2026-11-18), verbatim gist:**
  - distance-matched decoys "should obviously be the default and how decoys are generated";
  - "the module should generate a graph for this by default as a QC sanity check".
- **Lifecycle:**
  - Default, not alpha: the maintainer asked for a default change. The QC figures are written by default as well.
  - Frozen production (`/mnt/wd4tb/production_freeze_1116`) is **re-decoded only when the maintainer says so**. Old settings files keep the old decoys through `DEFAULT_CHANGES`.
- **Rules:**
  - The settings-file rerun gives byte-identical output, including the QC figures.
  - Each documented callable ships with its arrival contract and 9-locale records in the same commit.
  - Run the raw-reader, nested-function, documented-callable and hash-comment ratchets before handing over.
  - Code basis: origin/nightly.
- **Paper-critical:** yes. The decoy FDR is the false-call measure in every genotyping comparison.
- **Development evidence (script-made, not paper data):**
  - `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/decoy_precedent/figure/` (decoy panel);
  - `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/decoy_precedent/figure_guide_calling_rules/` (guide-calling rules panel);
  - literature: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/decoy_precedent/LITERATURE.md`.

## Gap
**Current decoy rule.** `spacr.ops_sbs.decoy_library` (`spacr/ops_sbs.py:780-822`):
- For each guide, it shuffles the guide's own bases k = 4 times.
- It keeps a shuffle only if it is ≥ 2 Hamming distance from every guide and every earlier decoy (≤ 200 tries per decoy, seed 1).
- This matches each decoy's overall base composition to its guide's. It does not match two other things:
  1. **Nearest-guide distance.** Decoys sit farther from the library than guides sit from each other:

     | Library | Guides at distance 2 / 3 | Decoys at distance 2 / 3 / 4 |
     |---|---|---|
     | Funk, 11-base barcodes, 20,445 guides | 93.4 / 6.6 % | 88.2 / 11.8 % |
     | PERISCOPE, 12-base prefixes, 80,862 guides | 95.8 / 4.2 % | 90.6 / 9.4 % / 1 decoy |

  2. **Per-position base composition.** A shuffle moves bases between positions. Position-specific composition is not checked today.

**Why it matters.** The decoy hit rate depends on distance, and in opposite directions in the two libraries:
- Funk: 22.7 decoy calls per 1,000 decoys at distance 2 vs 14.0 at distance 3.
- PERISCOPE: 2.0 vs 7.9.

Re-weighting the decoys to the guides' distance mix changes the FDR:
- Funk: +2.1 % (95 % CI +1.1 to +3.0). Today's FDR is slightly understated.
- PERISCOPE: −11.8 % (95 % CI −14.4 to −9.1). Today's FDR is conservative.

A simulation that sub-samples the existing decoys to the guide mix gives the same point estimates (+2.1 % and −11.3 %), with wider CIs.

**Literature.** Validity condition (d) in LITERATURE.md: decoys must not sit unusually close to or far from targets. Every in situ precedent takes its blanks from the same code as the real barcodes, so the two are exchangeable in distance (MERFISH, Xia 2019; Xenium; starFISH). spaCR's decoys are made after the library was designed, so this has to be enforced and shown.

**No QC figures today.** The genotyping step writes numbers (`decode_fdr_<well>.json`, `field_stats_<well>.csv`, `threshold_tuning_*`) but no QC figures. Every figure in the two development panels was made by a script.

## Spec

### 1. Decoy generation (new default)
1. **Function.** Add `decoy_library(library, k=4, seed=1, match='distance_position')` in `spacr/ops_sbs.py`. `match='shuffle'` keeps today's algorithm byte for byte.
2. **What `'distance_position'` matches.** The decoys must reproduce two properties of the guides:
   - **(a) Nearest-guide distance.** The distribution of the nearest-guide Hamming distance over decoys equals the guide-to-nearest-other-guide distribution, by quota per distance d.
   - **(b) Per-position base composition.** The base composition at each position equals the guides' (A/C/G/T fraction per position).
3. **Suggested algorithm** (the implementer may improve it if the test in 1.5 passes):
   - Draw candidates by mutating a random guide at d positions, choosing the substituted bases from that position's guide composition.
   - Accept a candidate when it is ≥ 2 from all guides and from all accepted decoys. Its nearest-guide distance must fill an open quota (quota(d) = round(k × N × p_guide(d))).
   - Correct composition drift with a final swap pass between accepted decoys that keeps the distances valid.
   - Keep the ≥ 2-edit minimum, so no read within one edit of a guide can be a decoy.
   - Fixed seed, deterministic order (sorted guides), no dependence on input order.
4. **Speed.** Use the masked-position hash already in `decoy_library`. Target: ≤ 60 s and ≤ 4 GB for 80,862 PERISCOPE prefixes × k = 4, on one core.
5. **Test** (`tests/test_ops_sbs_decoys.py`), on both real libraries (or seeded 1/10 subsets in CI):
   - The decoy nearest-guide distance distribution matches the guides': each distance bin within ±0.5 points; chi-square p > 0.05 against the guide counts.
   - Per-position composition: max |decoy − guide| fraction ≤ 0.01 at every position.
   - No decoy is within 1 edit of any guide or decoy, and none duplicates a guide.
   - `match='shuffle'` returns today's exact list (golden hash) for Funk and PERISCOPE.
   - The same seed gives the same list, independent of input order.
6. **Report.** Write the library's distance and composition summary (guides vs decoys, per d and per position) to `library_summary.json`, next to the existing prefix collision summary.

### 2. Settings
1. **New key:** `ops_decoy_match` ∈ {`'shuffle'`, `'distance_position'`}.
   - Default `'distance_position'`.
   - Used by both the full-barcode decode (`ops_decoy_library_k`) and the spacer-prefix decode (`ops_prefix_decoy_k`, `ops_prefix_decoy_seed`).
   - Add it to `ops_settings.py`, the type table, `settings_spec.py` and the 9-locale tooltips.
2. **New key:** `ops_genotype_qc_figures` (bool, default True; see section 3).
3. **Schema bump:**
   - `SETTINGS_SCHEMA_VERSION` = the item number.
   - `DEFAULT_CHANGES[<item>] = {'ops_decoy_match': 'shuffle', 'ops_genotype_qc_figures': False}`.
   - Settings files and run journals older than the item keep the old decoys and write no figures, so their output is byte-identical to today's.
4. **Reproducibility manifest and release note:**
   - A row in `docs/source/reproducibility_manifests.rst` under the item number.
   - A line in the next `docs/notes/release_notes_<v>.md`, in simple terms: "Decoy barcodes now match the guides' spacing and base make-up, so the decoy FDR is unbiased; the genotyping step now writes QC figures. Old settings files keep the old decoys."
5. **The FDR scale stays as it is.** Funk `ops_fdr_scale` = 1.268, PERISCOPE 1.0. It is re-checked in the proof (step 3.4). Changing it is a separate decision for the maintainer.
6. **Frozen production is not re-decoded automatically.** A re-decode of `production_freeze_1116` (Funk plates, PERISCOPE CP186A/B) happens only on the maintainer's explicit go.

### 3. Default QC figures written by the genotyping step
1. **Location and themes.**
   - Written to `<run>/results/ops_decode/qc/` (full-barcode) and `<run>/results/ops_prefix/qc/` (spacer-prefix), one set per well plus one per plate.
   - Each figure in both themes (`*_light.png`, `*_dark.png`, 300 dpi) plus `.svg`.
   - A source-data CSV per figure.
   - A short `qc_legend.md` that states the method, the formulas, the CI method and the reference rows.
2. **Graph rules** (the spacr-graphs skill):
   - bars plus jittered per-field points, both at about 60 % opacity;
   - spaCR cool palette (navy, teal, blue, grey), never brown;
   - Open Sans, plain theme, no titles on the plot;
   - a reference row in every chart;
   - field-bootstrap 95 % CIs (1,000 resamples, seed 0, the scheme of `plate_fdr_summary`);
   - medians as short black lines;
   - off-axis points as open triangles, counted in the legend;
   - abbreviations spelled out.
3. **Figure set:**

| # | Figure | Content | Reference row |
|---|---|---|---|
| Q1 | Decoy vs guide distance | Nearest-neighbour Hamming distance, guides→guides vs decoys→guides; per-position composition difference | Guides→guides |
| Q2 | Decoy hit rate by distance | Decoy calls per 1,000 decoys at d = 2, 3, ≥ 4; per-field points | All decoys pooled |
| Q3 | FDR with CI | FDR as it stands (and re-weighted to the guide mix while `ops_decoy_match='shuffle'`) | FDR target (`ops_prefix_fdr_target_pct` or tuner limit) |
| Q4 | Per-cycle mismatch rate | % of guide-matched reads with the mismatch at each cycle; per-field points | Random-read floor = (1-mismatch decoy reads / k) / L |
| Q5 | Substitution matrix | True → called base for one-mismatch guide reads, with the random-read share stated | Uniform 1/3 |
| Q6 | Read gap histogram | Read distance to the nucleus edge (guide- and decoy-matched), with `ops_footprint` marked; shares inside / 0–3 / 3–6 / 6–footprint | Decoy-matched reads |
| Q7 | Genotyped % vs tile-edge distance | Bins 0–10, 10–25, 25–40, 40–60, 60–100, > 100 px from the nucleus centroid to its own tile edge; registration margin stated | Interior > 100 px |
| Q8 | Guide multiplicity | Cells with 0 / 1 / 2 / 3 / ≥ 4 distinct guides (min reads 1 and 2), plus the neighbour-bleed share: of two-guide cells, the % whose runner-up guide is a neighbour's top guide (neighbours = disc edges within `ops_footprint`) | Decoy-barcode multiplicity; neighbour-bleed null (shuffled neighbours, mean and 95 % range of 500 permutations) |
| Q9 | Shuffle call rate | Barcode-shuffle call rate per 100 cells (`shuffle_call_rate`, existing seeds) next to the decoy FDR | Decoy FDR |

4. **Off-switch.** `ops_genotype_qc_figures=False` writes nothing. Its runtime cost must be ≤ 5 % of the genotyping step (target: ≤ 20 s per well).
5. **Byte reproducibility.** Rerunning from the settings file gives byte-identical PNG, SVG and CSV files:
   - fixed seeds;
   - `svg.hashsalt` fixed and `metadata={'Date': None}`;
   - PNG `metadata={'Software': None}`;
   - fonts embedded as paths, or `svg.fonttype='none'` with the font stated.

   A test hashes all outputs across two runs.
6. **Run report.** The run summary links the QC folder and prints one line per well: decoy-vs-guide distance match (max bin difference), FDR [CI], shuffle rate, worst cycle, and genotyped % at the edge vs the interior.

### 4. Reference implementation (scripts to port; read-only)
- **Distances:**
  - `/mnt/wd4tb/scratch_rules_1118/hamming.py`: guide nearest-neighbour distances;
  - `/mnt/wd4tb/scratch_rules_1118/decoys.py`: decoy distances and hit rate by distance.
- **Read classes, substitutions, gap bins:** `funk_reads.py`, `peri_reads.py`, same folder.
- **Edge bins:** `funk_edge_nuclei.py`, same folder.
- **Per-field cycles, gap histogram, edge bins per field:** `.../decoy_precedent/figure_guide_calling_rules/funk_field_cycles.py` and `peri_field_cycles.py`.
- **Figure logic (Q1–Q7):** `.../decoy_precedent/figure/make_decoy_figure.py` and `.../figure_guide_calling_rules/make_guide_calling_rules_figure.py`.
- **Multiplicity and neighbour bleed (Q8):** `/mnt/wd4tb/scratch_specs_1118/step0/step0_funk.py` and `step0_periscope.py`.

## Proof check (pre-set; judges spaCR only)
**Data.**
- Funk plate 20200202_6W-LaC024A well A1: 333 fields, frozen production settings.
- PERISCOPE CP186A Well1: 86 fields, the item-967 k0 settings.
- Everything is run from settings files on a worktree, not on frozen production.

1. **Decoys match.** The step 1.5 test passes on both full libraries. Q1 shows the decoy and guide distance mixes within ±0.5 points per bin.
2. **FDR with old vs new decoys.** For each dataset, at the production thresholds, report:
   - real calls, decoy calls and FDR [field-bootstrap CI] with `shuffle` decoys and with `distance_position` decoys;
   - the paired difference [CI];
   - the re-weighted estimate from the old decoys, next to them.

   Pre-set expectation, from the development panel:
   - Funk: new FDR within the old re-weighted CI (+1.1 to +3.0 % above the old FDR).
   - PERISCOPE: −9 to −15 % below the old FDR.

   If a result falls outside this range, report it and diagnose. The new default still lands only on the maintainer's OK.
3. **Yield with and without decoys in the matcher** (literature condition 6: target-decoy competition). Re-decode with k = 0 (no decoys) and with k = 4 (old and new decoys). Report:
   - genotyped % [CI] for each;
   - the paired differences;
   - the number of cells that changed call.

   Expected: a loss ≤ 0.3 points from adding decoys. Any loss is stated in the methods.
4. **Scale re-check.** With the new decoys, compare the decoy FDR with the barcode-shuffle call rate (Q9) on the same fields. Report the scale that `calibrate_fdr_scale` would choose. Do not change `ops_fdr_scale` without the maintainer.
5. **Literature validity conditions** (LITERATURE.md, "Conditions to state in the Methods"). Each one is shown in the proof report and in the QC set:
   1. **Decoys are never real:** no decoy sequence is in the library, and none is within 1 edit (test).
   2. **Exchangeability:** decoy→guide and guide→guide distance distributions and per-position composition match (Q1 and test).
   3. **The decoy-to-target ratio is corrected:** the FDR formula keeps the 1/k factor, and k is reported. If the distance quotas leave some guides with fewer than k decoys, k_eff = decoys / guides is used.
   4. **What decoys cannot see:** recombination, mispairing, segmentation and cell-assignment errors are stated as not measured. Q8 neighbour bleed and Q9 shuffle are shown alongside.
   5. **Variance:** bootstrap CIs on every FDR. Averaging over decoy seeds (Keich 2019) is reported, with 5 seeds giving a mean and range.
   6. **Competition:** yield with and without decoys (step 3).
6. **Off-switch.** With the pre-item `DEFAULT_CHANGES` values, output on 4 CP186A sites (3, 45, 64, 86) and 2 Funk A1 tiles is byte-identical to nightly before the change, and no `qc/` folder is written.
7. **Figures.**
   - The QC set renders for both datasets in both themes.
   - It is byte-identical across two runs.
   - It is inspected by eye for overlap, clipping and distortion.
   - The development panels' numbers are reproduced within rounding: Funk cycle 7 mismatch 0.59 %; Funk gap shares 21.9 / 53.7 / 21.2 / 3.2 %; Funk edge bins 0.2 / 26.5 / 64.1 / 69.6 / 70.9 / 76.8 %; Funk multiplicity 0 / 1 / 2 / 3 / ≥ 4 = 24.2 / 61.5 / 13.0 / 1.2 / 0.04 % (step-0 fields 150–157, min reads 1).
8. **Speed.** Decoy generation time and QC time per well, on the same machine, against the current nightly.

## Design card
- **Question:** do distance- and position-matched decoys change the decoy FDR as the re-weighting predicts, without costing yield?
- **Data:** the two wells above, chosen because the development panel and the production freeze already exist for them. All fields are used, because each run takes minutes.
- **Controls:** old decoys; re-weighted old decoys; barcode-shuffle call rate; k = 0 for yield.
- **Statistics:** field bootstrap (1,000 resamples, seed 0) for every FDR and yield; paired differences.
- **Pre-set criterion:** proof steps 1, 2 and 6 as written above.
- **Compute:** CPU only, ≤ 4 threads, ≤ 8 GB. About 10 min per Funk well decode and about 60 s per PERISCOPE site. No GPU.

## Open points for the maintainer
1. Should the new default also trigger a re-decode of the frozen production plates? The spec says only on your explicit go.
2. Should the FDR scale for Funk (1.268) be re-derived with the new decoys (proof step 4)? Or kept until the paper freeze?

## Effort
M–L: decoy algorithm and tests (M), QC figure module with byte-reproducible output (M), proof runs (S).
