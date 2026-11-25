# Item 1024: nucleus fragments and the nucleus–cell pairing rule (OPS phenotype segmentation)

- **Status:** SPEC ONLY, for the maintainer's decision. No build, not in L5. Filed 2026-11-25 (night; recommended option logged).
- **Evidence:** `audit_ops_20261124/identity/BRIEFLOW_ONLY_NUCLEI.md` and montages; scripts and data in `/mnt/firecuda2/codex/scratch-docs/bf_only_1124/`.
- **Data:** Funk plate A (LaC024A) well A1, spaCR 26bc722c15 (frozen run).
  - spaCR's own segmentation was rerun on 48 fields; re-pairing reproduces the stored masks pixel for pixel.
  - Development result, one well.

## Problem (numbers)
- spaCR's fixed nucleus–cell pairing (one nucleus per cell; extra nuclei dropped; nuclei without a cell dropped; nuclei clipped to their cell) removes **35,322 nuclei per well**: 1,622,756 → 1,587,434. Of these, **28,025** are an extra nucleus in a cell and **7,297** have no cell.
- Cellpose-SAM (Cellpose 4.2.1.1) often cuts one nucleus into two fragments. In class B1 (9,165 nuclei in the well), spaCR's raw mask has two fragments inside the Brieflow object in 261 of 309 inspected cases. **36 %** of fragment pairs show no DAPI dip between them, which fits one nucleus cut in two.
  - Confounder: fragment pairs sit closer together (29 vs 41 px) than separate neighbours, which on its own makes the dip shallower.
- Consequence vs Brieflow (published): 22,393 Brieflow-only nuclei after correct matching. Of these:
  - **47 %**: the same nucleus, outlined differently;
  - **37 %**: found by spaCR, then dropped as a second nucleus in the same cell;
  - **9 %**: found, then dropped for having no cell;
  - **6 %, about 1,270 per well**: never segmented. About a third of those look like Brieflow debris; the split rests on 44 checked nuclei.
- Per-cell effect: when spaCR keeps one fragment of a cut nucleus, the nucleus features (area, shape, intensity sums) describe half a nucleus, and the cell's nucleus/cytoplasm split is wrong.

## Options
- **(a) Merge fragments within a cell before pairing.** When ≥ 2 nucleus objects fall in one cell, merge them into one nucleus if they pass the fragment test (below). Otherwise treat the cell as binucleate: keep both and flag the cell.
- **(b) Cell-less nuclei.** Assign each to the nearest cell whose mask it overlaps or touches within d px. Otherwise keep it as a nucleus-only object (no cytoplasm features), flagged.
- **(c) Status quo** (drop), as a control arm.

## Fragment vs true binucleate: the check required before any default
- **Hand truth:** a stratified sample of multi-nucleus cells (about 300; strata by fragment distance, area ratio and DAPI dip), labelled "one nucleus cut" vs "two nuclei" by the maintainer or an annotator, blinded to the rule.
- **Candidate rule:** DAPI dip along the line joining the centroids (relative to the fragments' median intensity) + combined area against the cell's nucleus-size distribution + convexity of the union + the contact length between fragments. Thresholds are set on half the sample; accuracy is reported on the other half (balanced accuracy, 95 % CI).
- **Controls:**
  - pairs both pipelines call separate (known two-nucleus neighbours);
  - mitotic/telophase cells, flagged by the cell-cycle model (must not be merged);
  - a shuffled-pair baseline.
- **Downstream check on held-out data:**
  - per-gene profiles and the held-out clustering recall (round-3 scorer), (a)+(b) vs (c);
  - unique nuclei and genotyped cells vs Brieflow, by the audit's physical-cell rule;
  - judged by effect + CI.

## Where it goes in spaCR (same task = same code)
- The fragment merge and the cell-less handling go INTO the existing shared routine `spacr.measure._assign_nuclei_to_cells`: the merge-spanning-children rule plus `_assign_shared_nuclei`, from merge23. They never go into a second, OPS-only routine. Measure and OPS both get them through that one routine.
- OPS keying: OPS tables are keyed by nucleus.
  - A merged nucleus keeps the LOWEST fragment label as its key and gets `nucleus_fragment_count` (≥ 1), plus a `nucleus_merged` flag.
  - The other fragment labels are recorded in a small mapping table (well, site, kept_label, merged_label), so masks and crops stay traceable.
  - The ownership and the pheno↔SBS link use the merged nucleus's centroid.
- A nucleus-only object (option b, no cell) gets a key in the same nucleus key space; its cell/cytoplasm feature columns are NaN, and it has `has_cell` = False.

## Settings and defaults plan
- New settings (names indicative):
  - `ops_nucleus_fragment_merge` = off | rule (default off);
  - `ops_cellless_nuclei` = drop | nearest_cell | nucleus_only (default drop);
  - thresholds for the fragment test.
- They ship as alpha ("Show alpha features"). A DEFAULT_CHANGES row and a schema bump only if the maintainer later turns either on by default after the check.
- The same implementation is shared by OPS and Measure (one operation, one implementation).

## Cost
- Code: about 1–2 days (fragment test + merge in the pairing step + cell-less handling + unit tests with synthetic two-fragment and binucleate cells).
- Hand truth: about 1–2 h of labelling (300 cells).
- Evaluation on A1 from the existing raw masks, no re-segmentation: about 0.5 day CPU; the held-out clustering rerun is about 3–4 h.

## Open for the maintainer
1. Build (a) + (b) as alpha options and run the check (recommended), or leave the status quo.
2. Who labels the 300 cells: the maintainer, or agent-proposed labels with his review.
