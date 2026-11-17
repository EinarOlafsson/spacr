# Item 948: the gene-pairs consensus mode (SNF of three views → PHATE → Leiden), built fast (F4)

- **Number:** 948 (assigned by master 2026-11-15). **Lifecycle:** alpha (Show alpha features). Each new setting gets 9-locale strings; 948 is on the clusters line of the alpha deck (`tools/refresh_alpha_deck.py`).
- **Maintainer decisions, 2026-11-15:**
  1. "Accept B1+B2, keep going".
  2. "Build 948 as F4".
  3. Option 1, widen 948: spaCR had no pairs option, so 948 builds the whole gene-pairs consensus as a new alpha consensus mode, default off, with the fast search (F4) built in.
- **Evidence:**
  - `BMCspacr/reanalysis/design_f_strategies/pairs_speed/REPORT.md`: the B1 + B2 card.
  - `REPORT_followup_20261115.md`: the fast-search card, `experiment_designs/20261115_pairs_fast_search_DRAFT.md`.
  - F4: 821 CPU s per replicate against 9,025 s for the published option (same pinned run). Accuracy gate met on plate A (R = 20): pairs +30.2 [25.2, 34.9], complexes +11.4 [8.5, 14.7].
- **Workflow:** the 8-step feature workflow. The script stage is done (Design F + the two speed cards). This item covers steps 5–8.

## What is built
The option is Design F's pairs winner `cons_snfpl~hyb`. In the guide-bootstrap consensus of item 928 (B = 10 bootstraps), bootstrap b clusters a graph built from three views:
- the Hybrid profile draw;
- an interphase view;
- a mitosis view.

The interphase and mitosis views are the cell-weighted means of the same drawn guides' per-phase median cells (TVN PCs), robust z against the control rows.

The three views are fused by similarity network fusion (SNF; K 20, mu 0.5, 20 iterations). PHATE is built on the fused network (precomputed distance max(S) − S, knn 10), followed by Leiden at the matched count. The co-association of the 10 partitions is cut by average linkage to k, and the count is matched as in item 928.

1. **Library** (`spacr.ops_clusters`, documented callables with arrival contracts):
   - `snf_fuse(views, single=False)`;
   - `phase_guide_medians(cells, columns)`;
   - `phase_view(names, mask, guides, medians, seed, width)`;
   - `fused_phate_graph(fused, seed, knn)`;
   - `leiden_short(graph, resolution, seed, passes)`;
   - `pairs_bootstrap_consensus(draw, views, n_boot, k, mask, seed, knn, rows, search, workers, match)`;
   - `match_resolution(..., passes=-1)`: the new keyword; the default path is unchanged.
2. **Search modes** (`ops_consensus_pairs_search`):
   - `'exact'` is the published analysis: double-precision SNF and the full resolution bisection on every bootstrap.
   - `'fast'` (default) is F4:
     - B1: one resolution search per run, on bootstrap 0;
     - B2: no discarded final fusion, so the outer rows are only drawn;
     - Leiden with 2 passes during the bisection steps, then one full run at the chosen resolution;
     - 2-pass Leiden for bootstraps 1–9 at that resolution;
     - single-precision SNF (the fused matrix is cast to double before PHATE).
3. **B4:** `ops_consensus_workers` (default 1). Bootstraps after the first run in fork worker processes. The co-association is summed in bootstrap order, so the outputs are byte-identical to workers = 1. On platforms without fork the bootstraps run serially.
4. **Settings (alpha, `ALPHA_FEATURES[948]`):**
   - `ops_consensus_mode` (`'profile'` = item 928, the default | `'pairs'`);
   - `ops_consensus_pairs_search` (`'fast'` | `'exact'`);
   - `ops_consensus_workers` (1);
   - `ops_consensus_cells`: per-cell table(s) with gene, sgRNA, mitotic and PC_0 …; a control cell belongs to row `<prefix>_<sgRNA>`;
   - `ops_consensus_view_columns` (50).

   The phase `cluster_consensus` writes the same files as item 928. Every gene is assigned in the pairs mode.
5. **Schema:** no bump. The default (`'profile'`) leaves item 928's outputs byte-identical (the item 952 / read-mode convention), so no DEFAULT_CHANGES row is needed.

## Proof check (pre-set)
- **Exact mode = the published analysis:** spaCR, from its own guide bootstraps and its own phase views built from the per-cell table, reproduces Design F's stored `cons_snfpl~hyb` targeting labels md5-identically on plate A r = 0, 1, 2.
- **Fast mode = F4:**
  - plate A R = 20 md5-identical to the card's F4 labels (`pairs_speed/res/eval/ps_f4/`) under the same library versions (phate 2.0.0, leidenalg 0.12.0, igraph 1.0.0); otherwise the accuracy gate;
  - gate: candidate − baseline point estimates inside pairs [24.95, 34.78] and complexes [7.97, 14.63].
- **B4:** workers = 3 labels and co-association byte-identical to workers = 1.
- **Plate B, through spaCR,** R = 20 at plate B's gate: pairs [23.39, 33.13], complexes [8.98, 15.81].
- **Permuted control** (new rule): the mean over 500 label permutations, with its 95 % CI including 0.
- **Item 928 unchanged:** with `ops_consensus_mode` at its default, the 928 tests and the stage 5 proof stay identical.
- **Time:** ≤ 900 CPU s per replicate on the card's timing set (plate A r = 0–2, one pinned core), same machine.
- **Unit tests** (`tests/test_ops_pairs_consensus_948.py`, fixture from the reference scripts):
  - SNF double = Design F's `snf_fuse`, single = the card's `snf_fuse32`, both exactly;
  - phase views = Design F's `phase_view` exactly;
  - fast and exact search structure;
  - workers = 1 vs 3 byte-identical;
  - the phase writes item 928's outputs;
  - settings validation.

## Proof results (2026-11-15; `BMCspacr/reanalysis/spacr_948_pairs_proof/`)
- **Exact mode:** 3 of 3 replicates md5-identical to Design F's `cons_snfpl~hyb`.
- **Fast mode:** 20 of 20 md5-identical to the F4 fixture.
- **Plate A gate:**
  - pairs +30.17 [25.19, 34.91], complexes +11.43 [8.46, 14.65], both inside;
  - vs the published option: pairs +0.20 [−0.24, 0.71], complexes +0.22 [−1.07, 1.59].
- **Plate B through spaCR:**
  - pairs +28.51 [23.43, 33.09], complexes +13.36 [10.31, 16.93], both inside.
- **Permuted controls (500 permutations):** the mean CIs include 0 on both plates.
- **B4 (workers = 3):** md5-identical labels; wall 391 s against 570 s.
- **Time:** 567–625 CPU s per replicate on one pinned core (limit 900).
