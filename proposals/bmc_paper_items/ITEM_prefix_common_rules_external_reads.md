# ITEM DRAFT (no number yet): common assignment rules and external per-spot reads in the prefix decode

- **Number:** 943 (reserved 2026-11-14).

- **Code basis:** origin/nightly ad161c4242 (`spacr/ops_prefix.py`).
- **Found by:** PERISCOPE P1 benchmark (card §5 S2: "each method is scored with its own rule and with one
  common rule applied to each method's reads"), 2026-11-13.

## Gap
1. `prefix_cell_calls` has two rule sets (`spacr`, `recipe_simple`). The card's common rules from the SCALLOPS
   preprint are missing: **max-count** (unique top barcode by read count), **50 % purity** (top ≥ 50 % of
   matched reads and unique), **100 % purity** (one barcode only).
2. spaCR cannot read another pipeline's per-spot reads (PERISCOPE `Foci.csv` `Barcode_BarcodeCalled` joined
   to `BarcodeFoci.csv` `Parent_Cells`) and call them with its own matcher and rules. The benchmark did this in
   its scorer.

## Spec
- `ops_prefix_quality_method` gains `maxcount`, `purity50`, `purity100` (classes `called` / `ambiguous` /
  `none`).
- New phase `prefix_external` (cpg layout): settings `ops_external_reads` (path template with `{site}`),
  `ops_external_read_col` (`Barcode_BarcodeCalled`), `ops_external_cell_col` (`Parent_Cells`),
  `ops_external_join` (list of id columns and a second table, for the Foci/BarcodeFoci join). Reads go through
  `match_prefix_reads` and `prefix_cell_calls` exactly as spaCR's own reads, with the decoy library of the
  decoy item. Output `external_cells_<plate>_<well>.csv` beside spaCR's.

## Proof check (pre-set)
- On the 10 iterate sites, spaCR's external-read calls equal the P1 scorer's common-rule calls cell for cell.
- `recipe_simple` on PERISCOPE reads still reproduces the published classes (0 mismatches, as item 920).
