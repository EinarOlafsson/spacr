# Item 951: run the OPS genotyping path on a chosen subset of tiles

- **Number:** 951 (assigned by master 2026-11-15). **Lifecycle:** alpha (Show alpha features). New settings get 9-locale strings; the deck entry goes into an existing grouped line in `tools/refresh_alpha_deck.py`. Low priority, any lane.
- **Code basis:** origin/nightly cfb5c1584b (`spacr/ops_engine.py` `run_ops`, `_stitch`; `spacr/ops_layout.py` `round_well_layout`).
- **Found by:** PERISCOPE-on-Funk card (`experiment_designs/20261114_periscope_on_funk_A1_DRAFT.md`), 2026-11-15. The card's spaCR arms (S-def, S-def-raw, rule_a_s2, 942 auto) are defined on 50 random A1 tiles plus 3 pilot tiles.

## Gap
- `run_ops` always starts from the stitch. On a folder holding only some tiles of a well, `_stitch` fits the well layout from the highest site number and fails:
  - error: `ValueError: no round well holds exactly 168 fields at a half tile of 0.5857; the acquisition is not a circle, so its layout has to be measured rather than fitted`;
  - setup that triggered it: pilot tiles 46, 148 and 167 of A1.
- The `objects` and `decode` phases iterate over the stitch's placements. No setting restricts them to a list of sites.
- A tile-level benchmark therefore has to run the whole well: stitch, objects and decode on all 333 A1 tiles, for every arm and every pixel source. The 53 study tiles are then read from that run.

## Spec
- New setting `ops_sites` (list of site numbers, default None = every site). When set:
  - **stitch:** fits or measures the layout from the whole acquisition's file index, as today (the layout needs every site's name, not its pixels). It places only the listed sites, from the layout grid, without pixel registration to neighbours that are not listed; each placement's source is recorded.
  - **objects and decode:** run only on the listed sites.
  - **reports:** `ops_report.json` and the FDR report name the subset.
- Alternatively, `ops_decode_only_tiles` (bool): skip stitch and run objects and decode in each tile's own frame, with placements set to the tile origin. Decode needs no well frame, but the territory and footprint rules use the tile frame.
- Off by default; outputs are byte-identical without the setting.

- **Library column:** when the library has no `prefix`, `barcode` or `sequence` column, `run_ops` refuses with a message that names the columns it found and the accepted names, or offers a setting for the column name.

## Proof check (pre-set)
- On the 3 pilot tiles, the subset run's per-tile reads and calls equal those of the whole-well run on the same tiles. Exact equality is expected where no rule uses neighbouring tiles; otherwise state which rule differs and by how much.
- A run without `ops_sites` is byte-identical to today's.
