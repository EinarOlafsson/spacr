# Item 1004: OPS Live Align & stitch shows the phenotype field before and after tile-affine placement (alpha)

- **Number:** 1004, assigned by the master session on 2026-11-21; the deferred follow-up of item 1000 (maintainer 2026-11-20: "an Align & stitch 'before placement' picture of the phenotype field at the raster seed's own prediction, beside the placed field").
- **Base:** landing train 6 head 20e35b6846 (item 1000 in it). **Lands** by ws-offload-ws-1004 in train 8.
- **Alpha:** OPS Live is alpha (item 971 `OpsLiveSwitch`, hidden with "Show alpha features" off); this adds pictures to its Align & stitch tab only. No setting, default or run output changes.

## What the tab shows

Item 1000's Align & stitch tab draws the sampled phenotype field on the sequencing crop at the placement the phenotype phase stored: the placement seed's own prediction (`ops_ph_placement_seed`, `transform_source` 'seed') or, without a seed, the raster fit. Tile-affine placement (`ops_align_model` 'tile_affine', item 930) happens later, in merge, which the Align tab does not run.

With a finished tile-affine merge of that field, the tab now draws two pictures of the same field over the same sequencing crop, the same two-tone overlay (teal sequencing DAPI, blue phenotype DNA, white where they coincide):

1. **before** tile-affine placement, at the seed (or the raster fit), as item 1000 drew it;
2. **after** tile-affine placement, with the affine used ('tile group' when merge fitted one for the crop's sequencing tile, else 'field') and how far it moves the field at the crop centre, also a table row.

Both use the tab's image-quality slider and, in laptop mode, its Refresh view, like every picture of item 1000.

- **Where the merge comes from:** for the test data, its stored `reference/merge/tile_affine/ops_phenotype.parquet`; for a user's data, `ops_phenotype` of the run in `dst_root` (read, never written).
- **Frames:** that run's affine is in its own stitched well frame. The preview stitches its own block, so the affine is moved by the difference of the field's two pre-merge centres (preview and run), which come from the same seed or raster.
- **Without a finished tile-affine merge** the tab keeps item 1000's single picture and a row says to run merge with `ops_align_model` 'tile_affine' to compare.

## Code

`spacr/ops_live.py`: `_field_dna` (the sampled field's DNA, shared with `_placement`), `_merged_placement_row`, `_tile_affine_placement`; `_align` returns `placed_after`, and each field records its `source`. `spacr/qt/widgets/ops_live_preview.py` `_draw_align` draws the pair. New user strings go to Codex as `tr_1004_task.md` (orphan branch ws-offload-ws-tr1004task).

## Tests

`tests/qt/test_ops_live_before_placement_1004.py`: a known affine reads the field back in place in the preview frame (frame offset and shift), the crop's own tile group wins over the field affine, no merge gives no after picture, the test data's stored table is read, and the tab draws the pair (or the notice). Item 1000 and 971 tests unchanged. A run on the real Funk test data (A1) is reported with the hand-over.
