# Item spec: OPS phenotype placement seeded from a saved raster file (a subset run composes with the whole-well positions of an earlier run)

- **Lifecycle:** future (alpha), behind "Show alpha features". One `ALPHA_FEATURES[940]` entry covers the `ops_placement_seed*` settings and the export action.
- **Number:** 940 (reserved 2026-11-13 by the master session).
- **Related:**
  - 936 (align & stitch shared core; the code lives there);
  - 935 (OPS live mode; needs this setting for align and phenotype);
  - 930 (`tile_affine`), 931 (feathered composition), 932 (settings-schema version table);
  - the test-data card `BMCspacr/reanalysis/experiment_designs/20261113_ops_testdata_subset_DRAFT.md` (open question 2, recommended option);
  - 921 (large-site tiling), the stitch determinism commit 5797246603.
- **Overlap with an existing spec:** none. 936 merges the solve and the composition but still solves; 935 needs a seed but does not define one (its open question 3 asks for it). This item defines the seed file and the setting.
- **Rule:**
  - With `ops_placement_seed` unset (the default), every output is byte-identical to today.
  - With a seed, a subset run composes exactly the pixels the full run composed for the same area. The seed replaces the solve, not the pixel work.
  - Every new documented callable ships with its arrival contract and nine-locale records in the same commit.
- **Code basis:** origin/nightly (6afe2e8f8c), plus 931 and 936 as they land.
- **Paper-critical:** no, but the test-data end-to-end test and item 935 are blocked without it.

## Problem and goal
The OPS stitch places tiles by overlap phase correlation and one least-squares solve over the whole well (`ops_engine._stitch`, `ops_stitch.stitch_well`, `ops_solve`). It writes the result as `ops_geometry` rows in `measurements.db`. The phenotype and merge phases read those rows back (`_placements`).

A few fields of a well cannot reproduce that solve. The engine takes the field count from the highest site number, keeps only the largest connected group of tiles, and phenotype alignment anchors lie across the whole well. The maintainer decided on 2026-11-13: placement can be loaded from a saved file from an earlier run, instead of being solved.

**Goal:** a setting that loads tile positions (in whole-well coordinates) from a file. A subset run then:
1. skips registration and the solve for the loaded tiles;
2. composes only the listed fields, at the same global positions the full run used;
3. checks that the file belongs to this data, and records which file it was in the run journal.

## Current state (verified on origin/nightly)
- **Where placements live.** `_stitch` (`ops_engine.py` ~l. 700-760) writes one `ops_geometry` row per placed site. Columns: `plate`, `well`, `cycle`, `site`, `y`, `x` (relative to the lowest placed tile), `origin_y`, `origin_x`, `tile_height`, `tile_width`. `_placements(db, plate, well)` reads `{site: (y, x)}` and the tile shape back, and raises "run the stitch phase first" when the rows are missing.
- **Report of the stitch.** The per-well dict holds `sites`, `placed`, `edges_proposed`, `edges_accepted`, residuals, `canvas`, `expected_canvas`, `canvas_agrees`, `raster_overlap`.
- **No file is written.** There is no `<well>.stitch.json` today. The only persisted form is the `ops_geometry` table inside the database. So this item adds an export and an import of that table, and does not reuse an existing file.
- **Composition.** `_WINDOW = 1024` (`ops_engine.py:148`) and `windows_over(canvas, size, overlap)` (`ops_compose`) cut the canvas into windows. 931's feathered blend weights each tile by its own distance to its edge, so a window's pixels depend on which tiles cover it.
- **Run journal.** `spacr.run_journal` records settings and a git hash per run (`_git_hash`).

## Proposed design

### Settings (new; alpha)
| Setting | Type | Default | Meaning |
|---|---|---|---|
| `ops_placement_seed` | path or None | None | a placement file to load; None means solve as today |
| `ops_placement_seed_mode` | 'strict' \| 'subset' | 'subset' | 'strict': the file must list exactly the tiles of this run. 'subset': the run may use any non-empty subset of the listed tiles |
| `ops_placement_seed_export` | bool | False | write the placement file after a solved stitch (full runs) |
| `ops_placement_seed_check_pixels` | bool | False | also verify the loaded positions against the tiles of this run: re-register a few seam pairs and require agreement within `ops_placement_seed_tolerance` |
| `ops_placement_seed_tolerance` | float, px | 2.0 | tolerance for that optional check |

The default of every key keeps old behaviour. A seeded run needs no `ops_placement_seed_mode` change for the common case.

### File format
`<plate>_<well>.ops_placement.json`, one per well and acquisition kind. It is an export of that well's `ops_geometry` rows plus provenance. No new solver output is invented.

```json
{
  "schema": "spacr.ops_placement",
  "schema_version": 1,
  "plate": "20200202_6W-LaC024A", "well": "A1",
  "acquisition": "sbs",                 // 'sbs' | 'phenotype'
  "cycle": 1,
  "tile_shape": [1480, 1480], "dtype": "uint16",
  "n_tiles_in_well": 333,               // highest site + 1, the layout count
  "tiles": [{"site": 321, "y": 15142.0, "x": 24118.0}],
  "origin": [y0, x0],                   // origin_y, origin_x of ops_geometry
  "canvas": [H, W],
  "raster_overlap": 213,
  "provenance": {
    "spacr_version": "...", "git_hash": "...",
    "settings_hash": "...",             // hash of the stitch-relevant settings
    "tile_ids_sha256": "...",           // sha256 of the sorted site numbers
    "source_files_sha256": "...",       // sha256 over (site, file name, byte size) of the stitched planes; not the pixels
    "created": "2026-..."
  }
}
```

- **Coordinates.** Pixels in the whole-well canvas, as `ops_geometry` stores them (float, relative to the lowest placed tile). The origin is kept so that a subset canvas can be cropped to the same global frame.
- **Writer.** Atomic (temp file, then rename), keys sorted, floats written with `repr` so the round trip is exact.
- **Versioning.**
  - `schema_version` is an integer. A reader accepts its own version and older ones, in 'upgrade' functions kept next to the reader. A newer version raises: "placement file version N is newer than this spaCR (M); update spaCR".
  - Unknown keys in `provenance` are ignored. Unknown top-level keys are kept on a rewrite.
  - This file version is independent of 932's settings-schema version, which versions the settings file.
- **Also accepted.** The same tiles from a `measurements.db` (`ops_seed_from_db`), so a user with a finished database does not need the JSON.

### Validation (before any pixel is read)
All failures raise `ValueError` through `tr()`, name the file, the well, and the offending values, and say what to do. Nothing is silently repaired.

1. **Schema.** `schema` is `spacr.ops_placement`; `schema_version` is readable; required keys present.
2. **Identity.** `plate` and `well` match the run's (case and zero-padding normalised as `schema.parse_well` does).
3. **Tile IDs.**
   - 'strict': the set of sites of this run equals the file's set. Mismatch message lists up to 10 missing and 10 extra sites and the counts.
   - 'subset': every site the run uses must be in the file. Sites in the run but not in the file are an error listing them. Sites in the file but not in the run are fine and are not composed.
4. **Layout count.** The run's layout count (`max(site)+1`, the same rule as `round_well_layout(max(site)+1)`) equals `n_tiles_in_well`. This catches a subset without the last site against a file from a full well.
5. **Image shape and dtype.** The tile shape and dtype read from the run's first tile header equal `tile_shape` and `dtype`. Message shows both.
6. **Acquisition kind and cycle.** They equal the run's reference cycle for stitching, or the declared acquisition for phenotype.
7. **Source files (warning, not error).** If `source_files_sha256` of the used sites differs, a warning says the files changed since the file was written. It is an error under `ops_placement_seed_mode='strict'`.
8. **Optional pixel check.** With `ops_placement_seed_check_pixels`, re-register up to 6 seam pairs among the used tiles (the existing `register_edge`, integer shifts) and compare with the file's relative offsets. A disagreement larger than the tolerance raises with the pair and both offsets. It never alters the loaded positions.
9. **Journal.** On success the run journal records `placement_seed.path`, the file's sha256, `schema_version`, `tiles_used`, `tiles_in_file`, the seed's provenance block, and the check outcome. A seeded run is marked `placement: seeded` in the well report, so no output can be mistaken for a solved one.

### How a subset run uses it
1. The stitch phase, when a seed is set, does not call `stitch_well`. It writes `ops_geometry` rows for the listed sites in the **global** frame (the file's `origin`, not re-zeroed to the subset) and reports `placement: seeded`, `edges_proposed/accepted: None`.
2. `_largest_component` is not applied to seeded tiles: the loaded positions define the placement, and a subset of disconnected blocks is allowed (test-data card, open question 4).
3. Compose and phenotype phases use `_placements` unchanged. They compose only the sites in `ops_geometry` for this well, so the unlisted tiles contribute nothing.
4. The canvas for windowing is the global canvas of the file. Windows that contain no used tile are skipped (cheap: no read). Windows that do contain used tiles are cut on the same grid as the full run, so a window is the same window.
5. **Boundary rule.** A window whose area is also covered by an unlisted tile will differ from the full run where that tile contributed. The run reports these windows (`partial_windows`), and the identity tests below use windows covered only by listed tiles, or the explicit rule that those windows are excluded from the identity claim.
6. Phenotype alignment (930 `tile_affine`, `ops_merge`) takes its well-level seed from the same file when `acquisition='phenotype'`. Anchors are not searched across the well: the seed carries the transform per tile group. If a listed tile group has no stored transform, the run raises and names the group.

### Interaction with other items
- **931 (feathering).** Feather weights are per tile (distance to the tile's own edge, +1), so a composed window depends only on the tiles that cover it and their positions. A seeded subset therefore gives identical pixels in every window that the listed tiles fully cover. The seed file does not store `ops_compose_blend`; blend is a run setting, and the run journal records both. A seed written under averaging can be used under feathering and the reverse, since only positions are stored.
- **930 (`tile_affine`).** For the phenotype acquisition the file can carry per-tile-group affine parameters (`"tile_groups": [{"id", "matrix", "n_nuclei"}]`) from `tile_affine_transform`. `ops_align_model` stays a run setting. If a seeded run asks for `tile_affine` and the file has no groups, it raises, because a subset cannot fit one. With `ops_align_model='similarity'` only the well-level similarity is needed.
- **932 (settings-schema versioning).** `ops_placement_seed*` are new keys whose default (None / False) keeps old behaviour. A settings file written before this item gets the same results under the new code, so **no `DEFAULT_CHANGES` row is added**. If the maintainer later turns on `ops_placement_seed_export` by default, that is a default change and gets a row then. The placement file's own `schema_version` is separate and is tested by the round-trip and the "newer version" error.
- **936 (shared core).**
  - The reader, writer, validator and `seed_to_placements` live in `spacr/stitch_core.py` (no Qt, no torch at import).
  - `align_folder` with `align_engine='core'` accepts the same setting under its own key `align_placement_seed`, and the same file format with `acquisition='align'`.
  - `ops_engine._stitch` and `align` both call `stitch_core.load_placement_seed`. No second copy.
- **935 (live mode).** Live align and phenotype take the seed from the user's finished stitch (the file, or the database), or from a sparse seed (935's open question 3). This item makes the file the mechanism; 935 picks the subset.

### GUI
- OPS panel: "Placement seed file" (path, browse) in the stitch and phenotype groups, with a "Write placement file after stitch" checkbox. The strict/subset choice sits in an "Advanced" fold. Alpha-gated.
- Tooltips say that a seeded run does not re-solve placement and is not a replacement for a full run in a paper.

## Acceptance tests
All run with `CUDA_VISIBLE_DEVICES=''` and a synthetic well (tiles cut from a known canvas with known shifts, plus noise) and, in the opt-in tier, the test-data subset.

1. **Default unchanged.** With the setting unset, `ops_geometry`, compose output and the journal (apart from the new keys) are byte-identical to nightly on the existing OPS test wells.
2. **Seeded subset equals full-run crop.** Run the full well (stitch with `ops_placement_seed_export=True`, compose). Run a subset of fields with the exported file. Every window fully covered by the listed tiles has the same pixels as the full-run composite cropped to that area, byte-identical (uint16) or equal within 0 (float64 accumulation, 931). Windows partly covered by unlisted tiles are reported in `partial_windows` and excluded.
3. **Phenotype seeding.** With a phenotype file carrying tile groups, a subset run's per-field transform equals the full run's for the same fields (≤ 1e-9 px).
4. **Round trip.** `ops_geometry` rows → file → `ops_geometry` rows is identical (floats compared exactly). Writing twice gives the same bytes. A file written by version 1 is read by every later reader (fixture kept in the tests).
5. **Mismatch errors**, each with the exact message checked:
   - wrong plate or well;
   - a site in the run not in the file;
   - 'strict' with extra or missing sites;
   - layout count differing (subset without the last site against a full-well file);
   - tile shape or dtype differing;
   - wrong acquisition or cycle;
   - schema_version newer than the reader;
   - a corrupt or truncated file;
   - `tile_affine` requested with no groups in the file;
   - changed source files (warning under 'subset', error under 'strict').
6. **Pixel check.** With `ops_placement_seed_check_pixels` and a file whose offset of one tile is wrong by 5 px, the run raises and names the pair. With a correct file it passes and does not change any position.
7. **Journal.** A seeded run's journal contains the seed path, sha256, version, tiles used, and `placement: seeded`. A reproduce-from-journal run (`cli_repro`) refuses if the seed file's sha256 differs, and says so.
8. **Determinism and time.** A seeded subset run is identical across reruns. Report the seconds of a seeded subset against the solved subset on the same tiles, in the same run (time is an optimisation target).
9. **End-to-end test-data run** (after the subset exists): stitch → align → genotype → phenotype on plate A well A1 with the seed in `reference/inputs/`, producing the reference outputs. This is the only item that makes the test-data card's open question 2 resolvable.
10. **Ratchets.** The raw-reader, nested-function, documented-callable and hash-comment ratchets pass.

## Translations and catalog
In the same commit as the code (catalogue symbols rule):
- nine-locale records for every new setting label, tooltip, validation message and GUI string (`tr()` keys);
- an arrival contract for each new documented callable: `write_placement_seed`, `load_placement_seed`, `validate_placement_seed`, `seed_to_placements`, `ops_seed_from_db`, plus the GUI action;
- regenerated runtime and API catalogs, help index, caption arrivals, coverage and the setting consumer map and settings flow, as the nearby commits do;
- warm the API catalogs first (`build_documentation_i18n.py --assemble`) before running tests.

## Dependencies, sequence and size
- **Depends on:** 936 for the home of the code (`stitch_core`). If 936 is not yet merged, this item may land first as `spacr/ops_placement.py` and move into `stitch_core` with 936, keeping the old name as a re-export. The item needs no 931 or 930 code to exist for the position-only part. The tile-group part needs 930.
- **Lands:** inside 936 or directly after it.
- **Used by:** 935 (live align and phenotype), the test-data end-to-end test (items 933-937 and 939 tests), item 921 for repeated large-site windows.
- **Size:** M (about 600 lines of code, 400 lines of tests, 9-locale strings for about 25 keys). Two to three agent-days including catalogue work.

## Readiness (proposals/READINESS.md convention)
- **Status:** Under development. It may only be exercised by its own proof checks (tests 2, 3, 4 and 9 above, against synthetic wells and the test-data subset's references). It must not be used to generate paper results.
- **Moves to Ready** when test 2 passes on the full test wells and on the A1 subset, with the evidence recorded in `proposals/READINESS.md`.
- **Paper rule.** Paper data come from full solved runs. A seeded subset output is test data or live-mode output only, and its journal says `placement: seeded`.

## Open questions for the maintainer
1. Should the file be JSON (readable, diff-able, recommended) or a Parquet / database table only? JSON is small (a few hundred tiles per well).
2. Should 'subset' be the default mode (recommended, since the setting exists for subsets), or 'strict'?
3. Should windows partly covered by unlisted tiles be composed (with a flag in the journal) or skipped (recommended: composed, flagged)?
4. Should a seed from a different spaCR minor version be accepted with a warning (recommended) or refused?
5. Should `ops_placement_seed_export` be on by default for full runs, so every finished well leaves a seed? That is a default change with a `DEFAULT_CHANGES` row, and costs a few KB per well.
