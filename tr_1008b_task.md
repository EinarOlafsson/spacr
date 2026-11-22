# Item 1008B translation and API handover

Source branch: `ws-offload-codex-work-1008b`, based on `e2a1ace77c537e4d3783f57055214040a9272741`.

The maintainer deferred reviewed translation records and catalog rebuilds for the pilot in WORK_TASKS v74; v76 approved the mechanical API arrival bindings and kept Part B on the WS. This task supplies source prose for Home; it does not supply translations or change generated catalogs.

New source docstrings:

- `spacr/measure.py:4395`, `_otsu_by_label(flat, values, presorted=None)`: observed-value Otsu thresholds per label, first maximum, constant threshold zero, and optional shared sort.
- `spacr/measure.py:4450`, `measure_colocalisation_otsu(labels, image)`: raw overlap, slopes, Otsu Manders and dense-rank weighted coefficients, output columns and undefined-value rules.

There are no new runtime messages, setting labels, or tooltips. Column suffixes are measurement identifiers. Every new callable has a docstring.

The self-contained source manifest [tr_1008b_sources.json](tr_1008b_sources.json) includes the exact public canonical API source block, its SHA256, canonical ID, file and line, and the approved admission status. It also inventories the private helper and every test docstring with source hashes; those private and test entries need no records.

The exact public source docstring is in `tests/data/release_contracts/1008b_colocalisation_otsu_api_arrivals_2026-11-21.json`. The P3/P4 test checks that fixture against the source and the exact two-parameter signature. The mechanical five-site admission patch for `tests/test_docstring_correctness.py` is retained privately at `/home/carruthers/Documents/repo/spacr-codex-recovery-20261121/1008b/1008b-frozen-docstring-arrival-proposal.patch`; WORK_TASKS v76 explicitly approved it and it is applied in this branch.

## Measurement mapping for the documentation

All values use raw object pixels. Four channels produce six pairs and 54 new measurement columns per compartment, or 162 over cell, nucleus and cytoplasm. Existing Funk Pearson columns provide the remaining Brieflow correlation values.

| Output suffix | Definition | Undefined-value rule |
| --- | --- | --- |
| `lstsq_slope_ij` | Least-squares slope of j on i with an intercept | 0.0 for zero predictor variance |
| `lstsq_slope_ji` | Least-squares slope of i on j with an intercept | 0.0 for zero predictor variance |
| `raw_overlap` | `sum(i*j) / sqrt(sum(i*i)*sum(j*j))` | 0.0 if either channel sums to zero |
| `raw_k1`, `raw_k2` | `sum(i*j)/sum(i*i)` and `sum(i*j)/sum(j*j)` | 0.0 if either channel sums to zero |
| `otsu_m1`, `otsu_m2` | Raw intensity fraction in one channel where the other exceeds its object Otsu threshold | 0.0 if either channel sums to zero |
| `otsu_rwc1`, `otsu_rwc2` | Same fractions with weight `(R-abs(rank_i-rank_j))/R`, dense ranks and the larger maximum rank R | 0.0 if either channel sums to zero |

Otsu selects the first observed value that maximises the between-class variance, putting values equal to the threshold in the lower class. Constant objects have threshold zero. Integer thresholds agree with `mahotas.thresholding.otsu`; real intensities use the same observed-value rule without histogram bins. Zero replaces undefined coefficients so `filter_dataframe_features` retains the feature columns.

## Reference and shared implementation

The vendored numerical reference comes from `/media/carruthers/mnt1/brieflow_ws/brieflow/workflow/lib/external/cp_emulator.py`, SHA256 `b2ddc5d2cc027320ed1bf93fdd95ce2b058741ff9647c1581f73d399f18ff10f`: `lstsq_slope_all_multichannel` at line 612 and the Otsu branch of `measure_colocalization` at line 654. Tests preserve the multichannel least-squares calculation and the seven colocalisation expressions. The optional threshold fallback independently evaluates per-object splits if mahotas is absent; only the direct mahotas comparison skips without that package.

The private threshold helper centres and scales each object by a power of two, then accumulates lower-class sums with a segmented binary prefix scan. This prevents a bright earlier label from changing a later dim object. Its loop has log2(maximum object pixel count) rounds over arrays, with no Python iteration over objects.

The production function reuses `spacr/measure.py:4001` (`_sorted_by_label`), `:4029` (`_within_label_ranks`, including dense ranks), `:4060` (`_pearson_by_label` for centred pixel differences), `_profile_inputs` and `_foreground`. No existing operation was copied into another production module. The existing `_otsu_separation` and `_image_otsu` use binned skimage thresholds for different whole-image operations; their behavior does not meet the per-object observed-value rule of this feature and they are left unchanged.

Part A owns family registration and frame wiring; Part B adds only the public measurement and private helper after `measure_costes`. Settings, presets and schema are unchanged in this lane.

## Measured validation

The final source passed 66 tests in the scoped run: P3/P4, existing profile families and the 934 Measure/OPS parity module (actual exit 0; `numeric.log` in the private recovery directory). This includes the 200-object mahotas comparison, first-max ties, real intensity scales, label isolation, dense ties, zero/constant channels, empty masks and one-pixel objects.

A bounded synthetic call used a 640 x 800 uint16 image, four channels, 1,280 objects and 400 foreground pixels per object (512,000 total). One compartment call returned a 1,280 x 55 frame in 1.212213865 CPU seconds (1.236617497 wall seconds), or 94.704208203125 CPU seconds per 100,000 objects. This is a synthetic throughput check; final A-D integration timing is still owned by the WS session. The exact receipt is `benchmark.json` in the private recovery directory.

The six mandatory code-rule files passed 157 tests (actual exit 0). All four full gate groups passed with actual exit 0 and no unexpected failures, retaining the approved exact 30 catalog-only deferrals:

| Group | Actual result |
| --- | --- |
| Items | 1,248 passed, 3 skipped; 959 subtests passed |
| OPS | 1,132 passed, 28 skipped; 102 subtests passed |
| Settings | 614 passed, 2 skipped; 369 subtests passed |
| Ratchets | 4,072 passed, 76 skipped, 3 expected failures |

The generic gate report `full-gates/pilot-gate-result.json` records `accepted: true`, `actual_group_process_exit: 0` and an empty unexpected-failure list. Raw logs and native JUnit XML for each group are in that directory. The committed source manifest includes the full gate receipt and exact catalog deferral node IDs. The distinct API source-run fixture error below remains separately recorded.

The explicit API source run reported **3 passed + 1 deferred fixture error** (actual process exit 1). The passing nodes were the source-derived public callable census, its cross-check with the i18n extractor, and `test_public_docstrings_matches_reviewed_visible_coverage`. The separate node `tests/test_documentation_i18n.py::test_documentation_api_catalog_inventory_and_hashes_are_current[en]` stopped in `api_catalogs` fixture setup on the existing stale reviewed source `spacr.ome_zarr.OmeZarrImage#11`, `docs/i18n/reviewed/api/sv/2026-09-27-api-delta2-000.json`, before its test body. WORK_TASKS v74 defers these Home reviewed records; the parent accepted this exact error separately from the 30 catalog nodes excluded by the pilot gates. No fourth source-test pass is claimed. Raw proof: `api-arrival.log` (SHA256 `218a29dcf7855b42fd8a1f40558f706099c26a859b895de20de6768204649b48`); exact receipts: `api-arrival-result.json` and `api-record-deferred.json` in the private recovery directory. The committed source manifest lists all 30 catalog node IDs and the distinct deferred fixture node.
