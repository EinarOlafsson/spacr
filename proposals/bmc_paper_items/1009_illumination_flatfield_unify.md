# Illumination and flat-field consolidation

Read-only specification, 2026-11-21. Proposed implementation owner: the WS lane. This audit changes no production code or settings.

Use **`spacr/ops_flatfield.py` as the single numerical implementation owner**, with Measure and OPS retaining their public workflow adapters. Move the remaining Measure fitting operations into that existing module; do not create another implementation. Keep the distinct estimators and their arithmetic conventions. This proposal adds no fresh default change beyond the separately approved item 1006 change.

## Sources and interpretation

All source coordinates below are `revision:path:line`. **B** is [nightly `19a95cd7993e0ce44078c779828b91098820598f`](https://github.com/EinarOlafsson/spacr/tree/19a95cd7993e0ce44078c779828b91098820598f). **F** is [amended item 1006 `08a41759c71eaa575f1043ad6ce4d6c4a31be439`](https://github.com/EinarOlafsson/spacr/tree/08a41759c71eaa575f1043ad6ce4d6c4a31be439). Test citations describe existing assertions; this audit does not claim those tests were executed.

| Finding | Source evidence | Implication |
|---|---|---|
| Measure's nuclear route already calls OPS's `build_nuclear_flat` and `flat_planes`. | B:`spacr/illumination.py:1342`; B:`tests/test_illumination_nuclear_flatfield_933.py:95` | There is an existing shared fitter to extend. Three application paths do not imply three independent nuclear algorithms. |
| Item 1006 adds compartment selection, object collection, pending-map completion and defaults around the existing object-fit kernel. | F:`spacr/ops_flatfield.py:1032`, `:1057`, `:1097`; F:`spacr/ops_engine.py:5168`, `:5256`. Direct B/F comparison leaves `build_flat_maps`, `fit_nuclear_flat`, `build_nuclear_flat`, `flat_to_field` and `correct_field` unchanged. | Consolidate ownership and adapters before changing numerical methods. |
| F lacks B's native source-loader and saved-model replay changes. | B/F diff of `spacr/illumination.py`; B:`spacr/illumination.py:918`, `:3437`; B:`spacr/ops_engine.py:4850`, `:4872`; F:`spacr/ops_engine.py:5117`, `:5233` | Integrate into the final merged source. Do not replace B's whole files with F's versions. |

## Models and options

| Path | Observations and fit | Normalization, geometry and correction | Evidence |
|---|---|---|---|
| Measure polynomial | Block-average raw fields; normalize each field by its median; take the across-field median; fit a total-degree polynomial, default degree 4, with MAD rejection. Supplied dark is not subtracted before fitting this relative profile. | Floor at 0.05 times the fitted-surface median; normalize to whole-field mean one. Generic bin centres are `i*factor+(factor-1)/2`. Apply float32 subtraction and reciprocal multiplication, without offset restoration. | B:`spacr/illumination.py:814`, `:825`, `:843`, `:948`, `:1053`, `:2191` |
| Measure smooth | Same relative-profile observations; Gaussian smoothing with sigma `max(1,min(binned_shape)/16)`, then bilinear interpolation. | Same generic floor, mean normalization and application policy. | B:`spacr/illumination.py:896`, `:1053` |
| Legacy OPS pixel/polynomial | Raw 16-pixel block means; median across fields; offset from the 0.1th percentile of block means; Gaussian sigma 6 map cells. DNA additionally receives a degree-4 polynomial. | Normalize `(surface-b)` by the central 18-cell-square mean minus `b`; floor 0.25. Missing lookup metadata means endpoint-aligned zoom. Exact correction is float32 `(I-b)/F+b`. | F:`spacr/ops_flatfield.py:54`, `:68`, `:97`, `:278`, `:303` |
| OPS object fit, also used by Measure nuclear | Median of raw object **means** in each map cell, with object counts. Default count-weighted local quadratic fit of `log(clip(median-b,1))`; local-linear and Gaussian alternatives. Known finite offset required. | Default block 16, sigma 32 field pixels, floor 0.10, centre square 18 cells, centre offset 8.0, block-centre fit/lookup. Default ridge is `1e-6` for quadratic and `1e-9` for linear. | F:`spacr/ops_flatfield.py:541`, `:657`, `:837`; B:`spacr/illumination.py:1342` |
| Item 1006 cytoplasm | Same object-fit kernel. Cytoplasm excludes nucleus pixels from each cell; within-object statistic is a mean, positioned at the cell centroid. | Same fitting options. A peak ratio above 1.2 is reported; the fitted map is not capped. | F:`spacr/ops_flatfield.py:1005`, `:1013`, `:1057`; F:`spacr/ops_engine.py:5256` |
| OPS analytic features | Average inverse flat within each compartment; adjust location, integrated and spread features with their existing formulas. | This is not generally identical to exact pixel remeasurement when intensity and gain vary together inside an object. Keep it a separate existing mode. | F:`spacr/ops_flatfield.py:325`, `:440`; B:`tests/test_ops_ownership_flatfield_funk.py:278` |

| Current selection | Effective defaults or rules | Evidence |
|---|---|---|
| Measure | Correction off; estimator polynomial; degree 4; per-plate true; maximum 50 fields; dark 0; missing plate is an error. An existing model takes precedence over vendor data, then estimation. | B:`spacr/illumination.py:3342`, `:3532` |
| Measure nuclear | One configured nuclear channel; other channels use the generic estimator. It refuses a missing/nonfinite/zero nuclear offset and median occupancy below one nucleus per map cell. | B:`spacr/illumination.py:1307`, `:1342`, `:1451`, `:1481` |
| OPS before 1006 | Correction `none`. Explicit exact/analytic modes remain available. | B:`spacr/ops_settings.py:131`; B:`spacr/ops_engine.py:4960` |
| Fresh 1006 | Correction `exact`; channel-fit `None`; nuclear stains include `A594`; object fields `all`. DNA/configured nuclear stains use nuclear medians; other channels use cytoplasm medians. Explicit per-channel entries win. Existing `legacy` selects DNA polynomial and other-channel pixel medians. | F:`spacr/ops_settings.py:130`; F:`spacr/ops_engine.py:5059`, `:5126` |

Different floors, offset sources, normalization regions and arithmetic orders are part of these contracts. A common function name is insufficient evidence that their outputs can be substituted.

## I/O, formats and callers

| Boundary | Current contract | Evidence / migration requirement |
|---|---|---|
| Measure fields | Retained NPY paths provide inventory/layout; selected pixels can come through `load_field`. Some shape discovery still reads the NPY header. | B:`spacr/illumination.py:771`, `:918`, `:1171`, `:1468`; B:`spacr/measure.py:11130`, `:11172`. Preserve selected-source callbacks; the numerical core receives explicit shape, not a new reader. |
| Depth and vendor input | Generic estimation takes a median over Z and applies one YX map across Z; merged-mask nuclear extraction requires YXC. Harmony/vendor profiles have their own gain/background/restore conventions. | B:`spacr/illumination.py:918`, `:1171`, `:1968`. Keep these existing adapters and refusals; a universal `I/F` operation would lose their semantics. |
| Measure model | Compressed NPZ arrays plus JSON manifest, loaded without pickle. Generic format 1; nuclear format 2 adds coarse maps, restore flags and fit records. Channels are integer intensity-axis indices. | B:`spacr/illumination.py:632`, `:677`, `:724`. Keep both readers and existing model bytes; do not rewrite models during loading. |
| OPS maps | Float32 `flat_<plate>_<channel>.npy`, `flat_info.json`, `fields.json`; nuclear/object sidecars and object Parquet tables. Channels are named. Metadata chooses lookup/offset. | F:`spacr/ops_flatfield.py:179`, `:226`, `:789`, `:927`, `:1097`; F:`spacr/ops_engine.py:5265`. Preserve adapters and explicit channel mapping. |
| Pending lifecycle | Object maps are withheld until the post-segmentation phase completes them. Requested pending channels refuse loading; completed maps are reused. | F:`spacr/ops_flatfield.py:226`, `:1032`, `:1097`; F:`spacr/ops_engine.py:5219`. Shared fitting must not bypass sequencing or refit completed maps. |
| Measure application | Preprocessing hook delivered to spawned workers through the existing installer/environment. Measure requests float pixels; standalone integer correction has rounding, overflow rescaling and clipping. Crop windows use field coordinates. | B:`spacr/illumination.py:193`, `:347`, `:489`, `:2236`, `:2949`; B:`spacr/measure.py:11342`. Preserve dtype and worker delivery as adapter policies. |
| Model replay/provenance | Resolved model path is saved; crop metadata includes model-file SHA-256. Segmentation provenance guards against double correction. | B:`spacr/illumination.py:2631`, `:3437`; B:`spacr/measure.py:4968`. Loading a saved run must not estimate a replacement model. |

| Caller to preserve | Operation | Source |
|---|---|---|
| Measure preparation and worker | Prepare/enable correction, then apply the existing preprocessing hook; preserve unmixing order. | B:`spacr/measure.py:13172`, `:11335`, `:11342` |
| Segmentation paths | Correct private segmentation inputs while retaining raw measurement intensities and resume provenance. | B:`spacr/io.py:4703`; B:`spacr/pipeline_v2.py:1711`; B:`spacr/core.py:740` |
| Saved-model embedding and standalone UI | Load/apply existing model; dispatch the public preparation workflow. | B:`spacr/embedding_profiles.py:256`; B:`spacr/qt/bridge.py:2195`; B:`spacr/validate.py:119` |
| OPS object phase | After segmentation, collect raw compartment means and call `build_nuclear_flat`; persist per-channel results. | F:`spacr/ops_engine.py:5108`, `:5168`, `:5256`, `:11113` |
| OPS exact consumers | Measurement, merged-field export and crop creation. | B:`spacr/ops_engine.py:5284`, `:7978`, `:8850` |
| OPS gains/analytic consumers | Mask-pass inverse gains and feature aggregation corrections. | B:`spacr/ops_engine.py:5142`, `:7333`, `:7349`, `:7406`, `:7500` |

## Same-data numerical comparison

The bounded P2 comparison passed on 2026-11-21. The following protocol was fixed before execution in the owned audit helper `compare_1009_synthetic.py` (SHA-256 `edf02337ed6fa9e0cec91b53dac77a5906308560134cc7c90091485d84f1d18f`). It uses the actual F implementations, with one native thread and no production-source changes.

| Protocol component | Fixed value or operation | Interpretation |
|---|---|---|
| Fields | 320×320 pixels, two channels, training sites 0–7 and unseen held-out sites 8–9; each uses seed `1009 + site`. | Every estimator sees the same training pixels. Held-out fields do not enter fitting. |
| Objects and signal | One 11×11 cell and 5×5 nucleus per 16-pixel map cell; cell-centre jitter up to two pixels per axis and nucleus shifted one pixel in x. DAPI nuclear signal 1000; A488 cytoplasm 700 and nucleus 150. | Dense objects provide a controlled diagnostic, not a real microscopy sampling model. Compartment signal replaces the background at its pixels. |
| Illumination and noise | Signal vignette `exp(-a*r²)`, with `a=0.8` for DAPI and `0.6` for A488; background `60*sqrt(vignette)`; camera offset 100 and Gaussian noise sigma 2. Coordinates use the image half-width. | The log-quadratic signal surface favors the object fitter's model family. Do not generalize its ranking to other fields or real plates. |
| Methods | Raw baseline; Measure polynomial/smooth; legacy OPS polynomial/pixel median on both channels; object median using nuclear DAPI and cytoplasm A488. | Evaluating both legacy fits on both channels is a comparison choice, rather than the shipped DNA-versus-other-channel selection. |
| Matched spatial grid | Measure `grid=20`, explicitly overriding shipped 256 to obtain 16-pixel bins; object and legacy maps use block 16. Other fitting options follow the model table above. | This is not an unmodified-default benchmark. Eight training fields also fall below Measure's ten-field trust threshold. |
| Offset policy | Supply 100 to Measure/object paths; legacy OPS estimates its own offset with its normal percentile rule. Keep each path's original correction arithmetic and offset restoration. | Offset estimation and map shape are distinct sources of error. Do not force identical offsets or rescale applied maps merely to improve scores. |
| Map shape error | Normalize predicted and true maps separately to mean one, then report `sqrt(mean((predicted/true - 1)²))`. | This compares relative illumination shape independently of the native map normalization. Applied correction still uses each method's native map. |
| Held-out object bias | Subtract each path's restored baseline from compartment means; normalize by mean signal at radius fraction below 0.25. Use radial bins `[0,.25), [.25,.5), [.5,.7), [.7,.85), [.85,1.01)`, with radius divided by the centre-to-corner distance. | Report outer-bin relative signal and whole-held-out signal CV separately. This is not an absolute-intensity calibration. |
| Exact shared-path checks | Compare Measure's existing nuclear wrapper with direct OPS for coarse map, sampled plane, metadata, both held-out corrected fields and a crop window. | These assertions assess shared implementation identity, independently of accuracy against synthetic truth. |

Results below are percentages, rounded to two decimals. Outer/centre uses the final radial bin; 100% means no residual radial attenuation relative to the centre under this metric. Each channel has 800 held-out objects; the outer bins contain 43 DAPI nuclei and 44 A488 cytoplasms. Source tables are `shape_error.csv` and `heldout_radial_bias.csv` under the evidence directory below.

| Method | Channel | Shape RMSE (%) | Maximum absolute relative shape error (%) | Outer/centre signal (%) | Held-out signal CV (%) |
|---|---|---:|---:|---:|---:|
| Raw | DAPI | 42.87 | 206.63 | 29.57 | 31.35 |
| Raw | A488 | 29.45 | 129.50 | 40.17 | 23.87 |
| Measure polynomial | DAPI | 25.43 | 126.90 | 45.98 | 19.58 |
| Measure polynomial | A488 | 10.54 | 50.52 | 69.76 | 9.20 |
| Measure smooth | DAPI | 26.90 | 134.53 | 44.77 | 20.47 |
| Measure smooth | A488 | 12.30 | 61.34 | 66.87 | 10.48 |
| Legacy OPS polynomial | DAPI | 32.43 | 160.28 | 33.78 | 27.26 |
| Legacy OPS polynomial | A488 | 20.79 | 95.34 | 36.79 | 24.58 |
| Legacy OPS pixel median | DAPI | 32.49 | 158.55 | 33.80 | 27.28 |
| Legacy OPS pixel median | A488 | 20.84 | 94.17 | 36.81 | 24.60 |
| Object median | DAPI | 1.60 | 17.51 | 99.09 | 0.73 |
| Object median | A488 | 1.09 | 11.91 | 98.60 | 0.54 |

In this realization, object maps have the lowest shape error and most uniform held-out signal, while their maximum map error remains appreciable at some pixels. This is expected to favor the local log-quadratic fit given the planted truth. Legacy OPS estimated offsets of 148.833 DAPI and 210.448 A488, compared with the planted camera offset 100 (`shape_error.csv`); its result includes that offset-estimation policy. These observations support preserving named estimators and policies during consolidation rather than treating them as numerically interchangeable. They do not establish which method is best on a real plate.

| Existing shared nuclear-path check | Observed result | Scope |
|---|---|---|
| Coarse map and sampled plane | Exactly equal | Measure `_nuclear_planes` versus direct OPS on the same DAPI table/options. |
| Fit metadata | Exactly equal | Full returned metadata dictionary. |
| Corrected pixels | Exactly equal in both held-out fields | Measure `NuclearIlluminationField.correct_field` versus OPS `correct_field`. |
| Crop window | Exactly equal to the corresponding corrected-field slice | Rows 31:99, columns 147:263, both held-out fields. |
| Source and synthetic-input guards | Passed | Numerical source, imported spaCR files and fixture bytes remained pinned. |

Evidence on Home is `/mnt/wd4tb/worktrees/codex-1009-spec/.codex-audit/1009-S1/`: `report.json` (SHA-256 `fe28824b89f52ff4aedbb2bb92f5b4e63c5eddff1cbab0673639305e99ac67e1`), the two summary CSVs, `heldout_objects.csv` and `maps.npz`. The report records their hashes, options, input hashes and exact-output identities. NumPy 2.5.2, SciPy 1.18.0, pandas 2.3.3 and threadpoolctl 3.6.0 were used. Scheduler job 550 exited 0 in 5 seconds; the numerical helper took 4.32 seconds, with scope `memory.peak` 148,070,400 bytes under the 2 GiB cap (`1009-S1.time.txt`, `1009-S1.resource.txt`, scheduler log). These are whole-comparison timings, not per-estimator speed measurements.

Two held-out fields cannot support real-plate confidence intervals or a general accuracy ranking. Object fitting uses exact synthetic masks; segmentation errors are not modeled. This comparison does not exercise native-only readers, complete phase persistence, pending-map interruption recovery or saved-model replay; those remain implementation acceptance boundaries in the table below.

## Proposed shared ownership

Keep the current public entry points and persisted formats. The implementation change is a move into the existing `ops_flatfield.py`, with thin delegates where an old import must remain valid. Do not leave copied active fitting bodies behind. Keep exceptions/messages at the public adapters and avoid a reverse dependency from the numerical owner into `illumination` or GUI/workflow code.

| Numerical responsibility in the one owner | Existing implementation to reuse or move | Adapter responsibility retained |
|---|---|---|
| Pixel-profile estimation | Existing OPS `field_blocks`/`build_flat_maps`; move Measure relative-profile, trimmed polynomial and smooth-surface bodies. Preserve each method's reduction order and options. | Field discovery, deterministic sampling, native loader, per-plate grouping and scientific method selection. |
| Object observations and fit | Reuse `field_object_means`, `nuclear_median_grid`, `fit_nuclear_flat`, `build_nuclear_flat`; pass explicit intensity/centroid columns and field shape. | Raw/corrected provenance, compartment choice, holdout selection, occupancy policy and phase ordering. |
| Map evaluation | Reuse `flat_to_field`/`flat_planes`; move the generic full-field/window operations without changing coordinates or casts. | Channel identity, crop bounds, persisted-format decoding and missing-model policy. |
| Pixel application | Reuse OPS divide/add-back and move Measure's existing reciprocal-multiply numerical body as distinct operations in the same owner. | Existing restore metadata, standalone integer handling, float measurement contract and hook installation. |
| Feature correction | Retain `mask_pass`/`correct_features`; do not substitute them for exact pixels. | Which existing analytic/exact mode the saved settings select. |

An internal result must describe map values, offset and its provenance, field shape, block/centre geometry, lookup, normalization, estimator/options, application arithmetic and per-channel training/holdout provenance. Existing NPY/JSON and NPZ serializers remain adapters initially; a common in-memory result does not require another on-disk format or user-facing compatibility flag.

## Migration and byte preservation

Preserve original saved-file bytes by loading without rewriting them. Preserve their effective settings through the established era-folding boundary, then verify output bytes independently. Re-saving with a current schema stamp is a separate user action; the existing writer does not promise identical historical container bytes.

| Case or rule | Required behavior | Evidence |
|---|---|---|
| Final source integration | Retain B's migration row 949, format-only 950 and resolved-model support while integrating F's 952 row narrowly. Preserve other allocations in the actual landing source. | B:`spacr/settings.py:2549`, `:2581`; F:`spacr/settings.py:2578` |
| Pre-952 file, both OPS keys absent | Fold to `ops_flatfield='none'` and `ops_flatfield_channel_fit='legacy'`. | F:`spacr/settings.py:2578`; F:`tests/test_ops_flatfield_channel_fit_1006.py:297` |
| Pre-952 file, explicit correction `exact`, fit absent | Keep `exact`; insert only the old `legacy` fit selection. | Same migration row and tests. |
| Explicit mapping or explicit `None` | Keep the present value; fold only absent eligible keys. | B:`spacr/settings.py:2706` |
| Current/fresh settings | Preserve the effective current defaults of the final integrated source, including 1006's existing scientific selection. | F:`spacr/ops_engine.py:5059`; F:`spacr/ops_settings.py:130` |
| Saved Measure model/estimator | Preserve explicit selection and saved-model precedence; do not refit during replay. | B:`spacr/illumination.py:3354`, `:3437` |
| Loader ordering | Fold in the saved file's era before ordinary default filling can hide omissions. CLI/GUI/replay already do this; direct `ops_defaults` and `illumination_settings` do not recover a missing historical era. | B:`spacr/cli.py:1719`, `:1728`; B:`spacr/cli_repro.py:58`; B:`spacr/qt/screens/app_screen.py:13512`; B:`spacr/ops_settings.py:4142`; B:`spacr/illumination.py:3532` |
| Pure consolidation | Add no `DEFAULT_CHANGES` row because this proposal changes no default. Any later deliberate default change gets the next owner-allocated version with the **previous** value under the existing key. | B:`spacr/settings.py:2635`, `:2706` |

## Validation and implementation risks

| Focused acceptance for the implementation | Existing coverage / source | Required extension or limit |
|---|---|---|
| Shared nuclear map, sampled plane and corrected pixels are exactly equal. | B:`tests/test_illumination_nuclear_flatfield_933.py:95`, `:105`, `:124`, `:152`, `:173` | Keep byte assertions, the occupancy refusal and both lookup choices. Numerical equality is separate from file-container equality. |
| Every old estimator preserves its map and pixel arithmetic. | B:`tests/test_ops_ownership_flatfield_funk.py:193`, `:226`; B:`tests/test_illumination.py:457`, `:472`, `:492`, `:549`, `:562` | Compare independent pre-move outputs for polynomial, smooth, legacy OPS and object maps; include nonzero dark, float/integer output, corners and overflow. |
| Full-field and crop-window output agree. | B:`tests/test_illumination_corrects_a_crop_window_773.py:49`, `:102`, `:117` | Preserve padding, channel order and both map geometries. |
| Saved settings and model resume preserve prior output bytes. | B:`tests/test_settings_schema_version_931.py:75`; B:`tests/test_illumination.py:1876`, `:1930`, `:2074`; F:`tests/test_ops_flatfield_channel_fit_1006.py:297` | Add the era cases above through CSV, JSON, journal and GUI boundaries; prove the original input files remain unchanged and no fit runs for a saved model. |
| Native and legacy readers both reach the same fit. | B:`tests/test_measure_ome_zarr_inputs_986.py:174`, `:285`; B:`spacr/ops_engine.py:4850`, `:4872` | Port F's new phase through the existing native-mask adapters. Its phase fixture stubs image reading and does not prove native-only operation. |
| One-time pending completion and offset/field selection remain intact. | F:`tests/test_ops_flatfield_channel_fit_1006.py:131`, `:167`, `:195` | Keep persistence separate from numerical extraction. The current multi-file writer is not one crash-safe transaction. |
| Worker delivery and raw-intensity provenance survive. | B:`tests/test_illumination.py:742`, `:909`; B:`tests/test_pipeline_v2_illumination.py:205`, `:269`; B:`tests/test_segmentation_illumination_with_separate_preprocessing.py:136` | Retain spawn installation, no-double-correction checks and private segmentation arrays. |

| Risk to avoid | Evidence | Proposed handling |
|---|---|---|
| Losing newer readers/replay by replacing whole files | B/F source differences cited above | Integrate narrow changes on the merged baseline before moving numerical bodies. |
| Inferring geometry from sampled centroids | F:`spacr/ops_flatfield.py:891`; B:`spacr/illumination.py:1342` | Accept actual shape at the shared boundary; preserve any old caller behavior until an explicit correctness change is separately tested. |
| Treating recorded MD5 as verified current bytes | F:`spacr/ops_flatfield.py:272`; F:`spacr/ops_engine.py:4933` | Preserve provenance behavior; do not advertise tamper detection or silently rebuild maps. |
| Losing per-channel holdout provenance | F:`spacr/ops_flatfield.py:1138` | Carry the individual map records; top-level metadata can contain the last completed channel's split. |
| Broadening a numerical refactor into a persistence repair | F:`spacr/ops_flatfield.py:1129` | Keep pending multi-file recovery changes separately scoped; map fitting alone does not repair interruption handling. |

Deliver the implementation in two stages: first integrate the existing defaults/readers and preserve public/persisted contracts; then move numerical bodies and run the focused equality checks. A different estimator or fresh default is a separate scientific change, not a necessary part of consolidation.
