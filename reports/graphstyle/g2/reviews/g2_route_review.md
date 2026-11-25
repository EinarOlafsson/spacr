> Curated publication copy. Original source: `.codex-audit/g2_route_review.md`; original SHA256: `25bf2ee5dde8cf06ed755421b88486cbb8224fd2d75fc6a7e5b758aa4066c8fc`. Original observations retain their recorded historical scope. Only selected artifacts are linked; omitted artifacts are labeled local-only and are never retargeted to a later pass. Unlinked context-relative paths and code blocks retain the original review context, which can include local-only omitted outputs; they do not select a later artifact. The validation index identifies the final selected subsets.

# G2 route disposition review

Recommend removing exactly the 78 IDs listed below from the G2 deferred file for the approved partial G2a delivery. Retain the two Measure categorical replay creators, their unresolved orchestration paths, and every other entry outside this list. This recommendation combines the frozen route result, static review of the actual drawing branches and accepted producer/native evidence; it does not claim completion of all 160 STEP_A entries. No source, test, initial-plan, scanner or deferred-file changes were made in this review.

Reviewed job 1684 report SHA256 `5330334b58b37eeb6b9366f4ce3ee19060a51a2ef1190223d9468963abe782d8`, helper SHA256 `1b11faa112a659949413e3c4b6958850a577345c68f74103fa7f392360e6090a`, and checkpoint `9510143823dcc1b984ecc522058a1e3f15e985d8`. All 15 selected source hashes and all pinned input hashes match the current files. Root subsequently committed these same core bytes as `a18cc405b6d1099f15343dbe7121b784a6d2a976`; the report still describes its original checkpoint-based scan. The report retains the original 1,607 inventory identities, original 13 owners, all 160 STEP_A entries and untouched 44 STEP_B entries; the ten selected pinned non-plot primitive signatures match. Its one normal scan reports 121 routed and 39 uncovered STEP_A identities. The five-entry uncovered recommendation bucket is a subset of those 39, not their total.

The unchanged guard at `tests/test_graphstyle_function_routes.py:69` requires every original identity and every currently detected primitive creator to be routed, explicitly deferred, or covered by its unchanged pinned non-plot signature. Exceptions may only shrink, retaining exact remaining IDs and reasons. Its routing algorithm proves some owner path for each lexical definition variant; it does not prove every conditional output branch compliant. Root should run this normal guard after the bounded deletion. Job 1686 separately passed the six unchanged hash-comment/raw-tabular guards; that result does not substitute for route, appearance or replay evidence.

## Exact removal list: 78

The first 40 are `candidate_removal_ids` minus the two Measure replay creators. The following 38 comprise ten additional reviewed creators and 28 local-only wrappers/writers. They are not mechanically inferred from a shortest-path witness: their actual drawing calls have been checked as described below.

```text
spacr.image_quality._write_gallery
spacr.measure._confluency_figure
spacr.measure._dna_histogram_figure
spacr.measure._hazard_ratio_figure
spacr.measure._time_to_event_figure
spacr.measure._viability_dose_figure
spacr.measure._viability_threshold_figure
spacr.measure._wound_curve_figure
spacr.measure._wound_edge_figure
spacr.measure._wound_plate_figure
spacr.measure.img_list_to_grid
spacr.qt.screens.align.TileLayoutWidget.paintEvent
spacr.qt.screens.annotate._ProbabilityHistogram.paintEvent
spacr.qt.screens.annotate._Thumbnail.paintEvent
spacr.qt.screens.make_masks._LiveMagnifier.paint
spacr.qt.screens.make_masks._MaskCanvas._paint_boxes
spacr.qt.screens.make_masks._MaskCanvas._paint_recrop_boxes
spacr.qt.screens.make_masks._OtsuHistogramPlot.paintEvent
spacr.qt.widgets.measure_preview.MeasurePreviewPanel._render_grid
spacr.qt.widgets.qc_field_browser.QCFieldBrowser._render
spacr.qt.widgets.qc_field_browser._OverviewCellPicker.paintEvent
spacr.qt.widgets.qc_field_browser._OverviewLevelsHistogram.paintEvent
spacr.qt.widgets.segmentation_views.SegmentationViews._default
spacr.seg_qc._robustness_figure
spacr.sequencing.graph_sequencing_stats.find_and_visualize_fraction_threshold._line_plot
spacr.submodules._invasion_stacked_bars
spacr.submodules._invasion_threshold_panels
spacr.submodules._replication_stacked_bars
spacr.submodules.generate_score_heatmap.plot_multi_channel_heatmap
spacr.submodules.interpret_vision_model.create_extended_radar_plot
spacr.utils.GradCAMGenerator.plot_activation_grid
spacr.utils.SaliencyMapGenerator.plot_activation_grid
spacr.utils.class_visualization
spacr.utils.model_metrics
spacr.utils.plot_clusters
spacr.utils.plot_clusters_grid
spacr.utils.plot_embedding
spacr.utils.plot_grid
spacr.utils.process_masks.plot_clusters
spacr.utils.setup_plot
spacr.illumination._write_qc_figure
spacr.illumination.illumination_qc
spacr.illumination.prepare_illumination_correction
spacr.illumination.prepare_illumination_model
spacr.illumination.prepare_segmentation_illumination
spacr.measure._run_time_to_event_step
spacr.measure._save_viability_figures._keep
spacr.measure._time_to_event
spacr.measure._wound_closure_tables
spacr.measure.measure_crop.job_callback
spacr.measure.process_meassure_crop_results
spacr.measure.process_measure_crop_results
spacr.qt.widgets.measure_preview.MeasurePreviewPanel._cancel_extra_work
spacr.qt.widgets.measure_preview.MeasurePreviewPanel._on_crops_ready
spacr.qt.widgets.measure_preview.MeasurePreviewPanel._on_preview_scale
spacr.qt.widgets.measure_preview.MeasurePreviewPanel.refresh
spacr.qt.widgets.qc_field_browser._overview_export
spacr.sequencing._run_barcode_qc
spacr.sequencing.generate_barecode_mapping
spacr.sequencing.graph_sequencing_stats.find_and_visualize_fraction_threshold
spacr.sequencing_qc._save_figure
spacr.sequencing_qc.barcode_qc
spacr.sequencing_qc.plot_barcode_qc
spacr.sequencing_qc.plot_threshold_sweep
spacr.submodules.apply_cellpose_model
spacr.submodules.apply_cellpose_model.plot_cellpose_result
spacr.submodules.compare_reads_to_scores
spacr.submodules.compare_reads_to_scores.plot_line
spacr.submodules.generate_score_heatmap
spacr.submodules.interpret_vision_model
spacr.submodules.plot_cellpose_batch
spacr.submodules.post_regression_analysis
spacr.submodules.post_regression_analysis._analyze_and_visualize_grna_correlation
spacr.submodules.post_regression_analysis._compute_effect_sizes
spacr.submodules.test_cellpose_model
spacr.submodules.test_cellpose_model.plot_cellpose_resilts
spacr.submodules.train_cellpose
spacr.utils.process_masks
```

This leaves 82 STEP_A and all 44 STEP_B entries deferred. Native evidence limits are not additional removal blockers: Align has no callable native finalizer, so acceptance rests on actual painted geometry and interaction evidence; the Overview print-restoration diagnostic is a known observation-stage false positive, not proof of a defect in the accepted saved page. Existing histogram diagnostics for interactive clipping shades/drag handles and unavailable inputs remain visible. Neither these limits nor empty diagnostics imply universal platform, locale, interaction-state or scientific validation.

## Required Measure replay holds

Keep `spacr.measure._wound_half_closure_figure` and `spacr.measure._viability_controls_figure`. Both have genuine shared-style routes and accepted numerical/appearance evidence, but their protected portable-replay dependency is unresolved.

At `measure.py:10187`, wound half-closure requests `condition_plot(..., points=False)`, then retains original deterministic offsets at `10200`, original dropna-only means including infinity at `10208`, and `_spacr_measure_means` at `10215`. Viability controls request points=False at `11408`, retain local RNG draws over the original rows and panels at `11418`, and retain original means at `11438`. The single-axis capture gate in `figures/bundle.py:457` and replay at `838–860` do not capture these mean overrides or actual point offsets, and do not express the multi-panel viability filtering/readout contract. A generic regenerated recipe is therefore insufficient. Preserve the existing metadata, exact means, row order and RNG behavior; wait for the G1 owner to supply complete capture/replay support. PNG/PDF success is not portable-replay completion.

## Five uncovered recommendations

| Identity | Actual boundary and required disposition |
|---|---|
| `spacr.qt.widgets.measure_preview.MeasurePreviewPanel._start_crop_request` (`measure_preview.py:2689`) | Asynchronous crop/result submission through lambdas, with no direct drawing primitives. The scanner intentionally excludes lambda bodies and cannot infer the job queue receiver chain. Request source-bound dispatcher handling from G1; do not style the scientific crop worker or add a no-op owner call. |
| `spacr.qt.widgets.qc_field_browser._FieldOverview._export` (`qc_field_browser.py:11800`) | Asynchronous submission to the actual `_overview_export` creator through lambdas, preserving selected fields, channels, limits, seed and options. Same dispatcher-boundary decision; no numerical rerun or new producer styling is implied. |
| `spacr.utils.plot_image` (`utils.py:8332`) | Real `OffsetImage`/`AnnotationBbox` thumbnail artist on an embedding axes. The actual display metadata and image bytes exist, but the independently indexed helper has no pinned shared-owner route. This is a genuine contextual image-owner gap, not pure raw I/O. G1 should provide or approve source-bound handling of the actual thumbnail artist. Do not restyle the entire embedding axes, alter pixels/zoom/coordinates, or add unused style resolution. |
| `spacr.utils.plot_images_by_cluster` (`utils.py:8296`) | Calls the preceding helper while preserving noise exclusion, cap and original random selection. Its hold follows the thumbnail-owner boundary; retain selection and image bytes. |
| `spacr.utils.plot_umap_images` (`utils.py:8252`) | Calls the same thumbnail chain, preserving index alignment and original sampling. A routed parent embedding does not route backward into an independently indexed child. |

## Local-only wrapper decisions

Seven of the eight wrapper-bucket IDs are eligible. Six are straightforward wrappers of migrated local creators: `measure._wound_closure_tables:10273`, `qt.widgets.measure_preview.MeasurePreviewPanel._cancel_extra_work:2457`, `_on_crops_ready:2727`, `_on_preview_scale:2806`, `refresh:2589`, and `utils.process_masks:9908` (all identities prefixed `spacr.`). Their actual drawing branches respectively reach the wound-edge figure, actual preview grid, or nested mask-cluster plot. This preserves cancellation, result tokens, scientific mask processing, selected rows and optional-plot behavior; it does not remove separately indexed raw or asynchronous children from their own deferrals.

`spacr.submodules.interpret_vision_model:4014` is also eligible: feature/permutation branches call `_feature_importance_figure`, and the SHAP branch calls `shap.summary_plot(..., show=False)` then the actual `_shap_distribution_figure` before displaying. Their accepted family and replay evidence apply to those final producers; the separately indexed nested radar creator is independently eligible. A null old-family witness is not an unrouted new helper. Keep `spacr.utils.MLR:5811`: it has both local `model_metrics` and existing G1 `_reg_v_plot` routes, and local metrics evidence alone does not accept the volcano branch. Preserve OLS/refinement and all returned values.

The additional 21 eligible wrappers/writers from the external-path bucket have the following actual boundaries:

| Group | Checked drawing contract |
|---|---|
| Four illumination preparation/QC callers (`illumination_qc:3143`, `prepare_illumination_model:3383`, `prepare_illumination_correction:3487`, `prepare_segmentation_illumination:3526`) | Their QC branches terminate in accepted `_write_qc_figure:3251`. Estimation/loading/validation and correction hooks retain their original numerical roles; no independent external chart creator is introduced by these branches. |
| Five sequencing callers/writers (`sequencing._run_barcode_qc:1152`, `generate_barecode_mapping:1451`, the nested fraction-threshold wrapper `1765`, `sequencing_qc._save_figure:1137`, `barcode_qc:1548`) | QC dispatch reaches the accepted barcode and threshold creators; the nested fraction-threshold body adds its retained reference markers and measured external key to its accepted line plot before normal save. The separate outer `graph_sequencing_stats` remains held because it also calls `plot_plates` at `sequencing.py:1923`. |
| Six submodule orchestration IDs (`apply_cellpose_model:865`, `test_cellpose_model:645`, `train_cellpose:557`, `compare_reads_to_scores:3754`, `generate_score_heatmap:7514`, `post_regression_analysis:7745`) | Their application drawing branches delegate to accepted nested Cellpose views, `plot_cellpose_batch`, comparison line, multichannel heatmap, correlation heatmap and effect-size producer. Model inference/training, source selection, caps, tables and RNG are unchanged. The training function only requests its existing preview; no model rerun is required to verify this delegation. |
| Two time-to-event callers (`measure._time_to_event:15842`, `_run_time_to_event_step:15913`) | The optional survival and nonempty-Cox branches call the accepted survival/HR creators and normal writer; no categorical W4/V2 branch is involved. |
| Four save/result boundaries (`measure._save_viability_figures._keep:11685`, `process_measure_crop_results:16086`, deprecated `process_meassure_crop_results:16113`, `measure_crop.job_callback:14282`) | `_keep` only calls the normal writer and records the path. Result handling saves supplied figures normally, selects/displays them under `figure_style`, and closes them; the alias and completion callback only delegate that same drawing boundary. Removing these writer/dispatch IDs does not certify every supplied upstream figure or release any replay creator/orchestration hold. Unlike `_MaskWorkerPool._deliver_figure`, these bodies have no arbitrary rendering callback branch. |

## Ten additional accepted local creators

The common support-path flags do not hold these actual producers: `illumination._write_qc_figure`, `sequencing_qc.plot_barcode_qc`, `sequencing_qc.plot_threshold_sweep`, `qt.widgets.qc_field_browser._overview_export`, `submodules.compare_reads_to_scores.plot_line`, both indexed `submodules.post_regression_analysis` creators, `submodules.apply_cellpose_model.plot_cellpose_result`, `submodules.test_cellpose_model.plot_cellpose_resilts`, and `submodules.plot_cellpose_batch`. All prefixes are `spacr.`. Each has its own real shared-style route and accepted final family evidence. The last three deliberately use the existing `generate_mask_random_cmap` instance LUT; that raw palette function creates no external figure. Accepted final-width renders and exact pixel/norm/LUT identity support retaining it unchanged.

## External and raw-boundary holds

Thirty-one of the 45 reported external paths terminate at `spacr.plot._organelle_slot_colour:3365` through the optional normal-writer integrity/replay chain (`save_figure:536` -> `_integrity_report:2868` -> image replay). Six terminate at `spacr.plot.generate_mask_random_cmap:4458`, a raw instance-colour helper. These 37 support witnesses warranted the source review above, not automatic rejection or removal. Thirty-one of these IDs are now eligible: ten actual creators and 21 wrappers/writers. Six remain held: `_MaskWorkerPool._deliver_figure` has an arbitrary `on_figure` callback at `1366`; `measure._classify_viability`, `_run_viability_step`, `_save_viability_figures`, and `_run_wound_closure` still invoke unresolved categorical replay producers; and `sequencing.graph_sequencing_stats` separately calls `plot_plates`. The path index is a witness rather than an exhaustive branch audit, which is why these real additional branches were checked. Do not recolour label masks or change scientific inputs to eliminate a raw support dependency.

The other eight reported external witnesses reach real model/profile/crop drawing owners: two `deep_spacr.apply_model`, four `sp_stats._profile_measurements`, and two STEP_B `_plot_cropped_arrays` paths. The 60 explicit dependency/raw-boundary holds separately retain the documented G4 assay/report/plaque paths, STEP_B image owners, G1 volume callbacks, raw RGB/mask export, worker transport, interaction chrome and three genuine shared support helpers absent from the pinned owner set. Use [reports/graphstyle/g2/g2_step_a_dependencies.md](../g2_step_a_dependencies.md) for their source-bound contracts. No fake seed expansion, additional owner IDs, graph alpha on image pixels, altered inference or new scientific computation is warranted.

## Four factored primitive creators

The new identities `spacr.submodules._feature_importance_figure:152`, `spacr.submodules._shap_distribution_figure:186`, `spacr.utils._activation_grid:7081`, and `spacr.utils._class_visualization_preview:7406` all reach original pinned owners. They are actual production bodies already covered by the corresponding family evidence, not substitute renderers. The first two reach `apply_axes_style`; the latter two reach `figure_style`. No deferred entry or route seed is needed. The normal guard already includes newly detected primitive creators independently of the original inventory.

This review used only the existing report, static source/guard reads, exact hash comparisons and prior accepted evidence. It ran no Python, imports, tests, renderers, compute jobs or inventory scan. Root owns any deferred-file deletion and its subsequent unchanged normal guard.
