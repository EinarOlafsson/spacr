# G2 checkpoint scope and route reconciliation

Static review at `9510143823dcc1b984ecc522058a1e3f15e985d8`, 2026-11-23. No Python, imports, tests, renders, jobs, application edits or ratchet edits were performed. The findings below are source inspection, not a new route-test PASS or appearance-compliance claim.

## Decision resolved by HOME_TASKS v180

The master explicitly assigns all 204 identities to Home. Step A converts the 160 entries outside G1-owned files now. Step B converts the 44 `plot.py` entries only after G1 FINAL lands on nightly and is merged. The seven additional Qt files listed below are outside the named G1 avoidance list. If Step A completes substantially before G1 lands, the master permits separate FINAL branches `ws-offload-codex-home-graphstyle-g2a` and `ws-offload-codex-home-graphstyle-g2b`. Keep deferred entries until their actual owner is converted, and preserve the correctly excluded `split_wells` entry.

The original question and evidence below document why ownership was clarified; they are no longer an open scope question.

The five-file implementation scope and the checkpoint's G2 deferred assignment differ materially. The original audit row names `measure.py`, `sequencing_qc.py`, `illumination.py`, `utils.py` and `submodules.py` (`codex-graphstyle-spec/proposals/bmc_paper_items/graphstyle_audit.md:36`). HOME_TASKS v178 repeats these five files and explicitly reserves `plot.py` and the named G1 Qt/style files. v179 points Home to the entire `deferred_g2.json`, but does not explicitly rescind the file restrictions. G1 `CHECKPOINT.md` also says that the five reserved G2 files are unchanged.

**Minimal question for master:** “Does v179 assign all 204 exact `deferred_g2.json` identities to Home G2, explicitly allowing the 44 listed `plot.py` identities despite v178's avoidance rule and adding the other 60 out-of-five-file entries; or should Home retain the five-file scope while G1 reconciles ownership of those 104 extra entries in the pinned initial plan?”

Recommendation until answered: proceed with the five-file work already authorized; preserve all extra deferred entries and the initial plan. Do not treat deletion from an exception list as a conversion, mark an unrendered creator compliant merely because it reaches style, or alter a numerical/image-processing wrapper to manufacture a route. Where an extra entry is already a wrapper around a converted owner, it can later be assessed for removal without changing its source, under the unchanged ratchet and appearance/data evidence. Independent creators need an explicit owner.

## Exact count reconciliation

`reports/graphstyle/routes/inventory.csv` is byte-identical to the original audit `functions.csv`: SHA256 `8c6de45f4436996eb55140f319245e3a8d6eb46cda7f2804fac018f713672c1c`. `initial_plan.json` is `cc41aef6994ecf2d4968b194564631cfc1d70e021c56bfbbe56f86910f079e79`; `deferred_g2.json` is `efdaf5a0ab1eb5afc932216298683a356906bce21667d4242a758b598018b77f`. Its full mapping equals `initial_plan.maximum_deferred["2"]` exactly, including reasons/locations.

| Physical source file | G2 deferred identities | Scope relationship |
|---|---:|---|
| `spacr/measure.py` | 34 | Named five-file scope |
| `spacr/utils.py` | 21 | Named five-file scope |
| `spacr/submodules.py` | 36 | Named scope; 37th historical entry is explicit non-plot exclusion below |
| `spacr/illumination.py` | 5 | Named five-file scope |
| `spacr/sequencing_qc.py` | 4 | Named five-file scope |
| `spacr/plot.py` | 44 | Explicitly reserved to G1 in v178 |
| `spacr/_mask_workers.py` | 2 | Additional transport/writer wrappers |
| `spacr/batch.py` | 1 | Additional generic execution wrapper |
| `spacr/external_masks.py` | 2 | Additional Measure wrappers |
| `spacr/image_quality.py` | 1 | Additional independent image-report owner |
| `spacr/seg_qc.py` | 1 | Additional independent heatmap owner |
| `spacr/sequencing.py` | 5 | Two QC wrappers plus a separate three-identity threshold-plot family |
| `spacr/spacr_cellpose.py` | 3 | Additional inference wrappers around plot.py drawing |
| `spacr/qt/mask_engine.py` | 1 | Additional image/label export owner |
| `spacr/qt/screens/align.py` | 1 | Additional native tile layout |
| `spacr/qt/screens/annotate.py` | 4 | Additional native histogram/image decorations |
| `spacr/qt/screens/make_masks.py` | 19 | Additional native overlays, histogram, volume rendering and callbacks |
| `spacr/qt/widgets/measure_preview.py` | 12 | Additional crop-thumbnail/composition family |
| `spacr/qt/widgets/qc_field_browser.py` | 7 | Additional field display/export/comparison family |
| `spacr/qt/widgets/segmentation_views.py` | 1 | Additional image-view dispatch |
| **Total** | **204** | **100 named-file entries + 104 extras** |

“14 modules” is correct when grouped by the first component after `spacr` (all Qt collapsed to one). There are **20 physical files**, including seven Qt files. The extras are exactly **44 plot + 45 Qt + 15 other**.

The historical named-file total is **101**, not 100: `spacr.submodules.split_wells` at `submodules.py:2676` is the difference. It remains in the 1,607-row inventory; `initial_plan.non_plot["spacr.submodules.split_wells"]` explicitly records disposition “excluded” and `primitives: []`. Its body loads/crops well images and writes scientific TIFF arrays (`2702–2742`); it is not a missing graph migration. The phase-1 `g2_submodules.md` correctly retains all 37 historical identities, including that exclusion. Preserve this entry and its raw-image behavior; do not add it to deferred G2 or silently drop it from coverage.

## What the ratchet does—and does not—prove

`tests/test_graphstyle_function_routes.py:80–119` pins the initial-plan hash, pins the original inventory hash, requires the original 1,607 identities, requires each current deferred mapping to be a subset of its original mapping, rejects cross-group duplicate IDs, and checks non-plot primitive signatures. Moving an ID between group files would violate that fixed assignment without coordinated G1 changes. G2 must not edit the pinned plan/test to bypass this.

`tools/graphstyle_routes.py:165–181` accepts each lexical definition variant when it has a call path to a named owner. Its own docstring explicitly says that this is not proof that every conditional path produced a compliant figure. The README likewise separates routing from geometry/rendered compliance. Therefore:

- A wrapper may become routed through its unchanged callee; it need not acquire a decorative style call.
- A creator can already call `figure_style` yet still have local conflicting colors/geometry or an unrelated conditional drawing path. Exception membership is neither proof of a missing route nor proof of visual failure.
- Parent/caller routing does not automatically certify every separately indexed nested helper. Their lexical calls must satisfy the unchanged test.
- The locations in deferred reasons are audit coordinates. Some moved during G1: e.g. `_checked_savefig` is recorded at `plot.py:582` but now defined at `590`; `_plot_cropped_arrays` is recorded at `5186` but now at `5197`; `graph_importance` is recorded at `9033` but now at `9198`. Match full lexical IDs, then current source, not stale line numbers.

## The 15 extra non-Qt/non-plot identities

| Exact IDs (prefix `spacr.`) | Current source route and disposition |
|---|---|
| `_mask_workers._MaskWorkerPool._deliver_figure` | `_mask_workers.py:1363–1378`: forwards a supplied figure to an arbitrary callback, shared `figure_sink.publish`, or `plot.save_figure`. Shared sink explicitly calls `figures.style.finalize_figure` at `figure_sink.py:132–135`. **Transport/writer wrapper**, not an independent creator. Callback branch remains caller-dependent; no claim all branches are proved. Keep until routing/normal export evidence covers its actual path. |
| `_mask_workers._worker_figures.show` | `_mask_workers.py:635–642` forwards existing figures to nested `publish`, which pickles them (`615–630`). **Transport wrapper**; this is not a direct style route and callback/worker transport must not change scientific values or ownership. Clarify source-route ownership rather than add a fake plotting call. |
| `batch.inprocess_runner` | `batch.py:1330–1359` invokes `cli.main`; no local artists. **Generic dispatch wrapper**, not a dedicated Measure creator. Do not broaden G2 into CLI execution or run a batch to prove appearance. |
| `external_masks.run_external_masks`, `external_masks.prepare_external_masks` | `external_masks.py:924–925` calls `measure.measure_crop`; `prepare_external_masks:944–969` plans and forwards. **Named-owner wrappers**. Their data import/write behavior stays intact; appearance fixes belong to Measure/downstream plot owners. |
| `image_quality._write_gallery` | `image_quality.py:154–199` creates stretched Pillow thumbnails and inline dark HTML/CSS (`189–197`). **Independent image-report creator outside the five files**. No five-file style conversion changes that HTML. Needs scope owner; preserve raw metrics and image-display normalization. |
| `seg_qc._robustness_figure` | `seg_qc.py:2630–2658`: already calls shared `_figure_axes` (`figures/style.py:763–774` → `figure_style`), but creates its own heatmap, colorbar and 7-pt annotations. **Independent creator with an existing shared route**, not a sequencing/Measure wrapper. Needs ownership/appearance decision, not automatic exception removal. |
| `sequencing._run_barcode_qc`, `sequencing.generate_barecode_mapping` | `_run_barcode_qc:1177–1189` calls `sequencing_qc.barcode_qc`; main mapping calls it at `1662`. **Named-owner wrappers** for that optional branch; preserve reads, counts, thresholds and exception policy. |
| `sequencing.graph_sequencing_stats`, `sequencing.graph_sequencing_stats.find_and_visualize_fraction_threshold`, `sequencing.graph_sequencing_stats.find_and_visualize_fraction_threshold._line_plot` | `sequencing.py:1716`, `1761`, `1789`: separate threshold-sweep creator family; `_line_plot:1795–1808` draws its own line in `figure_style`, and parent adds reference lines/limits at `1828–1838`. **Independent family**, not a wrapper around `sequencing_qc.plot_threshold_sweep`. Numerical threshold selection must remain unchanged. |
| `spacr_cellpose.identify_masks_finetune`, `spacr_cellpose.generate_masks_from_imgs`, `spacr_cellpose.check_cellpose_models` | `spacr_cellpose.py:169`, `308`, `410`; first two import/call `plot.print_mask_and_flows` (`181/299`, `341/405`), the last is stock-model orchestration. **Inference wrappers around another reserved plot owner**. Do not run inference or rewrite inference to obtain style proof; coordinate the shared plot renderer. |

## All 44 `plot.py` extra identities

Every ID below has prefix `spacr.plot.`. These are expressly in the v178 avoided file. Some are called by the five files, but that makes them dependencies, not permission to edit the reserved file. The following groups account for every one of the 44 IDs once.

| IDs | Scope classification / source evidence |
|---|---|
| `_checked_savefig`; `_display_gif` | Existing writer/display support, not new scientific charts. Current `590` writes with integrity handling; `5350` displays an existing GIF. Shared writer changes belong to G1. |
| `_generate_mask_random_cmap`; `_get_colours_merged`; `_organelle_slot_colour`; `generate_mask_random_cmap`; `outline_palette_colours`; `random_cmap` | Palette/support owners for mask/outline views. Do not equate label colors with biological control roles, or change label IDs/random-state consumption to force style compliance. `submodules.py:735,865` already calls `generate_mask_random_cmap`; the owner is still plot.py. |
| `_plot_images_on_grid`; `_save_scimg_plot`; `_save_scimg_plot._visualize_scimgs`; `plot_image_grid`; `plot_images_and_arrays`; `plot_images_and_arrays.plot_from_file_dict` | Independent montage/galleries and their orchestration. `utils.py:2198` imports `_plot_images_on_grid`; current `plot_image_grid:9048–9069` still owns canvas/image placement inside shared style. Not fixed by a new context at a caller. |
| `_plot_merged_plot`; `plot_merged`; `overlay_masks_on_images`; `plot_object_outlines`; `plot_image_mask_overlay`; `plot_image_mask_overlay._plot_merged_plot`; `plot_image_mask_overlay._save_channels_as_tiff`; `plot_image_mask_overlay.random_color_cmap`; `plot_image_mask_overlay_magenta_outlines`; `plot_image_mask_overlay_magenta_outlines._plot_merged_plot`; `plot_image_mask_overlay_magenta_outlines._save_channels_as_tiff`; `plot_image_mask_overlay_magenta_outlines.random_color_cmap` | Independent overlay/composition family plus nested palette/raw-channel TIFF support. `submodules.py:2406,2472` calls the ordinary overlay. The two `_save_channels_as_tiff` entries are raw image export boundaries; a ratchet entry does not justify recoloring raw TIFFs. Preserve those outputs, seek an explicit non-plot/route disposition through G1 if necessary. |
| `_plot_cropped_arrays`; `_plot_cropped_arrays.plot_single_array`; `normalize_and_visualize`; `plot_cellpose4_output`; `plot_masks`; `plot_organelle_output`; `plot_resize`; `print_mask_and_flows`; `visualize_cellpose_masks`; `visualize_masks` | Independent image/mask comparison creators. Measure calls `_plot_cropped_arrays` at `12028–12033` and `12297–12298`; utils calls `plot_resize` at `5912–5916` and `plot_masks` at `6473–6523`. Current `_plot_cropped_arrays:5197/5224` owns label-vs-intensity selection and per-plane drawing; a Measure-only patch cannot replace this owner. |
| `graph_importance`; `jitterplot_by_annotation`; `plot_data_from_csv`; `plot_data_from_db`; `plot_comparison_results`; `plot_lorenz_curves` | General statistical/data-loading wrappers and independent comparisons, broader than the named QC row. `graph_importance:9198` hands data to `spacrGraph`; `plot_data_from_db:8669` and CSV `8815` also use that shared family. These are not Measure-only call sites. Preserve analysis/grouping/selection; G1 must retain or delegate ownership explicitly. |
| `plot_region`; `plot_region.save_figure_as_pdf` | Mixed image/activation grid orchestration plus writer (`8931/8952` current). Uses existing plot and activation-grid owners; preserve field selection and output semantics. No separate local style helper. |

## All 45 Qt extra identities

The Qt files below are **additional to the original five**, but are not the specifically enumerated v178 G1 Qt files (`control_chart`, `fast_plots`, `graph_builder`, `graph_spec`, `grouped_plot`, `pca_view`, `trellis_view`, `umap_*`, preferences). That is a scope expansion question, not proof that all Qt changes were explicitly prohibited. Their direct native painting/image routes are not automatically repaired by changing Measure's Matplotlib creators.

| Prefix and exact suffix IDs | Disposition and current source evidence |
|---|---|
| `spacr.qt.mask_engine.` → `_export_vvvv` | `mask_engine.py:3406–3442`: mixed scientific 16-bit label PNG, RGB/outline image, table, manifest and vvvv project export. **Image/export boundary**, not ordinary statistical chart. Preserve labels/tables/source bytes; any display policy needs explicit scope and must not alter the scientific products. |
| `spacr.qt.screens.align.` → `TileLayoutWidget.paintEvent` | `align.py:161`: direct native tile rectangles/empty state. **Independent GUI geometry/decorative view**, not a named-file plot wrapper. |
| `spacr.qt.screens.annotate.` → `_ProbabilityHistogram.paintEvent`; `_Thumbnail._paint_badge`; `_Thumbnail.paintEvent`; `_ZoomOverlay.paintEvent` | Direct native probability histogram and image/badge/zoom painting. Histogram `1712–1745` computes its own plot rectangle, bars/colors and threshold pen. **Independent UI creators**, with different image/chrome versus statistical-mark semantics; editing the five files does not reach them. |
| `spacr.qt.screens.make_masks.` → `_LiveMagnifier.paint`; `_MaskCanvas._layered_picture`; `_MaskCanvas._paint_boxes`; `_MaskCanvas._paint_crosshair`; `_MaskCanvas._paint_drag`; `_MaskCanvas._paint_gesture`; `_MaskCanvas._paint_layers`; `_MaskCanvas._paint_recrop_boxes`; `_OtsuHistogramPlot.paintEvent`; `_PromptSession.paint`; `_layers_picture_for`; `_render_volume_view` | Native image compositing, manipulation overlays, histogram and volume rendering owners. Ratchet coordinates: `9616,3060,3642,3629,3804,4166,3508,4130,10062,7984,2297,11392`. **Independent UI/image creator family**, not a wrapper to five-file statistical plotting. Preserve volume pixels/labels/picking geometry and thresholds; do not apply blanket mark opacity to data images. |
| `spacr.qt.screens.make_masks._VolumeStage.` → `_build_layer_dock`; `_finish_stroke`; `_grid_toggled`; `_render_request`; `_stroke_to`; `_submit_render`; `_sync_scale_slider` | Volume-view callbacks/request/painting orchestration, at `12361,13591,12928,12964,13558,13053,12537`. **Wrappers/support around that independent native family**. Any eventual removal needs that family's route/appearance evidence, not a Measure route. No GPU/volume jobs were run here. |
| `spacr.qt.widgets.measure_preview.` → `_rounded_image`; `_rounded_pixmap`; `_thumbnail_image`; `_thumbnail_images`; `_crop_pass`; `MeasurePreviewPanel._cancel_extra_work`; `MeasurePreviewPanel._crop_pixmap`; `MeasurePreviewPanel._on_crops_ready`; `MeasurePreviewPanel._on_preview_scale`; `MeasurePreviewPanel._render_grid`; `MeasurePreviewPanel._start_crop_request`; `MeasurePreviewPanel.refresh` | Twelve crop-preview creator/callback identities. They do use Measure to obtain crops, but the graph/image presentation is local: `_thumbnail_image:140–145` constructs/scales/rounds a QImage; `_crop_pass:192–201` obtains crops then calls local thumbnail rendering. `_render_grid:2827` owns local composition. **Mixed data wrapper and independent UI view**, not fixed by styling Measure charts. |
| `spacr.qt.widgets.qc_field_browser.` → `QCFieldBrowser._render`; `_FieldOverview._compare_groups.run`; `_FieldOverview._export`; `_OverviewCellPicker.paintEvent`; `_OverviewCollapseArrow.paintEvent`; `_OverviewLevelsHistogram.paintEvent`; `_overview_export` | Seven mixed field-display, native histogram/chrome and export/comparison identities. `_overview_export:4176–4193` constructs its own Matplotlib page and uses shared save/style; `_compare_groups.run:8844–8864` delegates statistical drawing to **sp_stats**, not one of the five files. Others paint local views. **Independent owner family plus external wrappers**, with raw-image/chrome exceptions requiring explicit decisions. |
| `spacr.qt.widgets.segmentation_views.` → `SegmentationViews._default` | `segmentation_views.py:517–536`: dispatches to local `render_overlay/render_labels/render_flows/render_cellprob`, not Measure's figure creators. **Image-view wrapper around another owner**, requiring a local image-display policy if assigned. |

## Recommended bounded execution boundary

The phase-1 `g2_measure.md`, `g2_utils.md`, `g2_submodules.md`, `g2_sequencing_illumination.md` and `g2_tutorials.md` remain the five-file conversion/data/tutorial plan. Their external shared-owner boundaries are real dependencies; do not silently turn them into permission to edit every downstream renderer.

1. Reconcile every named-file historical ID, including `split_wells` as an unchanged exclusion; remap current lines by lexical ID.
2. Convert only authorized creators through the checkpoint's shared owner, preserving scientific arrays/tables/statistics. Preserve wrappers unless their actual API wiring needs a change.
3. Use the existing ratchet without changing its initial plan, non-plot signatures, owner set or tests. Remove only individually justified G2 entries; do not claim the whole deferred file will empty under the five-file scope.
4. Obtain the explicit ownership answer for the 104 extras before editing reserved/additional creators. If the five-file scope remains, G1/master must reconcile the pinned assignment; Home cannot move IDs across groups under the existing subset rule.
5. Keep static reachability, both-theme appearance, data identity and tutorial screenshots as separate evidence. No clipping, full-render compliance or changed-image claim was inferred from source alone here.
