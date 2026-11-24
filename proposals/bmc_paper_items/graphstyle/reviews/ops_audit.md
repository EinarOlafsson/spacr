# OPS plotting audit — graphstyle, source 299179d

Read-only source audit at `299179d9530ac618d82f11cc823f3ad8abd1c001`, branch `ws-offload-codex-home-graphstyle-spec`. Scope: all 33 top-level `spacr/ops*.py` modules plus `spacr/power_ops.py` and `spacr/gene_measurement_compare.py` (35 files). No application edits, imports, Python, tests, renders, numerical analyses, jobs or publication were performed in this lane.

Authority: copied `proposals/bmc_paper_items/graphstyle/authority/SKILL.md` SHA256 `a2a199443c6a4bb0dafaf75fc503c908735a4d6b3646fa945d9a92d5813790c1`; companion `spacr_figstyle.py` SHA256 `6979fe7e9ed6aa31a41fc66b19447be50b9b901c861dd60f0982382a6913dca3`. Actual helper `SLOT_IN=0.22` controls, not older 0.25-inch prose. Eight-point minimum supersedes older smaller text. Task requires both light and dark output, while the copied helper alone implements light settings.

## Coverage and limits

The parent's one scheduled AST inventory has 729 files / 32,922 functions / 1,035 candidates and zero parse failures. The assigned scope contains **41 AST candidates**, all individually disposed in `ops_inventory.json`: 39 real plot/writer/wrapper routes and two nonplot false positives. Bounded text inspection adds nine report/image/layout/support functions and fifteen raw-data writers. Total **65 rows**: 16 plot creators, 22 wrappers, two image-report writers, one annotation-image creator, six support functions, 16 raw-I/O exclusions, two nonplot call-graph exclusions. All 65 IDs and definition lines match the parent's retained AST inventory; none of the 41 candidates is missing. No module-level plot creator was found in these files. Lexical nested functions were included in the parent's AST and the follow-up call search; none adds a plotting creator in this scope. The class method `FieldStager._stage` is explicitly excluded as raw-stack I/O.

The JSON contains the requested fields `id,file,line,name,purpose,chart_type,helper,findings,exposure,disposition`, including exact local source coordinates for each claim. The Markdown table below is a completeness index. The bounded text search is retained in `ops-text-candidates.txt`; scoped candidate list in `ops-candidates.tsv`.

Exposure ranks are estimates from public entry points, default-on reports and explicit optional phase routing, **not user telemetry**. No clipping, overlaps, leader crossings, illegible colors, actual font fallback or broken rendered scale bar was established without rendering. Variable sample-size, undefined-cell and same-range comparisons are conditional where the input governs them. A missing explicit geometry contract is distinguished from measuring an actual output's axes.

The other assigned modules contain numerical, registration, layout, storage or setting operations, without plot creators in the retained AST or bounded text pass. In particular, `np.histogram` and `stats.norm.pdf` in numerical code are not plotting; 'canvas' in stitch/compose is a pixel coordinate frame; `_st_analyse` produces tables/AnnData, not an extra chart. `ops_live._run_align` and `_run_merge` call `run_ops` with only stitch/phenotype or masks/merge/aggregate phases, so the propagated plotting candidates are false positives.

## Highest-impact findings

1. **Measurement comparison has two divergent implementations and a legacy shared restyler.** `gene_measurement_compare.plot:559` and `render_comparison:1114` use different box opacity (0.25/0.8 versus 0.30/0.85), violin opacity 0.55, fixed rather than physical category widths, unrestricted jitter overlays, and the old HOUSE palette. `gene_measurement_sweep.py:507–537` defines non-authority grey/rust/ochre and 7/6.2/6 pt fonts; its `_readable:631` forces those labels/ticks and removes top/right spines at 670–682. The newer `style_base.apply_page:219` also defaults to hiding those spines via `FigureStyle:100`. Fixing callers alone while leaving these routes unchanged would reintroduce violations.

2. **Scoring reports have directly verifiable small text and statistics presentation issues.** `ops_scoring._figures:768` uses 7 pt ticks/titles, 5.5 pt in-axes star strings, removes two spines, and writes 200 dpi PNG (795–814). The underlying supplied values are summaries, not necessarily independent replicates; don't convert them into purported replicate evidence. Preserve Holm-adjusted values in the inputs, but render exact adjusted probabilities through the shared bracket policy. `ops_scoring_figures` defaults true (`ops_settings.py:360`; caller `ops_scoring.py:1541`).

3. **Image UMAP is a parallel styling system.** `ops_image_umap._figure:692` directly creates Figure/Axes, has explicit light/dark colors, 4/6/7 pt labels, 0.25/0.35/0.8 opacity and a non-square 7.14-by-5.5-inch central axes. `utils.generate_colors:7915` uses Viridis (not rainbow, but outside the specified categorical palette); guides are foreground crosses rather than identity-teal. This route already preserves shared image ranges and raw crops (`shared_display_range:576`, caller 875–884), and exports PDF/PNG/SVG, so preserve those strengths while integrating styling. Crop panels lack scale bars/channel headings. Actual layout congestion remains unmeasured.

4. **Pipeline chart geometry and export policy drift is widespread.** `power_ops._figure:1160` and `gene_measurement_compare._etest_figure:5450` divide category width by comparison count and dodge bars; the authority instead prescribes consecutive fixed slots. `ops_maxproj._figure:505` already uses correct palette/0.6 opacity/0.6 bar fraction and an appropriate composition plot, but its canvas formula does not enforce actual 0.22-inch axes slots. Cluster/feature/cell-cycle routes similarly choose fixed canvas dimensions without the existing geometry helpers. Most local writers explicitly request fixed-name 120–200 dpi PNGs without vector/legend companions. `save_figure` can apply normal export choices and only cap requested DPI downward (`plot.py:121`); it does not silently raise 120/150 to 300. `_checked_savefig:582` is a checked raw savefig, not the full export/restyling service.

5. **Two actual Pareto charts need an explicit form decision.** `ops_decode.write_tuning_curve:1980` and `ops_grid_search._plot:603` intentionally plot yield versus error with a selected/default point. This is the tradeoff form the supplied authority rejects for comparisons; a later implementation should choose an approved presentation without discarding the curve/selected-setting semantics. Do not extend that ban to UMAP, spatial coordinates or valid continuous distributions.

6. **Spatial and annotation images need image-specific policy.** `ops_engine._st_draw_overlay:13379` uses white segmentation outlines, orange transcript marks, mixed opacity, per-image contrast and no scale bar/key. Actual transcript locations are data points, not a centroid-dot substitute, so the centroid-cross rule does not apply to them. `ops_cell_cycle.write_label_queue:1537` builds scientifically meaningful blinded triptychs using fixed raw bounds, own-nucleus dilation, measured micrometre calibration and a white scale bar, and records ranges/gamma/calibration in queue.json (1725–1784). It uses yellow nucleus outlines and default PIL missing-channel text. Change presentation only: keep the masking, identity blinding, fixed ranges, channel ordering and scale semantics intact. Raw .npy/TIFF files are not chart export failures.

## Existing shared route and bounded implementation boundary

Use **`spacr.figures.style` as the existing graph-policy owner**, not a third production styling module copied from the audit authority. It already has `rc:636`, `figure_style:720`, `condition_plot:935`, `size_condition_axes:1006`, `square_axes:1040`, `bracket:1078` and `figure_problems:1098`; constants at 893–904 match 0.22 inches, 0.6 opacity and the current palette. Its contextual rc settings preserve Open Sans, boxed axes and outward ticks when callers do not override them. `figure_style` is context-managed, unlike the copied helper's global apply_style.

Keep `plot.save_figure:426` as the normal export boundary, respecting `print_ready`, preference provenance and the explicit headless `pipeline_figures` guard (`ops_engine._in_pipeline_figures:11543`, `run_ops:11555`). `figure_style(theme_target())`, `figure_style` ink resolvers, `figure_style.py:854–904` and `style_base` already offer theme support. Forced `figure_style("print")` reports have no explicit dark-deliverable parameter; source inspection cannot certify both-theme outputs. UMAP has a distinct dark argument and must be reconciled with the same policy.

The existing `figure_problems` checker covers small text, overlaps, bounds, errorbar color and *already-square-marked* axes; it does not match every copied `check_figure` rule, and none of these scoped creators/writers invokes either by name. The image-integrity option in `save_figure/_checked_savefig` is not a replacement for chart-style checks and is off by default. Future work should check actual Figure artists/geometry at the normal export boundary with explicit exceptions, preserving public APIs and numerical outputs.

Preserve ordinary defaults and scientific inputs; style settings and export defaults are a separate migration decision. Do not change statistical tests, thresholds, fitted curves, control denominators, representative-cell selection, image ranges, pixel masking, phase routing or simulation seed/order to satisfy styling. Existing data CSVs/registered figure data can feed all demonstrations without recomputing screens.

## Authority conflicts to resolve before blanket changes

- Round5 says no n on graphs; later 2026-11-20 maintainer rules require absolute counts and explicitly permit annotated n. Inventory therefore flags current n placement as **policy-dependent**, not a certain violation.
- Earlier bars-everywhere advice is softened by Round7: choose the form for the question. Composition remains stacked bars; matrix stays heatmap; valid embeddings remain spatial/continuous charts.
- Current helper/condition_plot can still overlay very large point sets; prose requests violin/no points for thousands. Implementing the new policy requires an explicit automatic distribution branch rather than assuming shared helper use is sufficient.
- The requested dark theme cannot be obtained by blindly calling the light-only authority helper. Preserve equivalent semantics and neutral readable chrome in both themes; do not invert quantitative image intensities or recolor data opportunistically.

## Two bounded before-thumbnail proposals (not run)

1. **`gene_measurement_compare.render_comparison:1114`**, with an explicitly constructed `Comparison(measurement="Area (µm²)", level="well", frame=...)`: two named groups plus the existing `REST` label, four deterministic finite observations per group; no statistics requested. Use the default jitter-box style once, and capture the returned Figure through the normal exporter. This reveals fixed-width/opacity/legacy color/default-spine/count behavior without file readers, fitting or application startup. A second many-point synthetic case can be deferred until needed; don't run a parameter sweep. Both-theme appearance must use the existing theme/export path in an isolated preference scope, with no global preference writes.

2. **`ops_scoring._figures:768`**, four to six tiny summary rows with columns `level,pipeline,metric,value,shuffled_value`, one `Consistency...` metric in profile level and two or three pipeline names. Supply a predeclared `stats["complex_profile"]["q_holm"]` mapping (synthetic explanatory values, explicitly not measured p-values). This pure plotting function imports no optional scoring engine and writes at most one PNG into scratch. It exposes the known 5.5/7 pt text, stars, palette, frame and categorical geometry. Do not call `score_ops_profiles`, copairs, EFAAR or any pipeline. Its production route is print-only; do not fabricate a dark before-output by claiming an unsupported argument.

Both can run serially in the root's bounded scheduler slot with synthetic data, offscreen backend and native threads1; this audit has not scheduled or executed either.

## Exhaustive function index

Each exact function row, purpose, helper chain, source-supported findings and exposure rank is in `ops_inventory.json`. “Support” means it changes figure context/layout/display inputs rather than drawing a chart. “Excluded” keeps the candidate or tempting image-I/O match visible rather than silently dropping it.

| Exact source location | Function | Disposition | Purpose |
|---|---|---|---|
| `spacr/gene_measurement_compare.py:559` | `plot` | plot | Grouped measurement comparison; optional file export. |
| `spacr/gene_measurement_compare.py:680` | `save` | wrapper | Write comparison figures, tables/settings and optional cell PNGs. |
| `spacr/gene_measurement_compare.py:1114` | `render_comparison` | plot | Restylable grouped comparison on a new or supplied figure. |
| `spacr/gene_measurement_compare.py:5450` | `_etest_figure` | plot | OPS energy-test power and held-out false positives. |
| `spacr/gene_measurement_compare.py:5497` | `etest_phase` | wrapper | Run etest phase and write power_hits.png. |
| `spacr/ops_cell_cycle.py:1094` | `_png_data` | writer | Encode chart PNG into HTML data URI. |
| `spacr/ops_cell_cycle.py:1114` | `_dna_figure` | plot | Well DNA histograms, fitted components and gates. |
| `spacr/ops_cell_cycle.py:1156` | `_checks_figure` | plot | S1 field shares and S2 observed versus shuffled control. |
| `spacr/ops_cell_cycle.py:1253` | `run_cell_cycle_phase` | wrapper | Write plate classification report and its two figures. |
| `spacr/ops_clusters.py:1678` | `_sweep_figure` | plot | Metrics versus cluster count, real/shuffled and published partitions. |
| `spacr/ops_clusters.py:1766` | `cluster_sweep_phase` | wrapper | Write sweep output and figure. |
| `spacr/ops_clusters.py:3183` | `_consensus_figure` | plot | Sorted co-clustering probability matrix. |
| `spacr/ops_clusters.py:3210` | `cluster_consensus_phase` | wrapper | Write consensus matrix image. |
| `spacr/ops_clusters.py:3337` | `_bench_figure` | plot | Per-arm replicate excess over size-matched null. |
| `spacr/ops_clusters.py:3372` | `cluster_bench_test_phase` | wrapper | Write benchmark comparisons and figure. |
| `spacr/ops_decode.py:1980` | `write_tuning_curve` | plot | Write threshold-grid CSV and genotyped-versus-FDR image. |
| `spacr/ops_engine.py:339` | `_run_cpg` | wrapper | Dispatch cropped-site decode/grid phases. |
| `spacr/ops_engine.py:3747` | `_apply_auto_thresholds` | wrapper | Tune and write per-well decoding curve. |
| `spacr/ops_engine.py:3896` | `_decode` | wrapper | Decode and optionally tune thresholds. |
| `spacr/ops_engine.py:11555` | `run_ops` | wrapper | Top-level OPS dispatch and figure context. |
| `spacr/ops_engine.py:12139` | `_ops_admitted_run` | wrapper | Admitted OPS orchestration. |
| `spacr/ops_engine.py:13379` | `_st_draw_overlay` | plot | Spatial transcripts/spots and masks over microscopy. |
| `spacr/ops_engine.py:13722` | `_st_run` | wrapper | Write spatial analysis overlay. |
| `spacr/ops_engine.py:14425` | `run_spatial_map` | wrapper | Write highest-cell-field overlay after mapping. |
| `spacr/ops_feature_info.py:665` | `_plot` | plot | Rank corrected feature-family information with permutation null. |
| `spacr/ops_feature_info.py:705` | `_run_feature_screen` | wrapper | Write information table/chart/report. |
| `spacr/ops_grid_search.py:315` | `_default_points` | wrapper | Compute default point; optionally emits tuning chart through decode. |
| `spacr/ops_grid_search.py:603` | `_plot` | plot | Grid yield versus shuffle rate, default and Pareto front. |
| `spacr/ops_grid_search.py:639` | `_run_grid_search` | wrapper | Write OPS grid outputs and image. |
| `spacr/ops_image_umap.py:692` | `_figure` | plot | UMAP, hierarchical hulls, crop ring and dendrogram. |
| `spacr/ops_image_umap.py:770` | `run_ops_image_umap` | wrapper | Build and export image UMAP plus raw/display crops and provenance. |
| `spacr/ops_live.py:578` | `_run_align` | excluded_nonplot | Prepare alignment preview arrays. |
| `spacr/ops_live.py:1445` | `_run_merge` | excluded_nonplot | Prepare matching/aggregation preview tables. |
| `spacr/ops_maxproj.py:505` | `_figure` | plot | Cluster edge composition by feature block. |
| `spacr/ops_maxproj.py:540` | `cluster_maxproj_phase` | wrapper | Write edge-composition chart. |
| `spacr/ops_prefix.py:2585` | `run_prefix_decode` | wrapper | Prefix decode with optional per-well tuning output. |
| `spacr/ops_scoring.py:768` | `_figures` | plot | Profile/cluster consistency comparisons and significance summaries. |
| `spacr/ops_scoring.py:1441` | `score_ops_profiles` | wrapper | Write scoring tables/statistics and optional figures. |
| `spacr/ops_tvn.py:1451` | `run_profile_phases` | wrapper | Dispatch optional profile/cluster/etest report phases. |
| `spacr/power_ops.py:1160` | `_figure` | plot | Power by design settings with seed replicates and 80% reference. |
| `spacr/power_ops.py:1241` | `run_ops_power_sim` | wrapper | Write simulation tables and power chart. |
| `spacr/ops_cell_cycle.py:1203` | `_checks_rows` | support | Tidy rows attached to S1/S2 figure. |
| `spacr/ops_cell_cycle.py:1214` | `_report_html` | writer | Embed report images, checks/counts and F1 tables. |
| `spacr/ops_cell_cycle.py:1537` | `write_label_queue` | image_panel | Blinded annotation triptychs with nucleus outline and physical scale bar. |
| `spacr/ops_engine.py:11543` | `_in_pipeline_figures` | support | Install deterministic pipeline figure context. |
| `spacr/ops_engine.py:12716` | `_st_gray` | support | Prepare grayscale display range for spatial images. |
| `spacr/ops_image_umap.py:542` | `read_crop` | excluded_raw_io | Read centered medoid crop. |
| `spacr/ops_image_umap.py:576` | `shared_display_range` | support | Choose pooled/fixed range per channel across all displayed crops. |
| `spacr/ops_image_umap.py:602` | `composite` | support | Combine channels into RGB using supplied shared bounds. |
| `spacr/ops_image_umap.py:621` | `layout_crop_ring` | support | Assign group medoids to angular crop-ring slots. |
| `spacr/ops_engine.py:11047` | `_ph_neighbour_effects` | excluded_raw_io | 11161/11164 save numeric embeddings/names; no plot |
| `spacr/ops_engine.py:12682` | `_st_load_image` | excluded_raw_io | 12711–12713 mpimage.imread reads pixels; not imshow |
| `spacr/ops_engine.py:13451` | `_st_register` | excluded_raw_io | 13515 writes registered TIFF image; no annotations/chart |
| `spacr/ops_engine.py:14089` | `run_spatial_masks` | excluded_raw_io | 14255–14297 store source stacks/segmentation labels; no chart |
| `spacr/ops_flatfield.py:179` | `write_flat_maps` | excluded_raw_io | 207 saves correction-map arrays |
| `spacr/ops_flatfield.py:927` | `write_nuclear_flat` | excluded_raw_io | 974 saves numerical flat map |
| `spacr/ops_flatfield.py:1097` | `complete_pending_flat` | excluded_raw_io | 1129 saves numerical flat map |
| `spacr/ops_large_sites.py:1994` | `run_large_site_segmentation` | excluded_raw_io | 2159–2162 store integer masks |
| `spacr/ops_live.py:806` | `_write_crop` | excluded_raw_io | 819 writes float TIFF planes |
| `spacr/ops_modules.py:173` | `consensus_partitions` | excluded_raw_io | 209 np.savez stores partition labels, no figure |
| `spacr/ops_modules.py:620` | `run_module_phase` | excluded_raw_io | 678 np.savez stores reference numeric PCs/labels |
| `spacr/ops_run_infra.py:135` | `FieldStager._stage` | excluded_raw_io | 147 np.save stages raw stack |
| `spacr/ops_tvn.py:1322` | `_keep_cells` | excluded_raw_io | 1333 saves numeric embedding array |
| `spacr/ops_tvn.py:1656` | `_profile_state` | excluded_raw_io | 1688 np.savez stores profile/reference arrays |
| `spacr/power_ops.py:338` | `_calibrate_phenotype` | excluded_raw_io | 395 np.savez stores calibration pool arrays |
