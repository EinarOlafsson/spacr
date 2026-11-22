# Graph style across spaCR: audit and implementation specification

Audit date: 2026-11-22. Source: `299179d9530ac618d82f11cc823f3ad8abd1c001`, the remote `nightly` head when this task started. Train 11 had not reached that ref. This is an audit and proposal; no application code, settings, scientific calculations or tests were changed.

spaCR already has part of the requested policy. `spacr.figures.style` contains 0.22-inch condition slots, a 20-observation switch, 60% marks, control colors and scoped styling. Many plotting functions use its theme context and the shared writer. The remaining problem is inconsistent adoption, competing defaults, and renderer/export paths that preserve or introduce local exceptions. Copying another style module into the application would add a third policy source.

## Authority and scope

The authoritative merged [skill](graphstyle/authority/SKILL.md) and [reference helper](graphstyle/authority/spacr_figstyle.py) are copied verbatim. Their SHA-256 values are `a2a199443c6a4bb0dafaf75fc503c908735a4d6b3646fa945d9a92d5813790c1` and `6979fe7e9ed6aa31a41fc66b19447be50b9b901c861dd60f0982382a6913dca3`.

The inventory parses all 729 Python files under `spacr/`, including nested functions, without importing them. It retains 32,922 lexical function definitions and identifies 1,035 broad candidates through plotting/writing primitives and statically resolvable callers. Review adds dynamic drawing/dispatch functions and classifies false positives. For example, rendering CLI text, copying a raw image, painting a decorative widget and constructing the report's `Figure` dataclass do not create a statistical chart. Static analysis cannot prove every possible dynamic runtime route; the supplemental source review and explicit exclusions make that limit inspectable.

The complete function inventory is in [functions.csv](graphstyle/functions.csv): **1,607 unique reviewed entries across 213 files**, covering all 1,035 candidates plus 572 supplemental drawing/support entries. Reconciliation found no duplicate IDs, missing candidates or source-coordinate mismatches. The [coverage summary](graphstyle/inventory.md) gives disposition and review-family counts. Each row gives the qualified function, exact source location, purpose and chart type, shared helper route, findings, estimated exposure and disposition. [Review notes](graphstyle/reviews/README.md) retain the supporting source evidence by family. A writer/wrapper row explicitly inherits its drawing owner's findings; absence of local marks is not a compliance verdict. Excluded rows remain in the inventory so they cannot silently disappear from coverage.

Exposure is a qualitative estimate from production callers and UI/report routes, not usage telemetry. “High” means routine shared plotting, primary analysis/QC or frequently embedded results; “medium” means an optional analysis/export; “low” means a specialized, debug or uncalled route. A later source revision needs a delta review, not a claim that this baseline describes newer code unchanged.

## Rules to carry into one policy

| Rule | Required interpretation and audit boundary |
|---|---|
| Categorical geometry | Use the revised/helper value of **0.22 in per condition**, element width 0.6 slot (0.132 in), gap 0.4 slot (0.088 in). Older 0.25-in prose is superseded. Equal condition counts have equal plot-area widths; grouped series use consecutive slots rather than narrower dodged bars. |
| Continuous/matrix geometry | Continuous statistical axes are square; heatmaps have square cells and a declared normalization. Image/schematic aspect follows its content and must not crop scientific pixels merely to make a square. |
| Sample-count form | Default comparison: a mean bar plus every replicate through 20 finite observations, a box/violin above 20, and **violin with no points for thousands per group**. Large continuous scatter has its separate density/hexbin rule. Preserve experimental-unit aggregation and statistical values. Other questions may need trends, distributions, composition or maps; the later skill explicitly rejects making every panel a bar chart. |
| Opacity and colors | Effective data-mark alpha is 0.6, including face RGBA and artist alpha; axes/text are opaque. Use the reference palette, never brown. Reference/control identities are explicit and grey; avoid inferring identity from a label substring. A comparative image uses spaCR teal and comparator magenta outlines, with a key and shared display ranges. |
| Text and legends | At least 8 pt; bundled Open Sans faces; objective quantities/units and expanded abbreviations in the caption/key. Titles and explanatory prose belong in the legend. Apply the skill's specific missing-value/status/schematic exceptions. Its later absolute-count instruction partly qualifies the earlier blanket ban on counts, so count placement must be stated rather than every `n` automatically called a violation. |
| References and heatmaps | A meaningful reference must be explained. Do not add an artificial grey row to a numeric heatmap: the later heatmap rule requires colors from its own scale and marks undefined cells explicitly. Reference ticks/legend entries are often the appropriate representation. |
| Themes and layout | Test light and dark display/export modes with the same scientific data, role colors and geometry. Check actual legend colors, text bounds, legend/data overlap and contrast. Source review alone cannot prove clipping or its absence. Keep black error bars per the authority while choosing a readable plotting ground; do not silently substitute an unrecorded theme exception. |
| Images and deliverables | Comparative images need matched ranges/gamma, method identity, scale and provenance. Do not apply graph alpha to micrograph pixels. Paper/development deliverables need the declared status, legend and source/statistical data plus the required vector/raster formats. An operational thumbnail is not automatically a complete paper figure. |

## Priorities by user exposure

| Priority | Routes | Confirmed issue and proposed first change |
|---|---|---|
| 1 — shared policy and finalization | `figures/style.py:34,421,636,935,1098`; `figure_style.py:15`; `plot.py:426`; `figure_sink.py:107` | Existing scoped style and preference defaults disagree on colors, spines, DPI and geometry. Ordinary saving applies changed preferences, not a complete graph-rules check. Make one resolved style/semantic specification feed these existing adapters and add final validation. |
| 1 — primary plots and embedded charts | `plot.py:6837`; `figures/panels.py:220`; `figures/distributions.py:273`; `figures/sheet.py:163`; `qt/widgets/fast_plots.py` | The working single-column condition route already follows part of the rules, but sibling branches, regression panels and custom Qt primitives retain local geometry, colors, labels and opacity. Migrate builders and adapters through the same specification; test scientific-value preservation. |
| 1 — Measure, mask and sequencing QC | `measure.py:8606,10021,10057`; `sequencing_qc.py`; `illumination.py`; `utils.py`; `submodules.py` | Shared contexts are common, but local plots still use sub-8-pt text, opaque or .15/.25/.65 marks and undeclared reference roles. Fixed layouts without square-axis enforcement need plot-area measurement; canvas shape alone is not a geometry violation. Image normalization/overlay policy needs explicit comparison metadata, not blanket recoloring. |
| 1 — OPS report outputs | `ops_engine.py`; `ops_cell_cycle.py`; `ops_scoring.py`; `ops_image_umap.py`; `gene_measurement_compare.py` | Small local fonts, inconsistent opacity/geometry, legacy palettes and fixed low-DPI report images persist. Preserve each report's numerical decisions and captions while replacing only drawing/export policy. |
| 2 — assay, regression and model reports | `assay_panels.py:1067`; `regression_qc.py`; `classifier_evaluation.py:2119`; `scorecard.py:1080`; `sp_stats.py:5450` | Assay bars do not switch above 20; control identity is not passed to the color owner. Brown colors occur in real plot paths; dodged bars and on-panel values persist. Use explicit plot kind, experimental unit and identity metadata. |
| 2 — tracking and specialized analyses | `division_prediction.py:711`; `division_analysis.py:1902`; `timelapse.py:4090`; `gene_graph.py:2752`; `surrogate.py:941` | Source and rendered evidence show theme/geometry defects, horizontal forms rejected by the skill, inconsistent mark opacity and unexpanded terms. Do not transpose a scientifically different chart mechanically; retain the panel question and choose its allowed form. |
| 3 — optional/debug/recreation routes | Recreated bundles, old QC entry points, specialized overlays and developer helpers in the inventory | Keep route coverage and reproduction metadata. Address lower-exposure creators after shared and routine outputs. Decorative UI/resource generators are explicit exclusions, not a migration backlog. |

Specific brown-color evidence includes `guide_permutation.py` (`#6D2B00`) and `sp_stats._map_figure` (`#b5471b`); the function inventory supplies the exact local coordinates. The different gray, rust and coral entries in old `ROLES` are distinct from the newer reference palette and need semantic mapping, not unrestricted color replacement.

## Representative before renders

Three unchanged production drawing functions were run once in both themes with tiny synthetic inputs. No models, microscopy pipelines or statistical tests were run. The normal `save_figure` path used screen mode at 300 dpi; Qt settings and caches were isolated. Each folder contains PNG, PDF, SVG, data and a status-led legend. These deliberately preserve current violations and are **not** compliant final scientific figures.

| Existing function | Light / dark PNG | Measured and visually checked findings |
|---|---|---|
| `assay_panels.draw_bar_panel:1067` | [Light](graphstyle/thumbnails/before_v1/assay_conditions_light/before.png) / [dark](graphstyle/thumbnails/before_v1/assay_conditions_dark/before.png) | Three conditions × 25 synthetic wells still produce bars; the named control is blue. Plot area is 2.5476 in wide, versus 0.66 in for three slots. Bars and points use 0.6 alpha; error bars remain opaque. Dark navy points and black error bars have weak contrast. No canvas clipping was seen. |
| `plaque._colony_size_figure:2035` | [Light](graphstyle/thumbnails/before_v1/colony_distribution_light/before.png) / [dark](graphstyle/thumbnails/before_v1/colony_distribution_dark/before.png) | Thirty synthetic diameters produce an opaque histogram and an unexplained red median line. Plot area is 4.4392 × 2.6760 in, not square. Text adapts to both themes. No canvas clipping was seen. |
| `division_prediction.division_prediction_figure:711` | [Light](graphstyle/thumbnails/before_v1/division_prediction_light/before.png) / [dark](graphstyle/thumbnails/before_v1/division_prediction_dark/before.png) | Each five-condition axis is 2.4633 in wide rather than 1.10 in. Artist alpha values include 0.3, 0.6 and 1. The dark render has black title/ticks/labels on a dark canvas, white axes interiors and pale error bars on white. The age-only control is purple. AUROC/AUPRC are expanded in the audit legend, not by the production figure. |

The [render measurements](graphstyle/thumbnails/before_v1/measurements.json) retain the measured dimensions, artist properties, checker findings and PNG hashes. The unchanged reference `check_figure` returns **zero findings** for the assay and colony plots despite the visible/quantified violations, and ten nonblack-error-bar findings for each division figure. Its geometry checks depend on metadata registered by its own constructors. This demonstrates why “checker passed” cannot certify arbitrary spaCR figures. All six PNGs were opened and inspected; no whole-application or native-platform visual acceptance is claimed.

## One shared owner and its API

Extend **`spacr.figures.style`**. Do not import the copied reference helper into application modules: it forces the Agg backend, updates global rcParams and searches workstation-specific font paths. Keep `spacr.figure_font` for bundled fonts, `spacr.figure_style` for preference/export-color adapters, `graph_types` for graph-choice policy, and `plot.save_figure`/`figure_sink` for persistence and delivery. These adapters must consume the same resolved specification rather than hold independent policy constants.

Proposed API inside that existing owner; these are not implemented functions:

```python
resolve_style(target, kind, *, conditions=None, sample_counts=None,
              control=None, identities=None, comparison=None,
              legend=None, policy_version=None) -> StyleSpec
axes_geometry(spec, *, cell_count=None, panel_height=None) -> Geometry
finalize_figure(figure, spec, *, diagnostics=True) -> FigureDiagnostics
```

`StyleSpec` is immutable and carries graph kind, physical geometry, opacity, role colors, font faces/weights, experimental unit, quantity/unit labels, reference identity, abbreviation expansions and the caption. It uses the existing `figure_style`, `condition_plot`, `size_condition_axes`, `square_axes` and `figure_problems` primitives. The Matplotlib and Qt/pyqtgraph adapters consume it; Plotly creators use an adapter if present in the finalized inventory. It must not alter source arrays, p-values, analytical normalization, acceptance thresholds or experimental-unit aggregation. Declared display-only ranges, gamma and heatmap color normalization are permitted and recorded, with source arrays unchanged.

Finalization validates both local artist properties and semantic requirements that cannot be inferred from artists. A missing control/legend/range is an actionable diagnostic; the finalizer must not invent it. It checks effective opacity, slot/square-cell geometry and actual extents after layout. Save and live display call the same finalizer. `figures.scene` must preserve the source's four spines and alpha instead of hiding axes or dropping marker alpha. Export chrome remains a reversible operation.

Current manual preference values that violate mandatory rules must be reported explicitly and handled under the new policy; existing output must not be falsely relabeled compliant. Record the applied policy in bundle/recreation metadata. Any default or saved-format migration belongs to the implementation change and its existing schema process. No compatibility switch should preserve an incorrect scientific computation; this task changes presentation policy only.

For dark themes, test actual contrast on the final composited background. The before renders show that selecting foreground/background tokens alone is insufficient. Where black error bars or dark identity colors cannot be read, use a declared readable plot ground or another documented rendering treatment while preserving the mandated identities; do not let different modules make silent independent exceptions.

## Ratchet and acceptance proposal

1. Extend the existing `tests/test_figure_rules_spacr_graphs.py` and drawing/export cohorts. The tests already cover several shared primitives; their existence is further evidence of partial adoption.
2. Add a function-level AST route ratchet over creator/writer calls, resolving import aliases and excluding strings/docstrings. Every new creator must use the shared specification/finalizer, and every writer must use the shared sink. Keep a reviewed baseline of remaining legacy function IDs with reasons and owners; prohibit new exceptions and shrink it as modules migrate. This is not a blanket allowlist for all functions in an old file.
3. Test groups of 1, 20, 21 and 1,001 finite observations; two, three and six conditions; exact physical width/gap; grey identity controls; effective RGBA alpha; single-observation points; >20 switching; violin without points for the 1,001-observation case; 8-pt minimum and bundled font weights. The ratchet must check semantics/geometry, not merely the presence of a helper name.
4. Compare Matplotlib and Qt scene metadata for boxed line/marker, scatter, categorical and missing-cell heatmap cases. Check pens/brushes and physical dimensions; do not demand pixel identity across different antialiasing engines. Test light/dark export and display on their final backgrounds.
5. Exercise ordinary saving and bundle recreation through the existing paths. Preserve source arrays and statistical results exactly. Check stated units, reference identity, caption/status, image ranges/gamma/scale and required output formats for declared paper deliverables. Test manual preferences and pipeline isolation separately.
6. Update contradictory old style assertions deliberately. Existing tests that require opaque marks are not evidence that opacity meets this authority. Run the smallest affected cohort per migrated module and retain passing results; no whole-suite or docs build is needed for this audit.

Implementation order: shared style specification and validation; renderer/writer parity; routine plotting/Measure/OPS/GUI routes; assay/model/specialized plots; then lower-exposure legacy creators. Work Codex owns that implementation. This audit's completion requires the published inventory, copied authority and before renders, not changes to the application.

## Validation and limits

The static inventory ran under scheduler job 1527, P2, one CPU thread, 2 GiB RAM, zero GPU allocation: 43.16 s; no parse failures. The single six-image render job 1530 used the same limits and completed in 19.35 s. Assembly job 1539 reconciled the seven review families in 3.26 s under the same limits; all candidates, source coordinates, main links and six PNG hashes passed. Application source stayed unchanged. Chrome and graphics processes were not treated as compute jobs; the verified shared scheduler gates GPU measurements only for declared GPU work.

No scientific benchmarks, model training, production data, numerical acceptance tests, full test suite, docs build or GPU work was run. Source-derived findings are distinguished from the six rendered cases. Layout, dynamic paths and platform-specific behavior outside those cases remain unmeasured; the proposed ratchet and later module changes are responsible for their acceptance.
