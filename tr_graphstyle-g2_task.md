# FINAL G2a English source handoff

Source code revision: `d6c4e6a9615c55f91276c0a88035a914bf2c9784` on branch `ws-offload-codex-home-graphstyle-g2a`. The published branch may have a child commit containing this handoff and its evidence.
Proposed translation branch: `ws-offload-codex-tr-graphstyle-g2a`.
Manifest: `tr_graphstyle-g2_sources.json` (task identity `graphstyle-g2a`).

FINAL applies to this G2a source-branch handoff. It does not certify completion of the original 204-entry G2 backlog, G1 replay/integration, STEP_B, tutorials or required WS acceptance.
Comparison source: immutable G1 CHECKPOINT `9510143823dcc1b984ecc522058a1e3f15e985d8`, not G1 FINAL. The successful extraction used `a18cc405b6d1099f15343dbe7121b784a6d2a976` plus the exact later scanner patch; all selected source and tool blobs now match the final commit by SHA256. No API/runtime extraction was repeated for this commit binding.

## Translation assignment

Prepare source-bound reviewed records for sv, de, es, zh_CN, pt, hi, ko, is and fr. Use this exact review_kind template, substituting the actual review date: 'Codex (OpenAI) AI translation and review on <actual review date> of Measure and quality-control graph captions and API documentation. No human or native-speaker signoff.'
Start from fresh current spacr-translations main, fast-forward before publication, and run the normal scoped loaders immediately before push, as required by v176. Do not publish against a stale translation base.
Use the shipped source/context, protected-syntax, placeholder, label and target predicates, followed by normal scoped API and runtime reviewed-record loaders for all nine locales. Inspect the actual translation base and preserve exact valid same-source/context targets and their existing review evidence. Do not invent reuse, blanket retire by label, or weaken a predicate to admit a target.
Keep code, RST roles, placeholders, units, method names and quantitative meanings exact; translate surrounding scientific prose naturally. Preserve layout-only/literal blocks without creating unnecessary translations.
This is a reviewed-record task. Catalog builds, source PO writes, translation pins, source implementation changes and numerical/scientific reruns are outside it.

## Public API scope

The normal scoped collector admits 26 changed owners: 12 added and 14 revised. Their complete current layouts contain 186 blocks, of which 184 require translation. The two ordinary literal blocks remain in the manifest for correct reconstruction. Exact prior/current bodies, layouts, hashes, context strings and matching prior labels are retained; review inventory still determines actual new/moved/reused records.
Protected documentation/API inventory consumers were not edited or broadly regenerated. Their final source-arrival integration remains with the source owner; this handoff does not substitute for those tests.

- `spacr.figures.style.apply_colourbar_theme`: added; 5 current blocks.
- `spacr.figures.style.apply_embedding_style`: added; 18 current blocks.
- `spacr.figures.style.ensure_identity_contrast`: added; 7 current blocks.
- `spacr.figures.style.ensure_reference_contrast`: added; 7 current blocks.
- `spacr.figures.style.external_legend`: added; 15 current blocks.
- `spacr.figures.style.image_report_css`: added; 5 current blocks.
- `spacr.figures.style.native_histogram_geometry`: added; 2 current blocks.
- `spacr.figures.style.native_plot_font`: added; 2 current blocks.
- `spacr.figures.style.paint_native_histogram`: added; 2 current blocks.
- `spacr.figures.style.resolve_plot_theme`: added; 5 current blocks.
- `spacr.figures.style.styled_reference_line`: added; 9 current blocks.
- `spacr.qt.screens.align.TileLayoutWidget.resizeEvent`: added; 1 current blocks.
- `spacr.qt.widgets.segmentation_views.render_cellprob`: revised; 5 current blocks.
- `spacr.qt.widgets.segmentation_views.render_flows`: revised; 4 current blocks.
- `spacr.qt.widgets.segmentation_views.render_overlay`: revised; 7 current blocks.
- `spacr.qt.widgets.segmentation_views.to_rgb8`: revised; 4 current blocks.
- `spacr.submodules.interpret_vision_model.create_extended_radar_plot`: revised; 4 current blocks.
- `spacr.utils.GradCAMGenerator.plot_activation_grid`: revised; 8 current blocks.
- `spacr.utils.SaliencyMapGenerator.plot_activation_grid`: revised; 8 current blocks.
- `spacr.utils.generate_colors`: revised; 3 current blocks.
- `spacr.utils.plot_clusters`: revised; 16 current blocks.
- `spacr.utils.plot_clusters_grid`: revised; 11 current blocks.
- `spacr.utils.plot_embedding`: revised; 23 current blocks.
- `spacr.utils.plot_grid`: revised; 8 current blocks.
- `spacr.utils.process_masks.plot_clusters`: revised; 1 current blocks.
- `spacr.utils.show_cam_on_image`: revised; 6 current blocks.

## Runtime scope and deferred admission

There are 400 added literal callsites in the scoped comparison. Of these, 379 callsites represent 308 distinct source strings with normal per-file membership and a proven inventory path. Those 308 strings are candidate reviewed-record inputs, subject to normal current source binding, exact-source reuse and compact-catalog ownership checks; this is not a claim of 308 missing records.
The overall per-file delta contains 317 source strings newly admitted within the compared files. This scoped figure is not global novelty: 299 of the admitted callsite sources are in that scoped-new set, while nine already existed within the compared scope. No whole-repository novelty scan or blanket retirement authorization is implied.
The remaining 21 callsites in `spacr/figures/style.py` represent 20 unique strings. Their per-file literal membership is known, but the shipped runtime inventory path is the separate G1 v181 dependency. They remain explicitly listed in `runtime_admission_gaps` and `translation_assignment.deferred_runtime_sources`; do not translate or publish records for these unadmitted sources yet. Rebind only that admission after its source-owner fix.
The successful targeted scanner refresh called the unchanged normal per-file scanner only for Measure and Utils, once each. It restored exactly 115 callsites / 97 per-file source additions while preserving prior admitted sources. No API, global runtime or other-file scan was repeated.

## Provenance and limits

Retained compact extraction JSON SHA256: `ce4926048c31008cb7ad42eb46716b6e8ccd5a7a5845056d524bf1e60ae1f21e`. The full source-only evidence remains identified by SHA256 in `retained_evidence`; its unchanged API bodies, raw internal docs and old runtime inventories are not duplicated in this compact assignment.
Normal API callable bodies and policies were unchanged during source-only loading; only the unrelated eager reviewed-README initializer was omitted in memory, with guards against review/cache access and use of omitted README inputs. No reviewed target evidence was replaced by that import boundary.
Private/internal docstrings remain excluded; retained raw internal changes do not authorize translation records. Nonliteral and ambiguous duplicate callsite evidence remains bounded by the original report. No guide/README inventory, application import, model call or numerical recomputation ran.
The validated before/after evidence index remains separate from translation admission and does not turn unresolved route, replay or platform dependencies into completed work.
