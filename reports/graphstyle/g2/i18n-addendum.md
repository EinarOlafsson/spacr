# G2a caption and translation-identity addendum

**2026-11-23.** Source commit `167b08514b29d0132c1fef99dcea2a312027841a` follows immutable G2a FINAL `793a87027870d4693146b4ad331eca49edc1ee9e`. This bounded source correction supplements the [retained validation index](validation.md); it does not replace earlier results or complete the pending translation integration.

The translation preflight exposed valid captions rejected as copied English or protected syntax. The source change adds exactly 15 locale-specific identical-spelling entries to `MANUAL_TRANSLATIONS`, preserving existing locale maps. It registers only the full scientific template `{well}: n={count}; BC={coefficient}.` in `_IDENTITY_TEXT`. BC is defined by the separate translated caption “BC means bimodality coefficient.” The template now follows the normal source-owned identity route and is excluded from external review admission; it is not an untranslated-prose exception or a new external record.

The only application title change is `True Mask` → `Ground-truth mask` in the nested Cellpose result renderer at `spacr/submodules.py:684`. That panel displays the supplied ground-truth label array, separately from predicted masks. Clearer English removes the collision with protected Python `True`. Boolean protection, syntax/script predicates, model calls, data arrays, colormaps and normalization are unchanged.

## Bounded validation

| Evidence | Result and limits |
|---|---|
| Source tests, scheduled job 1701 | Three tests passed: 24 caption/locale pairs through the unchanged scoped runtime loader, nine BC identity decisions through the authoritative normal lookup, negative copied-English/locale/Boolean/format-field controls, and the existing Cellpose image-plane/panel-size regression. No global source scan was run by these tests. |
| Original caption observer, job 1702 | Failed with exit 1. Its geometry observer assumed the post-export canvas supplied `get_renderer`; that assumption was invalid after PDF export. The failed helper, receipts and partial artifacts remain local audit evidence and are not selected as acceptance images. |
| Caption observer v2, job 1703 | Passed in light and dark themes using the actual nested renderer and retained five displayed input planes. The observer temporarily attaches an Agg canvas only when required for measurement, then restores the original canvas. No production writer or source code changed between these observer runs. |
| Successor binding S0 | Passed one normal per-file `submodules.py` scan: `Ground-truth mask` added, `True Mask` removed, and BC excluded through its identity route. It proved exactly 15 policy triples, prior-map preservation, unchanged predicate function bodies and the sole application title change. Exact source/target/dependency checks retain 3,469 previously passing target decisions; affected decisions and final translation loaders remain separate. |

The [v2 report](evidence/cellpose-caption-i18n-v2/report.json), SHA256 `e0b879434b798ed21c299f7473324d62990ce1c7648f22ce81c2060b7c54ecef`, binds source file SHA256 `642d37657d4c2871ce025b23c2f733946cbeca528b3a15a4a63ccb6bbf05fcf5` and the [retained baseline report](evidence/submodules-after-width-roles/report.json). Both themes preserve input/artist data, masks, RNG state, normalization and categorical LUTs. The receipt records true 8 pt titles, non-overlapping title bounds, requested saved backgrounds and page dimensions within the existing paper limits. The assay reviewer inspected all four v2 PNGs individually. PDF geometry is checked; no independent PDF raster review is claimed here.

| Theme | Retained baseline producer PNG | Selected v2 producer PNG | Selected v2 ordinary exports |
|---|---|---|---|
| Light | [Before](evidence/submodules-after-width-roles/light/test_images/cellpose_result_000.png) | [After](evidence/cellpose-caption-i18n-v2/light/test_images/cellpose_result_000.png) | [PNG](evidence/cellpose-caption-i18n-v2/light/test_images/figure.png), [PDF](evidence/cellpose-caption-i18n-v2/light/test_images/figure.pdf) |
| Dark | [Before](evidence/submodules-after-width-roles/dark/test_images/cellpose_result_000.png) | [After](evidence/cellpose-caption-i18n-v2/dark/test_images/cellpose_result_000.png) | [PNG](evidence/cellpose-caption-i18n-v2/dark/test_images/figure.png), [PDF](evidence/cellpose-caption-i18n-v2/dark/test_images/figure.pdf) |

Local-only evidence remains under `/mnt/wd4tb/worktrees/codex-graphstyle-g2/.codex-audit/`: source-test receipts in `checks/g2a-i18n-tests/`, failed observer receipts in `checks/g2a-i18n-caption/`, v2 receipts in `checks/g2a-i18n-caption-v2/`, and successor binding in `tr-g2a/S0/report.json` (SHA256 `68cd00a15ddda1c3d9840e56ed81d4831ccf88a446a77a0d1e344650dd16ede5`). These local paths are not presented as published receipt links.

Translation-record integration and its required normal scoped loaders are pending. These English synthetic render checks do not establish multilingual layout, native-platform or tutorial acceptance. No model inference, segmentation, new scientific measurement or full application run was performed for this addendum.
