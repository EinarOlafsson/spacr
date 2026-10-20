# README text-panel links: implementation preflight

Bound published source: `a744a9fda2b11835f3290cdaf7cf88f5a97f400f`.
This is preparation, not completion or publication. No source, generator,
rendered artifact, test, site or Git mutation was made.

`inventory.json` maps every actual English README picture to its exact href,
phone/desktop assets, originating structured blocks and destination source
opening. Its 83 small source/destination/score-asset bindings are byte-identical
to the published commit. `panel-map.md` gives a row for every picture. Normal
prose SVG inspection reads only an 8192-character metadata prefix, not embedded
font payloads; the eighteen model score SVGs are individually bounded below
96 KiB. Destination opening observations are lexical source inspections, not
rendered/browser acceptance. The audit uses pure source parsing, never calls
compose/build/check/render or loads scientific/UI libraries.

## Current acceptance inventory

| Actual English output | Count | Required acceptance |
|---|---:|---|
| Prose/heading panels from English YAML | 48 | Each entire responsive picture links to readable text, whose first substantive content contains its complete exact source headings/paragraphs/list content. |
| Generated hardware/workflow/model-version caption panels | 7 | Same contract, including hardware table cells and legend; generation uses the owning structured table/caption source. |
| List-row panels (resources and citing papers) | 10 | Each row links to its own readable text first; retain original resource/paper URL as a continuation. |
| Model score heading | 1 | Linked model index starts with the heading's text, then existing index content. |
| Default model score charts | 8 | Existing model page starts with complete accessible chart text/values, before its existing explanations or repeated chart. |
| Navigation/link buttons | 18 | Already linked. Preserve their navigation and label accessibility; if the implementation applies a literal all-SVG text-first contract to these controls, add label-first continuations deliberately. |
| Deck cover | 1 | Screenshot/navigation image, separately scoped from structured text panels; preserve current deck destination. |

There are 65 prose/table/list panels: 46 are unlinked and 19 already linked.
All ten READMEs contain the same 92 text pictures and 46 unlinked pictures
(plus the shared deck cover). The JSON enumerates all 93 English images;
the table above is a category summary, not a substitute for per-panel checks.

### Existing targets requiring specific treatment

The 19 linked prose/list panels have these actual routes:

| Panel(s) | Current route | Minimal treatment |
|---|---|---|
| `spacr`, `other-resources-2-3` | `docs/source/features.rst` | Shared page cannot start with both distinct texts. Use owned per-panel text pages, followed by this original guide URL. |
| `install-spacr-2` | `docs/source/installer_guide.rst#container-images` | A fragment does not make the whole page start with this panel. Use a panel text continuation or an explicitly generated prefix for its destination section, depending on final first-content interpretation. |
| `pypi-installation-3` | `docs/source/installer_guide.rst` | Also shared with a navigation button and all locales. Prefer a per-panel continuation over changing the guide's global introduction. |
| `other-resources-2-2` | `docs/source/python_api.rst` | Unique English prose claim; can receive an owned generated first-content prefix if guide ownership permits. Per-language continuations avoid replacing the English opening for nine other languages. |
| `other-resources-2-5`, `language-translation` | `docs/source/localization.rst` and `#contextual-help` | Same physical page has distinct panel texts. Preserve the guide and its fragment as detail routes after per-panel text. |
| `language-translation-2` | `docs/i18n/REVIEW_SCOPE_2026-09-04.md` | Unique English prose claim but shared by locales/historical review ownership. Use a per-panel continuation; do not rewrite historical evidence merely for the README. |
| `acknowledgments` | `docs/i18n/TRANSLATION_MODELS.md` | Unique English prose claim with shared locale use. A normal owned prefix is possible only with that page owner's agreement; otherwise continuation. |
| `tutorials`, `other-resources-2-1` | Hosted tutorials root | Two distinct texts and independent tutorial ownership. Keep original hosted route as detail link after each owned text continuation. |
| `other-resources-2-4` | Hosted API index | Keep published API destination as a continuation; do not edit frozen/generated API pages by hand. |
| `contributing-and-support`, `licensing` | GitHub issues and LICENSE page | Destination has independent or non-prose semantics; use text continuation and retain original issue/license link. |
| `other-work-citing-spacr-2-1` through `-5` | ASM, Cell Reports, bioRxiv and Research Square articles | Third-party article introductions are outside our control. Generate each citation's readable text first, then original article URL. |

Eight default model pages are already unique owned destinations and can be
updated directly by `tools/build_model_cards.py:render_card` (418); the model
index can be updated by `render_index` (502). Their current first visible
heading is the long model name or `spaCR model cards`, followed by an SVG,
not the full score-panel words. Do not hand-edit those Markdown files.

## Concrete generator seam and edit batches

1. Add readable-text outputs to `tools/build_readme_images.py:Output` (1464),
   with one deterministic route based on language and stable image ID. Proposed
   owned location: `docs/readme_text/<language>/<image-id>.md`. Keep these as
   Markdown pages readable on GitHub/PyPI repository links. If hosted pages are
   required too, use a normal generated `docs/source` include/copy and hidden
   toctree; files outside `docs/source` are not automatically Sphinx pages.
   No YAML prose changes are necessary merely to add these routes. English
   canonical text stays in `packaging/readme_source.yaml`; nine translated
   sources retain their same block sequence and reviewed wording.
2. Introduce a shared route/text-output resolver at `_panel_markup` (1628) and
   `_rows_markup` (1648). It receives language, image ID, exact structured
   blocks and original detail targets. Make zero/one/many-target panels all
   link their entire existing `<picture>` to readable text. Keep original
   inline detail URLs in the text and existing multi-link button rows. Rebase
   relative URLs from the new text page, rather than copying README-relative
   strings. Prefix headings must use the drawn source text, not `alt_text`,
   which adds heading punctuation. Do not prepend an extra audit or README
   panel title before the required text.
3. Use the same seam for `convert_generated` (1931): the seven generated
   panels must not be missed just because `compose` copies marker HTML.
   Hardware source is `_hardware_table` in `generate_readme_visuals.py`;
   workflow captions are `_workflow_markup_for_readme`/`_grid_markup`;
   model-version captions are `_readme_model_zoo_block` (914). The committed
   `generated-*.rst` sidecars record their exact current words. Generate
   corresponding readable text at the same time as the SVG/marker block,
   including every table heading, value, row and legend.
4. Extend `build` (2057) and `check` (2079) to own and validate the readable
   pages as well as images/README. Bind full structured text, route and
   continuation targets in their digest. Validate missing/stale text,
   unavailable anchors, duplicate page claims, leading article text parity,
   all languages and obsolete outputs confined to the generator's own
   directory. `_panel_images` (1614) currently omits `linked`/`row` from its
   digest. If adding link arrows changes pixels, bind these flags and bump
   `RENDERER_VERSION`; do not silently accept visually stale images.
5. For the nine model score pictures, share a structured accessible chart
   representation between `tools/build_score_cards.py:render_svg` (607) /
   `render_heading` (740) and `build_model_cards.render_card` / `render_index`.
   Read values from the same zoo records and existing re-score overlay, not
   copied alt text. `alt_text` (484) is a summary, and does not contain every
   metric, value or legend. Keep the existing score image/grid and detailed
   Markdown sections after the first substantive text.

The score-card phone and desktop labels differ: `fields` versus `held-out
fields`, and `stock` versus `stock cpsam_v2`. SVG text-element order is not a
defined visual reading order. Preserve both current visual layouts and put
complete accessible chart text/values first. If literal layout-specific
wording is necessary, responsive leading text variants on an owned page are
an implementation option. That rendering choice is not yet chosen,
implemented or accepted; it is not a prerequisite for asking user permission
during this preflight.

## Visual invariants and owning checks

Preserve BODY13.6/TITLE16, font weights300/400, PHONE358/DESKTOP830,
`(min-width: 1280px)` responsive source, embedded fonts, full alt/desc text,
teal-rim dark-grey surfaces, current blue score bars and
`generate_readme_visuals.SCORE_CARDS_PER_ROW=2`. Preserve source grouping,
word wrapping, real copyable code blocks and both picture variants. Adding
clickability does not justify changing the accepted card design.

Existing checks are `tests/test_readme_is_built_from_images.py`,
`tests/test_readme_presentation.py`, `tests/test_pypi_readme_images.py`,
`tests/test_model_cards_are_current.py` and
`tests/test_every_model_has_a_score_card.py`. The current image suite checks
source grouping, currency, alt/desc, all locales, 13.6/16 text, true layout
sizes and palette, but lacks every-panel href/leading-page text assertions.
Its test `test_a_single_link_makes_the_whole_image_a_link_and_several_become_buttons`
explicitly asserts that a multiple-link panel has no anchor. Replace that
obsolete expected behavior with the new readable-page link requirement,
while retaining assertions for every original detail button/link.

Add meaningful checks for each of the 65 actual text panels and nine model
score pictures, across all ten README outputs: reachable owned destination,
first substantive content parity, retained detail URLs, zero/one/many links,
multi-row/table text, fragments, third-party routes and shared-page claims.
Keep the eighteen navigation buttons and cover explicitly enumerated.
Check PyPI's `setup.py:pypi_readme` preserves the new links when `<picture>`
is converted to its phone image. Inspect the actual first article content
after Markdown/Sphinx rendering, not just file startswith or alt text.
Physical phone/desktop review and normal source-current output checks remain
future acceptance; no rendering/test outcome is claimed by this plan.

## Normal generation sequence after release/held integration

Re-inventory on final released nightly first. In an already authorized focused
worktree, edit the owning generator/output/check contract and tests. Preserve
unchanged English/translated prose. Run `python tools/build_model_cards.py`
only if the model page/score text seam changes, then
`python tools/build_readme_images.py --refresh-generated`. That refresh uses
`generate_readme_visuals.write_readme_blocks` without redrawing every module
PNG. Full `python packaging/generate_readme_visuals.py` is needed only when
its actual logo/tile/registry assets change. Run the respective `--check`
commands and the focused owning suites under the coordinated background
resource/test slot, then verify GitHub/PyPI and both layouts before publish.

The old Docker1.5.1.4 commands in canonical YAML are a separate release
metadata issue. Root's existing normal release metadata workflow should
update tags only after verified publication; do not mix that repair with
panel-route behavior or assert a new tag is already published.
