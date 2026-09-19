# Notes from `spacr/qt/help_index.py`

Written by hand on 2026-09-19 for instruction 422, not lifted by
`tools/extract_source_notes.py`: the module was written without comments, so
its reasons were in docstrings from the start. This file holds the ones that
are about the DESIGN rather than about a function — the numbers, and the
things that were tried and measured before the shape settled.

## Why the index is 11,253 rows and a second to build

| kind | rows | source |
|---|---|---|
| `api` | 10,326 | `API_ENTRIES` in the generated `help_api_index` |
| `setting` | 767 | `resolve_default_settings` × 39 visible modules |
| `preference` | 121 | `PREFERENCE_ENTRIES` |
| `module` | 39 | `spacr.qt.app.APPS` |

Measured on this tree, 2026-09-19: `build_index()` takes 0.8–1.1 s, almost
all of it in the settings provider, which imports every module's defaults.
That is why the field builds it on a worker and says "Building the index…"
until it lands, and why importing `help_index` itself must stay free — the
providers import their registries inside themselves, and
`test_importing_the_index_does_not_import_qt` is what keeps that true.

The 767 setting rows are (module, key) pairs, not 767 distinct settings:
one setting that four modules expose is four rows. That is the maintainer's
answer to the open question the instruction left — "one result row per module
where a setting appears" — and it is why the subtitle always names the module
AND the category.

## The ranking, and the two measurements that shaped it

Five bands, highest first: the whole name, a word of the name, a prefix of the
whole name, a prefix of a word, the name containing the term, a fuzzy
subsequence of the last component, and finally the description.

**A description hit must never outrank a name hit.** Typing `cell_diameter`
with description hits ranked equally returns the twenty settings whose help
text mentions the diameter before the setting itself.

**The fuzzy band needed a ceiling.** Without one, `clldiam` matched
`spacr.qt.folder_metadata.save_filename_map` — every letter present, in order,
spread over forty characters — and ranked it above `cell_diameter`. A long
enough name contains almost any short sequence of letters. The span is capped
at three times the length of the term, and the subsequence is matched against
the LAST dotted component rather than the whole symbol, so `spacr.qt.…` cannot
contribute letters to a match on a function's name.

**A word of the name had to be its own band.** Before it, typing `umap` put
`umap_canvas_width` and eight API symbols above the Image UMAP module, because
`Image UMAP` starts with neither. Splitting a title on dots, underscores and
spaces alike is what lets one rule serve `spacr.qt.screens.mask`,
`cell_diameter` and `Image UMAP`.

## Why there is a per-kind cap

There are 10,326 API symbols against 39 modules. A plain top-40 by score is an
API list with the occasional module in it, and the instruction is explicit
that "one query can return several kinds at once". `PER_KIND_LIMIT = 8` is
what keeps the small kinds on screen. `search(..., per_kind=None)` lifts it,
which is what a test asking "is it in the index at all" wants.

## The duplicate that the shape removed

An earlier version emitted API rows from two providers: `api_entries` for the
published symbols and a second pass in `setting_entries` for the functions
that read each setting. `spacr.io.preprocess_img_data` then appeared eleven
times in one result list, once per setting it reads. Folding the consumer
information INTO the API row — one row per symbol, its subtitle saying which
settings it reads — is both shorter and the thing the instruction asked for,
since the row is still findable by the name of any setting it consumes.
