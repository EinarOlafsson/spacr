321 — THE API AND README STILL SHOW THE OLD MODULE STRUCTURE
============================================================

ASKED FOR (2026-08-31, verbatim)
--------------------------------
"fix the API and README to reflect the new module structure in spacr. the
buttons should be discribed in the API, just not as their own modules i
guess, and the README should only show buttons for the current modules."

WHAT CHANGED UNDERNEATH THEM
----------------------------
Instruction 318 cut Home from thirty tiles across seven sections to
TWENTY-ONE across four, and moved the rest to fold buttons or Help:

    Core   (6)  Mask, Measure, Annotate, Classify, Map Barcodes,
                Regression
    Data   (6)  Import, Run Compare, Experiment Design, Power / Design,
                Dose-Response, QC
    Tools  (5)  Make Masks, Align & Stitch, Image UMAP, Gate Editor,
                Graph Builder
    Assays (4)  Plaque, Recruitment, Invasion, Replication

Explore, Results & QC, Design and Segmentation models are gone as places.
`spacr.qt.app.tiled_apps()` is the live answer to "what has a tile", and
`SECTION_TILE_ORDER` is the order they draw in.

TWO SURFACES, TWO DIFFERENT RULES. Do not treat them alike.
-----------------------------------------------------------

README — ONLY THE TWENTY-ONE. It carries 38 `|App_*|` image
substitutions, laid out in rows under the OLD section headings (see the
`|App_...|` block and the rows that use it). Every module that is now a
button or a Help entry still appears there as though it were a place to
start, which is precisely what the restructure was for.

  * Show the 21 tiled modules, grouped and ordered exactly as
    `SECTION_TILE_ORDER` has them, under the four current headings.
  * Remove the badge rows for folded and Help-only modules. Their
    substitution definitions can go with them -- 17 of the 38 are now
    unused, and an unused substitution in RST is silent.
  * DERIVE the check, do not eyeball it. A test that reads
    `tiled_apps()` and asserts the README shows those and only those is
    the only version that survives the next restructure. This is the
    third time these badges have gone stale.
  * The nine translated READMEs under `docs/i18n/readme/` carry the same
    block and must move together, or the English one becomes the only
    correct page.

API — EVERY MODULE STAYS, INCLUDING THE FOLDED ONES. "The buttons should
be described in the API, just not as their own modules." A folded module
still has a screen, a factory, a key, a settings model and a headless
entry point; its API page is what a scripting user reads, and deleting it
would remove documentation for working code.

  * Keep the page. Change how it is REACHED and how it is described: it
    is documented as a button on its host, not as a top-level module.
  * Each folded module's page should say which host opens it and how --
    "opened from the Regression masthead", not "open Investigate Hit".
  * The API index should group by the four live sections, so the
    navigation matches what the user sees on Home. The folded ones belong
    under their HOST's section, which is already where
    `app_catalog`/`APPS` file them -- the section of a folded module is
    what says which host it sits behind.
  * `spacr-run --list` and the headless entry points are unaffected: a
    fold changes the door, not the API.

DONE MEANS
----------
* The README's badge set equals `tiled_apps()`, in `SECTION_TILE_ORDER`
  order, under the four current headings -- asserted by a test, not read
  by eye.
* No unused `|App_*|` substitution definitions remain.
* The nine translated READMEs agree with the English one.
* Every module still has an API page, folded or not.
* A folded module's page names its host and how to open it.
* Nothing in either surface names Explore, Results & QC, Design or
  Segmentation models as a place.

WATCH OUT
---------
The README badge block and the i18n README pipeline are covered by
`tools/build_documentation_i18n.py` and the caption ratchet. Changing
user-facing prose here MOVES THE RATCHET -- coordinate before re-pinning,
or the pin lands on a tree that is still changing.
