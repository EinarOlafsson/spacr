================================================================================
THE README FRONT PAGE: ONE MODULE GRID, A MODEL ZOO, AND ONE DOWNLOAD COMMAND
================================================================================

Status:    parts 1 and 2 done 2026-09-02; part 3 in progress.
Numbering: FILED AS 359, RENUMBERED TO 362 the same day. The other
           session had already taken 358 and 359 (09:18 and 10:08
           against 10:34 and 11:04), and two files sharing a number
           is worse than a gap -- the index lists both and neither
           can be referred to. Commits 5498a16a1 and e62fc3787 say
           "359" and cannot be rewritten; they mean this file.
Requested: 2026-09-02, in four messages:
  (a) "in the readme the app module tiles, make them all the same size and
      present them on an evenly spaced grid with 6 modules per row. with the
      current module number generating a grid with 4 rows the last with 3
      modules."
  (b) "remove the titles and arrows for the modulse and the title should be
      spaCR modules"
  (c) "add the uploaded modules in the current model zoo to readme in a model
      zoo section"
  (d) "and there should be a command to downlaode all test data at once with an
      explination in the readme. this command should allow the user to
      downloade only a section of the squensing data."

--------------------------------------------------------------------------------
PART 1 -- ONE GRID OF IDENTICAL TILES  [DONE 2026-09-02]
--------------------------------------------------------------------------------

(a) and (b) were ONE change, and that is the thing worth recording. The
modules were drawn as two different objects: a six-wide pipeline STRIP
joined by arrow glyphs, then three named bands ("Data", "Tools",
"Assays") of smaller tiles below it. That made three problems which were
all the same problem:

  * the two kinds of tile COULD NOT be the same size. The strip had to
    fit six buttons AND five arrows into the width the bands used for six
    buttons, so the arrows came out of the buttons -- 14.5% against an
    effective 15.5%. A comment in the generator explained this at length
    and called the difference deliberate.
  * the bands were 6, 5 and 4 wide, so three of the four rows ended short
    and the thing never read as a grid.
  * the band titles restated the Home screen's own grouping, and went
    stale every time Home was restructured.

Deleting the arrows is what removed the width difference, which is what
let every tile become one size, which is what made an even grid possible.
So: 21 modules, six per row, four rows, the last holding three. The six
pipeline modules lead the grid -- the arrows used to claim they were a
sequence, and position claims it now.

  WHAT WAS DELETED, and must not come back: `render_pipeline_tile`,
  `render_app_tile`, `render_pipeline_arrow`, `_app_column`, and the
  constants APP_COLUMNS, APP_COLUMN_STEP, APP_DISPLAY_PERCENT,
  PIPELINE_DISPLAY_PERCENT, ARROW_DISPLAY_PERCENT, ARROW_CANVAS_*,
  APP_TILE_SIZE and APP_TILE_PADDING. Two ways to ask for a tile is how
  the two sizes existed; there is one, `render_module_tile`. The test
  asserts every one of those names is GONE, because the sizes come back
  the moment there are two paths again.

  `arrow.png` is deleted from both `spacr/resources/icons/workflow/` and
  `docs/source/_static/workflow/`. An unreferenced PNG left in a resource
  tree is what a later change quietly starts using.

  The heading is "spaCR modules", translated into all nine locales in
  `REVIEWED_README_HEADINGS`. "spaCR" is not translated in any of them.

--------------------------------------------------------------------------------
PART 2 -- A MODEL ZOO SECTION  [DONE 2026-09-02]
--------------------------------------------------------------------------------

`spacr.model_zoo.catalogue(remote=True)` publishes three trained models,
each carrying `trained_on`, `trained_by`, `notes` and a `sha256`. None of
it was in the README, so a reader had no way to learn the models exist
without opening the GUI. There is now a "Model zoo" section under `Data`,
generated between `.. spacr-model-zoo-begin/end` markers.

  GENERATED, NOT TYPED. The catalogue is a literal in `spacr/model_zoo.py`
  and the README section must be written from it by
  `packaging/generate_readme_visuals.py`, between markers, the way the
  module grid and the hardware table already are. A hand-written table is
  a second copy that goes stale -- which is the exact failure this
  instruction's part 1 was cleaning up.

  SAY WHAT THE NOTES SAY. Each entry's `notes` carry the honest limits
  ("accuracy falls sharply above IoU 0.8 -- suited to counting and area
  rather than precise morphometry"). Those belong in the README. A model
  table that prints only the good number is the kind of claim 316 exists
  to prevent.

--------------------------------------------------------------------------------
PART 3 -- ONE COMMAND TO FETCH THE TEST DATA  [IN PROGRESS]
--------------------------------------------------------------------------------

Downloading example data means opening the GUI and pressing a button per
module. There is no headless route, which makes it useless on a cluster
and awkward to document.

`spacr-download` fetches the example data in one call. The pieces already
exist: `spacr/qt/hf_download.py` holds the three example repos and their
archives, and `spacr/screen_data.py` already expresses "only a section"
-- eight archives, four per-plate `measurements` (~0.5 GB each) and four
per-plate `crops` (~8 GB each), with sizes known BEFORE download and an
`is_present` check per piece.

  THE 33 GB MUST NOT BE THE DEFAULT. The screen is 33 GB. A bare
  `spacr-download` must fetch the small example sets and say how to ask
  for the screen; it must never start 33 GB because someone typed the
  command to see what it did.

  NO QT IN THE CLI. `spacr/qt/hf_download.py` imports PySide6 at module
  scope. A command that cannot run without a display is not a headless
  route, so the Qt-free half has to be reachable without importing that
  module.

--------------------------------------------------------------------------------
HOW IT WILL BE CHECKED
--------------------------------------------------------------------------------

* Part 1: `tests/test_readme_presentation.py` and
  `tests/test_the_readme_describes_the_build_that_ships.py` pass, and the
  grid renders 6/6/6/3 on the real GitHub page.
* Part 2: the section is regenerated from the catalogue, and a test fails
  if a published model is missing from it.
* Part 3: `spacr-download --help` works with no display and no torch;
  `--list` prints every piece with its size without downloading; a bare
  run does not start the screen; tests mock the network entirely.
