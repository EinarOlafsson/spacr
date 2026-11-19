# Figure-style parity between `spacr-run ops` and the Python driver (spec)

Task 5 of the 2026-11-19 WS offload. Status: **spec only, no code change.** The
fix would sit in `spacr/ops_decode.py` and possibly `spacr/cli.py`, and both are
on the do-not-touch list. The cause has not been reproduced either: the two
differing PNGs were not available on the WS.

## The symptom

The decode tuning-curve PNG, `threshold_tuning_<plate>_<well>.png`, has a
different title size depending on the entry point:
- `spacr-run ops`;
- the Python driver, which calls the same code from a script.

## What the code fixes, whatever the entry point

`spacr/ops_decode.py::write_tuning_curve` draws everything inside
`figures.style.figure_style("print")`, and pins the parts that would control
size:
- `Figure(figsize=(5.0, 3.8), dpi=150)`;
- `axis.set_title(..., fontsize=9)`, an explicit size, so `axes.titlesize` and
  every rcParams font-size setting are bypassed;
- `save_figure(..., fmt="png", dpi=150)`, an explicit DPI.

`plot.deliverable_dpi` only lowers the DPI for very large canvases, which never
applies at 5 x 3.8 in. The geometry is therefore identical on both paths. A title
that looks a different size can only come from:
- a different **font face** at the same point size, or
- a figure that did not go through this function at all, for example an older
  file left over from a pinned production commit.

## Ranked candidate causes

1. **Font fallback.** `figures.style.rc()` calls
   `figure_font.use_open_sans_for_figures()` and asks for the families
   `["Open Sans", "Helvetica", "Arial", "DejaVu Sans"]`.
   - `figure_font.bundled_faces()` returns `[]` when `font_dir()` is missing or
     unreadable. That failure is deliberately silent: "a figure drawn in the
     wrong font is a blemish".
   - Matplotlib then draws in DejaVu Sans, whose glyphs are visibly larger than
     Open Sans at the same 9 pt.
   - `font_dir()` resolves relative to the imported `spacr` package. So it can
     differ between the installed console script `spacr-run` and a driver that
     puts a worktree on `PYTHONPATH`, for example one without the font files or
     with an unhydrated LFS checkout.
   - The result is also cached per process (`_registered` / `_resolved`), so
     whichever path ran first decides for the whole process.
2. **Preference overrides.** `figures.style.user_overrides()` reads
   `spacr.qt.preferences.get_figure_style*()`. Any exception returns `{}`.
   - The preference store, through QSettings, HOME or XDG, can be readable on one
     path and not on the other. For example, the run journal swaps in a
     temporary `.spacr` directory under tests.
   - An override of `font.family` changes the face, as in cause 1.
   - On the WS driver path both preference layers are empty, so this needs a
     user who has changed figure preferences.
3. **Different figures.** The two PNGs came from different commits: production
   pinned at 4792e9d57e or 3edbf157ed, against nightly. Check before anything
   else.

On the WS, in a plain Python process (`spacr-wsws` worktree on 4ef66cb340),
`rc("print")` gives `font.size` 6.2, `axes.titlesize` 7.0 and
`font.sans-serif` `['Open Sans', 'Helvetica', 'Arial', 'DejaVu Sans']`, and
`user_overrides()` returns `{}`.

## Reproduce first (Home has the two PNGs)

1. Confirm both PNGs record the same commit; the run journal stores it.
2. For each entry point, print the title face that was actually used:

   ```python
   fig = ...  # the figure inside write_tuning_curve, or any figure under figure_style("print")
   title = fig.axes[0].title
   print(title.get_fontname(), title.get_size())
   from spacr import figure_font
   print(figure_font.font_dir(), figure_font.bundled_faces()[:2], figure_font.use_open_sans_for_figures())
   ```

   A difference in `get_fontname()` (Open Sans against DejaVu Sans) confirms
   cause 1.

## Proposed fix, once confirmed (outside the frozen files)

- `spacr/figure_font.py`:
  - make a failed Open Sans resolution visible once per process, as a
    `logging.warning` that names `font_dir()`;
  - add `require_figure_font()` for tests, which raises when the bundled faces
    are missing.
- `spacr/figures/style.py::rc()`: keep the family list, but put the resolved face
  first, so both paths render with whatever actually resolved, never a silent
  mix.
- Packaging: check that the font files ship in the wheel and are present in
  every worktree (`git check-attr filter` for LFS).
- No change to `ops_decode.py` or `cli.py`; they stay frozen.

## Acceptance test, to add with the fix

`tests/test_figure_style_entry_point_parity.py`:
- Render one small figure with `figure_style("print")` and
  `set_title("x", fontsize=9)`:
  - in-process, through the driver path;
  - in a subprocess started the way the console script starts (`python -c
    "from spacr.cli import main; ..."` is fine, since `cli.py` is not edited).
- Compare `title.get_fontname()`, `title.get_size()` and the PNG pixel size.
- Add a case where `font_dir()` is monkeypatched to an empty folder: it must
  give the logged warning, not a silent change of face.

## User-facing strings

None in this commit. The proposed fix adds one warning, which the master
translates.
