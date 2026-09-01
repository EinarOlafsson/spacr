329 — README, LOGOS, CHROME AND BACKDROP POLISH (2026-09-01)
============================================================

Status:    DONE, 2026-09-01. Every item below is implemented, tested and
           pushed to nightly. Filed after the fact at the maintainer's
           request so the batch is on the record.

ASKED FOR (2026-09-01, several messages)
----------------------------------------
A run of presentation fixes, plus one report that turned out to be a
real bug.

1.  README: state numbers, drop the fluff.
2.  The spaCR logo is invisible on GitHub in light mode.
3.  Core module buttons should be the same size as the other module
    buttons.
4.  Rename "What each configuration accelerates" to "Hardware support".
5.  Remove the "both drive the same modules / GPU where supported"
    sentence.
6.  Rename the long source heading to "Install from source (light)".
7.  Remove the "Objects and settings" section.
8.  "Choose the next page by what you want to do:" becomes a bold
    "Other resources".
9.  Section order: Hardware support, installers, PyPI, conda-forge,
    source, source (light), command-line entry points, "Core workflow"
    (renamed from "What you can do"), the modules, then the rest.
10. Setup: centre the capability table, keep it off the Back button,
    and show no "1 of 7" on the first slide.
11. The x / square / minus marks do not always match their container.
12. Backdrop defaults: orbit fold, supersampling 2, backend auto,
    scale 0.5, speed 1.
13. The API logo has the same white-on-white problem; give it a dark
    teal rounded square.
14. Module tooltips should only be shown at the bottom of the screen.

WHAT WAS DONE
-------------
1.  README prose cut from 1798 words to 1542 across ten sections. The
    install section reads "Full clone: 427 MB. Core clone: 76 MB." The
    previous version quoted 427 MB and 1.3 GB in one sentence without
    saying they measure different things -- tracked files against the
    checkout including .git.

2.  BOTH LOGOS now carry their own background, and the reason the
    obvious fix is unavailable is the same in both places: GitHub
    renders README.rst through docutils and passes no raw HTML, so
    there is no <picture> element to swap on prefers-color-scheme, and
    Sphinx serves ONE html_logo for both colour schemes. One image has
    to work in both themes.

    The README logo takes the workflow tile's surface and rim at a
    matching corner radius. The API logo is a dark teal rounded SQUARE,
    because a Sphinx sidebar gives the logo a square slot, written to
    docs/source/_static/logo_spacr_docs.png by the visuals generator.
    conf.py, setup_docs.sh and index.rst all point at it.

    The favicon deliberately stays the bare mark: it is drawn at 16px
    against the browser's own tab strip, which is dark in both themes on
    every platform that ships one, and a panel at that size is mostly
    panel. A test says so, so it is not "fixed" later.

3.  BUTTON SIZES could not match while both rows were sized to the same
    total width. The rows carry different things -- the core row is six
    buttons AND five arrows, an app row is six buttons -- so the arrows
    had to come out of the core buttons, leaving them at 14.5% against
    an effective 15.5%. The shared ROW WIDTH is dropped and the shared
    BUTTON SIZE kept. An app row now ends short of the right margin,
    which was already the normal case for three of the four bands.

10. The setup note is placed with setGeometry over the card rather than
    laid out, so nothing pushes back when it grows -- and it grew a
    capability table the day before. Its floor is now the nav row rather
    than the card's edge. The other end of the clamp is kept: a card too
    short for the band still lifts the note rather than losing it. The
    table is centred with a div align="center", because Qt rich text does
    not honour margin:auto.

11. The window marks were painted the menu bar's colour, read once from
    the theme at construction so they "cannot drift from the bar". They
    drifted anyway. A colour copied once is a SNAPSHOT, and the bar
    repaints for a theme change, a palette change, and on macOS for a
    translucency the copied value never had. Matching by copying was the
    bug; they are transparent now and cannot drift. Safe here and not
    for the bar itself -- these sit INSIDE the menu bar, which paints its
    own surface, whereas the bar is a top-level surface.

12. The pattern default was the one that mattered. It was `mandelbrot`,
    the only pattern with NO CPU renderer, so on every machine without a
    usable GPU the stated default could not be drawn and
    `pattern_for_this_machine` quietly substituted the orbit fold. The
    default now says what actually happens, and a test pins the default
    and the no-GPU fallback to the same pattern.

14. Home already routed descriptions to a hint bar and the reason is
    written on AppTile: these blurbs run to several hundred characters,
    fine in a fixed line and wrong in a box over the grid the user is
    reading to choose between. The sidebar and fold strip kept popping
    them. spacr/qt/module_hints.py diverts them into the status bar.

    The hook is QEvent.ToolTip, not hover: it fires when Qt has already
    decided to show a tooltip, so the text appears after the same delay
    the user is used to, and returning True suppresses the popup.
    Installed on the APPLICATION, because the fold strip builds its
    buttons lazily per host masthead and a per-widget filter would miss
    every one made later. It suppresses only when the text landed
    somewhere -- losing the description entirely is worse than the popup.

AND ONE REAL BUG, WHICH WAS MINE
---------------------------------
Reported from the Mac: opening Mask was slow and logged, repeatedly,

    ERROR spacr.qt.app: Could not translate the mask screen
    ModuleNotFoundError: No module named 'spacr.qt.i18n_catalogs'

The lightweight install (328) omits the catalogs. That exclusion rests
on a contract written in spacr/qt/i18n.py -- "External catalogs add
coverage; their absence must not make the compact core catalog
unavailable" -- and I verified that SOME catalog imports honoured it
rather than all of them.

retranslate_widget_tree imported setting_label unguarded, and its
enclosing except catches AttributeError, RuntimeError and TypeError,
none of which an ImportError is. So it escaped, once per screen change
and once per late settings panel.

Guarded, and tested two ways so the same class of mistake cannot repeat:
an AST sweep over EVERY i18n_catalogs import in the package, and a
behavioural test that blocks the import and drives the call that failed.
The behavioural one was vacuous at first -- it used a label the compact
catalog knows, so the external lookup never happened, and it needed the
Qt property names settingKey/settingsAppKey rather than the private
ones.

NOT DONE, AND WHY
-----------------
* THE macOS SETUP WINDOW IN DARK MODE. Reported as black text on a black
  background after switching a light-mode system to dark, and narrowed
  by the reporter to that window only -- "outside of spacr setup the dark
  theme works pretty well". Not reproducible on Linux, where the same
  switch repaints correctly. Needs a Mac session; see 314's open
  questions, which are blocked on the same machine.

* THE BACKDROP PAUSING WHILE A TOOLTIP IS UP, on macOS. Nothing in spaCR
  reacts to application-activation state -- there is no
  applicationStateChanged, ApplicationInactive or isActiveWindow handler
  anywhere in spacr/qt -- so this is not our own pause logic. The
  candidate is the `not self.isVisible()` early return in
  fractal_travel's _request_frame, which drops the loop to a 50 ms poll;
  whether macOS reports the widget as hidden while a native tooltip
  window is up cannot be checked from here. Needs a Mac session.

FILES
-----
    packaging/generate_readme_visuals.py   both logos, button widths
    packaging/source_install_excludes.txt  unchanged
    README.rst + 9 localized READMEs       order, renames, removals
    docs/source/conf.py, index.rst         the panelled API logo
    setup_docs.sh                          ditto
    tools/build_documentation_i18n.py      Core workflow heading table
    spacr/qt/i18n.py                       the guarded catalog import
    spacr/qt/fractal_defaults.py           backdrop defaults
    spacr/qt/widgets/setup_slides.py       table, clamp, counter
    spacr/qt/app.py                        transparent chrome, hints
    spacr/qt/widgets/fold_strip.py         module hint properties
    spacr/qt/module_hints.py               NEW

TESTS
-----
    tests/test_the_docs_logo_reads_on_a_white_page.py
    tests/qt/test_the_first_slide_does_not_sit_on_its_own_buttons.py
    tests/qt/test_the_window_buttons_show_the_bar_through.py
    tests/qt/test_the_backdrop_opens_on_the_asked_for_defaults.py
    tests/qt/test_a_missing_catalog_never_breaks_a_screen.py
    tests/qt/test_module_descriptions_go_to_the_bottom.py
    tests/test_readme_presentation.py                (updated)

Every fix mutation-checked: the old clamp fails the overlap test, an
always-on counter fails two, a literal colour back in the chrome
stylesheet fails two, and reverting the catalog guard fails both of its
tests.
