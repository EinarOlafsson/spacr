"""The Home-variant generators' shared bootstrap keeps a review sheet honest.

``spacr/resources/home/versions/_generators/common.py`` is the table of record
for the thirty candidate Home screens: which checkout they import, which fonts
their metrics assume, which themes are worth rendering, and where an app that
nobody has filed lands. Each of those is a quiet failure when it goes wrong --
a sheet rendered against the operator's installed spaCR, metrics that differ
per machine, a duplicate "space" render, or thirty variants that die together
on one uncategorised app key -- so the edges are driven here directly.

The module is loaded from its path under a private name (it is inside a
package-data directory, not an importable package) and every process-global it
touches -- ``sys.path``, the ``spacr`` entries of ``sys.modules`` -- is put
back afterwards.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

pytest.importorskip("PySide6")

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GENERATORS = os.path.join(REPO_ROOT, "spacr", "resources", "home", "versions",
                          "_generators")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(GENERATORS),
    reason="home-screen variant generators are not part of this checkout")

MODULE_NAME = "_cov_wf_home_common"


@pytest.fixture(scope="module")
def common():
    """The generators' ``common`` module, loaded under a private name.

    Loading it runs ``_prefer_checkout_package()`` and reads the live app
    registry, so ``sys.path`` and the private ``sys.modules`` entry are
    restored when the module's tests are done with it.
    """
    saved_path = list(sys.path)
    path = os.path.join(GENERATORS, "common.py")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(MODULE_NAME, None)
        sys.path[:] = saved_path


@pytest.fixture
def spacr_import_state():
    """Snapshot ``sys.path`` and every loaded ``spacr`` module, then restore.

    ``_prefer_checkout_package`` deletes the ``spacr`` entries of
    ``sys.modules`` when they came from somewhere else, which is exactly the
    behaviour under test and exactly what would poison the rest of the
    session. The same module objects go back in place afterwards, so nothing
    is re-imported and no other test sees a second copy of ``spacr``.
    """
    saved_path = list(sys.path)
    saved = {name: mod for name, mod in sys.modules.items()
             if name == "spacr" or name.startswith("spacr.")}
    try:
        yield saved
    finally:
        sys.path[:] = saved_path
        for name, mod in saved.items():
            sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Which spaCR the sheet is rendered from
# ---------------------------------------------------------------------------

def test_a_checkout_spacr_is_kept_and_a_stray_one_is_evicted(
        common, spacr_import_state, tmp_path):
    """The sheet must picture the checkout it is written into, not a wheel.

    A reviewer runs the generators from a working tree to see the widgets
    they just changed. If an already-imported ``spacr`` came from a
    site-packages install, every render silently shows the *installed* app and
    the review is worthless -- so a foreign ``spacr`` is dropped from
    ``sys.modules`` and re-imported from the checkout. A package that has no
    single ``__file__`` (a namespace-style import, or one assembled by a
    loader) still has to be judged, by its ``__path__`` entries.
    """
    checkout_pkg = os.path.join(common.repo_root(), "spacr")
    assert os.path.isdir(checkout_pkg), checkout_pkg

    local = types.ModuleType("spacr")
    local.__file__ = ""          # no single origin: only __path__ can answer
    local.__path__ = [checkout_pkg]
    sys.modules["spacr"] = local
    sys.modules["spacr.qt"] = types.ModuleType("spacr.qt")

    common._prefer_checkout_package()

    assert sys.modules["spacr"] is local, (
        "a spacr that already lives in the checkout must be left alone")
    assert "spacr.qt" in sys.modules
    assert sys.path[0] == common.repo_root(), (
        "the checkout has to be searched first for the re-import to find it")

    stray = types.ModuleType("spacr")
    stray.__file__ = ""
    stray.__path__ = [str(tmp_path / "site-packages" / "spacr")]
    sys.modules["spacr"] = stray

    common._prefer_checkout_package()

    assert "spacr" not in sys.modules, (
        "an installed spacr must be evicted so the checkout is imported")
    assert "spacr.qt" not in sys.modules, (
        "leaving a submodule behind would mix the two trees")


def test_the_checkout_root_is_the_only_copy_of_itself_on_sys_path(
        common, spacr_import_state):
    """A duplicated ``sys.path`` entry is how the wrong copy wins a race.

    The function prepends the checkout root; if it merely inserted, a root
    already present further down (pytest's rootdir insertion, a ``PYTHONPATH``
    entry, a symlinked duplicate) would stay behind it and any later
    reordering could resurrect the loser. It removes every equivalent entry
    first, including the ``''`` that means "the current directory".
    """
    root = common.repo_root()
    sys.path[:] = [root, "", os.path.join(root, "docs"), root]

    common._prefer_checkout_package()

    assert sys.path.count(root) == 1
    assert sys.path[0] == root
    assert os.path.join(root, "docs") in sys.path, (
        "unrelated entries are not the target and must survive")


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def test_only_real_font_files_are_registered_before_a_render(
        common, monkeypatch, tmp_path):
    """Text metrics decide every tile's height, so the faces must load.

    ``HTile`` sizes itself from the bundled Open Sans; if the faces are not
    registered the renders fall back to whatever the machine has and the sheet
    stops being comparable between reviewers. The directory also carries
    licence and README files, and handing one of those to
    ``addApplicationFont`` registers nothing but a warning, so only ``.ttf`` /
    ``.otf`` names -- in either case -- are passed on.
    """
    from PySide6 import QtGui

    registered = []

    class _FontDatabase:
        """Stand-in recording what would be handed to the real font database."""

        @staticmethod
        def addApplicationFont(path):
            registered.append(path)
            return len(registered)

    monkeypatch.setattr(QtGui, "QFontDatabase", _FontDatabase)

    root = tmp_path / "checkout"
    fonts = root / "spacr" / "qt" / "resources" / "fonts"
    fonts.mkdir(parents=True)
    for name in ("OpenSans-Regular.ttf", "OpenSans-Bold.OTF", "LICENSE.txt",
                 "README"):
        (fonts / name).write_bytes(b"stub")
    monkeypatch.setattr(common, "repo_root", lambda: str(root))

    common._load_fonts()

    assert registered == [str(fonts / "OpenSans-Bold.OTF"),
                          str(fonts / "OpenSans-Regular.ttf")], (
        "both font suffixes register, case-insensitively, and nothing else")

    bare = tmp_path / "no-fonts"
    bare.mkdir()
    monkeypatch.setattr(common, "repo_root", lambda: str(bare))
    registered.clear()

    common._load_fonts()

    assert registered == [], (
        "a checkout without the bundled fonts still has to render, not raise")


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

def test_space_is_offered_only_when_it_is_a_palette_of_its_own(
        common, monkeypatch):
    """A theme that renders as dark costs the reviewer a duplicate sheet.

    ``available_themes`` decides how many times all thirty variants are drawn.
    Older builds answer ``palette_for("space")`` with the dark palette itself
    (or with something that is not a palette at all); offering "space" then
    produces thirty more PNGs identical to the dark ones, and a reviewer who
    trusts the file names compares two renders of the same thing.
    """
    from spacr.qt import theme

    dark = {"bg": "#101214", "fg": "#e8e8e8"}

    def _answers(space_palette):
        """A ``palette_for`` that returns ``space_palette`` for "space"."""
        def palette_for(name):
            return dark if name == "dark" else space_palette
        return palette_for

    monkeypatch.setattr(theme, "palette_for",
                        _answers({"bg": "#04060d", "fg": "#cfd8ff"}))
    assert common.available_themes() == ("dark", "light", "space"), (
        "a distinct, painted space palette is worth its own renders")

    monkeypatch.setattr(theme, "palette_for", _answers(dark))
    assert common.available_themes() == ("dark", "light"), (
        "space aliased to dark would render the same sheet twice")

    monkeypatch.setattr(theme, "palette_for", _answers({"bg": ""}))
    assert common.available_themes() == ("dark", "light")

    monkeypatch.setattr(theme, "palette_for", _answers("space"))
    assert common.available_themes() == ("dark", "light"), (
        "a non-palette answer is not a theme")

    def _raises(name):
        """A theme module too old to know the name at all."""
        raise KeyError(name)

    monkeypatch.setattr(theme, "palette_for", _raises)
    assert common.available_themes() == ("dark", "light"), (
        "dark and light always render, whatever the theme module does")


# ---------------------------------------------------------------------------
# Late registrations
# ---------------------------------------------------------------------------

def test_an_app_nobody_filed_lands_in_the_declared_fallback_band(
        common, monkeypatch):
    """A newly registered app must not take the whole review sheet down.

    The categorisation tables are hand-written literals, while apps register
    themselves from their own modules. An app that registered after a table
    was written is in the registry and in no band, which makes
    ``check_coverage`` raise and costs all thirty renders. It is appended to
    the declared fallback band instead -- and only there, so the hand-made
    judgements about every other app are untouched.
    """
    monkeypatch.setattr(common, "all_keys",
                        lambda: ["mask", "measure", "brand_new"])
    literal = [("Prepare", ["mask"]), ("Review", ["measure"])]

    result = common._with_late_registrations(literal, fallback="Review")

    assert result == [("Prepare", ["mask"]), ("Review", ["measure",
                                                         "brand_new"])]
    assert literal == [("Prepare", ["mask"]), ("Review", ["measure"])], (
        "the literal table is a source file, not scratch space")
    common.check_coverage(result)

    filed = common._with_late_registrations(
        [("Prepare", ["mask", "brand_new"]), ("Review", ["measure"])],
        fallback="Review")
    assert filed == [("Prepare", ["mask", "brand_new"]), ("Review",
                                                          ["measure"])], (
        "an app somebody has already filed stays where it was put")


def test_a_fallback_band_that_no_table_carries_leaves_the_app_unplaced(
        common, monkeypatch):
    """A renamed band silently stops catching new apps, so it must be caught.

    The fallback is named by string. Rename the band in the table and forget
    the ``fallback=`` argument and the loop finds nothing to extend: the new
    app is dropped, and the failure surfaces later as ``check_coverage``
    refusing the table -- which is the message a maintainer needs to see, and
    what this pins.
    """
    monkeypatch.setattr(common, "all_keys", lambda: ["mask", "brand_new"])

    stale = common._with_late_registrations([("Prepare", ["mask"])],
                                            fallback="Screens & reports")

    assert stale == [("Prepare", ["mask"])], (
        "no band answers to that title, so nothing can be extended")
    with pytest.raises(AssertionError) as excinfo:
        common.check_coverage(stale)
    assert "brand_new" in str(excinfo.value)
    assert "not categorised" in str(excinfo.value)

    repaired = common._with_late_registrations([("Prepare", ["mask"])],
                                               fallback="Prepare")
    assert repaired == [("Prepare", ["mask", "brand_new"])]
    common.check_coverage(repaired)


def test_a_table_that_still_names_a_folded_app_fails_loudly(common, monkeypatch):
    """A retired key would draw a Home tile for a screen that no longer exists.

    Folding a module into its host removes its registry row, but the
    hand-written band keeps the name. Unlike a *new* app, a *retired* one is
    never quietly repaired: the tile would open nothing, so the table raises
    and names the keys to delete.
    """
    monkeypatch.setattr(common, "all_keys", lambda: ["mask", "measure"])

    with pytest.raises(AssertionError) as excinfo:
        common._with_late_registrations(
            [("Prepare", ["mask", "cell_extract"]),
             ("Review", ["measure", "sequencing"])],
            fallback="Review")

    message = str(excinfo.value)
    assert "cell_extract" in message and "sequencing" in message, (
        "the maintainer has to be told which rows to remove")
    assert "retired" in message

    live_only = common._with_late_registrations(
        [("Prepare", ["mask"]), ("Review", ["measure"])], fallback="Review")
    assert live_only == [("Prepare", ["mask"]), ("Review", ["measure"])]


# ---------------------------------------------------------------------------
# The shipped tables, against the live registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table_name", ["CATS_BROAD3", "CATS_STAGE5",
                                        "CATS_NARROW8", "CATS_QUESTIONS",
                                        "CATS_INTENT4"])
def test_every_shipped_categorisation_covers_the_live_registry(
        common, table_name):
    """A variant builder reads these tables, so a gap is thirty dead renders.

    ``check_coverage`` is the contract every builder assumes: each registered
    app appears exactly once, and no band names a key the registry does not
    define (``common.name_of`` would raise ``KeyError`` on it).
    """
    table = getattr(common, table_name)

    common.check_coverage(table)

    placed = [key for _title, keys in table for key in keys]
    assert sorted(placed) == sorted(common.all_keys())
    assert len(placed) == common.n_apps()


def test_every_registered_app_has_a_use_count_and_sorts_by_it(common):
    """Variant 14 indexes ``USE_COUNTS[key]`` for a badge on every tile.

    A key missing from the table is not "sorts last", it is a ``KeyError``
    that takes the whole sheet down -- so the module fills itself in from the
    live registry, giving a never-counted app the deliberately small default
    that puts it at the bottom of every frequency-ordered variant.
    """
    keys = common.all_keys()

    assert set(keys) <= set(common.USE_COUNTS), (
        "every registered app must carry a count")
    assert common.USE_COUNTS["mask"] == 412
    assert common.UNUSED_APP_COUNT == 4
    assert min(common.USE_COUNTS.values()) >= common.UNUSED_APP_COUNT, (
        "the default has to sit at or below every hand-written count")

    order = common.by_frequency()
    assert sorted(order) == sorted(keys)
    assert order[0] == "mask", "the most-used app leads the frequency variants"
    counts = [common.USE_COUNTS[key] for key in order]
    assert counts == sorted(counts, reverse=True)
