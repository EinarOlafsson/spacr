"""Image Scatter and PCA are projections of Image UMAP's table, not errands.

Three screens drew one point per object out of the same measurement
table: the UMAP embedding of the crops, any two measured columns, and the
principal components of the whole feature block. Same objects, same
click, three projections -- so the other two fold onto Image UMAP rather
than being two more tiles a user has to leave the screen to find.

What these tests protect:

* the button has to be recognisable as the module it replaced -- its own
  icon, its own sentence, and the maturity colour its TILE lit up in,
  read from the one table rather than retyped;
* the button has to open the module ITSELF, so each test names a
  capability only the real screen has -- the hover crop, the loadings
  biplot and its CSV export;
* and the source has to travel. A projection that made the user retype
  the database the screen beside it is already reading would be a second
  module rather than a second view, which is the whole reason for the
  fold.
"""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import image_umap
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.image_scatter import ImageScatterScreen
from spacr.qt.screens.pca import PCAScreen


def _database(folder) -> str:
    """A minimal measurements database at the canonical project location."""
    measurements = folder / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)
    path = measurements / "measurements.db"
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
            "fieldID TEXT, object_label INTEGER, cell_area REAL)")
        connection.execute(
            "INSERT INTO cell VALUES ('p1','A','1','f1',1,10.0)")
        connection.commit()
    finally:
        connection.close()
    return str(path)


def _host(qtbot, source: str = ""):
    """An Image UMAP screen with its fold strip installed."""
    screen = AppScreen(app_key="umap")
    qtbot.addWidget(screen)
    if source:
        assert screen.apply_settings_dict({"src": source}) == 1
    strip = image_umap.install_folds(screen)
    assert strip is not None, "no fold strip was installed on Image UMAP"
    return screen, strip


def _opened(qtbot, opener):
    """Press one fold opener and register what it opened."""
    window = opener.open()
    assert window is not None, f"{opener.key}: the button opened nothing"
    qtbot.addWidget(window)
    return window


# ---------------------------------------------------------------------------
# The strip
# ---------------------------------------------------------------------------

def test_image_umap_carries_the_two_projections_as_buttons(
        qtbot, qt_theme_applied):
    """Both appear on the masthead, in declared order.

    Two measured columns first, the whole feature block second -- the
    order a user narrows a table in -- so it is asserted rather than left
    to a set.
    """
    _screen, strip = _host(qtbot)
    assert list(strip.keys()) == list(image_umap.FOLDED_APPS)


def test_a_projection_button_is_its_module_icon_and_description(
        qtbot, qt_theme_applied):
    """No text, the module's own icon, the module's own line as tooltip.

    The sentence is read through
    :func:`~spacr.qt.screens.map_barcodes.fold_description` rather than
    straight out of ``APPS``: these two have no registry row left, and
    that function is the seam the strip itself restates the buttons
    through, so it is what has to be right.
    """
    from spacr.qt.screens.map_barcodes import fold_description

    _screen, strip = _host(qtbot)
    for key in image_umap.FOLDED_APPS:
        button = strip.button_for(key)
        assert button is not None
        assert button.text() == "", f"{key}: the fold button has a caption"
        assert not button.icon().isNull(), f"{key}: no icon on the button"
        description = fold_description(key)[1]
        assert description, f"{key}: nothing says what this button is"
        assert description in button.toolTip()


def test_a_projection_button_lights_in_the_stage_its_tile_lit_in(
        qtbot, qt_theme_applied):
    """The hover colour is the module's maturity, from the one table.

    Two colour tables drift, so the button's ``stage`` property -- what
    the stylesheet selects on -- is asserted against the one answer the
    strip itself uses. That is ``fold_description``, which reads
    ``app_stage`` while the row exists and the fallback afterwards; these
    two rows are gone, so it is the fallback that has to still say alpha.
    """
    from spacr.qt.screens.map_barcodes import fold_description
    from spacr.qt.theme import STAGE_HOVER, stylesheet

    _screen, strip = _host(qtbot)
    sheet = stylesheet()
    for key in image_umap.FOLDED_APPS:
        stage = fold_description(key)[2]
        assert stage == "alpha", f"{key}: the button no longer lights alpha"
        assert strip.button_for(key).property("stage") == stage
        assert stage in STAGE_HOVER
        rule = f'QPushButton#FoldButton[stage="{stage}"]:hover'
        assert rule in sheet, f"{key}: nothing lights the button on hover"


def test_the_fold_fallback_is_in_the_table_that_is_actually_read():
    """Each folded key is named in the one table ``fold_description`` reads.

    The host used to keep its own ``FOLD_FALLBACK`` beside its
    ``FOLDED_APPS``, which read well and answered nothing: the strip is
    restated through :func:`map_barcodes.restate_fold_button`, and that
    looks in ``map_barcodes.FOLD_FALLBACK`` and nowhere else. A fallback
    written anywhere else leaves the buttons mute the day the rows go.
    """
    from spacr.qt.screens import map_barcodes

    for key in image_umap.FOLDED_APPS:
        assert key in map_barcodes.FOLD_FALLBACK, (
            f"{key}: nothing would say what this button is once its row "
            f"is dropped")


def test_the_fold_fallback_says_what_the_module_says_about_itself():
    """The kept copy of each tile's line agrees with its module.

    The registry rows for these two are gone -- being folded is what
    dropping them means -- so the comparison that kept the fallback
    honest against ``APPS`` has nothing left to compare against. Their
    modules still declare the same ``APP_NAME`` and ``APP_DESCRIPTION``
    the tiles were built out of, and those are what the button now has
    to go on matching.
    """
    from spacr.qt.screens import image_scatter as image_scatter_module
    from spacr.qt.screens import map_barcodes
    from spacr.qt.screens import pca as pca_module

    modules = {"image_scatter": image_scatter_module, "pca": pca_module}
    for key, module in modules.items():
        name, description, stage = map_barcodes.FOLD_FALLBACK[key]
        assert name == module.APP_NAME
        assert description == module.APP_DESCRIPTION
        assert stage == "alpha"


def test_a_button_still_says_what_it_is_with_no_registry_row():
    """What the fallback exists for, asserted through the seam that uses it.

    ``fold_description`` answers out of the registry while a row is there
    and out of the fallback afterwards; these two rows are gone, so this
    is the fallback answering and nothing else.
    """
    from spacr.qt.app import APPS
    from spacr.qt.screens import map_barcodes

    live = {row[0] for row in APPS}
    for key in image_umap.FOLDED_APPS:
        assert key not in live, f"{key} still has a registry row"
        name, description, stage = map_barcodes.fold_description(key)
        assert name and description
        assert stage == "alpha"


def test_the_window_seam_finds_this_host(qtbot, qt_theme_applied):
    """The strip reaches a real window through the walker, not by hand.

    Every test above calls :func:`image_umap.install_folds` directly. The
    running window never does: ``shortcuts._install_window_hooks`` starts
    :func:`map_barcodes.install_window_hooks`, which walks the screen
    stack and looks each screen's key up in ``FOLD_HOST_MODULES``. A host
    missing from that table has no strip in the running app however well
    its own installer works -- and with these two rows dropped, a strip
    that never arrives is two modules with no way in at all.
    """
    from spacr.qt.screens import map_barcodes

    assert map_barcodes.FOLD_HOST_MODULES.get(image_umap.HOST_KEY) \
        == "image_umap"
    screen = AppScreen(app_key=image_umap.HOST_KEY)
    qtbot.addWidget(screen)

    strip = map_barcodes.install_folds_on(screen)

    assert strip is not None, "the walker gave Image UMAP no strip"
    assert list(strip.keys()) == list(image_umap.FOLDED_APPS)


def test_every_folded_key_has_a_builder():
    """A key in the strip with no builder would be a dead button."""
    assert set(image_umap.FOLDED_APPS) <= set(image_umap.BUILDERS)


# ---------------------------------------------------------------------------
# The source travels
# ---------------------------------------------------------------------------

def test_the_measurements_database_is_found_from_the_project(tmp_path):
    """Both layouts the source box actually holds resolve to the file.

    A project folder is what the UMAP screen is normally pointed at; the
    database itself is what a user who dropped one has. Anything else
    resolves to nothing, and the view then opens on its own Browse button
    rather than on a path that is not there.
    """
    database = _database(tmp_path)

    assert image_umap.measurements_database(str(tmp_path)) == database
    assert image_umap.measurements_database(database) == database
    assert image_umap.measurements_database(str(tmp_path / "nothing")) == ""
    assert image_umap.measurements_database("") == ""


def test_the_database_beside_the_project_is_found_too(tmp_path):
    """``<project>/measurements.db`` is the older layout and still opens."""
    path = tmp_path / "measurements.db"
    path.write_bytes(b"")

    assert image_umap.measurements_database(str(tmp_path)) == str(path)


def test_the_scatter_opens_on_the_database_the_umap_screen_is_reading(
        qtbot, qt_theme_applied, tmp_path):
    """Switching projection must not mean retyping the source.

    That is the difference between three views of one module and three
    modules that happen to read the same file.
    """
    database = _database(tmp_path)
    screen, _strip = _host(qtbot, source=str(tmp_path))

    view = _opened(qtbot, screen._fold_openers[0])

    assert isinstance(view, ImageScatterScreen)
    assert view.database() == database
    # The capability only this screen has: the crop of the object under
    # the cursor, resolved and cached for the whole plot.
    assert view._thumbs is not None
    assert view.canvas is not None


def test_pca_opens_on_the_same_database(qtbot, qt_theme_applied, tmp_path):
    """The third projection is seeded from the same place as the second."""
    database = _database(tmp_path)
    screen, _strip = _host(qtbot, source=str(tmp_path))

    view = _opened(qtbot, screen._fold_openers[1])

    assert isinstance(view, PCAScreen)
    assert view._path == database
    # The capability only this screen has: the loadings biplot behind the
    # scores, and the decomposition written back out as a table.
    assert view.pca is not None
    assert callable(view.export_csv)


def test_a_project_with_no_database_still_opens_the_view(
        qtbot, qt_theme_applied, tmp_path):
    """An unresolvable source costs the seed, not the screen.

    A user who has not chosen a project yet still gets the projection, on
    its own Browse button.
    """
    screen, _strip = _host(qtbot, source=str(tmp_path))

    view = _opened(qtbot, screen._fold_openers[0])

    assert isinstance(view, ImageScatterScreen)
    assert view.database() == ""


def test_a_blank_path_does_not_clear_a_source_already_typed(
        qtbot, qt_theme_applied, tmp_path):
    """"No path known" must not throw away a path the user typed.

    The seed is best-effort; the box is the user's.
    """
    database = _database(tmp_path)
    view = ImageScatterScreen(threaded=False)
    qtbot.addWidget(view)

    assert view.set_database(database) is True
    assert view.set_database("") is False
    assert view.database() == database


# ---------------------------------------------------------------------------
# The strip never costs the screen
# ---------------------------------------------------------------------------

def test_a_screen_that_is_not_the_host_gets_no_strip(qtbot, qt_theme_applied):
    """Installing into the wrong screen does nothing at all.

    The seam that calls this walks every module screen, so being asked
    about the wrong one has to be free rather than wrong.
    """
    screen = AppScreen(app_key="measure")
    qtbot.addWidget(screen)

    assert image_umap.install_folds(screen) is None


def test_a_second_install_returns_the_strip_the_first_one_made(
        qtbot, qt_theme_applied):
    """The stack watcher installs on every tab change; twice is not two."""
    screen, strip = _host(qtbot)

    assert image_umap.install_folds(screen) is strip


# ---------------------------------------------------------------------------
# The seed never costs the view
# ---------------------------------------------------------------------------
#
# Reading the host's source is best-effort: a projection that refused to
# open because it could not work out where the database was would be a
# worse screen than one that opens on its own Browse button.

def test_the_source_of_a_screen_that_has_none_is_empty():
    """Every way the source is unknown answers "", not a broken path."""
    class _Raises:
        def collect(self):
            raise RuntimeError("no settings")

    assert image_umap.source_path(None) == ""
    assert image_umap.source_path(object()) == ""
    assert image_umap.source_path(
        type("_S", (), {"_settings_model": _Raises()})()) == ""
    assert image_umap.source_path(
        type("_S", (), {"_settings_model": type(
            "_M", (), {"collect": lambda self: {"src": ""}})()})()) == ""


def test_a_list_of_plates_seeds_from_the_first():
    """Several plates through one run is one table per plate.

    A scatter over two plates' tables is two populations on one pair of
    axes, so the first plate is the one the other views can actually plot.
    """
    screen = type("_S", (), {"_settings_model": type(
        "_M", (), {"collect": lambda self: {
            "src": ["/data/plate1", "/data/plate2"]}})()})()

    assert image_umap.source_path(screen).endswith("plate1")


def test_a_file_that_is_not_a_database_resolves_to_nothing(tmp_path):
    """A source box holding some other file must not be handed to sqlite."""
    stray = tmp_path / "settings.csv"
    stray.write_text("a,b\n1,2\n")

    assert image_umap.measurements_database(str(stray)) == ""
