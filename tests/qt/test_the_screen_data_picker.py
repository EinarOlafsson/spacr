"""Choosing which pieces of the published screen to download.

The screen is 33 GB across four plates. The regression measurement and cell
functions read the DATABASES -- 2.2 GB for all four -- and the 30 GB of crops
are only needed to display images. One download would make trying one function
cost 33 GB, so every piece is fetched on its own and this is where they are
chosen.

EVERY ROW STATES ITS SIZE and the total updates as rows are ticked, because a
picker that lists eight items and then transfers an unstated number of
gigabytes is the thing it exists to prevent.
"""
from __future__ import annotations

import pytest

from spacr.screen_data import (SCREEN_ASSETS, SCREEN_REPO, assets_for,
                               human_size, total_size)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

def test_every_plate_has_both_pieces():
    for plate in (1, 2, 3, 4):
        kinds = {a.kind for a in assets_for(plate=plate)}
        assert kinds == {"measurements", "crops"}, plate


def test_the_databases_are_the_cheap_half():
    """The claim the picker's advice rests on."""
    databases = total_size(assets_for("measurements"))
    crops = total_size(assets_for("crops"))
    assert databases < crops / 10


def test_no_merged_folder_is_offered():
    """300 GB a plate, 1.2 TB for the screen -- past what a public dataset
    host takes without arrangement, and nothing in Regression reads them."""
    assert not [a for a in SCREEN_ASSETS if "merged" in a.archive]


def test_sizes_are_stated_for_every_piece():
    assert all(a.bytes > 0 for a in SCREEN_ASSETS)


@pytest.mark.parametrize("count, expected", [
    (0, "0 B"), (999, "999 B"), (1_500, "1.5 KB"),
    (590_741_504, "590.7 MB"), (8_300_000_000, "8.3 GB"),
])
def test_sizes_read_the_way_a_download_manager_reports_them(count, expected):
    """Decimal units, matching what a download manager and a disk vendor both
    say -- a user comparing our figure with either should not have to know
    which of two conventions each of us picked."""
    assert human_size(count) == expected


def test_a_piece_is_present_only_when_its_own_files_are(tmp_path):
    """Every piece unpacks into ONE shared plate folder, so the folder
    existing says nothing about which pieces are in it."""
    database = assets_for("measurements", plate=1)[0]
    crops = assets_for("crops", plate=1)[0]

    assert not database.is_present(tmp_path)
    assert not crops.is_present(tmp_path)

    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements" / "measurements.db").write_bytes(b"x")
    assert database.is_present(tmp_path)
    assert not crops.is_present(tmp_path), (
        "the database made the crops look downloaded")


def test_an_empty_data_folder_is_not_a_download(tmp_path):
    """A cancelled download can leave the folder behind."""
    crops = assets_for("crops", plate=1)[0]
    (tmp_path / "data").mkdir()
    assert not crops.is_present(tmp_path)


# ---------------------------------------------------------------------------
# The picker
# ---------------------------------------------------------------------------

@pytest.fixture
def picker(qapp, tmp_path):
    from spacr.qt.widgets.screen_data_picker import ScreenDataPicker

    made = ScreenDataPicker(folder=tmp_path)
    yield made
    made.close()
    made.deleteLater()
    qapp.processEvents()


def test_it_lists_every_piece(picker):
    assert picker._list.count() == len(SCREEN_ASSETS)


def test_each_row_shows_its_size(picker):
    for index in range(picker._list.count()):
        text = picker._list.item(index).text()
        asset = picker._list.item(index).data(0x0100)  # Qt.UserRole
        assert human_size(asset.bytes) in text, text


def test_nothing_is_selected_to_begin_with(picker):
    assert picker.chosen() == []
    assert not picker._download.isEnabled(), (
        "Download must not be pressable with nothing chosen")


def test_the_total_follows_the_selection(picker):
    picker._tick_kind("measurements")
    assert human_size(total_size(assets_for("measurements"))) \
        in picker._total.text()
    assert picker._download.isEnabled()


def test_clearing_empties_the_selection(picker):
    picker._tick_kind("crops")
    assert picker.chosen()
    picker._tick_kind(None)
    assert picker.chosen() == []


def test_selecting_a_row_ticks_it(picker):
    """One gesture, one meaning. A row highlighted but unticked would let the
    highlight -- which is what the eye reads -- lie about what is downloaded."""
    picker._list.item(0).setSelected(True)
    picker._follow_selection()
    assert len(picker.chosen()) == 1


def test_an_already_downloaded_piece_says_so(qapp, tmp_path):
    from spacr.qt.widgets.screen_data_picker import ScreenDataPicker

    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements" / "measurements.db").write_bytes(b"x")
    made = ScreenDataPicker(folder=tmp_path)
    try:
        texts = [made._list.item(i).text() for i in range(made._list.count())]
        assert any("already downloaded" in t for t in texts)
    finally:
        made.close()
        made.deleteLater()
        qapp.processEvents()


def test_an_already_downloaded_piece_can_still_be_re_fetched(qapp, tmp_path):
    """A re-download is how a truncated or edited copy gets repaired; a row
    that cannot be ticked gives no way to do that."""
    from PySide6.QtCore import Qt
    from spacr.qt.widgets.screen_data_picker import ScreenDataPicker

    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements" / "measurements.db").write_bytes(b"x")
    made = ScreenDataPicker(folder=tmp_path)
    try:
        item = made._list.item(0)
        assert item.flags() & Qt.ItemIsUserCheckable
        assert item.flags() & Qt.ItemIsEnabled
    finally:
        made.close()
        made.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

def test_cancelling_the_picker_downloads_nothing(qapp, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    class _Screen:
        screen_data_destination = lambda self: tmp_path
        load_the_screen_data = AppScreen.load_the_screen_data

    assert _Screen().load_the_screen_data(
        choose=lambda parent, folder, kind: [],
        ask=lambda *a, **k: pytest.fail("it downloaded after a cancel")) == {}


def test_each_kind_shows_only_its_own(qapp, tmp_path):
    """Filtered rather than greyed: a Feature download listing eight rows and
    refusing four of them would be four chances to start a 30 GB transfer by
    mistake."""
    from spacr.qt.widgets.screen_data_picker import ScreenDataPicker

    for kind in ("measurements", "crops"):
        made = ScreenDataPicker(folder=tmp_path, kind=kind)
        try:
            shown = {made._list.item(i).data(0x0100).kind
                     for i in range(made._list.count())}
            assert shown == {kind}
            assert made._list.count() == len(assets_for(kind))
        finally:
            made.close()
            made.deleteLater()
            qapp.processEvents()


def test_the_crops_dialog_says_what_they_cost(qapp, tmp_path):
    """30 GB is worth a sentence before it is started, not after."""
    from spacr.qt.widgets.screen_data_picker import ScreenDataPicker

    made = ScreenDataPicker(folder=tmp_path, kind="crops")
    try:
        advice = ScreenDataPicker.ADVICE["crops"]
        assert "8 GB per" in advice and "30 GB" in advice
        assert "only needed to DISPLAY images" in advice
    finally:
        made.close()
        made.deleteLater()
        qapp.processEvents()


def test_the_feature_dialog_says_what_reads_them(qapp, tmp_path):
    advice = __import__(
        "spacr.qt.widgets.screen_data_picker", fromlist=["x"]
    ).ScreenDataPicker.ADVICE["measurements"]
    assert "measurement and cell functions actually read" in advice


def test_the_four_download_buttons_exist():
    """Score and Count are cheap and fetch at once; Measurements (.db) and
    Image crops cost gigabytes and open a picker.

    The third button was called "Feature" until it was renamed for the file
    it fetches -- "Feature" is what the tables HOLD, and a user asking for
    "the measurements.db button" could not find it. This assertion still
    named the old caption, so it was red against a source that was right.
    """
    from pathlib import Path

    import spacr.qt.screens.app_screen as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'QPushButton("Measurements (.db)")' in source
    assert 'QPushButton("Image crops")' in source
    assert '("Score", "scores",' in source
    assert '("Count", "counts",' in source


def test_the_row_is_labelled_and_aligned_with_the_settings():
    """A row of buttons floating above the form reads as unrelated to it; one
    whose label sits in the same column reads as part of it.

    ``at_top=True`` since instruction 353 -- "the input tables sould start
    with download buttons not end wit them" -- which this exact-string
    assertion was left failing by. Matched WITHOUT the closing bracket so the
    next argument the row grows does not break it a second time; the label
    and the fact that it goes through ``add_prose_row`` are what this test is
    about, and where in the form it lands is asserted by
    ``test_the_download_buttons_sit_over_their_columns``.
    """
    from pathlib import Path

    import spacr.qt.screens.app_screen as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'section.add_prose_row("Download", row' in source


def test_the_labelled_row_is_not_a_setting():
    """`_row_widgets` is taken to hold labelled SETTINGS by the module smoke
    test, which asserts each field carries a settingKey."""
    import ast
    import inspect
    import textwrap

    from spacr.qt.widgets.section import Section

    # The CODE, not the prose: the docstring explains why `_row_widgets` is
    # not touched, and a text search reads that explanation as the thing it
    # warns against.
    tree = ast.parse(textwrap.dedent(inspect.getsource(Section.add_prose_row)))
    touched = {node.attr for node in ast.walk(tree)
               if isinstance(node, ast.Attribute)}
    assert "_row_widgets" not in touched
    assert "_form" in touched, (
        "it must still go through the section's own form, or the label will "
        "not line up with the settings")


def test_the_chosen_archives_are_what_is_fetched(qapp, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    wanted = assets_for("measurements")
    asked = {}

    class _Screen:
        _screen_data_button = None
        screen_data_destination = lambda self: tmp_path
        load_the_screen_data = AppScreen.load_the_screen_data

    _Screen().load_the_screen_data(
        choose=lambda parent, folder, kind: wanted,
        ask=lambda parent, dest, archives, repo, done: asked.update(
            archives=archives, repo=repo))

    assert asked["archives"] == [a.archive for a in wanted]
    assert asked["repo"] == SCREEN_REPO


# ---------------------------------------------------------------------------
# A piece that is not published cannot be ticked
# ---------------------------------------------------------------------------
#
# The manifest names eight pieces. A publish that has not finished, or one that
# failed, would otherwise be ticked, start, and fail after the user had
# committed to an eight-gigabyte download.


def _picker_with(qapp, tmp_path, published, kind=None):
    from spacr.qt.widgets import screen_data_picker as module

    original = module.__dict__.get("published_archives")
    made = None
    import spacr.screen_data as screen_data
    saved = screen_data.published_archives
    screen_data.published_archives = lambda *a, **k: published
    try:
        made = module.ScreenDataPicker(folder=tmp_path, kind=kind)
        return made
    finally:
        screen_data.published_archives = saved


def test_an_unpublished_piece_is_disabled(qapp, tmp_path):
    from PySide6.QtCore import Qt

    made = _picker_with(qapp, tmp_path, {"plate1-measurements.tar"})
    try:
        by_name = {made._list.item(i).data(Qt.UserRole).archive:
                   made._list.item(i) for i in range(made._list.count())}
        assert by_name["plate1-measurements.tar"].flags() & Qt.ItemIsEnabled
        assert not (by_name["plate1-data.tar"].flags() & Qt.ItemIsEnabled)
    finally:
        made.close()
        made.deleteLater()
        qapp.processEvents()


def test_an_unpublished_piece_says_so(qapp, tmp_path):
    made = _picker_with(qapp, tmp_path, {"plate1-measurements.tar"})
    try:
        texts = [made._list.item(i).text() for i in range(made._list.count())]
        assert any("not published yet" in t for t in texts)
    finally:
        made.close()
        made.deleteLater()
        qapp.processEvents()


def test_an_unpublished_piece_is_never_chosen(qapp, tmp_path):
    """What cannot be fetched must not be promised."""
    made = _picker_with(qapp, tmp_path, {"plate1-measurements.tar"})
    try:
        made._tick_kind("crops")
        assert made.chosen() == []
    finally:
        made.close()
        made.deleteLater()
        qapp.processEvents()


def test_a_lookup_that_failed_leaves_every_row_offered(qapp, tmp_path):
    """None means "could not tell" -- offline, or the hub did not answer --
    and is deliberately not the same as "nothing is published". Treating it as
    an absence would grey out every row and explain nothing."""
    from PySide6.QtCore import Qt

    made = _picker_with(qapp, tmp_path, None)
    try:
        assert all(made._list.item(i).flags() & Qt.ItemIsEnabled
                   for i in range(made._list.count()))
    finally:
        made.close()
        made.deleteLater()
        qapp.processEvents()
