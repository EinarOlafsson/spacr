"""The live preview's set table, its z-projection, and its stray helpers.

Written for instruction 60. The set table -- the grid of fields down the side
and channels across the top that Mask uses as its source selector -- was the
largest single unreached feature in ``live_preview.py``: its clicks, its
shift-clicks, its image cap and its MIP switch had no test between them.

The plate is Yokogawa-named and written to disk, because every one of these
paths asks a real question about real file names.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from spacr.qt.widgets import live_preview as LP  # noqa: E402
from spacr.qt.widgets.live_preview import LivePreviewPanel  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """QPixmap aborts the process when no QGuiApplication exists."""
    return qapp


def _tile(value: int = 7, shape=(8, 8)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint16)


@pytest.fixture
def flat_plate(tmp_path: Path) -> Path:
    """Three fields, three channels, one plane each.

    Its own directory, not ``tmp_path`` itself: a test that wants a flat
    folder and a stacked one gets ONE ``tmp_path``, and writing both into it
    makes the flat folder a stacked one.
    """
    root = tmp_path / "flat"
    root.mkdir()
    for field in range(1, 4):
        for chan in range(1, 4):
            tifffile.imwrite(
                root / f"plate1_A01_T0001F{field:03d}L01A01Z01C{chan:02d}.tif",
                _tile(field * 10 + chan))
    return root


@pytest.fixture
def stack_plate(tmp_path: Path) -> Path:
    """Two fields, two channels, four z-planes each."""
    root = tmp_path / "stacked"
    root.mkdir()
    for field in range(1, 3):
        for chan in range(1, 3):
            for z in range(1, 5):
                tifffile.imwrite(
                    root / f"plate1_A01_T0001F{field:03d}L01A01Z{z:02d}"
                           f"C{chan:02d}.tif",
                    _tile(z))
    return root


@pytest.fixture
def panel(qtbot, flat_plate: Path):
    widget = LivePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    widget.load_image(sorted(flat_plate.iterdir())[0])
    return widget


def _no_shift(monkeypatch):
    monkeypatch.setattr(LP.LivePreviewPanel, "_shift_held", lambda self: False)


def _shift(monkeypatch):
    monkeypatch.setattr(LP.LivePreviewPanel, "_shift_held", lambda self: True)


def _path_at(panel, row: int, col: int) -> str:
    item = panel._set_table.item(row, col)
    return item.data(Qt.UserRole) if item is not None else ""


# ---------------------------------------------------------------------------
# What the table is
# ---------------------------------------------------------------------------

def test_the_table_is_a_row_per_field_and_a_column_per_channel(panel):
    table = panel._set_table
    assert table.rowCount() == 3
    assert table.columnCount() == 3
    assert [table.horizontalHeaderItem(c).text() for c in range(3)] == \
        ["ch 0", "ch 1", "ch 2"]
    # Every cell names the file behind it, so a click never has to guess.
    for row in range(3):
        for col in range(3):
            assert Path(_path_at(panel, row, col)).is_file()


# ---------------------------------------------------------------------------
# The headers
# ---------------------------------------------------------------------------


def test_clicking_a_field_header_keeps_the_channel_and_changes_the_field(
        panel, monkeypatch):
    """A user comparing channel 2 across fields should not be dropped back to
    channel 1 by moving down a row."""
    _no_shift(monkeypatch)
    panel._on_set_cell_clicked(0, 2)
    panel._on_set_header_clicked(1)
    assert (panel._table_row, panel._table_col) == (1, 2)
    assert str(panel._image_path) == _path_at(panel, 1, 2)


def test_shift_clicking_a_field_header_takes_every_channel_of_that_field(
        panel, monkeypatch):
    _shift(monkeypatch)
    panel._max_images_box.setValue(4)
    panel._on_set_header_clicked(2)
    assert panel._selected_cells == [(2, 0), (2, 1), (2, 2)]


# ---------------------------------------------------------------------------
# The image cap
# ---------------------------------------------------------------------------


def test_a_cap_box_that_cannot_be_read_falls_back_to_the_default(panel):
    """The cap sizes a list the user is looking at; guessing zero would show
    nothing at all."""
    panel._max_images_box = None
    assert panel.max_images() == LP.DEFAULT_MAX_IMAGES


def test_cells_with_no_file_behind_them_are_dropped_and_duplicates_collapse(
        panel):
    kept = panel._cells_with_images([(0, 0), (0, 0), (99, 99), (1, 1)])
    assert kept == [(0, 0), (1, 1)]


def test_a_selection_of_nothing_leaves_the_view_where_it_was(panel,
                                                               monkeypatch):
    _no_shift(monkeypatch)
    panel._on_set_cell_clicked(0, 0)
    before = panel._image_path
    panel._set_selection([(99, 99)], extend=False)
    assert panel._image_path == before


# ---------------------------------------------------------------------------
# Asking the keyboard whether shift is down
# ---------------------------------------------------------------------------

def test_shift_is_read_from_the_keyboard_because_the_click_does_not_carry_it(
        panel, monkeypatch):
    """``cellClicked`` and ``sectionClicked`` carry no modifier."""
    from PySide6.QtWidgets import QApplication
    monkeypatch.setattr(QApplication, "keyboardModifiers",
                        staticmethod(lambda: Qt.ShiftModifier))
    assert panel._shift_held() is True
    monkeypatch.setattr(QApplication, "keyboardModifiers",
                        staticmethod(lambda: Qt.NoModifier))
    assert panel._shift_held() is False


def test_a_keyboard_that_cannot_be_asked_is_read_as_no_shift(panel,
                                                               monkeypatch):
    from PySide6.QtWidgets import QApplication
    monkeypatch.setattr(
        QApplication, "keyboardModifiers",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no app"))))
    assert panel._shift_held() is False


# ---------------------------------------------------------------------------
# The MIP switch
# ---------------------------------------------------------------------------

def test_a_flat_plate_disables_the_projection_switch_and_says_why(panel):
    toggle = panel._mip_toggle
    assert toggle.isEnabled() is False
    assert "nothing to project" in toggle.toolTip()


def test_projection_with_nothing_open_reads_no_file(qtbot):
    widget = LivePreviewPanel()
    qtbot.addWidget(widget)
    widget._image_path = None
    widget._reload_for_mip()            # returns without reading anything
    assert widget._image is None


def test_a_flat_field_is_read_as_one_plane_even_with_projection_on(panel):
    """The switch applies to stacks; a field with one plane per channel is
    read exactly as it was."""
    panel._mip_enabled = True
    path = Path(_path_at(panel, 0, 0))
    assert np.array_equal(panel._load_for_display(path),
                          LP.load_preview_image(path))


def test_a_file_the_sampler_does_not_know_is_read_as_one_plane(panel,
                                                                monkeypatch):
    panel._mip_enabled = True
    monkeypatch.setattr(
        panel._sampler, "set_for_path",
        lambda p: (_ for _ in ()).throw(KeyError("not enumerated")))
    path = Path(_path_at(panel, 0, 0))
    assert np.array_equal(panel._load_for_display(path),
                          LP.load_preview_image(path))


# ---------------------------------------------------------------------------
# The sample size and the status line
# ---------------------------------------------------------------------------


def test_a_cap_the_sampler_refuses_changes_nothing(panel, monkeypatch):
    before = panel._status.text()
    monkeypatch.setattr(panel._sampler, "set_max", lambda value: False)
    panel._on_max_sets_changed(999)
    assert panel._status.text() == before


# ---------------------------------------------------------------------------
# Small helpers with no home of their own
# ---------------------------------------------------------------------------

def test_a_settings_widget_that_is_not_there_reads_as_empty():
    assert LP._widget_text(None) == ""


def test_a_settings_widget_that_refuses_to_be_read_reads_as_empty():
    class _Broken:
        def text(self):
            raise RuntimeError("deleted")

    assert LP._widget_text(_Broken()) == ""
    assert LP._widget_text(object()) == ""


def test_a_mask_with_no_objects_has_no_outline():
    assert not LP._labelled_boundary(np.zeros((4, 4), dtype=np.int64)).any()
    assert not LP._labelled_boundary(np.zeros((4,), dtype=np.int64)).any()


def test_a_thicker_outline_covers_more_pixels_than_a_thin_one():
    mask = np.zeros((12, 12), dtype=np.int64)
    mask[3:9, 3:9] = 1
    thin = LP._labelled_boundary(mask, thickness=1)
    thick = LP._labelled_boundary(mask, thickness=3)
    assert int((thick > 0).sum()) > int((thin > 0).sum())
    # Still only that object's label -- a thicker line must not invent one.
    assert set(np.unique(thick)) <= {0, 1}


def test_colouring_no_objects_returns_no_colours():
    assert LP._random_outline_palette(np.array([], dtype=np.int64)).shape == \
        (0, 3)


# ---------------------------------------------------------------------------
# The load that comes back from the worker
# ---------------------------------------------------------------------------


def test_a_field_that_cannot_be_projected_still_shows_its_image(
        qtbot, stack_plate, monkeypatch):
    """Switching field or channel with MIP on used to drop silently back to
    one plane until the switch was toggled off and on again."""
    widget = LivePreviewPanel()
    qtbot.addWidget(widget)
    widget.load_image(sorted(stack_plate.iterdir())[0])
    widget._mip_enabled = True
    monkeypatch.setattr(
        LP.LivePreviewPanel, "_load_for_display",
        lambda self, path: (_ for _ in ()).throw(OSError("plane gone")))
    path = Path(sorted(stack_plate.iterdir())[0])
    widget._install_loaded_image(path, LP.load_preview_image(path))
    assert widget._image is not None
    assert int(np.max(widget._image)) == 1


def test_a_sampler_that_cannot_say_how_many_channels_leaves_the_combo_alone(
        panel, monkeypatch):
    """The channel count is a convenience; losing it must not cost the whole
    selector refresh that every image load goes through."""
    def _refuse(self):
        raise RuntimeError("not enumerated")

    monkeypatch.setattr(type(panel._sampler), "channels", property(_refuse))
    panel._refresh_source_selectors()
    assert panel._channel_box.count() >= 1


# ---------------------------------------------------------------------------
# The MIP toggle and the field dropdown, in their remaining states
# ---------------------------------------------------------------------------

def test_a_panel_with_no_projection_switch_refreshes_quietly(panel,
                                                                monkeypatch):
    """Same shape as the table: a panel without the switch must not pay for
    the enumeration that decides whether to enable it."""
    read = []

    def _sets(self):
        read.append(1)
        return ()

    monkeypatch.setattr(type(panel._sampler), "sets", property(_sets))
    panel._mip_toggle = None
    panel._refresh_mip_toggle()
    assert read == []


# ---------------------------------------------------------------------------
# Background removal, outline colours, model list
# ---------------------------------------------------------------------------

def test_no_channel_on_screen_means_no_background_to_remove(panel):
    assert panel._background_for_channel(None) is None


def test_nothing_to_show_is_returned_untouched(panel):
    assert panel._apply_display_background(None) is None


def test_an_outline_colour_asked_for_before_any_were_rolled_is_made_on_demand(
        panel):
    panel._auto_outline_colours = {}
    colour = panel._auto_outline_colour("cell")
    assert len(colour) == 3
    # And it sticks: re-asking must not recolour the objects on screen.
    assert panel._auto_outline_colour("cell") == colour


def test_a_normalise_setting_that_cannot_be_applied_does_not_stop_the_rest(
        panel, monkeypatch):
    """`apply_settings` seeds a whole panel; one unreadable value must not
    cost the other twenty."""
    monkeypatch.setattr(
        panel._normalise_check, "setChecked",
        lambda value: (_ for _ in ()).throw(RuntimeError("deleted")))
    panel.apply_settings({"normalize": True, "cell_channel": 2})
    assert int(panel._cell_channel.value()) == 2


def test_a_model_the_probe_found_is_added_without_disturbing_the_choice(panel,
                                                                         monkeypatch):
    """A value the user picked must not vanish under them because a probe
    came back thinner."""
    panel._model_box.setCurrentIndex(0)
    chosen = panel._model_box.currentText()
    existing = [panel._model_box.itemText(i)
                for i in range(panel._model_box.count())]
    monkeypatch.setattr(LP, "_model_menu", lambda: ["a_brand_new_model"] + existing)
    panel.refresh_model_choices()
    names = [panel._model_box.itemText(i)
             for i in range(panel._model_box.count())]
    assert "a_brand_new_model" in names
    assert set(existing) <= set(names)
    assert panel._model_box.currentText() == chosen


def test_a_mask_whose_objects_have_no_outline_pixels_draws_nothing(panel):
    """A single-pixel-wide label can boundary out to nothing; drawing an
    empty palette raised rather than skipping."""
    image = np.zeros((6, 6), dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=np.int64)
    out = LP.overlay_masks(image, {"cell": mask}, random_outline=True)
    assert out.shape[:2] == (6, 6)


def test_an_object_that_fills_the_whole_field_draws_no_outline(panel):
    """A 4-connected boundary needs something on the other side of it, so a
    mask covering every pixel has none -- which a huge Cellpose diameter
    produces routinely. Drawing an empty palette raised; skipping is right,
    and the image comes back unmarked rather than not at all."""
    image = np.arange(36, dtype=np.uint8).reshape(6, 6)
    mask = np.ones((6, 6), dtype=np.int64)
    assert not (LP._labelled_boundary(mask) > 0).any()
    out = LP.overlay_masks(image, {"cell": mask}, random_outline=True)
    bare = LP.overlay_masks(image, {}, random_outline=True)
    assert np.array_equal(out, bare)      # nothing was painted over it


def test_a_folder_that_cannot_be_grouped_still_shows_one_image(tmp_path,
                                                                monkeypatch):
    """The set grouping is a convenience; a folder whose names the regex
    cannot read is still a folder with a picture in it."""
    tifffile.imwrite(tmp_path / "not_a_yokogawa_name.tif", _tile(9))
    monkeypatch.setattr(
        LP, "enumerate_image_sets",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("unreadable names")))
    payload = LP.load_source_payload(tmp_path)
    assert payload["error"] == ""
    assert payload["sets"] is None
    assert Path(payload["path"]).name == "not_a_yokogawa_name.tif"
    assert int(np.max(payload["array"])) == 9
