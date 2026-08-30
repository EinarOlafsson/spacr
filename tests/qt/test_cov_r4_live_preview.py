"""Live preview — the branch a condition never took, one arc at a time.

``live_preview.py`` has no unexecuted *lines* left. What it still has is
the other side of a dozen conditions: the loop that never ran twice, the
guard that was never false, the ``for`` that always broke before it ran
out. This file goes after exactly those, and nothing else:

* :func:`first_supported_image` walking **past** a file it cannot use,
  and :func:`load_source_payload` when the sample draws nothing;
* the panel's constructor when one of the two canvases is missing;
* ``_load_for_display`` picking the projected channel — by channel name
  when the file is not the first channel's, and by falling back to the
  set's first channel when the enumeration does not name the file at all;
* ``shutdown`` with no runner to shut down;
* a **cap change with no image loaded**, which must enumerate nothing;
* the naming-dialect walk giving up after twelve ancestors;
* a selected cell that has **no file behind it** — skipped when the
  selection is painted, and dropped before it can be loaded;
* a z-stack with **no time axis**, whose tooltip must not mention one;
* the sample announcement when there is no sample yet;
* the channel dropdown with **nothing to re-select**;
* "All channels" cleaning a **second** channel's background as well as
  the first;
* the three ``hasattr(self, "_compartment_widgets")`` guards, which are
  what the panel looks like before ``_build_compartment_widgets`` runs.

Offscreen, offline, no Cellpose: every mask here is handed in through
``_on_worker_done``'s documented direct-call hatch.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget                            # noqa: E402

from spacr.qt.widgets import live_preview as LP                  # noqa: E402
from spacr.qt.widgets.preview_controls import (                  # noqa: E402
    ImageSet, populate_channel_combo,
)

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _qapp(qapp):
    """QPixmap aborts the process outright when no QGuiApplication exists."""
    return qapp


def _tile(value: int, shape=(8, 8)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint16)


def _name(field: int, chan: int, z: int = 1) -> str:
    return f"plate1_A01_T0001F{field:03d}L01A01Z{z:02d}C{chan:02d}.tif"


@pytest.fixture
def flat_plate(tmp_path: Path) -> Path:
    """Three fields, one channel, one plane each."""
    root = tmp_path / "flat"
    root.mkdir()
    for field in range(1, 4):
        tifffile.imwrite(root / _name(field, 1), _tile(field * 10))
    return root


@pytest.fixture
def stack_plate(tmp_path: Path) -> Path:
    """One field, two channels, three z-planes each.

    Channel 1's planes are 1/2/3 and channel 2's are 10/20/30, so the two
    projections are told apart by value alone.
    """
    root = tmp_path / "stacked"
    root.mkdir()
    for chan, step in ((1, 1), (2, 10)):
        for z in range(1, 4):
            tifffile.imwrite(root / _name(1, chan, z), _tile(z * step))
    return root


@pytest.fixture
def panel(qtbot):
    widget = LP.LivePreviewPanel()
    qtbot.addWidget(widget)
    return widget


@contextlib.contextmanager
def _without_compartment_widgets(panel):
    """Put the panel back into the state it is in *during* ``__init__``.

    ``_build_compartment_widgets`` is the line after ``_build_ui`` in the
    constructor, so there is no public moment at which a caller can observe
    a panel that has not run it yet — and three methods carry a
    ``hasattr(self, "_compartment_widgets")`` guard for exactly that state.
    The attribute is removed and put straight back rather than the panel
    being half-built by hand.
    """
    saved = panel._compartment_widgets
    del panel._compartment_widgets
    try:
        yield
    finally:
        panel._compartment_widgets = saved


# ---------------------------------------------------------------------------
# first_supported_image / load_source_payload
# ---------------------------------------------------------------------------

def test_the_first_image_is_found_past_the_files_that_are_not_images(tmp_path):
    """A folder whose first name sorts before any image still finds one."""
    ahead = tmp_path / "ahead"
    ahead.mkdir()
    (ahead / "a_readme.txt").write_text("not an image")
    (ahead / "b_notes.csv").write_text("also not an image")
    tifffile.imwrite(ahead / "c_scan.tif", _tile(5))

    found = LP.first_supported_image(ahead)
    assert found is not None and found.name == "c_scan.tif", (
        "the scan sorts third; the two files before it must be walked past, "
        "not treated as the answer")

    # And the case that never leaves the first iteration, so the assertion
    # above is about the walking rather than about sorting.
    straight = tmp_path / "straight"
    straight.mkdir()
    tifffile.imwrite(straight / "a_scan.tif", _tile(5))
    (straight / "z_readme.txt").write_text("not an image")
    assert LP.first_supported_image(straight).name == "a_scan.tif"


def test_a_draw_that_returns_nothing_leaves_the_first_image_in_place(
        flat_plate, monkeypatch):
    """An empty sample must not blank the path the discovery already found.

    ``sample_image_sets`` cannot return an empty list for a non-empty input
    as written (it returns the input itself whenever the cap does not bite),
    so the ``if picked:`` fallback is only reachable through the seam the
    function is imported by. Patching it is what makes the fallback — keep
    the first supported image — observable at all.
    """
    sets, _channels = LP.enumerate_image_sets(flat_plate, LP.SUPPORTED_SUFFIXES)
    assert len(sets) == 3
    drawn = LP.sample_image_sets(
        sets, 1, LP.sample_seed(flat_plate, len(sets), 1))
    assert len(drawn) == 1

    real = LP.load_source_payload(str(flat_plate), max_sets=1)
    assert Path(real["path"]) == drawn[0].path(), (
        "the normal path opens on the sampled set")

    monkeypatch.setattr(LP, "sample_image_sets", lambda *a, **k: [])
    empty = LP.load_source_payload(str(flat_plate), max_sets=1)
    assert Path(empty["path"]) == LP.first_supported_image(flat_plate), (
        "with nothing drawn the discovered image must survive, not be lost")
    assert empty["array"] is not None
    assert empty["sets"] is not None and len(empty["sets"]) == 3, (
        "the enumeration is still adopted even when the draw came back empty")


# ---------------------------------------------------------------------------
# __init__: turning drops off on the canvases
# ---------------------------------------------------------------------------

def test_a_missing_canvas_does_not_stop_the_other_refusing_drops(
        qtbot, monkeypatch):
    """The constructor skips a view it does not have and still does the rest.

    Both views are created unconditionally by ``_build_ui``, so the only way
    to reach the ``if _v is not None`` false side is to take one away between
    ``_build_ui`` and the loop — which is what the wrapper below does.
    """
    stash = {}
    real_build = LP.LivePreviewPanel._build_compartment_widgets

    def _drop_mask_view(self):
        real_build(self)
        stash["view"] = self._mask_view
        self._mask_view = None

    monkeypatch.setattr(LP.LivePreviewPanel, "_build_compartment_widgets",
                        _drop_mask_view)
    maimed = LP.LivePreviewPanel()
    qtbot.addWidget(maimed)
    monkeypatch.undo()

    assert maimed._src_view.acceptDrops() is False, (
        "the view that WAS there must still have had drops turned off")
    assert stash["view"].acceptDrops() is True, (
        "the view the panel could not see must have been skipped, not "
        "reached through")
    maimed._mask_view = stash["view"]

    whole = LP.LivePreviewPanel()
    qtbot.addWidget(whole)
    assert whole._src_view.acceptDrops() is False
    assert whole._mask_view.acceptDrops() is False, (
        "with both views present both refuse drops, so the panel's own "
        "handlers see them")


# ---------------------------------------------------------------------------
# _load_for_display: which channel gets projected
# ---------------------------------------------------------------------------

def test_a_projection_follows_the_channel_the_open_file_belongs_to(
        panel, stack_plate):
    """Opening channel 2 and switching MIP on projects channel 2's planes."""
    second = stack_plate / _name(1, 2, 1)
    assert panel.load_image(second) is True
    assert panel._mip_toggle.isEnabled(), (
        "three planes per channel is a stack; the switch has to offer itself")

    panel._mip_toggle.setChecked(True)

    assert panel._image.max() == 30, (
        "channel 2's planes are 10/20/30, so its projection is 30")
    assert panel._image.min() == 30
    # Channel 1's projection is 3. Reaching it would mean the loop stopped at
    # the first channel instead of walking on to the one the file names.
    assert panel._image.max() != 3


def test_a_set_that_does_not_name_the_open_file_projects_its_first_channel(
        panel, stack_plate, tmp_path, monkeypatch):
    """An enumeration that disagrees with the file on screen still projects.

    ``ImageSetSampler.set_for_path`` indexes by the very file names it stores
    in ``channels``, so it can never hand back a set that does not name the
    path it was asked about — the two ``for``s that walk off their ends are
    only reachable through that one method. Replacing it (and nothing else)
    is the smallest seam that produces the disagreement.
    """
    odd_dir = tmp_path / "elsewhere"
    odd_dir.mkdir()
    tifffile.imwrite(odd_dir / "alpha.tif", _tile(7))
    tifffile.imwrite(odd_dir / "beta.tif", _tile(900))
    stranger = ImageSet(key=("plate1", "A01", "001"), directory=str(odd_dir),
                        channels={"01": "alpha.tif"},
                        planes={"01": ["alpha.tif", "beta.tif"]})
    assert stranger.z_count == 2

    assert panel.load_image(stack_plate / _name(1, 1, 1)) is True
    monkeypatch.setattr(panel._sampler, "set_for_path", lambda _p: stranger)

    panel._mip_toggle.setChecked(True)

    assert panel._image.max() == 900, (
        "with no channel matching the file name the set's first channel is "
        "projected: max(7, 900)")
    assert panel._image.min() == 900
    assert panel._image.max() != 3, (
        "3 is the projection of the file's own real stack, which this "
        "sampler no longer knows about")


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------

class _Recorder:
    """Stands in for the panel's :class:`JobRunner` and counts shutdowns."""

    def __init__(self):
        self.shutdowns = 0

    def pending_jobs(self):
        return 0

    def shutdown(self):
        self.shutdowns += 1


def test_closing_a_panel_with_no_runner_shuts_nothing_down(panel):
    """``shutdown`` dispatches to the runner it has, and to nothing when it
    has none. A panel torn down twice must not resurrect the first runner."""
    real = panel._load_jobs
    recorder = _Recorder()
    panel._load_jobs = recorder

    panel.close()
    assert recorder.shutdowns == 1, (
        "closing must abandon the load in flight")

    panel._load_jobs = None
    panel.close()
    assert recorder.shutdowns == 1, (
        "with no runner the close reaches nothing at all — least of all the "
        "runner it used to have")
    assert panel._image_loaders == [], (
        "and the panel still reports no loads in flight")

    panel._load_jobs = real


# ---------------------------------------------------------------------------
# _refresh_source_selectors with nothing loaded
# ---------------------------------------------------------------------------

def test_a_cap_change_before_any_image_enumerates_nothing(panel, flat_plate):
    """Re-drawing the sample with no image loaded must not scan a folder.

    There is no folder to scan yet: ``_image_path`` is ``None`` until the
    first load, and the enumeration is keyed off it.
    """
    assert panel._image_path is None
    assert panel._sampler.directory is None

    panel._max_sets_box.setValue(7)

    assert panel._sampler.directory is None, (
        "nothing was loaded, so nothing may be enumerated")
    assert panel._fov_box.count() == 0
    assert panel.sample_note() == "showing all 0 image sets"

    # The same control, once there IS a path behind it, does enumerate.
    assert panel.load_image(flat_plate / _name(1, 1)) is True
    assert panel._sampler.directory == str(flat_plate)
    assert panel._fov_box.count() == 3


# ---------------------------------------------------------------------------
# _regex_config: how far up the tree it looks
# ---------------------------------------------------------------------------

class _FakeSettingsModel:
    """The two widgets ``_regex_config`` reads off a screen's settings model."""

    def __init__(self, metadata_type: str, custom_regex: str):
        self._widgets = {
            "metadata_type": LP._AsWritten(metadata_type),
            "custom_regex": LP._AsWritten(custom_regex),
        }


def test_the_naming_dialect_is_looked_for_twelve_ancestors_up_and_no_further(
        qtbot):
    """The walk is bounded at twelve, and the bound is where it says it is.

    Called directly: ``_regex_config`` is only reachable in passing, from an
    enumeration that would need a real folder at thirteen levels of nesting
    to say anything about the bound itself.
    """
    chain = [QWidget() for _ in range(13)]
    for child, parent in zip(chain[:-1], chain[1:]):
        child.setParent(parent)
    qtbot.addWidget(chain[-1])

    panel = LP.LivePreviewPanel(chain[0])
    model = _FakeSettingsModel("custom", "PLATE_(?P<wellID>.*)")

    # Thirteenth ancestor: one past the end of the walk.
    chain[12]._settings_model = model
    assert panel._regex_config() == (LP.DEFAULT_METADATA_TYPE, None), (
        "a settings model thirteen levels up is out of reach, so the "
        "defaults stand")

    # Twelfth ancestor: the last one the walk does reach.
    del chain[12]._settings_model
    chain[11]._settings_model = model
    assert panel._regex_config() == ("custom", "PLATE_(?P<wellID>.*)"), (
        "twelve levels up is inside the bound and must be read")


# ---------------------------------------------------------------------------
# The set table: cells with no file behind them
# ---------------------------------------------------------------------------

@pytest.fixture
def ragged_plate(tmp_path: Path) -> Path:
    """Two fields; the first one is missing channel 1.

    So the table's very first cell — the one ``_selected_cells`` starts on —
    has no item at all.
    """
    root = tmp_path / "ragged"
    root.mkdir()
    tifffile.imwrite(root / _name(1, 2), _tile(12))
    tifffile.imwrite(root / _name(2, 1), _tile(21))
    tifffile.imwrite(root / _name(2, 2), _tile(22))
    return root


def test_a_selected_cell_with_no_file_paints_no_selection(
        panel, ragged_plate, monkeypatch):
    """The starting cell can be empty; showing the selection must skip it.

    ``_selected_cells`` starts at ``(0, 0)`` before any table exists, and
    extending a selection carries the old entries over WITHOUT re-checking
    them — so a shift-click on a real cell leaves a file-less one in the
    list, and it is the paint pass that has to cope.
    """
    assert panel.load_image(ragged_plate / _name(2, 1)) is True
    table = panel._set_table
    assert (table.rowCount(), table.columnCount()) == (2, 2)
    assert table.item(0, 0) is None, "field 1 has no channel 1 to show"
    assert panel._selected_cells == [(0, 0)]

    # The cap defaults to one image, which would truncate the carried-over
    # cell straight back off the list again.
    panel._max_images_box.setValue(4)
    assert panel._selected_cells == [(0, 0)], (
        "re-applying the cap cannot promote a file-less cell either")

    monkeypatch.setattr(LP.LivePreviewPanel, "_shift_held", lambda self: True)
    table.cellClicked.emit(1, 0)

    assert panel._selected_cells == [(0, 0), (1, 0)], (
        "the empty cell is carried over by the extend, unvalidated")

    # Selecting loads, and loading rebuilds the table from the folder again,
    # which throws the freshly painted selection away with the old items. So
    # the paint pass -- the same call the click made, with the same stale
    # list still on the panel -- is asked once more, with nothing after it.
    panel._sync_table_selection()

    assert table.item(0, 0) is None, "still nothing behind the first cell"
    assert table.item(1, 0).isSelected(), "the real cell IS selected"
    assert len(table.selectedItems()) == 1, (
        "and it is the only one — the cell with no file contributed nothing")
    assert (table.currentRow(), table.currentColumn()) == (1, 0)


def test_the_cell_a_selection_activates_always_has_a_file_behind_it(
        panel, ragged_plate, monkeypatch):
    """``_set_selection`` filters before it activates, so it cannot load air.

    This is the proof for the un-taken ``if path:`` false arc at the end of
    ``_set_selection``: every candidate has already been through
    ``_cells_with_images``, which drops exactly the cells whose
    ``Qt.UserRole`` data is falsy, and the active cell is always the last of
    what survived. The re-read below is of the same item in the same table.
    """
    assert panel.load_image(ragged_plate / _name(2, 2)) is True
    monkeypatch.setattr(LP.LivePreviewPanel, "_shift_held", lambda self: False)

    # (0, 0) has no file; (1, 0) does. Both are offered together.
    panel._set_selection([(0, 0), (1, 0)], extend=False)

    assert panel._selected_cells == [(1, 0)], (
        "the file-less cell never becomes part of the selection, so it can "
        "never become the active one either")
    assert Path(panel._image_path).name == _name(2, 1), (
        "and the image loaded is the one the surviving cell names")

    # The filter is what does it, not the caller: offered nothing but the
    # file-less cell, the selection does not change at all.
    panel._set_selection([(0, 0)], extend=False)
    assert panel._selected_cells == [(1, 0)]
    assert Path(panel._image_path).name == _name(2, 1)


# ---------------------------------------------------------------------------
# The MIP switch's tooltip
# ---------------------------------------------------------------------------

def _stack_set(well: str, fieldid: str, first: str) -> ImageSet:
    return ImageSet(key=("plate1", well, fieldid), directory="/nowhere",
                    channels={"01": first},
                    planes={"01": [first, f"{first}_b"]})


def test_a_stack_with_no_field_axis_says_nothing_about_time(panel):
    """The time sentence is added only when the sets report a third key part.

    The sets are handed to the sampler through its public ``adopt`` — the
    same door the background loader uses — because the question is about
    what the tooltip says for a given enumeration, not about file names.
    """
    panel._sampler.adopt("/nowhere",
                         [_stack_set("A01", "", "a.tif"),
                          _stack_set("A02", "", "c.tif")], ["01"])
    panel._max_sets_box.setValue(1)

    assert panel._mip_toggle.isEnabled(), "two planes per field is a stack"
    quiet = panel._mip_toggle.toolTip()
    assert "2 z-planes" in quiet
    assert "time axis" not in quiet, (
        "nothing here reports a time axis, so nothing may be claimed about "
        "one")

    panel._sampler.adopt("/nowhere",
                         [_stack_set("A01", "001", "a.tif"),
                          _stack_set("A02", "001", "c.tif")], ["01"])
    panel._max_sets_box.setValue(2)

    talkative = panel._mip_toggle.toolTip()
    assert "2 z-planes" in talkative
    assert "time axis is" in talkative, (
        "with the third key part present the tooltip must say the time axis "
        "is left alone")


# ---------------------------------------------------------------------------
# Announcing the sample
# ---------------------------------------------------------------------------

def test_the_sample_is_announced_only_once_there_is_one(panel, flat_plate):
    """Toggling MIP before any load must not overwrite the status line."""
    panel._status.setText("SENTINEL")
    assert panel.sample_note() == "", "nothing has been enumerated yet"

    panel._mip_toggle.setChecked(True)

    assert panel._status.text() == "SENTINEL", (
        "there is no sample to restate, so the status line is left alone")

    panel._mip_toggle.setChecked(False)
    assert panel.load_image(flat_plate / _name(1, 1)) is True
    panel._max_sets_box.setValue(1)

    note = panel.sample_note()
    assert note.startswith("showing a random sample of 1 of 3")
    assert panel._status.text() == note[:1].upper() + note[1:], (
        "with a sample to restate the status line says it, capitalised")


# ---------------------------------------------------------------------------
# Localising the channel dropdown
# ---------------------------------------------------------------------------

def test_an_empty_channel_dropdown_has_no_selection_to_restore(panel):
    """Localising an emptied combo must not invent an index for it.

    Reached directly: every production caller localises straight after
    ``populate_channel_combo``, which always leaves at least the
    "All channels" entry behind, so the empty case has no route in.
    """
    box = panel._channel_box
    populate_channel_combo(box, 3)
    box.setCurrentIndex(2)

    panel._localise_channel_combo()
    assert box.count() == 4
    assert box.currentIndex() == 2, (
        "a real selection survives the re-captioning")
    assert box.itemData(2) == "Ch 1", (
        "and the English entry is what the item still carries")

    box.clear()
    panel._localise_channel_combo()
    assert box.count() == 0
    assert box.currentIndex() == -1, (
        "with nothing in the combo there is no index to put back")


# ---------------------------------------------------------------------------
# "All channels" background removal
# ---------------------------------------------------------------------------

@pytest.fixture
def three_channel_tif(tmp_path: Path) -> Path:
    """8x8x3: dim/bright pairs at 50/200, 60/300 and 70/400."""
    arr = np.zeros((8, 8, 3), dtype=np.uint16)
    for index, (dim, bright) in enumerate(((50, 200), (60, 300), (70, 400))):
        arr[:4, :, index] = dim
        arr[4:, :, index] = bright
    path = tmp_path / "three.tif"
    tifffile.imwrite(path, arr)
    return path


def test_every_channel_with_a_background_is_cleaned_not_just_the_first(
        panel, three_channel_tif):
    """A composite must not show a cleaned cell channel beside a raw nucleus.

    The second channel is the one that matters: the copy is taken on the
    first channel that has a background, and taking it again on the second
    would throw the first channel's cleaning away.
    """
    assert panel.load_image(three_channel_tif) is True
    assert panel._image.shape == (8, 8, 3)

    panel._object_box.setCurrentText("cell + nucleus")
    panel._cell_channel.setValue(0)
    panel._nucleus_channel.setValue(1)
    panel._common_widgets["remove_background"].setChecked(True)
    panel._common_widgets["background"].setValue(100)
    assert panel.display_channel() is None, "the combo is on All channels"

    shown = panel._display_image()

    assert shown[..., 0].min() == 0 and shown[..., 0].max() == 200, (
        "the cell channel is thresholded")
    assert shown[..., 1].min() == 0 and shown[..., 1].max() == 300, (
        "and so is the nucleus channel — cleaning it must not undo the "
        "cell channel by re-copying the source")
    assert shown[..., 2].min() == 70, (
        "channel 2 belongs to no selected object, so it is left alone")
    assert panel._image[..., 0].min() == 50, (
        "and none of it was written back through the view of the source")


# ---------------------------------------------------------------------------
# The three `hasattr(self, "_compartment_widgets")` guards
# ---------------------------------------------------------------------------

def test_settings_propagate_without_the_compartment_widgets(panel):
    """The channel/model half travels even before the filters exist."""
    panel._common_widgets["background"].setValue(321)
    panel._cell_channel.setValue(2)

    full = panel.settings_for_propagation()
    assert full["cell_background"] == 321
    assert full["adjust_cells"] is False

    with _without_compartment_widgets(panel):
        bare = panel.settings_for_propagation()

    assert "cell_background" not in bare and "adjust_cells" not in bare, (
        "the compartment half cannot be read before it is built")
    assert bare["cell_channel"] == 2 and bare["model_name"] == full["model_name"], (
        "everything that does not come from those widgets still travels")


def test_a_request_built_without_the_compartment_widgets_carries_the_cache(
        panel, flat_plate):
    """``_build_request`` falls back to exactly the cached settings dict.

    Built directly rather than through ``run_preview`` so no Cellpose import
    is needed to look at what the worker would have been handed.
    """
    panel.apply_settings({"my_marker": 42})
    assert panel.load_image(flat_plate / _name(1, 1)) is True

    full = panel._build_request()
    assert full.preprocess_settings["my_marker"] == 42
    assert "adjust_cells" in full.preprocess_settings, (
        "normally the compartment widgets are merged over the cache")

    with _without_compartment_widgets(panel):
        bare = panel._build_request()

    assert bare.preprocess_settings == {"my_marker": 42}, (
        "with no compartment widgets the cached settings are the whole of it")
    assert bare.postprocess_settings == bare.preprocess_settings
    assert bare.object_types == full.object_types


def test_a_recompute_without_the_compartment_widgets_filters_nothing(
        panel, flat_plate):
    """No filter widgets means no filter — every object survives.

    The masks are pushed in through ``_on_worker_done``'s documented
    direct-call hatch (token ``-1``), so no Cellpose is involved.
    """
    assert panel.load_image(flat_plate / _name(1, 1)) is True
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[0:4, 0:4] = 1      # 16 px
    mask[6:7, 6:8] = 2      # 2 px
    panel._cell_channel.setValue(0)
    panel._compartment_widgets["cell"]["min_area"].setValue(8)

    panel._on_worker_done({"cell": mask}, "", -1)
    assert set(np.unique(panel._masks["cell"]).tolist()) == {0, 1}, (
        "the 2-px object is below the 8-px minimum and must go")

    with _without_compartment_widgets(panel):
        panel._on_worker_done({"cell": mask}, "", -1)

    assert set(np.unique(panel._masks["cell"]).tolist()) == {0, 1, 2}, (
        "with no widgets to read the minimum from, nothing is filtered out")
    assert "cell=2" in panel._status.text(), (
        "and the count on the status line says so")
