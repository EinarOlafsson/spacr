"""A live preview never loads the whole experiment — execution list 5.3b.

The user's report: *"in the live preview for large experiments it seems like
the full experiment in live preview is causing lag and problems"*. It did. The
field-of-view dropdown listed every file in the folder, and the panels rebuild
their selectors on every image they load, so a 384-well plate at 16 fields and
4 channels — 24 576 files — cost a measured 292 ms **per change of field**, plus
182 ms to open the dropdown and 30 MB of resident memory to hold the model.

The fix, and what these tests hold it to:

* the folder is grouped into image **sets** from file names alone, using the
  project's own acquisition regex — nothing is opened to build the list;
* only a bounded random **sample** is loaded, 20 sets by default;
* the sample spans the plate instead of being the first N names, which on a
  plate-ordered folder is all of row A;
* the sample is stable — re-rendering must never silently swap it — and
  changing the maximum, an explicit act, redraws it;
* the panel says out loud that it is showing N of M sets;
* the maximum sits immediately **left** of the thing it sizes: the sets
  dropdown in Measure, Timelapse and Motility, and — since b840a914
  (2026-08-05) made the set table Mask's source selector — the ``Choose
  image…`` button in Mask, where what it sizes is the table below.

The file-opening assertions count **real** ``open`` and ``imread`` calls, not a
counter the code under test maintains.
"""
from __future__ import annotations

import builtins
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from PySide6.QtGui import QShowEvent

from spacr.qt.widgets import live_preview as LP
from spacr.qt.widgets.live_preview import LivePreviewPanel
from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
from spacr.qt.widgets.preview_controls import (
    DEFAULT_MAX_SETS, FLAT_CONTROL_NAME, FlatSpinBox, ImageSetSampler,
    enumerate_image_sets, sample_image_sets, sample_seed,
)
from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel

#: Well rows and columns of the synthetic plate. 24 x 8 fields x 3 channels is
#: 576 files in 192 sets — an order of magnitude past the 20-set cap, and the
#: names are ordered so "the first 20" is unmistakably "the top-left corner".
ROWS = "ABCDEFGH"
COLS = range(1, 4)
FIELDS = range(1, 9)
CHANNELS = range(1, 4)
TOTAL_SETS = len(ROWS) * len(COLS) * len(FIELDS)          # 192
TOTAL_FILES = TOTAL_SETS * len(CHANNELS)                  # 576


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """QPixmap aborts the process when no QGuiApplication exists."""
    return qapp


@pytest.fixture(scope="module")
def big_plate(tmp_path_factory) -> Path:
    """A Yokogawa-named plate far larger than the sample cap."""
    root = tmp_path_factory.mktemp("plate")
    tile = np.arange(64, dtype=np.uint16).reshape(8, 8)
    for row in ROWS:
        for col in COLS:
            for field in FIELDS:
                for chan in CHANNELS:
                    tifffile.imwrite(
                        root / f"plate1_{row}{col:02d}_T0001F{field:03d}"
                               f"L01A01Z01C{chan:02d}.tif", tile)
    assert len(list(root.iterdir())) == TOTAL_FILES
    return root


class OpenCounter:
    """Records every image file the process really opens.

    Wraps :func:`builtins.open` and :func:`tifffile.imread` — the two doors a
    preview can read a pixel through — so the assertions below are about the
    filesystem, not about a variable the widget keeps.
    """

    def __init__(self, monkeypatch):
        self.paths: list = []
        real_open, real_imread = builtins.open, tifffile.imread

        def counting_open(file, *a, **kw):
            self._note(file)
            return real_open(file, *a, **kw)

        def counting_imread(files, *a, **kw):
            self._note(files)
            return real_imread(files, *a, **kw)

        monkeypatch.setattr(builtins, "open", counting_open)
        monkeypatch.setattr(tifffile, "imread", counting_imread)
        monkeypatch.setattr(LP, "tifffile", tifffile, raising=False)

    def _note(self, path):
        try:
            text = os.fspath(path)
        except TypeError:
            return
        if isinstance(text, bytes):
            text = text.decode("utf8", "replace")
        if text.lower().endswith((".tif", ".tiff", ".png", ".npy")):
            self.paths.append(text)

    @property
    def unique(self) -> set:
        return set(self.paths)

    def reset(self) -> None:
        self.paths = []


@pytest.fixture
def counter(monkeypatch) -> OpenCounter:
    return OpenCounter(monkeypatch)


def _entries(combo) -> list:
    return [combo.itemText(i) for i in range(combo.count())]


# ---------------------------------------------------------------------------
# Enumeration reads names, never pixels
# ---------------------------------------------------------------------------

def test_enumeration_groups_the_plate_without_opening_one_file(
        big_plate, counter):
    """576 files become 192 sets of 3 channels, and nothing is opened."""
    sets, channels = enumerate_image_sets(big_plate, LP.SUPPORTED_SUFFIXES)

    assert len(sets) == TOTAL_SETS
    assert channels == ["01", "02", "03"]
    assert all(len(s.channels) == len(CHANNELS) for s in sets)
    # The whole point: grouping a plate is metadata work.
    assert counter.unique == set(), (
        f"enumeration opened {len(counter.unique)} image files")


def test_enumeration_reuses_the_projects_own_acquisition_regex(big_plate):
    """Sets are keyed by the regex's plate/well/field, not by ad-hoc parsing."""
    sets, _ = enumerate_image_sets(big_plate, LP.SUPPORTED_SUFFIXES)
    assert sets[0].key[0] == "plate1"
    assert {s.key[1] for s in sets} == {
        f"{r}{c:02d}" for r in ROWS for c in COLS}
    assert {s.key[2] for s in sets} == {f"{f:03d}" for f in FIELDS}


def test_enumeration_does_not_drag_in_the_scientific_stack():
    """Grouping file names must not import ``spacr.utils``.

    ``spacr.utils`` costs a measured 3.2 s and ~900 MB of RSS because it pulls
    in torch and cellpose. An earlier draft of this feature imported it just to
    read a filename pattern and took time-to-usable from 572 ms to 3957 ms —
    it made the lag it was meant to remove seven times worse.
    """
    code = (
        "import sys;"
        "from spacr.qt.widgets.preview_controls import enumerate_image_sets;"
        "enumerate_image_sets('.', ('.tif',));"
        "print('spacr.utils' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env={**os.environ,
                                         "QT_QPA_PLATFORM": "offscreen"})
    assert out.stdout.strip().endswith("False"), out.stdout + out.stderr


# ---------------------------------------------------------------------------
# Only the sample is ever opened
# ---------------------------------------------------------------------------

def test_preview_opens_only_the_sample_never_the_plate(
        big_plate, counter, qtbot):
    """Walking every entry opens one file per sampled set — not 576."""
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    counter.reset()

    assert panel.load_source_async(big_plate)
    qtbot.waitUntil(lambda: panel._image is not None, timeout=20000)

    sample_size = panel._fov_box.count()
    assert sample_size == DEFAULT_MAX_SETS
    # Landing on the folder decodes exactly the one field being shown.
    assert len(counter.unique) == 1
    shown = _entries(panel._fov_box)

    for index in range(sample_size):
        panel._fov_box.setCurrentIndex(index)
        # The FOV dropdown loads asynchronously, so a user stepping through
        # the list is modelled by waiting for each field to arrive. Firing all
        # twenty into the same event-loop turn would instead measure the
        # supersede logic -- correct behaviour, but not the thing this test is
        # about, and it would under-count the opens.
        qtbot.waitUntil(lambda: not panel._image_loaders, timeout=20000)

    opened = counter.unique
    assert len(opened) == sample_size, (
        f"walked {sample_size} sets but opened {len(opened)} files")
    assert len(opened) < TOTAL_FILES / 10
    # Browsing must not shift the list under the user.
    assert _entries(panel._fov_box) == shown
    # And every file opened really was one of the sampled sets.
    for path in opened:
        name = Path(path).name
        # Well IDs can themselves contain an "F" (F03), so anchor on _T...F.
        found = re.fullmatch(
            r"plate1_(?P<well>[A-Z]\d{2})_T\d+F(?P<field>\d{3})L.*\.tif", name)
        assert found, name
        label = f"{found['well']} f{found['field']} ({len(CHANNELS)}ch)"
        assert label in shown, (
            f"opened {name}, which is not one of the {sample_size} shown sets")


def test_changing_the_maximum_opens_nothing_at_all(big_plate, counter, qtbot):
    """Re-sampling is a list operation; it must not touch the disk."""
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_image(sorted(big_plate.iterdir())[0])
    counter.reset()

    panel._max_sets_box.setValue(60)

    assert panel._fov_box.count() >= 60
    assert counter.unique == set(), (
        f"re-sampling opened {sorted(counter.unique)}")


# ---------------------------------------------------------------------------
# The sample represents the plate
# ---------------------------------------------------------------------------

def test_sample_is_drawn_across_the_plate_not_the_first_n(big_plate):
    """A sample that is the first N names is the top-left corner, not a plate."""
    sets, _ = enumerate_image_sets(big_plate, LP.SUPPORTED_SUFFIXES)
    seed = sample_seed(big_plate, len(sets), DEFAULT_MAX_SETS)
    picked = sample_image_sets(sets, DEFAULT_MAX_SETS, seed)

    assert len(picked) == DEFAULT_MAX_SETS
    keys = [s.key for s in picked]
    assert keys != [s.key for s in sets[:DEFAULT_MAX_SETS]]

    rows = {s.key[1][0] for s in picked}
    assert len(rows) >= len(ROWS) // 2, (
        f"20 sets landed in only rows {sorted(rows)} of {len(ROWS)}")
    # It must reach past the alphabetically-first block entirely.
    assert max(s.key[1] for s in picked) > sets[DEFAULT_MAX_SETS].key[1]
    # ...while still being listed in plate order, so the dropdown reads
    # front-to-back rather than shuffled.
    assert keys == sorted(keys)


def test_panel_opens_on_a_sampled_field_not_on_the_first_file(
        big_plate, qtbot):
    """Auto-loading a folder must land inside the sample it is showing."""
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    assert panel.load_source_async(big_plate)
    qtbot.waitUntil(lambda: panel._image is not None, timeout=20000)

    loaded = panel._sampler.set_for_path(panel._image_path)
    assert loaded is not None
    assert loaded.key in [s.key for s in panel._sampler.sample()]
    assert panel._fov_box.count() == DEFAULT_MAX_SETS


# ---------------------------------------------------------------------------
# Stability: only an explicit act redraws the sample
# ---------------------------------------------------------------------------

def test_rerendering_never_silently_changes_the_sample(big_plate, qtbot):
    """A user comparing settings must keep looking at the same 20 fields."""
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_image(sorted(big_plate.iterdir())[0])
    before = _entries(panel._fov_box)

    for _ in range(5):
        panel._refresh_source_selectors()
    panel._on_display_channel_changed()
    panel.load_image(sorted(big_plate.iterdir())[0])

    assert _entries(panel._fov_box) == before


def test_changing_the_maximum_redraws_the_sample(big_plate, qtbot):
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_image(sorted(big_plate.iterdir())[0])
    before = _entries(panel._fov_box)

    panel._max_sets_box.setValue(40)
    after = _entries(panel._fov_box)

    assert len(after) >= 40
    assert after != before
    # Back to 20 returns the very sample the user started with.
    panel._max_sets_box.setValue(DEFAULT_MAX_SETS)
    assert _entries(panel._fov_box) == before


def test_seed_is_reproducible_across_instances_and_processes(big_plate):
    """The same folder at the same cap always yields the same sets."""
    sets, channels = enumerate_image_sets(big_plate, LP.SUPPORTED_SUFFIXES)
    first, second = ImageSetSampler(20), ImageSetSampler(20)
    first.adopt(big_plate, sets, channels)
    second.adopt(big_plate, sets, channels)

    assert first.seed == second.seed
    assert [s.key for s in first.sample()] == [s.key for s in second.sample()]
    # Seeded from the folder, the count and the cap — nothing per-process.
    assert first.seed == sample_seed(big_plate, TOTAL_SETS, 20)
    # ...and from the folder's *name*, so the same plate previews the same
    # fields whether it is read from a local copy or the NAS it came off.
    assert sample_seed("/nas/acq/plate1", TOTAL_SETS, 20) == \
        sample_seed("/scratch/copy/plate1", TOTAL_SETS, 20)
    assert sample_seed("/x/plate1", TOTAL_SETS, 20) != \
        sample_seed("/x/plate2", TOTAL_SETS, 20)
    # An explicit reshuffle is the other way to move.
    first.reshuffle()
    assert [s.key for s in first.sample()] != [s.key for s in second.sample()]


def test_enumeration_is_cached_so_stepping_through_fields_rescans_nothing(
        big_plate, monkeypatch, qtbot):
    """The 292 ms-per-field cost was a re-listing on every load. Not any more."""
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_image(sorted(big_plate.iterdir())[0])

    scans = []
    real_scandir = os.scandir
    monkeypatch.setattr(
        os, "scandir",
        lambda *a, **kw: (scans.append(a[0] if a else None),
                          real_scandir(*a, **kw))[1])

    for index in range(panel._fov_box.count()):
        panel._fov_box.setCurrentIndex(index)

    assert scans == [], f"re-scanned the folder {len(scans)} times"


# ---------------------------------------------------------------------------
# The panel says it is showing a sample
# ---------------------------------------------------------------------------

def test_ui_states_n_of_m_sets(big_plate, qtbot):
    """A sampled preview must never be mistaken for the whole plate."""
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_image(sorted(big_plate.iterdir())[0])

    # The control itself reads "20 of 192 sets".
    assert panel._max_sets_box.value() == DEFAULT_MAX_SETS
    assert panel._max_sets_box.suffix() == f" of {TOTAL_SETS} sets"

    note = panel.sample_note()
    assert str(DEFAULT_MAX_SETS) in note and str(TOTAL_SETS) in note
    assert "sample" in note.lower()
    # The status line the user is already reading says it too.
    assert str(TOTAL_SETS) in panel._status.text()
    # And so does the dropdown's own tooltip.
    assert str(TOTAL_SETS) in panel._fov_box.toolTip()


def test_a_small_folder_is_not_described_as_a_sample(tmp_path, qtbot):
    """Below the cap nothing is hidden, and the panel must not imply it is."""
    tile = np.zeros((8, 8), dtype=np.uint16)
    for field in range(1, 4):
        tifffile.imwrite(
            tmp_path / f"plate1_A01_T0001F{field:03d}L01A01Z01C01.tif", tile)
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_image(sorted(tmp_path.iterdir())[0])

    assert panel._fov_box.count() == 3
    assert panel.sample_note() == "showing all 3 image sets"
    # The cap cannot claim a bigger plate than exists.
    assert panel._max_sets_box.maximum() == 3


def test_a_clamped_cap_is_pushed_back_into_the_sampler(
        big_plate, tmp_path, qtbot):
    """A small folder clamps the cap, and the dropdown must follow the box.

    Otherwise the box reads "12 of 12 sets" while the dropdown still holds the
    50 the user asked for on the previous, larger folder.
    """
    tile = np.zeros((8, 8), dtype=np.uint16)
    for field in range(1, 6):
        tifffile.imwrite(
            tmp_path / f"plate1_A01_T0001F{field:03d}L01A01Z01C01.tif", tile)

    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_image(sorted(big_plate.iterdir())[0])
    panel._max_sets_box.setValue(50)
    assert panel._sampler.max_sets == 50

    panel.load_image(sorted(tmp_path.iterdir())[0])

    assert panel._max_sets_box.value() == 5      # clamped to what exists
    assert panel._sampler.max_sets == 5          # and the sampler agrees
    assert panel._fov_box.count() == 5
    assert panel.sample_note() == "showing all 5 image sets"


def test_a_folder_the_regex_cannot_parse_still_lists_file_names(
        tmp_path, qtbot):
    """Ad-hoc folders keep the exact labels the FOV dropdown always showed."""
    tile = np.zeros((8, 8), dtype=np.uint16)
    for name in ("alpha.tif", "beta.tif"):
        tifffile.imwrite(tmp_path / name, tile)
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_image(tmp_path / "alpha.tif")

    assert _entries(panel._fov_box) == ["alpha.tif", "beta.tif"]


# ---------------------------------------------------------------------------
# Placement and styling, in all four panels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    MeasurePreviewPanel, TimelapsePreviewPanel, MotilityPreviewPanel,
])
def test_max_sets_control_sits_left_of_the_sets_dropdown(factory, qtbot):
    """The cap sits on the dropdown it sizes, in the three panels that show one.

    LivePreviewPanel was in this list until b840a914 (2026-08-05), "the table
    is the source selector", which took the field dropdown out of its pick row
    — there is no dropdown there to be left of any more. It has its own
    placement test below; it was not dropped.
    """
    panel = factory()
    qtbot.addWidget(panel)
    row = panel._pick_row
    widgets = [row.itemAt(i).widget() for i in range(row.count())]

    assert panel._max_sets_box in widgets
    index = widgets.index(panel._max_sets_box)
    assert index < widgets.index(panel._fov_box)
    # Immediately left of it — nothing gets between them.
    assert widgets[index + 1] is panel._fov_box


def test_max_sets_control_caps_the_mask_preview_table_from_beside_choose(
        tmp_path, qtbot):
    """The Mask preview's cap sizes the set table, and sits last before Choose.

    b840a914 (2026-08-05) made the table the source selector, so the number
    that used to say how many entries the field dropdown offered now says how
    many **rows** the table has. The control did not move house: it is still
    the last thing in the pick row before ``Choose image…``, still to the
    right of the images cap it reads next to.
    """
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    row = panel._pick_row
    widgets = [row.itemAt(i).widget() for i in range(row.count())]

    assert panel._fov_box not in widgets            # the redesign, pinned
    index = widgets.index(panel._max_sets_box)
    assert widgets.index(panel._max_images_box) < index
    # Immediately left of the Choose control — nothing gets between them.
    assert widgets[index + 1] is panel._pick_btn

    tile = np.zeros((8, 8), dtype=np.uint16)
    for field in range(1, 6):
        tifffile.imwrite(
            tmp_path / f"plate1_A01_T0001F{field:03d}L01A01Z01C01.tif", tile)
    panel.load_image(sorted(tmp_path.iterdir())[0])
    assert panel._max_sets_box.value() == 5         # clamped to what exists
    assert panel._set_table.rowCount() == 5         # a row per set

    panel._max_sets_box.setValue(2)
    # Two drawn, plus the set on screen if the draw missed it — the open image
    # always keeps a row to click back to. Never all five: the cap bit.
    assert panel._set_table.rowCount() in (2, 3)
    # The table is a readable form of the same sample the dropdown lists, not
    # a second, differently-populated view of the folder.
    assert panel._set_table.rowCount() == panel._fov_box.count()


@pytest.mark.parametrize("factory", [
    LivePreviewPanel, MeasurePreviewPanel, TimelapsePreviewPanel,
    MotilityPreviewPanel,
])
def test_max_sets_control_wears_the_flat_look(factory, qtbot, qt_theme_applied):
    """Same chrome-free styling as the dropdowns it sits beside."""
    from spacr.qt.theme import active_palette

    panel = factory()
    qtbot.addWidget(panel)
    box = panel._max_sets_box

    assert isinstance(box, FlatSpinBox)
    assert box.objectName() == FLAT_CONTROL_NAME
    box.showEvent(QShowEvent())            # a theme switch must land
    qss = box.styleSheet()
    assert active_palette()["fg"] in qss
    assert "background: transparent" in qss
    assert "border: none" in qss
    # The spin arrows are chrome too, and are stripped like the combo's.
    assert "::up-button" in qss and "::down-button" in qss


@pytest.mark.parametrize("factory", [
    LivePreviewPanel, MeasurePreviewPanel, TimelapsePreviewPanel,
    MotilityPreviewPanel,
])
def test_every_panel_can_state_its_sample(factory, qtbot):
    panel = factory()
    qtbot.addWidget(panel)
    assert callable(panel.sample_note)
    assert isinstance(panel.sample_note(), str)


# ---------------------------------------------------------------------------
# The other three panels sample too
# ---------------------------------------------------------------------------

def test_measure_preview_samples_a_big_merged_folder(tmp_path, counter, qtbot):
    """A measure run's ``merged`` folder holds one array per field of view."""
    merged = tmp_path / "merged"
    merged.mkdir()
    for row in ROWS:
        for field in range(1, 13):
            np.save(merged / f"plate1_{row}01_{field}.npy",
                    np.zeros((8, 8, 8), dtype=np.uint16))
    total = len(ROWS) * 12
    assert total > DEFAULT_MAX_SETS

    panel = MeasurePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_array(str(sorted(merged.iterdir())[0]))
    counter.reset()

    assert panel._fov_box.count() in (DEFAULT_MAX_SETS, DEFAULT_MAX_SETS + 1)
    assert panel._max_sets_box.suffix() == f" of {total} sets"
    assert str(total) in panel.sample_note()

    before = _entries(panel._fov_box)
    panel._refresh_source_selectors()
    assert _entries(panel._fov_box) == before
    panel._max_sets_box.setValue(40)
    assert _entries(panel._fov_box) != before
    assert counter.unique == set()


def test_motility_preview_samples_a_big_plate(tmp_path, qtbot):
    """The groups dropdown lists a sample of the plate's time series."""
    merged = tmp_path / "plate" / "merged"
    merged.mkdir(parents=True)
    for row in ROWS:
        for field in range(1, 13):
            for time in range(1, 3):
                np.save(merged / f"plate1_{row}01_{field}_{time}.npy",
                        np.zeros((3, 8, 8), dtype=np.uint16))
    total = len(ROWS) * 12

    panel = MotilityPreviewPanel()
    qtbot.addWidget(panel)
    assert panel.load_folder(str(tmp_path / "plate"))

    assert len(panel._groups) == total
    assert panel._fov_box.count() in (DEFAULT_MAX_SETS, DEFAULT_MAX_SETS + 1)
    assert str(total) in panel.sample_note()
    # Dropdown entries still carry the (plate, well, field) key as item data.
    assert panel._fov_box.itemData(0) in panel._groups

    before = _entries(panel._fov_box)
    panel._populate_group_box()
    assert _entries(panel._fov_box) == before
    panel._max_sets_box.setValue(50)
    assert _entries(panel._fov_box) != before
    assert panel._fov_box.count() >= 50


def test_timelapse_preview_samples_sibling_sequences(tmp_path, counter, qtbot):
    """A plate of per-field frame folders is sampled like everything else."""
    tile = np.zeros((8, 8), dtype=np.uint16)
    for row in ROWS:
        for field in range(1, 13):
            fov = tmp_path / f"plate1_{row}01_f{field:02d}"
            fov.mkdir()
            for frame in range(2):
                tifffile.imwrite(fov / f"t{frame:03d}.tif", tile)
    total = len(ROWS) * 12

    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    assert panel.load_sequence(tmp_path / f"plate1_{ROWS[0]}01_f01")
    counter.reset()

    assert panel._fov_box.count() in (DEFAULT_MAX_SETS, DEFAULT_MAX_SETS + 1)
    assert panel._max_sets_box.suffix() == f" of {total} sets"
    assert str(total) in panel.sample_note()

    before = _entries(panel._fov_box)
    panel._refresh_source_selectors()
    assert _entries(panel._fov_box) == before
    assert counter.unique == set(), "re-sampling read frames off disk"


# ---------------------------------------------------------------------------
# 5.3 must not regress
# ---------------------------------------------------------------------------

def test_fov_and_channel_dropdowns_still_work(big_plate, qtbot):
    """The dropdowns 5.3 added still select, and still change the image."""
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.load_image(sorted(big_plate.iterdir())[0])

    # 20, or 21 when the file opened by hand is one the draw happened to miss.
    assert panel._fov_box.count() in (DEFAULT_MAX_SETS, DEFAULT_MAX_SETS + 1)
    first = panel._image_path
    panel._fov_box.setCurrentIndex(panel._fov_box.count() - 1)
    qtbot.waitUntil(lambda: panel._image_path != first, timeout=20000)
    assert panel._image is not None
    assert panel.display_channel() is None      # "All channels"
