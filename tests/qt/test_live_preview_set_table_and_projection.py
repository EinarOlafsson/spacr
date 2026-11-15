"""The live preview panel: its set table, its z-projection switch and the loads behind them.

The set table (fields down the side, channels across the top) is Mask's source
selector: clicks, shift-clicks, header clicks and the image cap decide which
files are shown.  The MIP switch projects z-stacks the way the ingest does.
Loads come back from a worker and only the newest one may change the panel.
The plates are Yokogawa-named TIFFs written to disk, because every one of
these paths asks a real question about real file names.  One general test per
behaviour family, each walking a case table with pytest's ``subtests``
fixture; every case builds its own panel.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from spacr.qt.widgets import live_preview as LP  # noqa: E402
from spacr.qt.widgets.live_preview import LivePreviewPanel  # noqa: E402
from spacr.qt.widgets.preview_controls import ImageSet  # noqa: E402

pytestmark = pytest.mark.qt

NO_STACK_TIP = ("No z-stacks here — every field has one plane per channel, "
                "so there is nothing to project.")


def _stack_tip(planes):
    return (f"Max-intensity projection over {planes} z-planes per field and channel "
            f"— the same projection the ingest applies before masking.")


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """QPixmap aborts the process when no QGuiApplication exists."""
    return qapp


def _tile(value: int = 7, shape=(8, 8)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint16)


def _name(field: int, chan: int, z: int = 1) -> str:
    return f"plate1_A01_T0001F{field:03d}L01A01Z{z:02d}C{chan:02d}.tif"


def _flat_plate(root: Path) -> Path:
    """Three fields, three channels, one plane each (value field*10+chan)."""
    root.mkdir()
    for field in range(1, 4):
        for chan in range(1, 4):
            tifffile.imwrite(root / _name(field, chan), _tile(field * 10 + chan))
    return root


def _stack_plate(root: Path, steps=((1, 1), (2, 1)), fields=2, planes=4) -> Path:
    """``fields`` fields; channel ``c`` plane ``z`` holds ``z * step``."""
    root.mkdir()
    for field in range(1, fields + 1):
        for chan, step in steps:
            for z in range(1, planes + 1):
                tifffile.imwrite(root / _name(field, chan, z), _tile(z * step))
    return root


@pytest.fixture
def plates(tmp_path):
    """A flat and a stacked plate in separate folders."""
    return _flat_plate(tmp_path / "flat"), _stack_plate(tmp_path / "stacked")


@pytest.fixture
def make_panel(qtbot):
    """A fresh panel, optionally with an image already loaded."""
    def make(path=None, **kwargs):
        widget = LivePreviewPanel(**kwargs)
        qtbot.addWidget(widget)
        if path is not None:
            assert widget.load_image(path) is True
        return widget
    return make


def _path_at(panel, row: int, col: int) -> str:
    item = panel._set_table.item(row, col)
    return item.data(Qt.UserRole) if item is not None else ""


def _set_shift(monkeypatch, held):
    monkeypatch.setattr(LP.LivePreviewPanel, "_shift_held", lambda self: held)


def test_first_supported_image_is_the_first_visible_image_in_sorted_order(tmp_path, monkeypatch, subtests):
    """Non-images and hidden files are walked past; nothing usable is None; walk errors are raised."""
    def folder(name, files, dirs=()):
        root = tmp_path / name
        root.mkdir()
        for sub in dirs:
            (root / sub).mkdir(parents=True)
        for rel, content in files:
            target = root / rel
            if content is None:
                tifffile.imwrite(target, _tile(5))
            else:
                target.write_text(content, encoding="utf-8")
        return root

    cases = [
        ("non-images before the image", folder("ahead", [
            ("a_readme.txt", "x"), ("b_notes.csv", "x"), ("c_scan.tif", None)]), "c_scan.tif"),
        ("image first", folder("straight", [
            ("a_scan.tif", None), ("z_readme.txt", "x")]), "a_scan.tif"),
        ("settings before the field", folder("settings", [
            ("0_settings.csv", "a,b\n"), ("1_readme.txt", "x"), ("2_field.tif", None)]),
         "2_field.tif"),
        ("first subfolder in sorted order", folder("subfolders", [
            ("B_later/z.tif", None), ("A_first/a.tif", None)], dirs=("B_later", "A_first")),
         "A_first/a.tif"),
        ("hidden files in subfolders", folder("hidden", [
            ("plate/.hidden.tif", None), ("plate/visible.tif", None), ("notes.txt", "x")],
            dirs=("plate",)), "plate/visible.tif"),
        ("only unsupported files", folder("unsupported", [
            ("settings.csv", "a,b\n"), ("notes.txt", "x")]), None),
        ("neither file nor folder", tmp_path / "never-existed", None),
    ]
    for case_id, source, expected in cases:
        with subtests.test(msg=case_id):
            found = LP.first_supported_image(source)
            assert found == (None if expected is None else source / expected)

    with subtests.test(msg="a folder that cannot be walked"), monkeypatch.context() as m:
        def _walk(source, topdown=True, onerror=None, followlinks=False):
            if onerror is not None:
                onerror(PermissionError("nope"))
            return iter(())

        m.setattr(os, "walk", _walk)
        empty = tmp_path / "unwalkable"
        empty.mkdir()
        with pytest.raises(OSError) as caught:
            LP.first_supported_image(empty)
        assert str(caught.value) == f"Could not inspect {empty}: nope"


def test_load_preview_mip_folds_planes_or_refuses(tmp_path, subtests):
    """One plane is returned unchanged; planes fold by maximum; mismatched or no planes are refused."""
    one = tmp_path / "a.tif"
    tifffile.imwrite(one, _tile(3))
    z1, z2, big = tmp_path / "z01.tif", tmp_path / "z02.tif", tmp_path / "big.tif"
    tifffile.imwrite(z1, _tile(1))
    tifffile.imwrite(z2, _tile(4))
    tifffile.imwrite(big, _tile(2, (16, 16)))

    for case_id, paths, expected in [("one plane", [one], _tile(3)),
                                     ("max over planes", [z1, z2, one], _tile(4))]:
        with subtests.test(msg=case_id):
            np.testing.assert_array_equal(LP.load_preview_mip(paths), expected)

    for case_id, paths, error, message in [
            ("planes that disagree", [z1, big], ValueError,
             "plane big.tif is (16, 16), expected (8, 8) — these files are not one z-stack"),
            ("no planes", [], FileNotFoundError, "no readable planes")]:
        with subtests.test(msg=case_id):
            with pytest.raises(error) as caught:
                LP.load_preview_mip(paths)
            assert str(caught.value) == message


def test_set_table_clicks_headers_and_cap_choose_the_shown_cells(make_panel, plates, tmp_path, monkeypatch, subtests):
    """Cells, headers, shift and the image cap decide exactly which files are selected and shown."""
    flat, stacked = plates

    def flat_panel():
        return make_panel(sorted(flat.iterdir())[0], threaded=False)

    with subtests.test(msg="table layout and stack depth"):
        panel = flat_panel()
        table = panel._set_table
        assert (table.rowCount(), table.columnCount()) == (3, 3)
        assert [table.horizontalHeaderItem(c).text() for c in range(3)] == ["ch 0", "ch 1", "ch 2"]
        assert "(4z)" not in table.item(0, 0).text()
        deep = make_panel(sorted(stacked.iterdir())[0])
        assert "(4z)" in deep._set_table.item(0, 0).text()

    with subtests.test(msg="clicking a cell"), monkeypatch.context() as m:
        panel = flat_panel()
        _set_shift(m, False)
        panel._on_set_cell_clicked(1, 2)
        assert str(panel._image_path) == _path_at(panel, 1, 2)
        assert panel._selected_cells == [(1, 2)]
        assert (panel._table_row, panel._table_col) == (1, 2)

    with subtests.test(msg="clicking an empty cell"), monkeypatch.context() as m:
        panel = flat_panel()
        _set_shift(m, False)
        panel._on_set_cell_clicked(0, 0)
        before = panel._image_path
        panel._on_set_cell_clicked(99, 99)
        assert panel._image_path == before
        panel._set_table.setItem(2, 2, type(panel._set_table.item(0, 0))(""))
        panel._on_set_cell_clicked(2, 2)
        assert panel._image_path == before
        assert panel._selected_cells == [(0, 0)]

    for case_id, cap, expected in [("shift-click under a cap of two", 2, [(0, 0), (1, 1)]),
                                   ("shift-click at the default cap", None, [(1, 1)])]:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            panel = flat_panel()
            if cap is None:
                assert panel.max_images() == 1
            else:
                panel._max_images_box.setValue(cap)
            _set_shift(m, False)
            panel._on_set_cell_clicked(0, 0)
            _set_shift(m, True)
            panel._on_set_cell_clicked(1, 1)
            assert panel._selected_cells == expected
            assert str(panel._image_path) == _path_at(panel, 1, 1)

    with subtests.test(msg="channel header keeps the field"), monkeypatch.context() as m:
        panel = flat_panel()
        _set_shift(m, False)
        panel._on_set_cell_clicked(2, 0)
        panel._on_channel_header_clicked(1)
        assert (panel._table_row, panel._table_col) == (2, 1)
        assert str(panel._image_path) == _path_at(panel, 2, 1)

    for case_id, cap, expected in [("shift channel header takes the column", 4, [(0, 1), (1, 1), (2, 1)]),
                                   ("cap keeps the most recent cells", 2, [(1, 1), (2, 1)])]:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            panel = flat_panel()
            _set_shift(m, True)
            panel._max_images_box.setValue(cap)
            panel._on_channel_header_clicked(1)
            assert panel._selected_cells == expected

    with subtests.test(msg="lowering the cap re-applies it"), monkeypatch.context() as m:
        panel = flat_panel()
        _set_shift(m, True)
        panel._max_images_box.setValue(3)
        panel._on_channel_header_clicked(0)
        assert panel._selected_cells == [(0, 0), (1, 0), (2, 0)]
        panel._max_images_box.setValue(1)
        assert panel._selected_cells == [(2, 0)]

    with subtests.test(msg="a carried-over cell with no file paints no selection"), \
            monkeypatch.context() as m:
        ragged = tmp_path / "ragged"
        ragged.mkdir()
        tifffile.imwrite(ragged / _name(1, 2), _tile(12))
        tifffile.imwrite(ragged / _name(2, 1), _tile(21))
        tifffile.imwrite(ragged / _name(2, 2), _tile(22))
        panel = make_panel(ragged / _name(2, 1))
        table = panel._set_table
        assert (table.rowCount(), table.columnCount()) == (2, 2)
        assert table.item(0, 0) is None
        assert panel._selected_cells == [(0, 0)]
        panel._max_images_box.setValue(4)
        assert panel._selected_cells == [(0, 0)]
        _set_shift(m, True)
        table.cellClicked.emit(1, 0)
        assert panel._selected_cells == [(0, 0), (1, 0)]
        panel._sync_table_selection()
        assert table.item(1, 0).isSelected()
        assert len(table.selectedItems()) == 1
        assert (table.currentRow(), table.currentColumn()) == (1, 0)


def test_set_table_population_survives_an_awkward_sample(make_panel, plates, monkeypatch, subtests):
    """No table, a refusing sampler, pins and missing channels each give an exact table."""
    flat, _stacked = plates

    def flat_panel():
        return make_panel(sorted(flat.iterdir())[0], threaded=False)

    with subtests.test(msg="no table builds nothing"), monkeypatch.context() as m:
        panel = flat_panel()
        sampled = []
        m.setattr(panel._sampler, "sample", lambda: sampled.append(1) or [])
        panel._set_table = None
        panel._populate_set_table()
        assert sampled == []

    with subtests.test(msg="sampler refuses"), monkeypatch.context() as m:
        panel = flat_panel()
        m.setattr(panel._sampler, "sample",
                  lambda: (_ for _ in ()).throw(RuntimeError("no enumeration")))
        panel._populate_set_table()
        assert panel._set_table.rowCount() == 0

    with subtests.test(msg="a pinned set the draw missed keeps its row"), monkeypatch.context() as m:
        panel = flat_panel()
        all_sets = list(panel._sampler.sample())
        pinned = all_sets[-1]
        m.setattr(panel._sampler, "sample", lambda: all_sets[:1])
        m.setattr(panel._sampler, "set_for_path", lambda p: pinned)
        panel._pin_path = Path(_path_at(panel, 0, 0))
        panel._populate_set_table()
        labels = [panel._set_table.verticalHeaderItem(r).text()
                  for r in range(panel._set_table.rowCount())]
        assert labels == [all_sets[0].label, pinned.label]

    with subtests.test(msg="a pin the sampler cannot place"), monkeypatch.context() as m:
        panel = flat_panel()
        m.setattr(panel._sampler, "set_for_path",
                  lambda p: (_ for _ in ()).throw(KeyError("unknown")))
        panel._pin_path = Path("/nowhere/x.tif")
        panel._populate_set_table()
        assert panel._set_table.rowCount() == 3

    with subtests.test(msg="a channel a field lacks leaves its cell empty"), monkeypatch.context() as m:
        panel = flat_panel()
        sets = list(panel._sampler.sample())
        sets[0].channels.pop(sorted(sets[0].channels)[-1], None)
        m.setattr(panel._sampler, "sample", lambda: sets)
        panel._populate_set_table()
        assert panel._set_table.item(0, 2) is None
        assert Path(_path_at(panel, 0, 1)).name == _name(1, 2)


def test_the_projection_switch_follows_the_stack_and_the_channel_on_screen(make_panel, plates, tmp_path, monkeypatch, qtbot, subtests):
    """MIP is offered only for stacks, projects the open channel, and never claims a time axis."""
    flat, stacked = plates

    with subtests.test(msg="flat plate disables it"):
        panel = make_panel(sorted(flat.iterdir())[0])
        assert panel._mip_toggle.isEnabled() is False
        assert panel._mip_toggle.toolTip() == NO_STACK_TIP

    with subtests.test(msg="stacked plate enables and redraws projected"):
        panel = make_panel(sorted(stacked.iterdir())[0])
        assert panel._mip_toggle.isEnabled() is True
        assert panel._mip_toggle.toolTip() == _stack_tip(4)
        np.testing.assert_array_equal(panel._image, _tile(1))
        panel._mip_toggle.setChecked(True)
        assert panel._mip_enabled is True
        qtbot.waitUntil(lambda: not panel._image_loaders)
        np.testing.assert_array_equal(panel._image, _tile(4))
        panel._mip_toggle.setChecked(False)
        qtbot.waitUntil(lambda: not panel._image_loaders)
        np.testing.assert_array_equal(panel._image, _tile(1))

    with subtests.test(msg="projection folds only the open channel"):
        panel = make_panel(sorted(stacked.iterdir())[0])
        panel._mip_enabled = True
        assert panel._sampler.set_for_path(panel._image_path).z_count == 4
        np.testing.assert_array_equal(panel._load_for_display(Path(panel._image_path)), _tile(4))

    with subtests.test(msg="projection follows the channel the file belongs to"):
        two_channel = _stack_plate(tmp_path / "two_channel", steps=((1, 1), (2, 10)),
                                   fields=1, planes=3)
        panel = make_panel()
        panel._cell_channel.setValue(1)
        assert panel.load_image(two_channel / _name(1, 2, 1)) is True
        assert panel._mip_toggle.isEnabled() is True
        panel._mip_toggle.setChecked(True)
        qtbot.waitUntil(lambda: not panel._image_loaders)
        np.testing.assert_array_equal(panel._image, _tile(30))

    with subtests.test(msg="a redraw that fails keeps the switch on"), monkeypatch.context() as m:
        panel = make_panel(sorted(stacked.iterdir())[0])
        m.setattr(LP.LivePreviewPanel, "_load_for_display",
                  lambda self, path: (_ for _ in ()).throw(OSError("unreadable")))
        panel._on_mip_toggled(True)
        assert panel._mip_enabled is True

    with subtests.test(msg="stacked then flat turns it off"):
        panel = make_panel(sorted(stacked.iterdir())[0])
        panel._mip_toggle.setChecked(True)
        assert panel._mip_toggle.isChecked() is True
        assert panel.load_image(sorted(flat.iterdir())[0]) is True
        assert (panel._mip_toggle.isChecked(), panel._mip_toggle.isEnabled()) == (False, False)
        assert panel._mip_toggle.toolTip() == NO_STACK_TIP

    with subtests.test(msg="a field identifier is never a time axis"):
        def stack_set(well, fieldid, first):
            return ImageSet(key=("plate1", well, fieldid), directory="/nowhere",
                            channels={"01": first}, planes={"01": [first, f"{first}_b"]})

        panel = make_panel()
        for fieldid, cap in (("", 1), ("001", 2)):
            panel._sampler.adopt("/nowhere", [stack_set("A01", fieldid, "a.tif"),
                                              stack_set("A02", fieldid, "c.tif")], ["01"])
            panel._max_sets_box.setValue(cap)
            assert panel._mip_toggle.isEnabled() is True
            assert panel._mip_toggle.toolTip() == _stack_tip(2)


def test_only_the_newest_load_or_a_new_field_changes_the_panel(make_panel, plates, monkeypatch, subtests):
    """Payloads and dropdown picks are applied only when current and different, with exact status text."""
    flat, _stacked = plates

    def flat_panel():
        return make_panel(sorted(flat.iterdir())[0], threaded=False)

    for case_id, payload, expected in [
            ("failed load", {"error": "permission denied"}, "Load failed: permission denied"),
            ("no supported image", {"path": None, "array": None},
             "No supported preview image found.")]:
        with subtests.test(msg=case_id):
            panel = flat_panel()
            panel._on_source_payload(panel._image_load_token, payload)
            assert panel._status.text() == expected

    with subtests.test(msg="superseded or malformed payload"):
        panel = flat_panel()
        before = panel._status.text()
        panel._on_source_payload(panel._image_load_token - 1, {"error": "stale"})
        panel._on_source_payload(panel._image_load_token, "not a dict")
        assert panel._status.text() == before

    with subtests.test(msg="raising the set cap restates the sample"):
        panel = flat_panel()
        panel._max_sets_box.setValue(2)
        note = panel.sample_note()
        assert note.startswith("showing a random sample of 2 of 3")
        assert panel._set_table.rowCount() == 2
        assert panel._status.text() == note[:1].upper() + note[1:]

    def recording(m):
        loads = []
        m.setattr(LP.LivePreviewPanel, "load_source_async",
                  lambda self, source, **k: loads.append(source))
        return loads

    with subtests.test(msg="a field change while one is loading"), monkeypatch.context() as m:
        panel = flat_panel()
        loads = recording(m)
        panel._loading_fov = True
        panel._on_fov_changed()
        panel._loading_fov = False
        assert loads == []

    with subtests.test(msg="a dropdown entry with no path"), monkeypatch.context() as m:
        panel = flat_panel()
        loads = recording(m)
        panel._fov_box.addItem("no data behind me", None)
        panel._fov_box.setCurrentIndex(panel._fov_box.count() - 1)
        panel._on_fov_changed()
        assert loads == []

    with subtests.test(msg="the exact file already open"), monkeypatch.context() as m:
        panel = flat_panel()
        loads = recording(m)
        m.setattr(panel._sampler, "set_for_path", lambda p: None)
        path = _path_at(panel, 0, 0)
        panel._image_path = Path(path)
        panel._fov_box.addItem("same file", path)
        panel._fov_box.setCurrentIndex(panel._fov_box.count() - 1)
        panel._on_fov_changed()
        assert loads == []

    with subtests.test(msg="another channel of the set on screen"), monkeypatch.context() as m:
        panel = flat_panel()
        loads = recording(m)
        index = panel._fov_box.findData(_path_at(panel, 0, 0))
        assert index == 0
        panel._image_path = Path(_path_at(panel, 0, 1))
        panel._fov_box.setCurrentIndex(index)
        panel._on_fov_changed()
        assert loads == []

    with subtests.test(msg="a different field is loaded"), monkeypatch.context() as m:
        panel = flat_panel()
        loads = recording(m)
        panel._fov_box.setCurrentIndex(1)          # the dropdown's own signal loads it
        assert [str(s) for s in loads] == [panel._fov_box.itemData(1)]


class _Recorder:
    """Stands in for the panel's ``JobRunner`` and counts shutdowns."""

    def __init__(self):
        self.shutdowns = 0

    def pending_jobs(self):
        return 0

    def shutdown(self):
        self.shutdowns += 1


def test_the_panel_builds_closes_and_cleans_every_channel(make_panel, tmp_path, monkeypatch, subtests):
    """Drops are refused on every view present, closing reaches only the runner it has, all channels are cleaned."""
    with subtests.test(msg="a missing canvas is skipped"), monkeypatch.context() as m:
        stash = {}
        real_build = LP.LivePreviewPanel._build_compartment_widgets

        def _drop_mask_view(self):
            real_build(self)
            stash["view"] = self._mask_view
            self._mask_view = None

        m.setattr(LP.LivePreviewPanel, "_build_compartment_widgets", _drop_mask_view)
        maimed = make_panel()
        m.undo()
        assert maimed._src_view.acceptDrops() is False
        assert stash["view"].acceptDrops() is True
        maimed._mask_view = stash["view"]
        whole = make_panel()
        assert (whole._src_view.acceptDrops(), whole._mask_view.acceptDrops()) == (False, False)

    with subtests.test(msg="closing with and without a runner"):
        panel = make_panel()
        real = panel._load_jobs
        recorder = _Recorder()
        panel._load_jobs = recorder
        panel.close()
        assert recorder.shutdowns == 1
        panel._load_jobs = None
        panel.close()
        assert recorder.shutdowns == 1
        assert panel._image_loaders == []
        panel._load_jobs = real

    with subtests.test(msg="all channels with a background are cleaned"):
        arr = np.zeros((8, 8, 3), dtype=np.uint16)
        for index, (dim, bright) in enumerate(((50, 200), (60, 300), (70, 400))):
            arr[:4, :, index] = dim
            arr[4:, :, index] = bright
        path = tmp_path / "three.tif"
        tifffile.imwrite(path, arr)
        panel = make_panel(path)
        assert panel._image.shape == (8, 8, 3)
        panel._object_box.setCurrentText("cell + nucleus")
        panel._cell_channel.setValue(0)
        panel._nucleus_channel.setValue(1)
        panel._common_widgets["remove_background"].setChecked(True)
        panel._common_widgets["background"].setValue(100)
        assert panel.display_channel() is None
        shown = panel._display_image()
        expected = arr.copy()
        expected[:4, :, 0] = 0
        expected[:4, :, 1] = 0
        np.testing.assert_array_equal(shown, expected)
        np.testing.assert_array_equal(panel._image, arr)
