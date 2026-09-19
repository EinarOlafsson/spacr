"""The live preview's image table gives each channel a column of its own.

GitHub issue #119, jak18015, spaCR 1.5.0.8 on macOS: in the Mask
generation live preview, after loading test images from a path, "Columns
for channels don't all show up reliably and images are all under a column
called 'ch' and there are no individual channel columns."

Measured on nightly 5b9207b21 by setting ``src`` on a built Mask screen with
the preview open:

* a folder of three-channel TIFFs whose names the naming dialect cannot read
  showed ONE column captioned "ch " -- a channel ID that was the empty
  string -- while the channel dropdown beside it offered Ch 0, Ch 1, Ch 2;
* a cellvoyager folder loaded while the form said ``cq1`` showed the same
  single column, and choosing ``cellvoyager`` afterwards changed nothing
  until something else reloaded the folder;
* the columns were the channels of the SAMPLED fields only, so a channel
  that some fields lack came and went with the random draw.

Each test below loads real files and clicks the table the way a user does.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
tifffile = pytest.importorskip("tifffile")

pytestmark = pytest.mark.qt


def _headers(panel):
    """The column captions the user reads."""
    table = panel._set_table
    return [table.horizontalHeaderItem(c).text()
            for c in range(table.columnCount())]


def _click(qtbot, panel, row, column):
    """Click one cell of the table with the mouse."""
    from PySide6.QtCore import Qt

    table = panel._set_table
    item = table.item(row, column)
    assert item is not None, f"no cell at {row}, {column}"
    table.scrollToItem(item)
    rect = table.visualItemRect(item)
    qtbot.mouseClick(table.viewport(), Qt.LeftButton, pos=rect.center())


def channel_of(panel):
    """The plane the preview is showing, before any display processing."""
    from spacr.qt.widgets.preview_controls import channel_view

    return channel_view(panel._image, panel.display_channel())


def _panel(qtbot):
    """A preview panel that loads inline, shown at a usable size."""
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    panel = LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.resize(1000, 700)
    panel.show()
    return panel


def _multichannel_folder(tmp_path, n_files=3, planes=3):
    """Files in no known naming, each holding ``planes`` channels."""
    folder = tmp_path / "multichannel"
    folder.mkdir()
    for index in range(n_files):
        image = np.zeros((32, 32, planes), dtype=np.uint16)
        for plane in range(planes):
            image[..., plane] = 100 * (plane + 1) + index
        tifffile.imwrite(folder / f"field{index + 1}.tif", image)
    return folder


def _cellvoyager_folder(tmp_path, fields=12, rare_field=5):
    """Yokogawa-named files; channel 04 exists for ONE field only."""
    folder = tmp_path / "cellvoyager"
    folder.mkdir()
    plane = np.ones((32, 32), dtype=np.uint16)
    for field in range(1, fields + 1):
        channels = [1, 2, 3] + ([4] if field == rare_field else [])
        for channel in channels:
            tifffile.imwrite(
                folder / (f"plateA_B02_T0001F{field:03d}"
                          f"L01A01Z01C{channel:02d}.tif"),
                plane * channel)
    return folder


def test_multichannel_files_get_a_column_per_channel(qtbot, tmp_path):
    """The reported folder: every file under 'ch', no channel columns."""
    panel = _panel(qtbot)
    folder = _multichannel_folder(tmp_path)

    assert panel.load_source_async(folder)

    assert _headers(panel) == ["ch 0", "ch 1", "ch 2"]
    assert panel._set_table.rowCount() == 3


def test_clicking_a_channel_column_shows_that_plane(qtbot, tmp_path,
                                                    monkeypatch):
    """A plane column opens its file once and then only switches plane."""
    from pathlib import Path

    from PySide6.QtCore import Qt

    panel = _panel(qtbot)
    folder = _multichannel_folder(tmp_path)
    panel.load_source_async(folder)
    expected = Path(panel._set_table.item(1, 2).data(Qt.UserRole))

    _click(qtbot, panel, 1, 2)

    assert panel._image_path == expected
    assert panel.display_channel() == 2
    np.testing.assert_array_equal(channel_of(panel),
                                  tifffile.imread(expected)[..., 2])

    reads = []
    original = panel.load_image
    monkeypatch.setattr(panel, "load_image",
                        lambda p: reads.append(p) or original(p))
    _click(qtbot, panel, 1, 0)
    assert reads == [], "moving along a row re-read the same file"
    assert panel.display_channel() == 0


def test_a_file_the_naming_cannot_read_is_not_captioned_ch(qtbot, tmp_path):
    """One plane per file and no channel in the name: the column is 'image'."""
    panel = _panel(qtbot)
    folder = tmp_path / "flat"
    folder.mkdir()
    for index in range(3):
        tifffile.imwrite(folder / f"img{index}.tif",
                         np.zeros((32, 32), dtype=np.uint16))

    panel.load_source_async(folder)

    assert _headers(panel) == ["image"]
    assert "ch " not in _headers(panel)


def test_a_channel_some_fields_lack_keeps_its_column(qtbot, tmp_path):
    """The columns come from the folder, not from the random draw."""
    panel = _panel(qtbot)
    folder = _cellvoyager_folder(tmp_path)
    panel.load_source_async(folder)
    panel._max_sets_box.setValue(1)

    for _attempt in range(30):
        sampled = panel._sampler.sample()
        if all("04" not in s.channels for s in sampled):
            break
        panel._sampler.reshuffle()
    else:
        pytest.fail("every draw held the one field with channel 04")
    panel._populate_set_table()

    assert _headers(panel) == ["ch 01", "ch 02", "ch 03", "ch 04"]


def test_choosing_the_naming_after_loading_regroups_the_table(
        qtbot, tmp_path):
    """Load under the wrong naming, then correct it on the Mask form."""
    from PySide6.QtWidgets import QApplication

    from spacr.qt.screens.app_screen import AppScreen

    folder = _cellvoyager_folder(tmp_path)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    screen._on_preview_switch(True)
    model = screen._settings_model
    naming = model._widgets["metadata_type"]
    naming.setCurrentIndex(naming.findText("cq1"))
    model._widgets["src"].setText(str(folder))
    panel = screen._live_preview
    qtbot.waitUntil(lambda: panel._image is not None
                    and panel._set_table.columnCount() > 0, timeout=10000)
    qtbot.waitUntil(lambda: not panel._image_loaders, timeout=10000)
    assert _headers(panel) == ["image"]

    naming.setCurrentIndex(naming.findText("cellvoyager"))
    qtbot.waitUntil(lambda: _headers(panel)[:1] == ["ch 01"], timeout=10000)
    QApplication.processEvents()

    assert _headers(panel) == ["ch 01", "ch 02", "ch 03", "ch 04"]
    assert panel._set_table.rowCount() == 12
