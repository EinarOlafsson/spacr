"""FOV and channel dropdowns in every live preview, asserted by rendering.

Execution list 5.3. Four modules ship a live preview — Mask, Measure,
Timelapse and Motility — and each now carries a field-of-view dropdown and a
channel dropdown sitting immediately *left* of its ``Choose …`` control, all
three flat and chrome-free like the **Live** toggle beside them.

The tests assert the layout order by index, the flat look against the very
palette entries :class:`AiToggleLabel` paints itself with, and — the part that
matters — that moving either dropdown changes the **pixels on screen**, read
back out of the rendered pixmap rather than out of the setting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from PySide6.QtGui import QImage

from spacr.qt.widgets import live_preview as LP
from spacr.qt.widgets.preview_controls import (
    ALL_CHANNELS, FLAT_CONTROL_NAME, FlatButton, FlatComboBox,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _qapp(qapp):
    """QPixmap aborts the process when no QGuiApplication exists."""
    return qapp


def _rendered(view) -> np.ndarray:
    """Return the RGB pixels currently displayed by a ``_ZoomView``."""
    item = view._pixmap_item
    assert item is not None, "nothing has been rendered into this canvas"
    image = item.pixmap().toImage().convertToFormat(QImage.Format_RGB888)
    width, height = image.width(), image.height()
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8)
    return buffer.reshape(height, image.bytesPerLine())[:, :width * 3] \
        .reshape(height, width, 3).copy()


def _pixel(view, x: int, y: int):
    return tuple(int(v) for v in _rendered(view)[y, x])


def _label_pixels(pixmap) -> np.ndarray:
    image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
    width, height = image.width(), image.height()
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8)
    return buffer.reshape(height, image.bytesPerLine())[:, :width * 3] \
        .reshape(height, width, 3).copy()


def _mask_panel(qtbot):
    panel = LP.LivePreviewPanel()
    qtbot.addWidget(panel)
    return panel


@pytest.fixture
def two_fields(tmp_path: Path):
    """Two single-channel tiles in one folder — two fields of view."""
    first = np.full((32, 32), 100, np.uint16)
    first[4:12, 4:12] = 4000
    second = np.full((32, 32), 100, np.uint16)
    second[20:28, 20:28] = 4000
    a, b = tmp_path / "plate1_A01_1.tif", tmp_path / "plate1_A01_2.tif"
    tifffile.imwrite(str(a), first)
    tifffile.imwrite(str(b), second)
    return a, b


@pytest.fixture
def three_channel_tif(tmp_path: Path) -> Path:
    """A tile whose three channels carry visibly different intensities."""
    arr = np.zeros((24, 24, 3), np.uint16)
    arr[..., 0] = 400
    arr[..., 1] = 12000
    arr[..., 2] = 40000
    path = tmp_path / "channels.tif"
    tifffile.imwrite(str(path), arr)
    return path


# ---------------------------------------------------------------------------
# 5.3 — the controls exist, in the right order, in every preview
# ---------------------------------------------------------------------------

def _mask(qtbot):
    return _mask_panel(qtbot)


def _measure(qtbot):
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
    panel = MeasurePreviewPanel()
    qtbot.addWidget(panel)
    return panel


def _timelapse(qtbot):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    panel = TimelapsePreviewPanel()
    qtbot.addWidget(panel)
    return panel


def _motility(qtbot):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
    panel = MotilityPreviewPanel()
    qtbot.addWidget(panel)
    return panel


#: Every screen that has a live preview, with the label on its Choose control.
PREVIEW_PANELS = (
    ("mask", _mask, "Choose image…"),
    ("measure", _measure, "Choose merged array…"),
    ("timelapse", _timelapse, "Choose sequence…"),
    ("motility", _motility, "Choose plate folder…"),
)


@pytest.mark.parametrize("app_key,build,choose_text", PREVIEW_PANELS,
                         ids=[p[0] for p in PREVIEW_PANELS])
def test_every_live_preview_has_fov_and_channel_left_of_choose(
        qtbot, app_key, build, choose_text):
    panel = build(qtbot)
    row = panel._pick_row
    fov = row.indexOf(panel._fov_box)
    channel = row.indexOf(panel._channel_box)
    choose = row.indexOf(panel._pick_btn if app_key != "timelapse"
                         else panel._seq_btn)
    assert fov >= 0 and channel >= 0 and choose >= 0
    # FOV, then channel, then the Choose control — dropdowns on its left.
    assert fov < channel < choose
    assert (panel._pick_btn if app_key != "timelapse"
            else panel._seq_btn).text() == choose_text


@pytest.mark.parametrize("app_key,build,_choose", PREVIEW_PANELS,
                         ids=[p[0] for p in PREVIEW_PANELS])
def test_the_selectors_wear_the_live_toggle_look(qtbot, app_key, build,
                                                 _choose):
    """Text only: the theme's foreground, 600 weight, body size, no chrome."""
    from spacr.qt.theme import FONT_SIZE, active_palette
    from spacr.qt.widgets.ai_toggle_label import AiToggleLabel

    panel = build(qtbot)
    live = AiToggleLabel(text="Live")
    qtbot.addWidget(live)
    palette = active_palette()
    controls = [panel._fov_box, panel._channel_box,
                panel._seq_btn if app_key == "timelapse" else panel._pick_btn]
    for control in controls:
        assert control.objectName() == FLAT_CONTROL_NAME
        qss = control.styleSheet()
        # The same three declarations the Live toggle paints itself with.
        assert f"color: {palette['fg']}" in qss
        assert f"font-size: {FONT_SIZE['body']}px" in qss
        assert "font-weight: 600" in qss
        # ... and no box chrome of any kind.
        assert "background: transparent" in qss
        assert "border: none" in qss
        assert "border-radius: 0px" in qss
    # The Live toggle really does declare those same values.
    live_qss = live.styleSheet()
    assert f"color: {palette['fg']}" in live_qss
    assert f"font-size: {FONT_SIZE['body']}px" in live_qss
    assert "font-weight: 600" in live_qss


def test_flat_controls_never_translate_their_entries(qtbot):
    """The entries are data read back by text; a language pass must skip them."""
    combo = FlatComboBox()
    qtbot.addWidget(combo)
    assert combo.property("i18nSkipItems") is True


def test_flat_controls_repaint_for_the_theme_in_force_when_shown(qtbot,
                                                                 monkeypatch):
    """A theme changed while the panel was hidden lands on the next show."""
    from spacr.qt.widgets import preview_controls as PC

    button = FlatButton("Choose image…")
    qtbot.addWidget(button)
    assert "#00ff00" not in button.styleSheet()
    real = PC.active_palette

    def _green_fg():
        palette = dict(real())
        palette["fg"] = "#00ff00"
        return palette

    monkeypatch.setattr(PC, "active_palette", _green_fg)
    button.show()
    assert "color: #00ff00" in button.styleSheet()


# ---------------------------------------------------------------------------
# 5.3 — Mask: the dropdowns change the pixels on screen
# ---------------------------------------------------------------------------

def test_mask_fov_dropdown_lists_the_folder_and_switches_the_image(
        qtbot, two_fields):
    first, second = two_fields
    panel = _mask(qtbot)
    assert panel.load_image(first) is True
    assert [panel._fov_box.itemText(i)
            for i in range(panel._fov_box.count())] == [first.name, second.name]
    assert panel._fov_box.currentData() == str(first)

    before = _rendered(panel._src_view)
    panel._fov_box.setCurrentIndex(1)

    # The FOV dropdown now decodes off the GUI thread, so the new image lands
    # a turn or two later rather than inside setCurrentIndex.
    qtbot.waitUntil(lambda: panel._image_path == second, timeout=20000)
    after = _rendered(panel._src_view)
    assert not np.array_equal(before, after), \
        "picking another field of view did not change the displayed image"
    # And it is genuinely the second tile: its bright square moved.
    assert _pixel(panel._src_view, 24, 24) > _pixel(panel._src_view, 6, 6)


def test_mask_channel_dropdown_switches_the_displayed_plane(
        qtbot, three_channel_tif):
    panel = _mask(qtbot)
    panel._normalise_check.setChecked(False)   # raw view: no percentile stretch
    assert panel.load_image(three_channel_tif) is True
    assert [panel._channel_box.itemText(i)
            for i in range(panel._channel_box.count())] == [
        ALL_CHANNELS, "Ch 0", "Ch 1", "Ch 2"]

    all_channels = _pixel(panel._src_view, 5, 5)
    panel._channel_box.setCurrentText("Ch 0")
    dim = _pixel(panel._src_view, 5, 5)
    panel._channel_box.setCurrentText("Ch 2")
    bright = _pixel(panel._src_view, 5, 5)

    assert panel.display_channel() == 2
    assert dim != all_channels
    assert bright != dim
    # Ch 0 is the dimmest plane and Ch 2 the brightest; a single plane renders
    # as grey, so all three components move together.
    assert dim[0] == dim[1] == dim[2]
    assert bright[0] == bright[1] == bright[2]
    assert bright[0] > dim[0]


def test_mask_channel_selection_also_drives_the_overlay(qtbot,
                                                        three_channel_tif):
    panel = _mask(qtbot)
    panel._normalise_check.setChecked(False)
    panel.load_image(three_channel_tif)
    mask = np.zeros((24, 24), np.int32)
    mask[6:18, 6:18] = 1
    panel._masks = {"cell": mask}
    panel._outline_colour.setCurrentText("red")

    panel._channel_box.setCurrentText("Ch 0")
    background_ch0 = _pixel(panel._mask_view, 1, 1)
    panel._channel_box.setCurrentText("Ch 2")
    background_ch2 = _pixel(panel._mask_view, 1, 1)

    assert background_ch0 != background_ch2       # the overlay follows too
    assert _pixel(panel._mask_view, 6, 6) == (240, 60, 60)   # outline survives


# ---------------------------------------------------------------------------
# 5.3 — Measure
# ---------------------------------------------------------------------------

def _measure_array(path: Path, offset: int = 0, flip: bool = False) -> str:
    """A merged array whose channels carry *different* gradients.

    Uniform channels would all rescale to the same flat crop, so a channel
    change would be invisible for the wrong reason. Channel 0 ramps left to
    right, channel 2 ramps top to bottom, and the rest are flat.
    """
    data = np.zeros((48, 48, 5), np.float32)
    ramp = np.linspace(10.0, 250.0, 48, dtype=np.float32)
    horizontal = np.tile(ramp, (48, 1))
    vertical = horizontal.T
    if flip:
        horizontal, vertical = vertical, horizontal
    data[..., 0] = horizontal
    data[..., 1] = 40.0
    data[..., 2] = vertical
    data[..., 3] = 40.0
    mask = np.zeros((48, 48), np.int32)
    mask[4 + offset:24 + offset, 4 + offset:24 + offset] = 1
    data[..., 4] = mask
    np.save(str(path), data)
    return str(path)


def test_measure_fov_dropdown_switches_the_loaded_array(qtbot, tmp_path):
    first = _measure_array(tmp_path / "plate1_A01_f1.npy")
    second = _measure_array(tmp_path / "plate1_A01_f2.npy", flip=True)
    panel = _measure(qtbot)
    panel._mask_dim.setValue(4)
    assert panel.load_array(first) is True
    assert [panel._fov_box.itemText(i)
            for i in range(panel._fov_box.count())] == [
        "plate1_A01_f1.npy", "plate1_A01_f2.npy"]

    before = panel._crops[0]["crop"].copy()
    panel._fov_box.setCurrentIndex(1)

    assert panel._data_path == second
    assert not np.array_equal(before, panel._crops[0]["crop"]), \
        "the crop grid still shows the previous field of view"


def test_measure_channel_dropdown_changes_the_rendered_crops(qtbot, tmp_path):
    path = _measure_array(tmp_path / "plate1_A01_f1.npy")
    panel = _measure(qtbot)
    panel._mask_dim.setValue(4)
    panel._normalise.setChecked(False)
    assert panel.load_array(path) is True
    assert [panel._channel_box.itemText(i)
            for i in range(panel._channel_box.count())] == [
        ALL_CHANNELS, "Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4"]

    panel._channel_box.setCurrentText("Ch 0")
    horizontal = panel._crops[0]["crop"].copy()
    panel._channel_box.setCurrentText("Ch 2")
    vertical = panel._crops[0]["crop"].copy()

    assert panel.display_channel() == 2
    assert not np.array_equal(horizontal, vertical)
    # A single channel is written to all three crop planes, so the crop is
    # grey — and each one carries its own channel's gradient direction.
    assert (vertical[..., 0] == vertical[..., 2]).all()
    # Sampled inside the object (rows/cols 4..23 of the 34px crop), so the
    # masked-out background cannot answer for the gradient.
    plane = horizontal[..., 0].astype(int)
    assert plane[10, 20] > plane[10, 6]          # ch 0 ramps left → right
    plane = vertical[..., 0].astype(int)
    assert plane[20, 10] > plane[6, 10]          # ch 2 ramps top → bottom
    # The grid the user looks at was rebuilt from it.
    thumb_rgb = _label_pixels(panel._crop_pixmap(vertical))
    assert int(thumb_rgb.max()) > 0


def test_measure_channel_dropdown_never_rewrites_the_run_settings(qtbot,
                                                                  tmp_path):
    """It is a view control: png_dims for a real run must not move."""
    path = _measure_array(tmp_path / "plate1_A01_f1.npy")
    panel = _measure(qtbot)
    panel._mask_dim.setValue(4)
    panel.load_array(path)
    before = panel.settings_for_propagation()["png_dims"]
    panel._channel_box.setCurrentText("Ch 3")
    assert panel.settings_for_propagation()["png_dims"] == before


# ---------------------------------------------------------------------------
# 5.3 — Timelapse
# ---------------------------------------------------------------------------

def _frame_folder(root: Path, name: str, value: int) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    for t in range(3):
        frame = np.full((24, 24), value, np.uint16)
        frame[2 + t: 8 + t, 2:8] = value * 3
        tifffile.imwrite(str(folder / f"f{t}.tif"), frame)
    return folder


def test_timelapse_fov_dropdown_switches_the_sequence(qtbot, tmp_path):
    root = tmp_path / "plate"
    first = _frame_folder(root, "field1", 300)
    second = _frame_folder(root, "field2", 9000)
    panel = _timelapse(qtbot)
    panel._normalise.setChecked(False)
    assert panel.load_sequence(str(first)) is True
    assert [panel._fov_box.itemText(i)
            for i in range(panel._fov_box.count())] == ["field1", "field2"]

    before = _rendered(panel._src_view)
    panel._fov_box.setCurrentIndex(1)

    assert panel._sequence.label == str(second)
    after = _rendered(panel._src_view)
    assert not np.array_equal(before, after), \
        "picking another field of view did not change the displayed frame"


def test_timelapse_channel_dropdown_drives_display_and_segmentation(qtbot,
                                                                    tmp_path):
    folder = tmp_path / "multi"
    folder.mkdir()
    for t in range(3):
        frame = np.zeros((24, 24, 3), np.uint16)
        frame[..., 0] = 400
        frame[..., 1] = 12000
        frame[..., 2] = 45000
        tifffile.imwrite(str(folder / f"f{t}.tif"), frame)
    panel = _timelapse(qtbot)
    panel._normalise.setChecked(False)
    assert panel.load_sequence(str(folder)) is True
    assert [panel._channel_box.itemText(i)
            for i in range(panel._channel_box.count())] == [
        "Ch 0", "Ch 1", "Ch 2"]

    panel._channel_box.setCurrentText("Ch 0")
    dim = _pixel(panel._src_view, 5, 5)
    panel._channel_box.setCurrentText("Ch 2")
    bright = _pixel(panel._src_view, 5, 5)

    assert dim != bright and bright[0] > dim[0]
    # One channel, two surfaces: the segmentation spinner followed along.
    assert int(panel._channel.value()) == 2
    panel._channel.setValue(1)
    assert panel._channel_box.currentText() == "Ch 1"


# ---------------------------------------------------------------------------
# 5.3 — Motility
# ---------------------------------------------------------------------------

def _motility_plate(root: Path) -> str:
    """Two fields, six planes; object drift differs between the fields."""
    merged = root / "merged"
    merged.mkdir(parents=True)
    for field, drift in ((1, 3), (2, 0)):
        for t in range(4):
            arr = np.zeros((6, 32, 32), np.float32)
            for channel in range(4):
                arr[channel] = 10 * (channel + 1)
            cell = np.zeros((32, 32), np.int32)
            yy, xx = np.ogrid[:32, :32]
            cell[(yy - 8) ** 2 + (xx - (6 + drift * t)) ** 2 <= 9] = 1
            arr[4] = cell
            arr[5] = np.zeros((32, 32), np.int32)
            np.save(str(merged / f"plate1_A01_{field}_{t}.npy"), arr)
    return str(root)


def test_motility_fov_dropdown_lists_the_groups_and_redraws(qtbot, tmp_path):
    panel = _motility(qtbot)
    assert panel.load_folder(_motility_plate(tmp_path / "plate")) is True
    assert panel._fov_box.count() == 2
    assert panel._fov_box is panel._group_box       # historical name kept

    with qtbot.waitSignal(panel.preview_ready, timeout=20000):
        panel.run_preview()
    first = _label_pixels(panel._plot.pixmap())

    panel._fov_box.setCurrentIndex(1)
    assert panel._points is None, "the stale point table survived"
    with qtbot.waitSignal(panel.preview_ready, timeout=20000):
        panel.run_preview()
    second = _label_pixels(panel._plot.pixmap())

    assert not np.array_equal(first, second), \
        "the figure still shows the previous field of view"


def test_motility_channel_dropdown_is_bound_to_the_tracked_plane(qtbot,
                                                                 tmp_path):
    panel = _motility(qtbot)
    assert panel.load_folder(_motility_plate(tmp_path / "plate")) is True
    assert [panel._channel_box.itemText(i)
            for i in range(panel._channel_box.count())] == [
        f"Ch {i}" for i in range(6)]
    assert panel._channel_box.currentText() == \
        f"Ch {int(panel._tracked_plane.value())}"

    panel._channel_box.setCurrentText("Ch 5")
    assert int(panel._tracked_plane.value()) == 5
    assert panel.display_channel() == 5
    panel._tracked_plane.setValue(4)
    assert panel._channel_box.currentText() == "Ch 4"


