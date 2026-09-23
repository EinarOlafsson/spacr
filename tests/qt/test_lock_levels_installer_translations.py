"""Current controls display source-bound translations through their real Qt paths."""
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("de", "es", "fr", "sv", "pt", "is", "zh_CN", "ko", "hi")


@pytest.fixture(params=LANGUAGES)
def locale(request, monkeypatch):
    from spacr.qt import i18n

    language = request.param
    monkeypatch.setattr(i18n, "current_language", lambda: language)
    targets = {}
    for filename in ("2026-09-22-lock-levels-starplast-controls.json",
                     "2026-09-22-shared-ruler.json"):
        payload = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                              filename).read_text())
        targets.update({row["source"]: row["translation"] for row in payload["records"]})
    return language, targets


def test_installer_controls_and_callback_prefixes_are_localized(qtbot, tmp_path, locale):
    from spacr.qt.starplast import StarplastInstallDialog

    language, targets = locale
    dialog = StarplastInstallDialog(root=tmp_path)
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == targets["Install Starplast (alpha)"]
    assert dialog.start_button.text() == targets["Install and open"]
    assert dialog.browse.text() == targets["Choose checkout…"]
    assert dialog.status.text() == targets["Ready to install when you choose Install and open."]
    assert str(tmp_path / "starplast") in dialog.explanation.text()
    assert all(value in dialog.explanation.text() for value in ("4 GB", "7 GB", "12 GB"))
    for source in ("Create Starplast environment", "Install pip", "Install Starplast and dependencies"):
        dialog._progress(2, 7, source + ": command output /tmp/example --flag")
        assert dialog.status.text() == targets[source] + ": command output /tmp/example --flag"
    assert dialog._thread is None


def test_levels_labels_and_worker_result_preserve_formatted_values(qtbot, locale):
    from PySide6.QtWidgets import QLabel
    from spacr.qt.screens.make_masks import _LevelsDialog

    language, targets = locale
    dialog = _LevelsDialog(np.arange(100, dtype=np.uint16).reshape(10, 10), (0, 100))
    qtbot.addWidget(dialog)
    try:
        qtbot.waitUntil(lambda: dialog.ready)
        assert dialog.windowTitle() == targets["Levels"]
        labels = {label.text() for label in dialog.findChildren(QLabel)}
        assert {targets["Black cutoff"], targets["White cutoff"]} <= labels
        assert dialog.reset.text() == targets["Reset levels"]
        caption = targets["Full-field intensity histogram · black {low:.4g}, white {high:.4g}"]
        assert dialog.caption.text() == caption.format(low=0, high=99)
        dialog.set_percentiles(20, 80)
        assert dialog.caption.text() == caption.format(low=19.8, high=79.2)
    finally:
        dialog.close()
        qtbot.waitUntil(lambda: dialog._worker.close(timeout=0.05))


def test_shortcut_panel_uses_current_lock_gesture(qtbot, locale):
    from spacr.qt.screens.make_masks import MakeMasksScreen

    language, targets = locale
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    try:
        key, explanation = screen._shortcut_rows["Ctrl+L+right click"]
        assert key.text() == targets["Ctrl+L+right click"]
        assert explanation.text() == targets["Lock / unlock box"]
    finally:
        screen._close_levels()
        screen._magnifier.close()
        screen._canvas.close_enhancer()


def test_ruler_readout_keeps_units_spacing_and_localized_errors(qapp, locale):
    from spacr.qt.widgets.image_ruler import ImageRuler

    language, targets = locale
    ruler = ImageRuler()
    assert ruler.label() == ""
    ruler.start, ruler.end = (0, 0), (3, 4)
    assert ruler.label() == "5.00 px"
    ruler.set_spacing(2, 3)
    assert ruler.label() == "5.00 px · 13.42 µm"
    with pytest.raises(ValueError) as error:
        ruler.set_spacing(0)
    assert str(error.value) == targets["Pixel spacing must be finite and positive."]
    ruler.set_spacing()
    assert ruler.label() == "5.00 px"


def test_preview_ruler_button_and_help_are_localized(qtbot, locale):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    language, targets = locale
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    assert panel._ruler_btn.text() == targets["Ruler"]
    source = ("Drag a line on either image to measure its length in image pixels. "
              "Right-click with Ruler selected to clear it. Turn Ruler off to pan.")
    assert panel._ruler_btn.toolTip() == targets[source]
    panel.close()
