"""Saved figures keep their bytes while actionable integrity findings stay visible."""

import hashlib
import json

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit, QPushButton

from spacr import plot
from spacr.qt.widgets import save_figure_dialog as saving


@pytest.fixture(autouse=True)
def notices(monkeypatch):
    """Use the real checker and retire every modeless notice after its test."""
    monkeypatch.setenv("SPACR_FIGURE_INTEGRITY", "1")
    yield
    for notice in list(saving._integrity_notices):
        notice.close()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not saving._integrity_notices


def _figure(problem):
    """Real independently sampled panels; only the named defect is planted."""
    figure = Figure(figsize=(4, 2), dpi=80)
    for index in range(2):
        data = np.random.default_rng(index).integers(100, 900, (64, 64), dtype=np.uint16)
        if problem == "saturation" and index == 1:
            data.flat[::20] = 65535
        high = 1000 if problem != "saturation" else 65535
        if problem == "display_range" and index == 1:
            high = 600
        figure.add_subplot(1, 2, index + 1).imshow(data, cmap="gray", vmin=0, vmax=high)
    return figure


def _dialog(qtbot, problem="display_range"):
    """Keep the real preview cloning and writer in the acceptance path."""
    figure = _figure(problem)
    dialog = saving.SaveFigureDialog(figure)
    qtbot.addWidget(dialog)
    dialog.dpi.setValue(80)
    return dialog, figure


@pytest.mark.parametrize("problem,suffix", [
    ("display_range", "png"), ("saturation", "png"), ("lossy_format", "jpg"),
])
def test_actual_save_button_writes_then_shows_actionable_findings(
        qtbot, monkeypatch, tmp_path, problem, suffix):
    """Chooser alone is supplied; the button, preview, checker and files are real."""
    dialog, figure = _dialog(qtbot, problem)
    before = [(a.images[0].get_array().copy(), a.images[0].get_clim()) for a in figure.axes]
    target = tmp_path / f"checked.{suffix}"
    monkeypatch.setattr(saving.QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    dialog.show()
    button = dialog._save
    assert button.isVisible() and button.isEnabled()
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert not dialog.isVisible()
    with Image.open(target) as image:
        image.load()
        assert image.width > 0 and image.height > 0
    sidecar = target.with_name(target.name + ".provenance.json")
    report = json.loads(sidecar.read_text())
    assert report["figure_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert any(f["check"] == problem for f in report["integrity"]["findings"])
    notice, = saving._integrity_notices
    assert notice.isVisible() and not notice.isModal()
    assert QApplication.activeModalWidget() is None
    text = notice.findChild(QPlainTextEdit).toPlainText()
    assert str(target) in text and str(sidecar) in text
    for finding in report["integrity"]["findings"]:
        if finding["severity"] == "warning":
            assert finding["message"] in text
    assert {"display_range": "use one range", "saturation": "no display setting recovers",
            "lossy_format": "Use PNG, TIFF or PDF"}[problem] in text
    for axes, (data, clim) in zip(figure.axes, before):
        np.testing.assert_array_equal(axes.images[0].get_array(), data)
        assert axes.images[0].get_clim() == clim
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (target, sidecar)]
    opened = []
    monkeypatch.setattr("PySide6.QtGui.QDesktopServices.openUrl", lambda url: opened.append(url) or True)
    notice.findChild(QPushButton, "FigureIntegrityOpenFolder").click()
    assert opened[0].toLocalFile() == str(tmp_path)
    notice.close()
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in (target, sidecar)] == hashes


@pytest.mark.parametrize("case", ["clean", "disabled", "line"])
def test_unflagged_exports_do_not_interrupt_the_user(qtbot, monkeypatch, tmp_path, case):
    """Notes alone, disabled checking and non-image figures stay quiet."""
    dialog, figure = _dialog(qtbot, "clean")
    if case == "disabled":
        monkeypatch.setenv("SPACR_FIGURE_INTEGRITY", "0")
    if case == "line":
        figure.clear()
        figure.add_subplot().plot([1, 2, 3])
        dialog.refresh()
    target = tmp_path / "quiet.png"
    assert dialog.save(str(target)) == str(target)
    assert target.is_file()
    assert not saving._integrity_notices


@pytest.mark.parametrize("raises", [False, True])
def test_sidecar_failure_retains_figure_and_says_what_to_fix(
        qtbot, monkeypatch, tmp_path, raises):
    """A failed provenance write must not look like a failed image write."""
    dialog, _ = _dialog(qtbot, "clean")

    def fail_sidecar(*_args):
        """Represent both documented failure forms from the provenance writer."""
        if raises:
            raise OSError("sidecar denied")
        return None

    monkeypatch.setattr(plot, "_finish_integrity", fail_sidecar)
    target = tmp_path / "kept.png"
    assert dialog.save(str(target)) == str(target)
    assert target.is_file()
    notice, = saving._integrity_notices
    text = notice.findChild(QPlainTextEdit).toPlainText()
    assert "Check folder permissions and save again" in text
    assert "Provenance:" not in text
    assert str(target) in text


def test_write_failure_or_cancel_never_announces_a_saved_figure(qtbot, monkeypatch, tmp_path):
    """Failed/cancelled writes produce no misleading success notice."""
    dialog, _ = _dialog(qtbot)
    assert dialog.save(str(tmp_path / "absent" / "failed.png")) == ""
    assert not saving._integrity_notices
    monkeypatch.setattr(saving.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    assert dialog.save() == ""
    assert not saving._integrity_notices


def test_notice_lifetime_is_independent_of_save_dialog(qtbot, tmp_path):
    """Closing the parentless save window cannot discard unread warnings."""
    dialog, _ = _dialog(qtbot)
    dialog.save(str(tmp_path / "saved.png"))
    notice, = saving._integrity_notices
    dialog.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert notice.isVisible()
    notice.reject()
    assert not saving._integrity_notices


def test_notice_presentation_failure_does_not_change_save_result(qtbot, monkeypatch, tmp_path):
    """A display failure after export cannot turn a saved file into an error."""
    dialog, _ = _dialog(qtbot)

    def fail_notice(*_args):
        """Simulate an unavailable notification surface."""
        raise RuntimeError("window gone")

    monkeypatch.setattr(saving, "_show_integrity_notice", fail_notice)
    target = tmp_path / "saved.png"
    assert dialog.save(str(target)) == str(target)
    assert target.is_file()
    assert target.with_name(target.name + ".provenance.json").is_file()
