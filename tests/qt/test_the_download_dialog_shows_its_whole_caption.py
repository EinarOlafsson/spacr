"""The demo-data download dialog clipped the filename and the count line.

Reported 2026-08-31, opening "load example data" in Mask Generation: "the
text number and % test is cut off".

A ``QProgressDialog`` takes its width from the label it is CONSTRUCTED
with. This one is constructed with "Preparing…" -- eleven characters --
and then spends the entire download showing

    Downloading <a Yokogawa-length filename>
    (3/6 files)

which is both wider and taller. The filename was clipped at the dialog
edge and the second line fell outside the dialog altogether.

Everything here is MEASURED against font metrics rather than asserted
structurally. "The label has word wrap set" is a property of the code;
"the text fits in the space it is given" is a property of the rendering,
and only the second one is the bug.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt import hf_download


@pytest.fixture
def dialog(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """The real dialog, built the way ``download_toxo_mito_demo`` builds it.

    Driven through the module's own construction path rather than
    rebuilt here -- a hand-made QProgressDialog would pass these tests
    while the shipped one still clipped.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    parent = QWidget()
    qtbot.addWidget(parent)
    made = {}

    def capture(dlg, thread, worker, host, done):
        made["dlg"] = dlg
        made["ui"] = _RealUI(dlg, thread, worker, host, done)
        return made["ui"]

    _RealUI = hf_download._HFDownloadUI
    monkeypatch.setattr(hf_download, "_HFDownloadUI", capture)
    # The worker thread must not actually start and hit the network.
    monkeypatch.setattr(hf_download.QThread, "start", lambda self: None)
    hf_download.download_toxo_mito_demo(parent, str(tmp_path), lambda *a: None)
    dlg = made["dlg"]
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    return dlg, made["ui"]


def test_the_caption_wraps_instead_of_running_off_the_edge(dialog):
    """A non-wrapping label is what clipped the filename."""
    dlg, _ui = dialog
    label = dlg.spacr_caption
    assert isinstance(label, QLabel)
    assert label.wordWrap() is True


def test_the_longest_filename_fits_across_the_dialog(dialog):
    """The width is chosen from the WIDEST caption, not the first one.

    Measured with the label's own font metrics, because the defect was
    that the dialog was sized for "Preparing…" and never grew.
    """
    dlg, _ui = dialog
    label = dlg.spacr_caption
    needed = label.fontMetrics().horizontalAdvance(hf_download._WIDEST_CAPTION)
    assert dlg.minimumWidth() >= needed, (
        f"the dialog is {dlg.minimumWidth()} px wide and the longest "
        f"caption needs {needed} px, so the filename is clipped")


def test_the_count_line_is_inside_the_dialog_during_a_download(dialog,
                                                               qtbot):
    """The reported symptom, driven: a real progress update, then measure.

    ``(3/6 files)`` is the second line, and it is the one that fell
    outside the dialog entirely. Asserted as "the caption needs no more
    height than the label has", which is the same statement as "nothing
    is cut off the bottom".
    """
    dlg, ui = dialog
    ui.on_progress("plate1_A01_T0001F001L01A01Z01C01.tif", 3, 6)
    qtbot.wait(10)
    label = dlg.spacr_caption
    # The caption reads "50%  (3/6)  <name>" now: the percentage leads,
    # because the name is the part that can be long and a window too narrow
    # for all of it must still show the number.
    assert "(3/6)" in label.text()
    needed = label.heightForWidth(label.width())
    assert label.height() >= needed, (
        f"the caption needs {needed} px of height and the label has "
        f"{label.height()}; the count line is cut off")
    assert label.width() >= label.fontMetrics().horizontalAdvance(
        "(3/6 files)")


def test_the_caption_still_fits_after_the_longest_filename(dialog, qtbot):
    """Every caption the download shows fits, not just the first one.

    Written after a mutation test showed the obvious version was
    vacuous. The first attempt at this fix also GREW the dialog by hand
    on every caption change; deleting that code changed no assertion
    here, because a word-wrapped label inside the dialog's own layout
    already grows to fit. The hand-rolled growth was doing nothing, so
    it is gone -- and this test now measures the property that survived
    rather than the code that did not.
    """
    dlg, ui = dialog
    label = dlg.spacr_caption
    for name, done in (("a.tif", 1),
                       ("plate1_A01_T0001F001L01A01Z01C01.tif", 3),
                       ("a_" + "very_" * 10 + "long_name.tif", 5)):
        ui.on_progress(name, done, 6)
        qtbot.wait(10)
        assert label.height() >= label.heightForWidth(label.width()), (
            f"the caption for {name!r} is cut off vertically")
        assert f"({done}/6)" in label.text()
