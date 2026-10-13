"""The download window: bar on top, percentage centred, Cancel beside it.

A QProgressDialog was used and its layout is not arrangeable -- label ABOVE the
bar, window sized from whatever caption it was built with. That caption is
"Preparing…" and the window then spends the download showing a file name and a
percentage, so the text was clipped at the window edge. Reported twice, most
recently 2026-09-01: "when downloading the % text is cut off fix that so the %
text is alligned to the center of the window to the left of the cancel button
with the blue bar above it".

Widening it was tried and was never going to be enough: the longest caption
contains a FILE NAME, and there is no longest file name. The text wraps.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from spacr.qt.hf_download import _DownloadDialog, _HFDownloadUI


@pytest.fixture
def dialog(qapp):
    made = _DownloadDialog("Downloading spaCR demo dataset")
    made.resize(560, 130)
    made.show()
    qapp.processEvents()
    yield made
    made.close()
    made.deleteLater()
    qapp.processEvents()


def test_the_bar_is_above_the_text(dialog):
    assert dialog._bar.geometry().bottom() <= dialog.spacr_caption.geometry().top()


def test_the_bar_spans_the_window(dialog):
    assert dialog._bar.width() > dialog.width() * 0.8


def test_the_text_is_centred(dialog):
    assert bool(dialog.spacr_caption.alignment() & Qt.AlignmentFlag.AlignHCenter)


def test_cancel_is_to_the_right_of_the_text(dialog):
    assert dialog._cancel.geometry().left() >= \
        dialog.spacr_caption.geometry().right()


def test_the_text_wraps_rather_than_being_clipped(dialog):
    """The longest caption holds a file name, so no width is wide enough."""
    assert dialog.spacr_caption.wordWrap()


def test_a_long_name_does_not_widen_the_window_past_the_screen(dialog, qapp):
    before = dialog.width()
    dialog.setLabelText("99%  (8300/8300)  " + "a" * 300 + ".tar")
    qapp.processEvents()
    assert dialog.width() <= before + 4, "a long name stretched the window"


# ---------------------------------------------------------------------------
# What the caption says
# ---------------------------------------------------------------------------

def _say(dialog, name, done, total):
    ui = _HFDownloadUI.__new__(_HFDownloadUI)
    ui._dlg = dialog
    _HFDownloadUI.on_progress(ui, name, done, total)
    return dialog.spacr_caption.text()


def test_a_percentage_is_shown(dialog):
    assert _say(dialog, "plate1-data.tar", 2500, 8300).startswith("30%")


def test_the_percentage_comes_first(dialog):
    """The name is the part that can be long, so a window too narrow for all
    of it still shows the number."""
    text = _say(dialog, "a-very-long-archive-name.tar", 1, 4)
    assert text.index("25%") < text.index("a-very-long")


def test_the_counts_are_shown_too(dialog):
    assert "(2500/8300)" in _say(dialog, "x.tar", 2500, 8300)


def test_a_zero_total_does_not_divide_by_zero(dialog):
    assert _say(dialog, "x.tif", 0, 0).startswith("0%")


def test_progress_past_the_total_is_clamped(dialog):
    """A stream that reports more than it promised must not read as 130%."""
    assert _say(dialog, "x.tar", 99, 10).startswith("100%")


def test_the_bar_follows_the_numbers(dialog):
    _say(dialog, "x.tar", 3, 4)
    assert dialog._bar.value() == 3
    assert dialog._bar.maximum() == 4


def test_the_bar_has_no_text_of_its_own(dialog):
    """One percentage on screen, not two that can disagree."""
    assert not dialog._bar.isTextVisible()


# ---------------------------------------------------------------------------
# The API the download flow drives it through
# ---------------------------------------------------------------------------

def test_cancel_reports_itself(dialog):
    seen = []
    dialog.canceled.connect(lambda: seen.append(True))
    assert not dialog.wasCanceled()

    dialog._cancel.click()

    assert seen == [True]
    assert dialog.wasCanceled()


def test_reaching_the_maximum_closes_it(dialog):
    """Otherwise a stuck modal blocks the main thread and Qt shows the
    "Application not responding" prompt."""
    dialog.setAutoClose(True)
    dialog.setMaximum(4)
    dialog.setValue(4)
    assert not dialog.isVisible()


def test_auto_close_off_leaves_it_open(dialog):
    dialog.setAutoClose(False)
    dialog.setMaximum(4)
    dialog.setValue(4)
    assert dialog.isVisible()
