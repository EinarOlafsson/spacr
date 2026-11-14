"""A barcode regex is only usable if it captures all three groups from a read.

Compiling is not enough. The sequencing pipeline writes one mapping row per
read from ``columnID``, ``grna`` and ``rowID``, so an expression that compiles
and defines all three groups can still be useless in two ways this widget has
to catch before the run starts:

* it does not match the read at all -- every read is discarded and the mapping
  table comes out empty;
* it matches but one group captures nothing -- the mapping table comes out
  full of rows with a blank barcode, which is worse, because that looks like
  data.

Saving is refused in both cases: a dialog that accepted an unusable expression
would put it into the settings the run then reads.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

_ALL_THREE = r"(?P<columnID>[ACGT]{2})(?P<grna>[ACGT]*)(?P<rowID>[ACGT]{2})"


def test_the_dialog_refuses_to_save_an_unusable_regex(qtbot):
    """Save on an invalid expression neither accepts nor records it."""
    from spacr.qt.widgets.barcode_regex import BarcodeRegexDialog

    dialog = BarcodeRegexDialog("(?P<columnID>[ACGT]{2})")
    qtbot.addWidget(dialog)
    dialog._sample_input.setPlainText("ACGT")

    dialog._save()

    assert dialog.regex == ""
    assert dialog.result() != BarcodeRegexDialog.Accepted


def test_the_dialog_saves_a_usable_regex(qtbot):
    """The same button does record the expression once it validates."""
    from spacr.qt.widgets.barcode_regex import BarcodeRegexDialog

    dialog = BarcodeRegexDialog(_ALL_THREE)
    qtbot.addWidget(dialog)
    dialog._sample_input.setPlainText("ACGGGGTT")

    dialog._save()

    assert dialog.regex == _ALL_THREE


def test_an_accepted_tester_replaces_the_field(qtbot, monkeypatch):
    """What the tester accepted becomes the field's value."""
    from PySide6.QtWidgets import QDialog

    from spacr.qt.widgets import barcode_regex as mod

    def _accept(self):
        self.regex = _ALL_THREE
        return QDialog.Accepted

    monkeypatch.setattr(mod.BarcodeRegexDialog, "exec", _accept)

    widget = mod.BarcodeRegexWidget("(?P<columnID>x)")
    qtbot.addWidget(widget)
    seen = []
    widget.valueChanged.connect(seen.append)

    widget._open_tester()

    assert widget.get_value() == _ALL_THREE
    assert seen == [_ALL_THREE]


def test_a_cancelled_tester_leaves_the_field_alone(qtbot, monkeypatch):
    """Cancelling the tester does not touch what was already typed."""
    from PySide6.QtWidgets import QDialog

    from spacr.qt.widgets import barcode_regex as mod

    monkeypatch.setattr(mod.BarcodeRegexDialog, "exec",
                        lambda self: QDialog.Rejected)

    widget = mod.BarcodeRegexWidget("(?P<columnID>x)")
    qtbot.addWidget(widget)

    widget._open_tester()

    assert widget.get_value() == "(?P<columnID>x)"
