"""The global modal guard in ``tests/qt/conftest.py`` covers every blocking call.

WHY THIS FILE EXISTS. Four separate full-suite runs of ``tests/qt`` have been
stopped by the same thing: a modal dialog opened in a test, with nobody to
click it, blocking for ever. Three were ``exec()`` calls and were fixed by
making ``QDialog.exec`` / ``QMessageBox.exec`` raise. The fourth, found on
2026-08-12 by running the suite under a real X server (``xvfb-run`` with
``QT_QPA_PLATFORM=xcb``), was ``QMessageBox.warning`` — a STATIC convenience
call, which builds the box and spins its event loop entirely inside C++ and
so never consults a Python-level ``exec`` override. Six tests in
``test_make_masks_canvas.py`` hung on it and killed their xdist worker.

The guard is only worth what it covers, and what it covers is invisible: a
missing entry shows up as a run that never finishes, hours later, on
whichever test drew the short straw. So the coverage is asserted here
directly.

Each test checks the MARKER before making a call, so that a regression fails
in milliseconds instead of reproducing the hang this file is about.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QFileDialog,
    QFontDialog,
    QInputDialog,
    QMessageBox,
)


# Every static that constructs a modal and runs its own event loop. Adding a
# name here without adding it to the fixture is the failure this file exists
# to make loud.
BLOCKING_STATICS = [
    (QMessageBox, "about"),
    (QMessageBox, "aboutQt"),
    (QMessageBox, "critical"),
    (QMessageBox, "information"),
    (QMessageBox, "question"),
    (QMessageBox, "warning"),
    (QFileDialog, "getExistingDirectory"),
    (QFileDialog, "getOpenFileName"),
    (QFileDialog, "getOpenFileNames"),
    (QFileDialog, "getSaveFileName"),
    (QInputDialog, "getDouble"),
    (QInputDialog, "getInt"),
    (QInputDialog, "getItem"),
    (QInputDialog, "getMultiLineText"),
    (QInputDialog, "getText"),
    (QColorDialog, "getColor"),
    (QFontDialog, "getFont"),
]


@pytest.mark.parametrize("cls, name", BLOCKING_STATICS,
                         ids=[f"{c.__name__}.{n}" for c, n in BLOCKING_STATICS])
def test_every_blocking_static_dialog_is_guarded(cls, name):
    """A static modal constructor must be replaced, not merely shadowed.

    ``QMessageBox.warning(...)`` does not go anywhere near
    ``QMessageBox.exec`` — Qt builds the box and calls ``exec`` on the C++
    side — so the guard has to replace the static itself.
    """
    assert getattr(getattr(cls, name), "_spacr_modal_guard", False), (
        f"{cls.__name__}.{name} is not covered by _no_unguarded_modals in "
        "tests/qt/conftest.py. A test that reaches it will HANG the whole "
        "run rather than fail.")


def test_a_guarded_static_raises_instead_of_opening_a_dialog():
    """The marker is not decoration: the call itself must refuse.

    The marker is checked first so that a broken guard fails here rather
    than blocking on the very dialog it was supposed to prevent.
    """
    assert getattr(QMessageBox.warning, "_spacr_modal_guard", False)
    with pytest.raises(AssertionError, match="headless test"):
        QMessageBox.warning(None, "title", "text")


def test_exec_is_still_guarded_on_both_dialog_classes(qtbot):
    """The three older hangs were ``exec`` calls; keep them covered.

    ``qtbot`` is here for the QApplication — constructing a QWidget without
    one aborts the process rather than raising.
    """
    for cls in (QDialog, QMessageBox):
        dialog = cls()
        qtbot.addWidget(dialog)
        with pytest.raises(AssertionError, match=r"exec\(\) was called"):
            dialog.exec()


def test_a_test_can_still_answer_a_dialog_it_drives(monkeypatch):
    """The guard must not lock out the tests that legitimately drive a modal.

    ``monkeypatch`` in a test body runs after the autouse fixture, so a test
    that wants to answer a box says so and wins. Without this property the
    guard would be unusable and the next person would delete it.
    """
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    assert QMessageBox.question(None, "t", "?") == QMessageBox.Yes
