"""Preferences explains a control in its strip, in paragraph form."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox, QLabel

from spacr.qt import preferences as P
from spacr.qt.widgets.hint_bar import HintBar


def test_the_strip_sits_above_the_buttons(qtbot):
    """Asked for 2026-08-28: above Defaults / Close / Open, not below."""
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    layout = dlg.layout()
    bar = dlg.findChildren(HintBar)[0]
    box = dlg.findChildren(QDialogButtonBox)[0]
    assert layout.indexOf(bar) < layout.indexOf(box)


def test_the_standing_sentences_are_gone(qtbot):
    """Two paragraphs sat under the tabs on every visit."""
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    said = " ".join((l.text() or "") for l in dlg.findChildren(QLabel))
    assert "apply instantly" not in said
    assert "Colour-blind mode affects plot colours" not in said


def test_the_resource_buttons_explain_in_a_paragraph():
    """The bulleted promise belongs to the confirmation, not to a hover."""
    from spacr.qt import resource_cleanup

    for action in ("ram", "vram", "cpu", "disk"):
        short = resource_cleanup.summary_text(action)
        long = resource_cleanup.confirmation_text(action)
        assert "•" not in short, f"{action} still hovers a bulleted list"
        assert len(short) < len(long)
        # The limit is the part a user is uneasy about; it must survive.
        assert short.strip().endswith(".")
    # And the long form is untouched, because the confirmation still needs it.
    assert "•" in resource_cleanup.confirmation_text("ram")


def test_the_strip_cannot_grow_without_bound(qtbot):
    """A strip that resized made the dialog jump as the pointer moved."""
    bar = HintBar()
    qtbot.addWidget(bar)
    assert bar.maximumHeight() < 200
    tall = bar.maximumHeight()
    bar.setText("word " * 400)
    assert bar.maximumHeight() == tall


def test_the_help_is_justified(qtbot):
    """Ragged right is most visible in a narrow popup of prose."""
    bar = HintBar()
    qtbot.addWidget(bar)
    assert bar.alignment() & Qt.AlignJustify

    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    tip = HoverTooltip()
    qtbot.addWidget(tip)
    assert tip._label.alignment() & Qt.AlignJustify


def test_nothing_in_preferences_pops_a_floating_tooltip(qtbot):
    """The strip is the answer, not a second one that the window covers."""
    from PySide6.QtWidgets import QWidget

    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    bar = dlg.findChildren(HintBar)[0]

    still_popping = [
        w for w in dlg.findChildren(QWidget)
        if (w.toolTip() or "").strip() and not isinstance(w, HintBar)]
    assert still_popping == [], (
        f"{len(still_popping)} controls answer twice, e.g. "
        f"{[type(w).__name__ for w in still_popping[:5]]}")

    # And the help was MOVED, not merely deleted.
    assert len(getattr(bar, "_hints", {})) > 100
