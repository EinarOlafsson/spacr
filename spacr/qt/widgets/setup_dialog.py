"""The first-run setup screen (instruction 221).

THE INSTALLER IS THE WRONG PLACE. It runs once, often unattended, sometimes
by an administrator who is not the user, and it asks its questions before
the person has seen a single screen of the thing they are configuring --
every answer is a guess. The first RUN is the first moment the questions
mean anything, and the only moment when the answers can be changed by the
person they affect.

NINE QUESTIONS IS A LOT FOR A FIRST SCREEN, so they are GROUPED and every
one has a working default: the screen can be dismissed without answering
anything, and nothing is worse for having been skipped.

INVARIANTS 10: decoration must never be load-bearing. If the blur, the
translucency or the corner accent cannot be drawn, the dialog is a plain
dialog with the same controls and the same answers -- see `_backdrop`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFormLayout, QGridLayout,
                               QGraphicsBlurEffect, QGraphicsScene,
                               QGraphicsPixmapItem, QGraphicsView, QLabel,
                               QVBoxLayout, QWidget)

LOG = logging.getLogger("spacr.qt.setup_dialog")

#: How the questions are grouped, and the order the groups appear in.
#:
#: NOT A LIST IN A COLUMN. They are not equal: LANGUAGE AND THEME CHANGE
#: WHAT THE NEXT SCREEN LOOKS LIKE, so they come first and their effect is
#: immediate; the reproducibility hash is a decision nobody can evaluate
#: before their first run, so it sits under a heading that says what it is
#: for rather than beside the theme.
GROUPS: List[tuple] = [
    ("How it looks", ("language", "theme", "colour_blind")),
    ("How it runs", ("spacr_mode", "hash_inputs")),
    ("The assistant", ("ai_provider", "ai_default")),
    ("When something breaks", ("issue_prompt", "share_logs")),
]

#: How far the home screen behind is blurred.
BLUR = 18.0


class SetupDialog(QDialog):
    """The setup screen, over a blurred snapshot of what is behind it."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Set spaCR up")
        self.setModal(True)
        self._editors: Dict[str, QWidget] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._backdrop_view = self._backdrop(parent)
        if self._backdrop_view is not None:
            outer.addWidget(self._backdrop_view)

        from .setup_card import SetupCard

        self.card = SetupCard(self)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(14)

        title = QLabel("<b>Set spaCR up</b>")
        card_layout.addWidget(title)
        blurb = QLabel(
            "Every one of these has a working default, so you can close this "
            "and change any of them later in Preferences.")
        blurb.setWordWrap(True)
        blurb.setObjectName("Muted")
        card_layout.addWidget(blurb)

        self._build_groups(card_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        # "Not now" rather than "Cancel": nothing is being cancelled, and a
        # user who reads Cancel as "undo what I already have" will not press
        # it even when it is the right button.
        buttons.button(QDialogButtonBox.Cancel).setText("Not now")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        card_layout.addWidget(buttons)

        if self._backdrop_view is None:
            outer.addWidget(self.card)
        else:
            self.card.setParent(self)
            self.card.raise_()
        self.resize(680, 560)

    # ------------------------------------------------------------- the box

    def _build_groups(self, layout) -> None:
        """One form per group, from the model's own question list."""
        from ..setup_screen import current, questions

        asked = {q[0]: q for q in questions()}
        answers = current()
        for heading, keys in GROUPS:
            here = [asked[k] for k in keys if k in asked]
            if not here:
                # A GROUP WITH NOTHING IN IT IS NOT DRAWN. The provider
                # question removes itself when no CLI is installed, and an
                # empty "The assistant" heading would read as a bug.
                continue
            label = QLabel(f"<b>{heading}</b>")
            layout.addWidget(label)
            form = QFormLayout()
            form.setContentsMargins(12, 0, 0, 8)
            for key, caption, _get, _set, choices in here:
                editor = self._editor(key, choices, answers.get(key))
                self._editors[key] = editor
                form.addRow(caption, editor)
            layout.addLayout(form)

    @staticmethod
    def _editor(key: str, choices, value):
        """A combo for a choice, a checkbox for a flag."""
        if choices:
            box = QComboBox()
            for data, caption in choices:
                box.addItem(str(caption), data)
            index = box.findData(value)
            box.setCurrentIndex(index if index >= 0 else 0)
            return box
        box = QCheckBox()
        box.setChecked(bool(value))
        return box

    def answers(self) -> Dict[str, Any]:
        """What the boxes currently say."""
        out: Dict[str, Any] = {}
        for key, editor in self._editors.items():
            if isinstance(editor, QComboBox):
                out[key] = editor.currentData()
            else:
                out[key] = bool(editor.isChecked())
        return out

    def accept(self) -> None:
        """Write the answers and record that the screen was answered.

        ONE SETTING'S REFUSAL MUST NOT LOSE THE OTHERS -- `setup_screen.apply`
        writes each on its own and reports what failed, and this reports it
        rather than swallowing it.
        """
        from ..setup_screen import apply, current_version, mark_answered

        trouble = apply(self.answers())
        if trouble:
            LOG.warning("some setup answers were refused: %s",
                        "; ".join(trouble))
        mark_answered(current_version())
        super().accept()

    def reject(self) -> None:
        """Dismissed. STILL MARKED ANSWERED.

        The screen has a working default for everything, so a user who closes
        it has chosen the defaults -- and reopening it on every launch until
        they fill it in would make dismissing it impossible.
        """
        from ..setup_screen import current_version, mark_answered

        mark_answered(current_version())
        super().reject()

    # -------------------------------------------------------- the backdrop

    def _backdrop(self, parent) -> Optional[QWidget]:
        """A blurred still of what is behind, or ``None``.

        CACHED AS A PIXMAP, NOT A LIVE BLUR. The corner accent is a
        pointer-position readout and has to keep up with the mouse; blurring
        a live widget underneath would repaint the whole backdrop on every
        move. This is grabbed once.

        NONE IS A FINE ANSWER (INVARIANTS 10). Decoration must never be
        load-bearing: with no parent, or on a platform where the grab or the
        blur fails, the dialog is a plain dialog with the same controls and
        the same answers.
        """
        if parent is None:
            return None
        try:
            shot = parent.grab()
            if shot.isNull():
                return None
            scene = QGraphicsScene(self)
            item = QGraphicsPixmapItem(QPixmap(shot))
            blur = QGraphicsBlurEffect()
            blur.setBlurRadius(BLUR)
            item.setGraphicsEffect(blur)
            scene.addItem(item)
            view = QGraphicsView(scene, self)
            view.setFrameShape(QGraphicsView.NoFrame)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            return view
        except Exception:                                    # noqa: BLE001
            LOG.debug("no blurred backdrop on this platform", exc_info=True)
            return None

    def resizeEvent(self, event):               # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self._backdrop_view is not None:
            self._backdrop_view.setGeometry(self.rect())
            margin = 48
            self.card.setGeometry(self.rect().adjusted(
                margin, margin, -margin, -margin))
            self.card.raise_()


def open_setup_if_needed(parent=None) -> Optional[SetupDialog]:
    """Show the setup screen if it has not been answered. Returns it, or None.

    THE CALLER DOES NOT DECIDE WHETHER TO ASK -- `should_open` does, from the
    recorded version. A screen each caller gated for itself is a screen that
    appears twice on one launch and never on another.
    """
    from ..setup_screen import should_open

    if not should_open():
        return None
    dialog = SetupDialog(parent)
    dialog.exec()
    return dialog
