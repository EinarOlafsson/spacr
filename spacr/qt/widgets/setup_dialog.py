"""Present optional application preferences on the first spaCR launch.

Questions are grouped by purpose and initialized with functional defaults, so
the dialog can be dismissed without additional configuration. The blurred
background is decorative; all controls remain available when the platform
cannot render it.
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

#: Ordered groups used to organize first-run preference questions.
GROUPS: List[tuple] = [
    ("How it looks", ("language", "theme", "colour_blind")),
    ("How it runs", ("spacr_mode", "hash_inputs")),
    ("The assistant", ("ai_provider", "ai_default")),
    ("When something breaks", ("issue_prompt", "share_logs")),
]

#: How far the home screen behind is blurred.
BLUR = 18.0


class SetupDialog(QDialog):
    """Collect optional first-run preferences.

    Parameters
    ----------
    parent : QWidget, optional
        Parent window. When available, a blurred snapshot is used as a
        decorative background.
    """

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
        """Return the current value of every displayed preference control."""
        out: Dict[str, Any] = {}
        for key, editor in self._editors.items():
            if isinstance(editor, QComboBox):
                out[key] = editor.currentData()
            else:
                out[key] = bool(editor.isChecked())
        return out

    def accept(self) -> None:
        """Apply displayed preferences and record setup completion.

        Each preference is applied independently. Rejected values are logged
        without discarding other valid selections.
        """
        from ..setup_screen import apply, current_version, mark_answered

        trouble = apply(self.answers())
        if trouble:
            LOG.warning("some setup answers were refused: %s",
                        "; ".join(trouble))
        mark_answered(current_version())
        super().accept()

    def reject(self) -> None:
        """Dismiss the dialog while retaining defaults and mark setup complete."""
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
    """Open the first-run setup dialog when required.

    Parameters
    ----------
    parent : QWidget, optional
        Parent application window.

    Returns
    -------
    SetupDialog or None
        Executed dialog, or ``None`` when setup has already been completed for
        the current version.
    """
    from ..setup_screen import should_open

    if not should_open():
        return None
    dialog = SetupDialog(parent)
    dialog.exec()
    return dialog
