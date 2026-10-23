"""Present optional application preferences on the first spaCR launch.

Questions are grouped by purpose and initialized with functional defaults, so
the dialog can be dismissed without additional configuration. The blurred
background is decorative; all controls remain available when the platform
cannot render it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFormLayout, QGridLayout,
                               QGraphicsBlurEffect, QGraphicsScene,
                               QGraphicsPixmapItem, QGraphicsView, QLabel,
                               QStyle, QStyleOptionComboBox, QVBoxLayout, QWidget)

LOG = logging.getLogger("spacr.qt.setup_dialog")

#: Ordered groups used to organize first-run preference questions.
GROUPS: List[tuple] = [
    ("How it looks", ("language", "theme", "colour_blind")),
    ("How it runs", ("spacr_mode", "hash_inputs", "spacr_ai")),
]

#: How far the home screen behind is blurred.
BLUR = 18.0


class _SetupComboBox(QComboBox):
    """Reserve full native text slots for every offered setup choice.

    Minimum floors follow current fonts, style and model contents; neither
    maximum dimensions nor fixed sizes restrict later layout growth.
    """

    def __init__(self, parent=None):
        """Bind model updates after Qt construction, before choices are added."""
        self._fit_ready = False
        self._fitting = False
        super().__init__(parent)
        self._fit_ready = True
        self.model().rowsInserted.connect(self._fit_contents)
        self.model().rowsRemoved.connect(self._fit_contents)
        self.model().dataChanged.connect(self._fit_contents)
        self.model().modelReset.connect(self._fit_contents)

    def event(self, event):
        """Recalculate after Qt has refreshed its font or style hint caches."""
        result = super().event(event)
        if event.type() in (QEvent.Polish, QEvent.FontChange, QEvent.StyleChange):
            self._fit_contents()
        return result

    def _fit_contents(self, *_unused):
        """Measure fresh native hints and all text slots in actual pixels.

        Ignore construction-time and reentrant polish events; never derive a
        new floor from the previously imposed minimum, so shrinking still works.
        """
        if not self._fit_ready or self._fitting or not self.count():
            return
        self._fitting = True
        try:
            from ..gui_scale import _set_actual_dim

            self.ensurePolished()
            hint = super().sizeHint().expandedTo(super().minimumSizeHint())
            option = QStyleOptionComboBox()
            option.initFrom(self)
            option.rect = QRect(0, 0, hint.width(), hint.height())
            option.frame = self.hasFrame()
            option.editable = self.isEditable()
            option.currentText = self.currentText()
            option.currentIcon = self.itemIcon(self.currentIndex())
            option.iconSize = self.iconSize()
            field = self.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, self)
            flags = int(Qt.AlignVCenter | Qt.AlignLeft | Qt.TextSingleLine)
            metrics = self.fontMetrics()
            widths, heights = [], []
            for index in range(self.count()):
                text = self.itemText(index)
                bounds = self.style().itemTextRect(metrics, field, flags, self.isEnabled(), text)
                widths.append(max(bounds.width(), metrics.horizontalAdvance(text)))
                heights.append(max(bounds.height(), metrics.boundingRect(text).height()))
            width = max(hint.width(), hint.width() - field.width() + max(widths))
            height = max(hint.height(), hint.height() - field.height() + max(heights))
            _set_actual_dim(self, "minw", width)
            _set_actual_dim(self, "minh", height)
        finally:
            self._fitting = False


class SetupDialog(QDialog):
    """Collect optional first-run preferences."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Build the first-run setup dialog.

        :param parent: parent widget, or ``None``; also where the backdrop
            behind the card is taken from.
        """
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
        self._fit_opening_to_screen()

    def _fit_opening_to_screen(self) -> None:
        """Scroll the grouped card, not its snapshot, for an oversized opening.

        Preserve natural actual-pixel content floors and the ordinary floating
        card path when it fits; only the opening is capped, never later growth.
        """
        screen = self.screen()
        if screen is None:
            return
        opening = self.size()
        available = screen.availableGeometry().size()
        if opening.width() <= available.width() and opening.height() <= available.height():
            return
        from ..dialogs import (OPENS_AT, let_the_content_scroll,
                               make_the_window_resizable, open_at_its_natural_size)
        from ..gui_scale import _set_actual_dim

        self.card.ensurePolished()
        self.card.layout().activate()
        natural = self.card.sizeHint().expandedTo(QSize(max(1, opening.width() - 96),
                                                      max(1, opening.height() - 96)))
        _set_actual_dim(self.card, "minw", natural.width())
        _set_actual_dim(self.card, "minh", natural.height())
        outer = self.layout()
        if self._backdrop_view is not None:
            outer.removeWidget(self._backdrop_view)
            self._backdrop_view.setParent(self)
            self._backdrop_view.lower()
        self._card_scrolls = True
        outer.addWidget(self.card)
        let_the_content_scroll(self)
        make_the_window_resizable(self)
        self.setProperty(OPENS_AT, opening)
        open_at_its_natural_size(self)


    def _build_groups(self, layout) -> None:
        """One form per group, from the model's own question list."""
        from ..setup_screen import current, questions

        asked = {q[0]: q for q in questions()}
        answers = current()
        for heading, keys in GROUPS:
            here = [asked[k] for k in keys if k in asked]
            if not here:
                continue
            label = QLabel(f"<b>{heading}</b>")
            layout.addWidget(label)
            form = QFormLayout()
            form.setContentsMargins(12, 0, 0, 8)
            for key, caption, _get, _set, choices in here:
                editor = self._editor(key, choices, answers.get(key))
                self._editors[key] = editor
                form.addRow(caption, editor)
                if key == "spacr_ai":
                    from ..i18n import tr
                    form.labelForField(editor).setToolTip(tr(
                        "Turning this off selects no AI provider and turns AI Error report off. "
                        "You can later enable GitHub error reports in Preferences while "
                        "keeping the provider set to None."))
            layout.addLayout(form)

    @staticmethod
    def _editor(key: str, choices, value):
        """A combo for a choice, a checkbox for a flag."""
        if choices:
            box = _SetupComboBox()
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
        """Re-lay the dialog for the new size.

        :param event: the Qt resize event.
        """
        super().resizeEvent(event)
        if self._backdrop_view is not None:
            self._backdrop_view.setGeometry(self.rect())
            if not getattr(self, "_card_scrolls", False):
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
