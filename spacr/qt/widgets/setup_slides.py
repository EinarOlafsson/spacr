"""Present first-run preferences as a short sequence of explained choices.

Each slide covers one preference group and writes through the existing setup
model. The animated backdrop, translucent card, and pointer-responsive border
are decorative; preference editing and persistence remain available when
those effects cannot be rendered.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel,
                               QPushButton, QStackedWidget, QVBoxLayout,
                               QWidget)

LOG = logging.getLogger("spacr.qt.setup_slides")

#: Setup slides as ``(title, explanation, setting keys)`` tuples.
#:
#: The order moves from interface choices, through execution preferences, to
#: assistant and data-sharing choices.
SLIDES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("Language",
     "Every label, tooltip and message spaCR shows you. You can change it "
     "later in Preferences, and nothing about your data depends on it.",
     ("language",)),
    ("Theme",
     "How spaCR looks, and whether its colours are chosen to stay "
     "distinguishable without colour vision. Both take effect as you pick "
     "them, so you can see what you are choosing.",
     ("theme", "colour_blind")),
    ("How it runs",
     "The mode decides how much spaCR does for you before asking. The "
     "reproducibility hash records what went into a run, so a result can be "
     "traced back to the exact inputs that produced it.",
     ("spacr_mode", "hash_inputs")),
    ("The assistant",
     "spaCR can explain an error or a result through a coding assistant you "
     "already subscribe to. It uses the vendor's own command-line tool, so "
     "nothing is sent anywhere you have not already logged in to.",
     ("ai_provider", "ai_default")),
    ("When something breaks",
     "What may leave this machine, and under whose name. Nothing is ever "
     "sent without you seeing it first and pressing send yourself.",
     ("issue_prompt", "share_logs")),
    ("Done",
     "That is everything. All of it is in Preferences if you change your "
     "mind.",
     ()),
)

#: A localized greeting for every language offered on the language slide.
#:
#: The greeting provides immediate confirmation without redrawing the window
#: beneath the setup dialog.
GREETINGS: Dict[str, str] = {
    "en": "Hello", "sv": "Hej", "de": "Hallo", "es": "Hola",
    "fr": "Bonjour", "pt": "Olá", "is": "Halló", "hi": "नमस्ते",
    "ko": "안녕하세요", "zh_CN": "你好",
}

#: The providers offered as logo buttons, and the CLI each one needs.
#: A DROPDOWN OF THREE NAMES IS A DROPDOWN; three logos is a choice somebody
#: makes in one glance.
PROVIDERS: Tuple[Tuple[str, str, str], ...] = (
    ("claude", "Claude", "claude"),
    ("gpt", "GPT", "codex"),
    ("gemini", "Gemini", "gemini"),
)

#: How much faster the backdrop runs than the ambient default.
BACKDROP_SPEED = 1.5

#: Ambient theme used for the stratified, independently drifting backdrop.
#:
#: Reusing the application theme keeps the setup backdrop synchronized with
#: the active palette.
BACKDROP_THEME = "aurora"


def greeting_for(code: str) -> str:
    """"Hello" in ``code``, falling back to English."""
    return GREETINGS.get(str(code or ""), GREETINGS["en"])


class SetupSlides(QDialog):
    """The setup screen: one question per slide, over a moving backdrop."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Set spaCR up")
        self.setModal(True)
        self._editors: Dict[str, QWidget] = {}
        self._index = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._backdrop = self._install_backdrop()

        from .setup_card import SetupCard

        self.card = SetupCard(self)
        # THE RIM FOLLOWS THE POINTER, so the card has to see it move even
        # when no button is down -- which is not the default.
        self.card.setMouseTracking(True)
        self.setMouseTracking(True)
        column = QVBoxLayout(self.card)
        column.setContentsMargins(28, 28, 28, 22)
        column.setSpacing(12)

        # CENTRED, NOT TOP-ALIGNED. One question in a card this size sat
        # against the ceiling with a void under it, which reads as a page
        # that failed to load the rest of itself. A slide is one thing, and
        # one thing belongs in the middle.
        column.addStretch(1)

        self._title = QLabel("")
        self._title.setObjectName("CardTitle")
        column.addWidget(self._title)
        self._blurb = QLabel("")
        self._blurb.setObjectName("Muted")
        self._blurb.setWordWrap(True)
        column.addWidget(self._blurb)

        self._pages = QStackedWidget()
        column.addWidget(self._pages)
        self._build_pages()

        column.addStretch(1)

        row = QHBoxLayout()
        self._back = QPushButton("‹ Back")
        self._back.clicked.connect(self.previous)
        row.addWidget(self._back)
        self._where = QLabel("")
        self._where.setObjectName("Muted")
        self._where.setAlignment(Qt.AlignCenter)
        row.addWidget(self._where, 1)
        self._next = QPushButton("Next ›")
        self._next.clicked.connect(self.next)
        row.addWidget(self._next)
        column.addLayout(row)

        self._show_slide(0)
        self.resize(720, 560)

    # --------------------------------------------------------- the slides

    def _build_pages(self) -> None:
        """One page per slide, from the model's own question list."""
        from ..setup_screen import current, questions

        asked = {q[0]: q for q in questions()}
        answers = current()
        for title, _blurb, keys in SLIDES:
            page = QWidget()
            form = QVBoxLayout(page)
            form.setContentsMargins(0, 8, 0, 0)
            form.setSpacing(14)
            for key in keys:
                if key not in asked:
                    # A QUESTION THAT REMOVED ITSELF LEAVES NO GAP. The
                    # provider question is absent when no CLI is installed,
                    # and an empty labelled row would read as a broken
                    # control rather than as a question that does not apply.
                    continue
                form.addLayout(self._row(asked[key], answers.get(key)))
            self._pages.addWidget(page)

    def _row(self, question, value) -> QHBoxLayout:
        key, caption, _get, _set, choices = question
        row = QHBoxLayout()
        label = QLabel(str(caption))
        row.addWidget(label)
        row.addStretch(1)
        editor = self._editor(key, choices, value)
        self._editors[key] = editor
        row.addWidget(editor)
        return row

    def _editor(self, key: str, choices, value) -> QWidget:
        """A logo strip, a combo, or a slider. NEVER A CHECKBOX.

        "aslo in the startup all the booleans should be sliders" -- a tick
        box is a form control and this is not a form. A slider reads as a
        STATE rather than as a task, which is what these settings are.
        """
        if key == "ai_provider":
            return self._provider_buttons(value)
        if choices:
            box = QComboBox()
            for data, caption in choices:
                box.addItem(str(caption), data)
            index = box.findData(value)
            box.setCurrentIndex(index if index >= 0 else 0)
            if key == "language":
                box.currentIndexChanged.connect(self._say_hello)
            if key in ("theme", "colour_blind"):
                # APPLIED AS CHOSEN, for the same reason the greeting is:
                # the only way to know a look took is to see it.
                box.currentIndexChanged.connect(
                    lambda _i, k=key: self._apply_look(k))
            return box

        from .toggle import Toggle

        # THE APPLICATION'S OWN SLIDER, not a second one, so the gesture and
        # the look are the ones the user meets everywhere else.
        slider = Toggle()
        slider.setChecked(bool(value))
        return slider

    def _provider_buttons(self, value) -> QWidget:
        """Claude, GPT and Gemini as buttons rather than a list."""
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        holder._chosen = str(value or "")
        holder._buttons = {}
        for code, label, command in PROVIDERS:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(holder._chosen == code)
            ready = self._provider_is_installed(command)
            button.setEnabled(ready)
            button.setToolTip(
                f"Use {label}." if ready else
                f"{label} is not set up on this machine: spaCR drives the "
                f"vendor's own `{command}` command, and it is not installed "
                f"or not logged in. Installing it is all that is needed.")
            button.clicked.connect(
                lambda _c=False, k=code, h=holder: self._choose_provider(h, k))
            row.addWidget(button)
            holder._buttons[code] = button
        return holder

    @staticmethod
    def _provider_is_installed(command: str) -> bool:
        """Whether the vendor CLI is on PATH.

        AN UNINSTALLED PROVIDER SAYS SO rather than being offered as though
        it were ready: choosing it would leave the assistant silently
        unavailable, and the user would blame spaCR.
        """
        import shutil

        return shutil.which(str(command)) is not None

    @staticmethod
    def _choose_provider(holder, code: str) -> None:
        holder._chosen = str(code)
        for name, button in holder._buttons.items():
            button.setChecked(name == code)

    # ------------------------------------------------------- what it shows

    def _say_hello(self, *_args) -> None:
        """Greet in the language just chosen."""
        box = self._editors.get("language")
        if box is None:
            return
        self._blurb.setText(
            f"{greeting_for(box.currentData())}\n\n{SLIDES[0][1]}")

    def _apply_look(self, key: str) -> None:
        """Put the theme or colour-blind choice into effect immediately."""
        from ..setup_screen import questions

        setter = next((q[3] for q in questions() if q[0] == key), None)
        editor = self._editors.get(key)
        if setter is None or editor is None:
            return
        try:
            setter(editor.currentData())
        except Exception:                                    # noqa: BLE001
            # A LOOK THAT WILL NOT APPLY MUST NOT LOSE THE ANSWER. It is
            # still written with the rest on accept.
            LOG.debug("could not apply %s live", key, exc_info=True)

    def _show_slide(self, index: int) -> None:
        index = max(0, min(int(index), len(SLIDES) - 1))
        self._index = index
        title, blurb, _keys = SLIDES[index]
        self._pages.setCurrentIndex(index)
        self._title.setText(f"<b>{title}</b>")
        self._blurb.setText(blurb)
        if index == 0:
            self._say_hello()
        self._where.setText(f"{index + 1} of {len(SLIDES)}")
        self._back.setEnabled(index > 0)
        self._next.setText("Start spaCR" if index == len(SLIDES) - 1
                           else "Next ›")

    # ------------------------------------------------------------ moving

    def next(self) -> int:
        """Forward one slide, and one CLOCKWISE circuit of the rim."""
        if self._index >= len(SLIDES) - 1:
            self.accept()
            return self._index
        self.card.circuit(clockwise=True)
        self._show_slide(self._index + 1)
        return self._index

    def previous(self) -> int:
        """Back one slide, and one ANTICLOCKWISE circuit.

        THE DIRECTION IS THE MESSAGE: it tells the user which way they went,
        which is worth more than the animation.
        """
        if self._index <= 0:
            return self._index
        self.card.circuit(clockwise=False)
        self._show_slide(self._index - 1)
        return self._index

    def slide(self) -> int:
        """Which slide is showing, counting from zero."""
        return self._index

    def mouseMoveEvent(self, event):            # noqa: N802 - Qt naming
        """Aim the rim at the pointer. Ignored while a circuit runs."""
        try:
            self.card.flow_towards(
                self.card.mapFrom(self, event.position().toPoint()))
        except Exception:                                    # noqa: BLE001
            pass
        super().mouseMoveEvent(event)

    # ----------------------------------------------------------- answers

    def answers(self) -> Dict[str, Any]:
        """What the slides currently say."""
        out: Dict[str, Any] = {}
        for key, editor in self._editors.items():
            if isinstance(editor, QComboBox):
                out[key] = editor.currentData()
            elif hasattr(editor, "_chosen"):
                out[key] = editor._chosen
            else:
                out[key] = bool(editor.isChecked())
        return out

    def accept(self) -> None:
        from ..setup_screen import apply, current_version, mark_answered

        trouble = apply(self.answers())
        if trouble:
            LOG.warning("some setup answers were refused: %s",
                        "; ".join(trouble))
        mark_answered(current_version())
        super().accept()

    def reject(self) -> None:
        """Dismissed at any slide. STILL MARKED ANSWERED.

        Every question has a working default, so a user who closes this has
        chosen them -- and reopening on every launch until it is filled in
        would make dismissing it impossible.
        """
        from ..setup_screen import apply, current_version, mark_answered

        apply(self.answers())
        mark_answered(current_version())
        super().reject()

    # --------------------------------------------------------- decoration

    def _install_backdrop(self):
        """Stratified layers drifting at 1.5x, or ``None``.

        NONE IS A FINE ANSWER (INVARIANTS 10). With no ambient engine
        available the slides are slides on a plain dialog, and every answer
        they write is the same.
        """
        try:
            from .ambient import install_ambient

            return install_ambient(self, theme=BACKDROP_THEME,
                                   speed=BACKDROP_SPEED)
        except Exception:                                    # noqa: BLE001
            LOG.debug("no ambient backdrop on this platform", exc_info=True)
            return None

    def resizeEvent(self, event):               # noqa: N802 - Qt naming
        super().resizeEvent(event)
        margin = 44
        self.card.setGeometry(self.rect().adjusted(
            margin, margin, -margin, -margin))
        self.card.raise_()


def open_setup_if_needed(parent=None) -> Optional[SetupSlides]:
    """Show the setup slides when the recorded setup state requires them.

    The centralized :func:`spacr.qt.setup_screen.should_open` check prevents
    independent callers from opening duplicate dialogs during one launch.
    """
    from ..setup_screen import should_open

    if not should_open():
        return None
    dialog = SetupSlides(parent)
    dialog.exec()
    return dialog
