"""Present first-run preferences as a short sequence of explained choices.

Each slide covers one preference group and writes through the existing setup
model. The animated backdrop, translucent card, and pointer-responsive border
are decorative; preference editing and persistence remain available when
those effects cannot be rendered.

:mod:`spacr.qt.setup_screen` holds the model and is the only writer of a
preference; this module is presentation alone. ``setup_dialog`` is the
earlier grouped-form layout of the same questions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import (QEasingCurve, QEvent, QObject, QPointF,
                            QPropertyAnimation, Qt, QTimer)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout,
                               QGraphicsOpacityEffect,
                               QHBoxLayout, QLabel, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

LOG = logging.getLogger("spacr.qt.setup_slides")

#: Setup slides as ``(title, explanation, setting keys)`` tuples.
#:
#: The order moves from interface choices, through execution preferences, to
#: the terms of use.
#: Space between a question and the control that answers it.
#:
#: A NUMBER, NOT WHATEVER IS LEFT OVER. The rows used to put a
#: stretch between the two, which set them 771 px apart on a
#: 980 px card -- the caption on one edge and its control on the
#: other, with the whole width of the slide in between.
FORM_GAP_PX = 24

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
     "How much of this machine spaCR keeps between runs: processes, "
     "caches and GPU memory. Laptop keeps the least — for 8 GB or on "
     "battery; Workstation the most. The same spaCR code runs at every "
     "level, and the reproducibility hash records what each run used.",
     ("spacr_mode", "hash_inputs", "spacr_ai")),
    ("Terms of use",
     "Review the terms of use and full license. Use is noncommercial by default; commercial use requires separate written permission.",
     ()),
    ("Done", "Welcome to spaCR", ()),
)

#: The title of the slide carrying the terms of use.
#:
#: Named rather than matched on, because that slide is the one page in the
#: sequence that is neither a form nor the closing word: it builds itself and
#: it refuses to be left.
TERMS_SLIDE = "Terms of use"

#: The caption on the animation question, and the one row of the theme
#: slide that is not one of the setup model's own questions.
#:
#: "in the startup, under theme should be annimation, degault to blobs." It
#: is asked here rather than added to :func:`spacr.qt.setup_screen.questions`
#: because it is not written through that screen's apply pass: the backdrop
#: has one seam, :func:`spacr.qt.preferences.set_ambient_animation`, which
#: both stores the choice and turns the backdrop on or off, and a second
#: writer for the same preference is how a stored None ends up drawing.
ANIMATION_LABEL = "Animation"

#: How close to the bottom of the terms counts as having reached it.
#:
#: PIXELS, because a scroll bar does not always land exactly on its maximum:
#: a wheel notch, a fling on a touchpad and a drag all stop where they stop,
#: and a gate that demands the exact maximum is a gate that stays shut for a
#: reader who is looking at the last line.
TERMS_END_SLACK = 4

#: A localized greeting for every language offered on the language slide.
#:
#: The greeting provides immediate confirmation without redrawing the window
#: beneath the setup dialog.
GREETINGS: Dict[str, str] = {
    "en": "Hello", "sv": "Hej", "de": "Hallo", "es": "Hola",
    "fr": "Bonjour", "pt": "Olá", "is": "Halló", "hi": "नमस्ते",
    "ko": "안녕하세요", "zh_CN": "你好",
}

#: How much faster the backdrop runs than the ambient default.
BACKDROP_SPEED = 1.5

BACKDROP_THEME = "data_art_impulse_lens"

#: Milliseconds one slide takes to fade into the next.
#:
#: A CROSS-FADE, NOT A CUT. `QStackedWidget.setCurrentIndex` swaps the page
#: between two frames, so the card's contents changed instantly under a rim
#: that took half a second to travel -- two speeds in one gesture, which is
#: what read as unfinished. 260 ms is long enough to see and short enough
#: that six slides do not feel like waiting.
FADE_MS = 260

#: Milliseconds the greeting is held before the first Next takes effect.
#:
#: "there should be a lag after the first next click to make time for Hello
#: in the chosen language" -- the greeting is the only proof the language
#: choice took, and it appears on the slide the user is leaving. Without a
#: pause it is on screen for one frame of a fade.
#:
#: THE HOLD INCLUDES THE FADE. At 850 ms the word reached full opacity at
#: about 500 and was gone by 850, so it was properly legible for a third of
#: a second and was reported as never appearing at all.
GREETING_MS = 1600

#: The face every slide is set in.
#:
#: LIGHT, AND IT REALLY IS LIGHT: `OpenSans-Light.ttf` ships in
#: `spacr/resources/font/open_sans/static`, so the weight resolves to the
#: face rather than to a synthesised one -- provided the bundled fonts have
#: been registered, which is why `_use_the_light_face` loads them itself
#: instead of assuming the application already did.
SLIDE_FONT = "Open Sans"

#: Point size of the word on the closing slide.
#: Corner radius of the setup window, in pixels.
#:
#: ONE NUMBER FOR TWO SURFACES. The card and the ambient backdrop behind it
#: are the same rectangle, so they must round by the same amount or the
#: backdrop's corners show past the card's and the dialog looks like two
#: stacked windows. SetupCard's own default is 18; this names it so the
#: backdrop can be told the same thing.
CARD_RADIUS = 18

DONE_POINTS = 44

#: Point size of the greeting.
#:
#: BIG ENOUGH TO BE SEEN AT ALL. At body size it was one short word in the
#: corner of a card, at full opacity for about a third of a second -- which
#: is why it was reported as not appearing. It is the answer to the question
#: just asked, so it is the size of an answer.
GREETING_POINTS = 30

#: How far down the card the greeting floats, as a fraction of its height.
#:
#: A BAND NOTHING ELSE USES. The question rows sit in the upper half and the
#: buttons along the bottom edge, so this is empty on every slide -- which
#: is what lets the word leave slowly instead of being switched off to make
#: room for what comes next.
#:
#: It sits TWO rows higher than it did, to leave room beneath it for what
#: the machine can actually run.
#:
#: Reported as the greeting on the start screen needing to sit one or two rows up so it does not overlap with the GPU text". The note grew a
#: capability table on 2026-08-31 and got taller; the greeting had not
#: moved since, so the two met.
GREETING_BAND = 0.50

#: How far down the card the GPU note sits: the row the greeting left.
GPU_NOTE_BAND = 0.78

#: What spaCR needs a GPU for, said once on the first slide.
#:
#: NOT A WARNING AND NOT A GATE. Everything else works without one -- the
#: measurements, the regression, every figure -- so this says which two
#: steps are the ones that will be slow or impossible, and leaves the
#: decision to the reader.
#: NOT "AN NVIDIA GPU" ANY MORE, and the old wording was not merely
#: imprecise -- it was wrong on the machine that reported it. spaCR now
#: dispatches to CUDA, ROCm, Apple Metal (which drives Apple Silicon AND
#: AMD cards in Intel Macs) and Intel XPU.
GPU_REQUIREMENT = (
    "spaCR tasks are GPU accelerated and are compatible with NVIDIA, AMD, "
    "Apple, and Intel GPUs. GPU acceleration is orders of magnitude faster "
    "than CPU for matrix multiplication tasks.")

#: The backend that drives each accelerator, for the "GPU: <name>: <lib>"
#: line. The LIBRARY, not the vendor -- a user reading "Metal" beside an
#: AMD card learns the thing that explains why ROCm is irrelevant on their
#: machine. Naming the vendor instead is the mistake this avoids.
GPU_LIBRARIES = {
    "cuda": "CUDA",
    "rocm": "ROCm",
    "mps": "Metal",
    "xpu": "XPU",
    "directml": "DirectML",
}

#: The table's rows: ``(library, capability prefix, task)``.
#:
#: The middle column is GPU or CPU per row and is DERIVED -- matched
#: against the start of what `accelerator.capabilities()` returns, so a
#: detail sentence can be reworded without silently emptying a row.
#:
#: "Live backdrop" is what the renderer is called in `capabilities()` and
#: in the code; the request said "Lave", which is that word typed in a
#: hurry rather than a different thing.
GPU_TABLE_ROWS = (
    ("Cellpose", "Segmentation", "Segmentation"),
    ("Torch models", "Model inference", "Classification"),
    ("Live backdrop", "Live backdrop", "Visualization"),
    ("UMAP / t-SNE / cluster", "UMAP", "Machine learning"),
)

#: The colours the verdict is drawn in on a dark theme.
GPU_YES_INK = "#3FB950"
GPU_NO_INK = "#F85149"


def verdict_ink(ok: bool) -> str:
    """The colour a GPU verdict is drawn in on the theme in force.

    :data:`GPU_YES_INK` and :data:`GPU_NO_INK` are the dark theme's green
    and red, and on a light page that green is under 2.5:1. A light theme
    draws the verdict in its own ``success`` and ``error`` instead, which
    its contrast rules hold readable.

    :param ok: whether the verdict is good news.
    :returns: a hex colour.
    """
    try:
        from ..theme import active_palette, relative_luminance

        palette = active_palette()
        if relative_luminance(palette["bg"]) > relative_luminance(
                palette["fg"]):
            return palette["success"] if ok else palette["error"]
    except Exception:                                        # noqa: BLE001
        LOG.debug("no palette for the GPU verdict", exc_info=True)
    return GPU_YES_INK if ok else GPU_NO_INK

#: What to say when the card is there and torch cannot use it.
#:
#: A DIFFERENT PROBLEM FROM NO CARD, with a different fix, so it gets a
#: different sentence. A CPU-only torch build and a driver older than the
#: CUDA runtime torch was built against both present as "cuda not
#: available", and `spacr-doctor` is what tells the two apart -- it exists
#: for this case and says which one it is.
GPU_DOCTOR_HINT = (
    "The card is there but torch cannot use it. "
    "Run spacr-doctor to find out which part of CUDA is missing.")

GPU_MAC_CPU_HINT = (
    "PyTorch cannot use this Mac's graphics, so spaCR runs these tasks on the CPU. "
    "Nothing needs to be installed.")

#: Milliseconds the greeting takes to fade AWAY.
#:
#: Slower than it arrives. A word that leaves at the speed it came reads as
#: being taken away; one that lingers reads as being finished with.
GREETING_LEAVE_MS = 700

#: Milliseconds the greeting takes to fade up.
#:
#: IT ARRIVES, it does not appear. A word that is simply switched on reads
#: as a label that was always going to be there; one that fades up reads as
#: an answer to what was just chosen, which is what it is.
GREETING_FADE_MS = 420


def greeting_for(code: str) -> str:
    """"Hello" in ``code``, falling back to English.

    :param code: language code, a key of :data:`GREETINGS` such as ``"sv"``
        or ``"zh_CN"``; unknown or empty codes give ``"Hello"``.
    """
    return GREETINGS.get(str(code or ""), GREETINGS["en"])


def _gpu_library() -> str:
    """The backend driving this machine's accelerator, or "".

    CUDA, ROCm, Metal, XPU, DirectML -- the library, not the vendor. A
    user reading "Metal" beside an AMD card learns the thing that
    explains why ROCm is irrelevant on their machine. Naming the vendor
    instead is the mistake this avoids.
    """
    try:
        from ...accelerator import resolve

        return GPU_LIBRARIES.get(resolve().kind, "")
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not name the accelerator library", exc_info=True)
        return ""


def graphics_card() -> Tuple[bool, str]:
    """``(usable, name)`` for the machine's graphics card.

    Usable means torch can reach it: a card spaCR cannot run on is not a
    compatible card however well the driver reports it. Torch is asked first
    for that reason, and NVML second because it names the card even when torch
    was built without CUDA -- which is the case worth telling apart, since the
    answer there is "install a CUDA build", not "buy a card".

    :returns: ``(True, 'NVIDIA GeForce RTX 3090')`` when segmentation can
        run on it; ``(False, name)`` when it cannot, with the best name
        available; ``(False, '')`` when nothing could be identified.
    """
    name = ""
    usable = False
    try:
        import torch as _torch_module

        from ...accelerator import inspect_torch

        found = inspect_torch(_torch_module)
        if found.is_gpu:
            return True, found.name or found.label
        if found.detected and not found.usable:
            return False, found.name or found.label
    except Exception:                                        # noqa: BLE001
        LOG.debug("the accelerator resolver could not be asked",
                  exc_info=True)
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count():
            usable = True
            name = str(torch.cuda.get_device_name(0))
    except Exception:                                        # noqa: BLE001
        LOG.debug("torch could not be asked about the GPU", exc_info=True)
    if name:
        return usable, name
    try:
        from ..widgets.home import _nvml

        nvml = _nvml()
        if nvml is not None and nvml.nvmlDeviceGetCount():
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            raw = nvml.nvmlDeviceGetName(handle)
            name = raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:                                        # noqa: BLE001
        LOG.debug("NVML could not name the GPU", exc_info=True)
    return usable, name


def _say(text: str, **values) -> str:
    """Translate one caption, falling back to the English it was given.

    A local shim rather than a module-level `from ..i18n import tr`: this
    module is imported during application start-up, before the language
    preference has necessarily been read, so the lookup happens per call.
    """
    try:
        from ..i18n import tr

        return tr(text, **values)
    except Exception:                                        # noqa: BLE001
        LOG.debug("no translation available", exc_info=True)
        return text.format(**values) if values else text


def _still_a_widget(widget: object | None) -> bool:
    """Whether a Python Qt wrapper still owns its C++ object."""
    if widget is None:
        return False
    try:
        from shiboken6 import isValid
    except Exception:                                        # noqa: BLE001
        return True
    return bool(isValid(widget))


class _CardPlacement(QObject):
    """Coalesce overlay placement until the scroll-owned card finishes resizing."""

    def __init__(self, card, callback):
        """Own the timer and callback connection for exactly the card's lifetime."""
        super().__init__(card)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(callback)
        card.installEventFilter(self)

    def eventFilter(self, watched, event):
        """Defer placement until the card's current resize event is complete."""
        if event.type() == QEvent.Resize:
            self._timer.start(0)
        return False


class SetupSlides(QDialog):
    """The setup screen: one question per slide, over a moving backdrop.

    :param parent: parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Build the first-run slide deck.

        Frameless, like the rest of the shell: the card it draws has rounded
        corners, and a square window frame around them is the seam this avoids.

        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setWindowTitle("Set spaCR up")
        self.setModal(True)
        self.setProperty("spacrIndependentBackdrop", True)
        self._editors: Dict[str, QWidget] = {}
        self._index = 0

        self._go_frameless()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._backdrop = self._install_backdrop()

        from .setup_card import SetupCard

        self.card = SetupCard(self, radius=CARD_RADIUS)
        self.card.setMouseTracking(True)
        self.setMouseTracking(True)
        column = QVBoxLayout(self.card)
        column.setContentsMargins(28, 28, 28, 22)
        column.setSpacing(12)

        column.addStretch(1)

        self._title = QLabel("")
        self._title.setObjectName("CardTitle")
        column.addWidget(self._title)
        self._blurb = QLabel("")
        self._blurb.setObjectName("Muted")
        self._blurb.setWordWrap(True)
        column.addWidget(self._blurb)

        self._greeting = QLabel("", self.card)
        self._greeting.setObjectName("CardTitle")
        self._greeting.setAlignment(Qt.AlignCenter)
        self._greeting.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._greeting.setVisible(False)

        self._gpu_note = QLabel("", self.card)
        self._gpu_note.setObjectName("Muted")
        self._gpu_note.setAlignment(Qt.AlignCenter)
        self._gpu_note.setWordWrap(True)
        self._gpu_note.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._gpu_note.setVisible(False)
        self._say_what_the_gpu_is()

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

        self._clear_the_containers()
        self._use_the_light_face()

        self._fade = None
        self._hello = None
        self._pending = None
        #: Whether the greeting has already been waited for once.
        self._greeted = False
        self._show_slide(0)
        self.resize(720, 560)
        self._fit_opening_to_screen()

    def _fit_opening_to_screen(self) -> None:
        """Fit undersized content naturally and scroll an oversized opening.

        An otherwise screen-fitting but undersized window reserves the GPU note
        in the layout. Oversized openings retain their natural overlay coordinates;
        the backdrop stays outside scrolling and no maximum limits later growth.
        """
        screen = self.screen()
        if screen is None:
            return
        opening = self.size()
        available = screen.availableGeometry().size()
        self.card.ensurePolished()
        self.card.layout().activate()
        if (opening.width() <= available.width() and opening.height() <= available.height()
                and (opening.width() < self.card.minimumSizeHint().width()
                     or opening.height() < self.card.minimumSizeHint().height())):
            from ..gui_scale import QWIDGETSIZE_MAX, _resize_actual, _set_actual_dim

            self._gpu_note_in_layout = True
            _set_actual_dim(self._gpu_note, "minw", 0)
            _set_actual_dim(self._gpu_note, "maxw", QWIDGETSIZE_MAX)
            column = self.card.layout()
            column.insertWidget(column.count() - 1, self._gpu_note)
            self._card_in_layout = True
            self.layout().addWidget(self.card)
            column.activate()
            opening = opening.expandedTo(self.card.sizeHint()).expandedTo(self.card.minimumSizeHint())
            _resize_actual(self, opening)
        if opening.width() <= available.width() and opening.height() <= available.height():
            return
        from ..dialogs import (OPENS_AT, let_the_content_scroll,
                               make_the_window_resizable, open_at_its_natural_size)
        from ..gui_scale import _set_actual_dim

        _set_actual_dim(self.card, "minw", opening.width())
        _set_actual_dim(self.card, "minh", opening.height())
        self._card_scrolls = True
        self._card_placement = _CardPlacement(self.card, self._place_the_greeting)
        if not getattr(self, "_card_in_layout", False):
            self.layout().addWidget(self.card)
        let_the_content_scroll(self)
        make_the_window_resizable(self)
        self.setProperty(OPENS_AT, opening)
        open_at_its_natural_size(self)

    def _use_the_light_face(self) -> None:
        """Set every slide in Open Sans Light.

        SET ON THE CARD, not on each label: Qt propagates a font to children
        that have not asked for one of their own, so one call covers the
        titles, the prose, the controls and the buttons -- and the two
        labels that DO want a weight of their own (the closing word and the
        greeting) set it after this and keep it.

        The bundled faces are registered here rather than assumed: this
        dialog is the FIRST thing a fresh profile sees, and on that path it
        can be built before anything else has loaded them.
        """
        try:
            from ..app import _load_bundled_fonts

            _load_bundled_fonts()
        except Exception:                                    # noqa: BLE001
            LOG.debug("bundled fonts are not loadable here", exc_info=True)
        face = QFont(SLIDE_FONT)
        face.setWeight(QFont.Light)
        self.card.setFont(face)

    def _clear_the_containers(self) -> None:
        """Stop the layout containers painting over the backdrop."""
        try:
            from ..theme import make_transparent
        except Exception:                                    # noqa: BLE001
            LOG.debug("no theme helper for transparency", exc_info=True)
            return
        holders = [self._pages]
        holders += [self._pages.widget(i) for i in range(self._pages.count())]
        holders += [w for w in self.findChildren(QWidget)
                    if w.property("spacrClearContainer")]
        try:
            make_transparent(*[w for w in holders if w is not None])
        except Exception:                                    # noqa: BLE001
            LOG.debug("a container would not go transparent", exc_info=True)


    def _build_pages(self) -> None:
        """One page per slide, from the model's own question list."""
        from ..setup_screen import current, questions

        asked = {q[0]: q for q in questions()}
        answers = current()
        for index, (title, blurb, keys) in enumerate(SLIDES):
            if title == TERMS_SLIDE:
                self._pages.addWidget(self._terms_page())
                continue
            if index == len(SLIDES) - 1 and not keys:
                self._pages.addWidget(self._closing_page(title, blurb))
                continue
            page = QWidget()
            form = QFormLayout(page)
            form.setContentsMargins(0, 8, 8, 0)
            form.setVerticalSpacing(14)
            form.setHorizontalSpacing(FORM_GAP_PX)
            form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
            form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
            for key in keys:
                if key not in asked:
                    continue
                form.addRow(*self._row(asked[key], answers.get(key)))
                if key == "theme":
                    animation = self._animation_row()
                    if animation is not None:
                        form.addRow(*animation)
            self._pages.addWidget(page)

    def _go_frameless(self) -> bool:
        """Drop the title bar and let the card's rounded corners show.

        The setup screen is dismissed by its own buttons and by Escape, so
        the close and minimise buttons were chrome around chrome. It stays
        movable: `glass._DragByBackground` drags a window by its empty
        background, which is what the title bar used to be for.
        """
        from PySide6.QtCore import Qt

        try:
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setWindowFlags(self.windowFlags()
                                | Qt.FramelessWindowHint)
            from .glass import _DragByBackground, _paint_nothing_behind_the_card

            _paint_nothing_behind_the_card(self)
            _DragByBackground(self)
            return True
        except Exception:                                    # noqa: BLE001
            LOG.debug("the setup screen would not go frameless",
                      exc_info=True)
            return False

    #: The greeting's fade-out, held so it is not collected mid-animation.
    _goodbye = None

    def _closing_page(self, title: str, blurb: str) -> QWidget:
        """The last slide: one word, centred, with its sentence under it.

        The shared header is HIDDEN for this slide rather than repeated --
        the title is the word in the middle, and having it twice on one
        screen is the layout saying it does not know which one is the
        heading.
        """
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)
        column.addStretch(1)

        self._done_word = QLabel(str(title))
        self._done_word.setAlignment(Qt.AlignCenter)
        try:
            from ..theme import active_palette

            ink = f"color: {active_palette()['accent']}; "
        except Exception:                                    # noqa: BLE001
            LOG.debug("no palette for the closing word", exc_info=True)
            ink = ""
        self._done_word.setStyleSheet(
            f"{ink}font-family: '{SLIDE_FONT}'; font-weight: 300; "
            f"font-size: {DONE_POINTS}pt;")
        column.addWidget(self._done_word)

        under = QLabel(str(blurb))
        under.setObjectName("Muted")
        under.setAlignment(Qt.AlignCenter)
        under.setWordWrap(True)
        column.addWidget(under)
        column.addStretch(1)
        return page

    def _terms_page(self) -> QWidget:
        """The terms, an acceptance gated on having read them, and the line
        that says what is missing.

        THE ACCEPTANCE IS DISABLED UNTIL THE END OF THE TERMS HAS BEEN ON
        SCREEN, and the terms are greyed with it: one state drawn on both
        halves, so the page reads as one thing waiting rather than as a
        switch that happens to be dead. The enabling IS the evidence that
        the text went past the reader.

        THE NEXT BUTTON IS NOT DISABLED HERE. A control that does nothing
        and says nothing leaves the reader to guess which of the things on
        the page is stopping them, so the button stays live and answers the
        press with :data:`spacr.qt.terms.WHY_NOT_YET` in the accent colour.
        """
        from PySide6.QtWidgets import QScrollArea

        from .. import terms as terms_module
        from .toggle import Toggle

        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)

        body = QLabel(terms_module.terms_text())
        body.setObjectName("Muted")
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._terms_body = body

        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setProperty("spacrClearContainer", True)
        scroll.viewport().setProperty("spacrClearContainer", True)
        self._terms_scroll = scroll
        bar = scroll.verticalScrollBar()
        if bar is not None:
            bar.valueChanged.connect(self._look_at_the_terms_gate)
            bar.rangeChanged.connect(self._look_at_the_terms_gate)
        column.addWidget(scroll, 1)

        where = QLabel(
            f'<a href="{terms_module.LICENSE_URL}">'
            f"{terms_module.LICENSE_NAME}</a> &nbsp;·&nbsp; "
            f"{terms_module.REQUIRED_NOTICE}")
        where.setObjectName("Muted")
        where.setOpenExternalLinks(True)
        where.setWordWrap(True)
        column.addWidget(where)

        self._scroll_hint = QLabel(_say(terms_module.SCROLL_HINT), page)
        self._scroll_hint.setObjectName("Muted")
        self._scroll_hint.setWordWrap(True)
        column.addWidget(self._scroll_hint)

        self._agree = Toggle(_say(terms_module.AGREE_LABEL), page)
        self._agree.toggled.connect(self._on_agreement_toggled)
        column.addWidget(self._agree)

        self._agree_note = QLabel("", page)
        self._agree_note.setWordWrap(True)
        self._agree_note.setVisible(False)
        try:
            from ..theme import active_palette

            self._agree_note.setStyleSheet(
                f"color: {active_palette()['accent']};")
        except Exception:                                    # noqa: BLE001
            LOG.debug("no palette for the terms note", exc_info=True)
        column.addWidget(self._agree_note)
        self._terms_read = False
        self._draw_the_terms_gate(False)
        return page


    def _look_at_the_terms_gate(self, *_args) -> None:
        """Re-read the gate and redraw it. The signal handler."""
        self._draw_the_terms_gate(self.terms_were_read())

    def terms_were_read(self) -> bool:
        """Always ``True``: the acceptance is not gated on scrolling.

        There is no scroll gate. Dragging a scroll bar to the bottom of a long
        document does not prove it was read, so scrolling is not a condition of
        acceptance.

        The acceptance remains. The full text is still on the page and still
        scrollable for anyone who wants it, the checkbox is still explicit, and
        :func:`spacr.qt.terms.record_agreement` still records the version and
        the moment. Only the greying is removed.

        Kept as a method rather than deleted because the slide, the Next
        button and the tests all ask this question, and one answer in one
        place is easier to be sure of than a gate removed from four.
        """
        self._terms_read = True
        return True

    def _draw_the_terms_gate(self, read: bool) -> None:
        """Put the gate's one state on both halves of the page."""
        box = getattr(self, "_agree", None)
        if _still_a_widget(box):
            box.setEnabled(bool(read))
        body = getattr(self, "_terms_body", None)
        if _still_a_widget(body):
            body.setStyleSheet("" if read else f"color: {self._dim_ink()};")
        hint = getattr(self, "_scroll_hint", None)
        if _still_a_widget(hint):
            hint.setVisible(not read)

    @staticmethod
    def _dim_ink() -> str:
        """The palette's dim ink, or a grey that works without one."""
        try:
            from ..theme import active_palette

            palette = active_palette()
            return str(palette.get("fg_dim")
                       or palette.get("fg_muted") or "#6b6f76")
        except Exception:                                    # noqa: BLE001
            LOG.debug("no palette for the greyed terms", exc_info=True)
            return "#6b6f76"

    def _on_agreement_toggled(self, agreed: bool) -> None:
        """Drop the complaint the moment the box is ticked."""
        note = getattr(self, "_agree_note", None)
        if note is not None and agreed:
            note.setVisible(False)

    def agreed_to_terms(self) -> bool:
        """Whether the acceptance box is ticked on this screen."""
        box = getattr(self, "_agree", None)
        return bool(box is not None and box.isChecked())

    def _refuse_to_leave_the_terms(self) -> int:
        """Say what is missing and stay put. Returns the slide still shown.

        TWO THINGS CAN BE MISSING and they need different sentences. An
        unticked switch is answered by :data:`spacr.qt.terms.WHY_NOT_YET`; a
        switch that cannot be ticked yet is answered by that AND by the
        reason it is greyed, because "tick the box above" is not actionable
        advice about a box that will not take a tick.
        """
        from .. import terms as terms_module

        read = self.terms_were_read()
        note = getattr(self, "_agree_note", None)
        if note is not None:
            said = _say(terms_module.WHY_NOT_YET)
            if not read:
                said = f"{_say(terms_module.SCROLL_HINT)} {said}"
            note.setText(said)
            note.setVisible(True)
        target = getattr(self, "_agree" if read else "_terms_scroll", None)
        if target is not None:
            target.setFocus()
        return self._index

    def _row(self, question, value) -> Tuple[QLabel, QWidget]:
        """The caption and the control, for one row of the page's form.

        Returned as a pair rather than as a finished layout so that every
        row on a page shares ONE form: two columns, one gap, and captions
        that line up with the captions above and below them.
        """
        key, caption, _get, _set, choices = question
        label = QLabel(str(caption))
        if key == "spacr_ai":
            label.setToolTip(_say(
                "Turning this off selects no AI provider and turns AI Error report off. "
                "You can later enable GitHub error reports in Preferences while "
                "keeping the provider set to None."))
        editor = self._editor(key, choices, value)
        self._editors[key] = editor
        return label, editor

    def _editor(self, key: str, choices, value) -> QWidget:
        """A combo or a slider. NEVER A CHECKBOX.

        "aslo in the startup all the booleans should be sliders" -- a tick
        box is a form control and this is not a form. A slider reads as a
        STATE rather than as a task, which is what these settings are.
        """
        if choices:
            from .setup_dialog import _SetupComboBox

            box = _SetupComboBox()
            for data, caption in choices:
                box.addItem(str(caption), data)
            index = box.findData(value)
            box.setCurrentIndex(index if index >= 0 else 0)
            if key == "language":
                box.currentIndexChanged.connect(self._say_hello)
                box.currentIndexChanged.connect(self._apply_language)
            if key in ("theme", "colour_blind"):
                box.currentIndexChanged.connect(
                    lambda _i, k=key: self._apply_look(k))
            return box

        from .toggle import Toggle

        slider = Toggle()
        slider.setChecked(bool(value))
        return slider

    def _animation_row(self) -> Optional[Tuple[QLabel, QWidget]]:
        """The backdrop question, asked under the theme it belongs with.

        EVERY CHOICE THE APPLICATION HAS, ``None`` included -- the list is
        :data:`spacr.qt.widgets.ambient.ANIMATION_CHOICES`, so a reader who
        finds the motion distracting can say so on the way in rather than
        going looking for it afterwards.

        IT OPENS ON WHAT IS ALREADY TRUE. The default is spaCR field
        (``data_art_impulse_lens``), as defined by
        :data:`spacr.qt.widgets.ambient.DEFAULT_THEME` and what
        :func:`spacr.qt.preferences.get_ambient_animation` falls back to --
        the slide shows the application's own default rather than a second
        opinion about it.

        :returns: the row, or ``None`` when there is no ambient module to
            ask -- in which case the theme slide is the two questions it
            was, rather than a labelled row with nothing in it.
        """
        try:
            from .ambient import (ANIMATION_CHOICES, DEFAULT_THEME,
                                  animation_label, animation_note)
        except Exception:                                    # noqa: BLE001
            LOG.debug("no ambient module to ask about the backdrop",
                      exc_info=True)
            return None
        try:
            from ..preferences import get_ambient_animation

            chosen = get_ambient_animation()
        except Exception:                                    # noqa: BLE001
            LOG.debug("the stored animation could not be read", exc_info=True)
            chosen = DEFAULT_THEME

        box = QComboBox()
        box.setObjectName("SetupAnimation")
        for index, name in enumerate(ANIMATION_CHOICES):
            box.addItem(_say(animation_label(name)), name)
            try:
                box.setItemData(index, _say(animation_note(name)),
                                Qt.ToolTipRole)
            except Exception:                                # noqa: BLE001
                LOG.debug("no note for animation %s", name, exc_info=True)
        where = box.findData(chosen)
        if where < 0:
            where = box.findData(DEFAULT_THEME)
        if where < 0:
            LOG.debug("no %s among the animations offered", DEFAULT_THEME)
            where = 0
        box.setCurrentIndex(where)
        box.currentIndexChanged.connect(self._apply_animation)
        self._animation = box

        return QLabel(_say(ANIMATION_LABEL)), box

    def animation_choice(self) -> str:
        """Which backdrop the slide is showing, or ``""`` with no row."""
        box = getattr(self, "_animation", None)
        return "" if box is None else str(box.currentData() or "")

    def _apply_animation(self, *_args) -> None:
        """Store the backdrop choice, through the one seam that owns it.

        :func:`spacr.qt.preferences.set_ambient_animation` both records the
        choice and turns the backdrop on or off, so writing the theme key
        directly would leave a profile that chose None with the animation
        still enabled -- and one that chose Blobs after switching it off
        with silence.
        """
        name = self.animation_choice()
        if not name:
            return
        try:
            from ..preferences import set_ambient_animation

            set_ambient_animation(name)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not store the animation choice", exc_info=True)

    def _say_hello(self, *_args) -> None:
        """Set the greeting text for the language currently chosen.

        Setting it is not showing it: :meth:`_show_the_greeting` is what
        puts it on screen, and only the first Next calls that.
        """
        box = self._editors.get("language")
        if box is None:
            return
        self._greeting.setText(greeting_for(box.currentData()))

    def _show_the_greeting(self) -> None:
        """Fade the greeting up, in the accent colour.

        BLUE, FROM THE PALETTE rather than a literal: the accent is what the
        theme calls its own, so the greeting matches the rim that is running
        round the card as it appears.
        """
        self._say_hello()
        try:
            from ..theme import active_palette

            accent = active_palette()["accent"]
            self._greeting.setStyleSheet(
                f"color: {accent}; font-family: '{SLIDE_FONT}'; "
                f"font-weight: 300; font-size: {GREETING_POINTS}pt;")
        except Exception:                                    # noqa: BLE001
            LOG.debug("no palette for the greeting", exc_info=True)
        self._place_the_greeting()
        self._greeting.setVisible(True)
        try:
            effect = QGraphicsOpacityEffect(self._greeting)
            self._greeting.setGraphicsEffect(effect)
            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(GREETING_FADE_MS)
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.OutCubic)
            self._hello = animation
            animation.start()
        except Exception:                                    # noqa: BLE001
            LOG.debug("no fade for the greeting", exc_info=True)
            self._hello = None

    def _fade_the_greeting_away(self) -> None:
        """Take the greeting off gently, and only if it is on.

        The abrupt version was ``setVisible(False)`` the instant the slide
        changed. Nothing depends on the word being gone -- it floats over a
        band no slide uses -- so it can be given the time to leave.
        """
        greeting = getattr(self, "_greeting", None)
        card = getattr(self, "card", None)
        if greeting is None or card is None:
            return
        if not greeting.isVisibleTo(card):
            return
        try:
            effect = QGraphicsOpacityEffect(greeting)
            greeting.setGraphicsEffect(effect)
            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(GREETING_LEAVE_MS)
            animation.setStartValue(1.0)
            animation.setEndValue(0.0)
            animation.setEasingCurve(QEasingCurve.InCubic)
            animation.finished.connect(
                lambda: greeting.setVisible(False))
            self._goodbye = animation
            animation.start()
        except Exception:                                    # noqa: BLE001
            LOG.debug("no fade for the greeting", exc_info=True)
            greeting.setVisible(False)
            self._goodbye = None

    def _place_the_greeting(self) -> None:
        """Put the greeting in its band, and the GPU note in its own.

        TWO INDEPENDENT PLACEMENTS. The greeting arrives only when the
        language is confirmed, so it does not exist while the first slide
        is being read -- and placing the note behind a check for the
        greeting left it at the geometry a QLabel is born with, the top
        left corner of the card, where it read as stray text above the
        language list until the first Next took it away.
        """
        card = getattr(self, "card", None)
        if card is None:
            return
        if self._greeting:
            height = self._greeting.sizeHint().height()
            self._greeting.setGeometry(
                0, int(card.height() * GREETING_BAND), card.width(), height)
            self._greeting.raise_()
        self._place_the_gpu_note()

    def _place_the_gpu_note(self) -> None:
        """Use the undersize layout's placement, or place the floating note's band."""
        if getattr(self, "_gpu_note_in_layout", False):
            return
        card = getattr(self, "card", None)
        note = getattr(self, "_gpu_note", None)
        if card is None or note is None or note.isHidden():
            return
        margin = 28
        width = max(1, card.width() - 2 * margin)
        from ..gui_scale import _set_actual_dim

        _set_actual_dim(note, "minw", width)
        _set_actual_dim(note, "maxw", width)
        height = note.heightForWidth(width)
        if height <= 0:
            height = note.sizeHint().height()
        top = int(card.height() * GPU_NOTE_BAND)
        floor = card.height() - margin
        back = getattr(self, "_back", None)
        if back is not None and back.parent() is not None:
            placed = False
            try:
                top_of_buttons = back.mapTo(card, back.rect().topLeft()).y()
                if top_of_buttons > 0:
                    floor = min(floor, top_of_buttons - 12)
                    placed = True
            except (AttributeError, RuntimeError):
                pass
            if not placed:
                try:
                    reserved = back.sizeHint().height()
                except (AttributeError, RuntimeError):
                    reserved = 0
                if reserved > 0:
                    floor = min(floor, card.height() - margin - reserved - 12)
        top = max(0, min(top, floor - height))
        note.setGeometry(margin, top, width, height)
        note.raise_()

    def _say_what_the_gpu_is(self) -> None:
        """Write the requirement and this machine's verdict into the note.

        GREEN OR RED, and the card is named either way: "no compatible
        GPU" on its own leaves the reader wondering whether spaCR looked.
        """
        usable, name = graphics_card()
        hint = ""
        if name:
            library = _gpu_library()
            tail = f": {library}" if library else ""
            line = (f'{_say("GPU")}: <span style="color:{{ink}};">{name}</span>'
                    f'{tail}')
            if not usable:
                hint = _say(GPU_MAC_CPU_HINT if library == "Metal" else GPU_DOCTOR_HINT)
        else:
            usable = False
            line = (f'{_say("GPU")}: '
                    f'<span style="color:{{ink}};">'
                    f'{_say("none detected")}</span>')
        ink = verdict_ink(usable)
        html = [f'<div>{_say(GPU_REQUIREMENT)}</div>',
                f'<div style="font-weight:600;">{line.format(ink=ink)}</div>']
        if hint:
            html.append(f'<div>{hint}</div>')
        html.extend(self._what_this_machine_can_do())
        self._gpu_note.setText("".join(html))
        self._place_the_gpu_note()

    @staticmethod
    def _cellpose_label() -> str:
        """"Cellpose 4", with the 4 taken from the installed package.

        A hardcoded major version is a claim the next Cellpose falsifies,
        and this label sits on the first screen a new user sees.
        """
        try:
            import cellpose

            version = str(getattr(cellpose, "version", None)
                          or getattr(cellpose, "__version__", ""))
            major = version.split(".")[0]
            return f"Cellpose {major}" if major.isdigit() else "Cellpose"
        except Exception:                                    # noqa: BLE001
            return "Cellpose"

    @staticmethod
    def _what_this_machine_can_do() -> list:
        """The capability TABLE: library, GPU-or-CPU, task.

        A table rather than four sentences. The content was already
        per-task -- one verdict cannot be honest
        here, because on Metal the segmentation and the classifier are
        accelerated while the cuML reductions are not -- but four
        sentences in a column read as four separate remarks rather than
        one answer with an axis.

        The middle column is DERIVED from `accelerator.capabilities()`,
        the same function `spacr-doctor` and the README table render, so
        the three surfaces cannot disagree about the same machine.

        Failures are swallowed. This is decoration on a setup slide, and
        a machine with a strange accelerator must still reach the button
        at the bottom of it.
        """
        try:
            from ...accelerator import capabilities, neural_engines
        except Exception:                                    # noqa: BLE001
            LOG.debug("capabilities unavailable", exc_info=True)
            return []
        rows = []
        try:
            answers = {task: (ok, detail)
                       for task, ok, detail in capabilities()}

            def _answer(prefix):
                """The first answer whose task starts with ``prefix``."""
                for task, (ok, detail) in answers.items():
                    if task.startswith(prefix):
                        return ok, detail
                return None, ""

            cells = []
            partial_details = []
            for library, prefix, task in GPU_TABLE_ROWS:
                accelerated, _detail = _answer(prefix)
                if accelerated is None:
                    LOG.debug("no capability row starts with %r", prefix)
                    continue
                if library.startswith("Cellpose"):
                    library = SetupSlides._cellpose_label()
                if prefix == 'UMAP' and _detail.startswith('PCA/KMeans on Metal'):
                    library = 'PCA / KMeans'
                    partial_details.append(_detail)
                ink = verdict_ink(bool(accelerated))
                where = _say("GPU") if accelerated else _say("CPU")
                cells.append(
                    f'<tr>'
                    f'<td align="right" style="padding-right:14px;">'
                    f'{_say(library)}</td>'
                    f'<td style="padding-right:14px; color:{ink}; '
                    f'font-weight:600;">{where}</td>'
                    f'<td style="opacity:0.85;">{_say(task)}</td>'
                    f'</tr>')
            if cells:
                rows.append(
                    '<div align="center">'
                    '<table style="margin-top:6px; border-collapse:collapse;">'
                    + "".join(cells) + '</table></div>')
                for detail in partial_details:
                    rows.append(f'<div align="center">{_say(detail)}</div>')
            for engine in neural_engines():
                rows.append(
                    f'<div style="opacity:0.75;">• {engine} — '
                    f'{_say("detected, not used by spaCR")}</div>')
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not render the capability list", exc_info=True)
            return []
        return rows

    def _apply_language(self, *_args) -> None:
        """Put the chosen language into effect and redraw this screen in it.

        The preference is written NOW rather than on accept, because
        everything that renders text after this point -- the rest of the
        slides, the tooltips, any dialog the screen opens -- reads the
        stored language rather than being handed one.
        """
        box = self._editors.get("language")
        if box is None:
            return
        code = str(box.currentData() or "")
        if not code:
            return
        try:
            from .. import preferences as prefs

            prefs.set_language(code)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not store the language", exc_info=True)
        self.retranslate()

    def retranslate(self) -> None:
        """Redraw every caption on this screen in the current language.

        Through the catalog walker rather than through a `tr()` at each
        call site: this screen builds its slides from tables, and a walker
        that remembers each widget's English source can switch from
        Swedish to Korean without translating a translation.
        """
        try:
            from ..i18n import retranslate_widget_tree

            retranslate_widget_tree(self)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not retranslate the setup screen",
                      exc_info=True)
        self._show_slide(self._index)
        self._say_hello()

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
            LOG.debug("could not apply %s live", key, exc_info=True)

    def _show_slide(self, index: int, *, fade: bool = False) -> None:
        """Show one slide, translating its text as it goes.

        The slide table holds the English the catalogue is keyed on, so the
        translation happens here rather than in the table -- which is what lets
        a slide re-shown after a language change pick up the new rendering.

        :param index: the slide to show; clamped to the deck.
        :param fade: cross-fade into it rather than switching.
        """
        index = max(0, min(int(index), len(SLIDES) - 1))
        self._index = index
        title, blurb, _keys = SLIDES[index]
        self._pages.setCurrentIndex(index)
        closing = index == len(SLIDES) - 1
        self._title.setVisible(not closing)
        self._blurb.setVisible(not closing)
        self._title.setText(f"<b>{_say(title)}</b>")
        self._blurb.setText(_say(blurb))
        note = getattr(self, "_gpu_note", None)
        if note is not None:
            note.setVisible(index == 0)
            self._place_the_gpu_note()
        if index != 0:
            self._fade_the_greeting_away()
        self._where.setText(
            "" if index == 0 else
            _say("{n} of {total}", n=index + 1, total=len(SLIDES)))
        self._back.setEnabled(index > 0)
        self._next.setText(_say("Start spaCR") if index == len(SLIDES) - 1
                           else _say("Next ›"))
        self._drop_the_fade()
        if fade:
            self._fade_in()

    def showEvent(self, event):                 # noqa: N802 - Qt naming
        """Schedule terms-gate evaluation after the window has been laid out.

        Whether the end of the terms document is visible depends on its
        rendered viewport. The zero-delay callback runs on the next event-loop
        turn, after Qt has completed layout for the show event.

        :param event: the show event, passed to the base class first.
        """
        super().showEvent(event)
        try:
            QTimer.singleShot(0, self._settle_opening_layout)
            QTimer.singleShot(0, self._look_at_the_terms_gate)
        except Exception:                                    # noqa: BLE001
            LOG.debug("the terms gate could not be re-measured",
                      exc_info=True)

    def _settle_opening_layout(self):
        """Lay out the card after native font metrics and window geometry settle."""
        if not (getattr(self, '_card_scrolls', False)
                or getattr(self, '_card_in_layout', False)):
            self.card.setGeometry(self.rect())
        self.card.ensurePolished()
        self.card.layout().invalidate()
        self.card.layout().activate()
        self._place_the_greeting()
        self._place_the_gpu_note()

    def _fade_in(self) -> None:
        """Bring the new slide up from transparent.

        HELD ON THE INSTANCE. A QPropertyAnimation with no owner is
        collected the moment the method returns, and the fade never runs --
        which looks exactly like not having written one.

        The effect is REMOVED when the fade finishes rather than left in
        place: a QGraphicsOpacityEffect renders its widget into an offscreen
        pixmap on every repaint, and leaving six of them alive over a
        drifting backdrop is a cost paid on every frame for an animation
        that has ended.
        """
        try:
            effect = QGraphicsOpacityEffect(self._pages)
            self._pages.setGraphicsEffect(effect)
            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(FADE_MS)
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.OutCubic)
            animation.finished.connect(self._drop_the_fade)
            self._fade = animation
            animation.start()
        except Exception:                                    # noqa: BLE001
            LOG.debug("no cross-fade on this platform", exc_info=True)
            self._fade = None

    def _drop_the_fade(self) -> None:
        """Stop any running fade and take the effect off the page stack.

        STOPPED, not just forgotten: an animation still running would go on
        driving an effect this is about to delete, and the page would be
        left at whatever opacity it had reached.
        """
        animation, self._fade = self._fade, None
        if animation is not None:
            try:
                animation.stop()
            except Exception:                                # noqa: BLE001
                pass
        try:
            self._pages.setGraphicsEffect(None)
        except Exception:                                    # noqa: BLE001
            pass


    def next(self) -> int:
        """Forward one slide, and one clockwise circuit of the rim.

        Leaving the language slide waits briefly, so the greeting in the chosen
        language stays visible; it is on the page being left, so without a
        pause it would be on screen for one frame of a fade. The rim starts its
        circuit immediately, so the click is answered at once and only the page
        change is held.

        The wait happens once. A pause on every return to the first slide would
        be a delay the user has already sat through.

        The terms slide is the one page this will not leave. Pressing Next
        there without the acceptance ticked stays on the slide and says why,
        rather than greying the button and leaving the reader to work out which
        control is holding them.
        """
        if SLIDES[self._index][0] == TERMS_SLIDE and not self.agreed_to_terms():
            self.card.circuit(clockwise=True)
            return self._refuse_to_leave_the_terms()
        if self._index >= len(SLIDES) - 1:
            self.accept()
            return self._index
        self.card.circuit(clockwise=True)
        if self._index == 0 and not self._greeted:
            self._greeted = True
            return self._advance_after_the_greeting()
        self._show_slide(self._index + 1, fade=True)
        return self._index

    def _advance_after_the_greeting(self) -> int:
        """Hold the greeting, then move on. Returns the slide still showing.

        The Next button is disabled for the wait rather than left live: a
        second click during the pause would queue a second advance and skip
        a slide.
        """
        self._next.setEnabled(False)
        self._show_the_greeting()
        try:
            self._pending = QTimer(self)
            self._pending.setSingleShot(True)
            self._pending.timeout.connect(self._finish_the_greeting)
            self._pending.start(GREETING_MS)
        except Exception:                                    # noqa: BLE001
            LOG.debug("no timer for the greeting pause", exc_info=True)
            self._finish_the_greeting()
        return self._index

    def _finish_the_greeting(self) -> None:
        """The pause is over: move to the next slide."""
        self._pending = None
        self._next.setEnabled(True)
        self._show_slide(min(self._index + 1, len(SLIDES) - 1), fade=True)

    def previous(self) -> int:
        """Back one slide, and one anticlockwise circuit.

        The direction shows the user which way they went.
        """
        if self._index <= 0:
            return self._index
        self.card.circuit(clockwise=False)
        self._show_slide(self._index - 1, fade=True)
        return self._index

    def slide(self) -> int:
        """Which slide is showing, counting from zero."""
        return self._index

    def mouseMoveEvent(self, event):            # noqa: N802 - Qt naming
        """Aim the rim at the pointer. Ignored while a circuit runs.

        :param event: the mouse move event; its position, mapped into the
            card, is what the rim flows towards. It is then passed on to the
            base class.
        """
        try:
            self.card.flow_towards(
                self.card.mapFrom(self, event.position().toPoint()))
        except Exception:                                    # noqa: BLE001
            pass
        super().mouseMoveEvent(event)


    def answers(self) -> Dict[str, Any]:
        """What the slides currently say."""
        out: Dict[str, Any] = {}
        for key, editor in self._editors.items():
            if isinstance(editor, QComboBox):
                out[key] = editor.currentData()
            else:
                out[key] = bool(editor.isChecked())
        return out

    def accept(self) -> None:
        """Close the slides and record that they have been seen.

        Recorded, so first-run guidance does not greet a returning user as a
        new one.
        """
        from ..setup_screen import apply, current_version, mark_answered

        trouble = apply(self.answers())
        if trouble:
            LOG.warning("some setup answers were refused: %s",
                        "; ".join(trouble))
        self._record_the_agreement()
        mark_answered(current_version())
        super().accept()

    def reject(self) -> None:
        """Dismissed at any slide. Still marked answered.

        Every question has a working default, so a user who closes this has
        chosen them -- and reopening on every launch until it is filled in
        would make dismissing it impossible.

        The terms are the exception, and they are not marked. A dismissal is a
        choice of defaults; it is not an acceptance of a licence, so nothing is
        recorded and `open_setup_if_needed` asks again.
        """
        from ..setup_screen import apply, current_version, mark_answered

        apply(self.answers())
        self._record_the_agreement()
        mark_answered(current_version())
        super().reject()

    def _record_the_agreement(self) -> None:
        """Store the accepted terms version, if the box was ticked.

        WRITTEN ONLY WHEN IT WAS GIVEN. Recording an acceptance on the way
        out of a screen that was closed would make the record say something
        the user never did, which is worse than having no record.
        """
        if not self.agreed_to_terms():
            return
        try:
            from .. import terms as terms_module

            terms_module.record_agreement(terms_module.TERMS_VERSION)
        except Exception:                                    # noqa: BLE001
            LOG.warning("the terms acceptance could not be recorded",
                        exc_info=True)


    def _install_backdrop(self):
        """spaCR's own field at 1.5x, or ``None``.

        NONE IS A FINE ANSWER (INVARIANTS 10). With no ambient engine
        available the slides are slides on a plain dialog, and every answer
        they write is the same.
        """
        try:
            from .ambient import install_ambient

            widget = install_ambient(self, theme=BACKDROP_THEME,
                                     palette="spacr", speed=BACKDROP_SPEED,
                                     corner_radius=CARD_RADIUS)
            widget.setProperty("spacrSetupBackdrop", True)
            return widget
        except Exception:                                    # noqa: BLE001
            LOG.debug("no ambient backdrop on this platform", exc_info=True)
            return None

    def resizeEvent(self, event):               # noqa: N802 - Qt naming
        """Re-lay the slide for the new size.

        :param event: the Qt resize event.
        """
        super().resizeEvent(event)
        if not (getattr(self, "_card_scrolls", False) or getattr(self, "_card_in_layout", False)):
            self.card.setGeometry(self.rect())
        self.card.raise_()
        self._look_at_the_terms_gate()
        try:
            from .glass import round_the_corners

            round_the_corners(self, CARD_RADIUS)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not round the setup window", exc_info=True)
        self._place_the_greeting()


def _catalogue_this_screen() -> None:
    """Put the terms slide's captions in the translation catalogs.

    Called once when this module is imported, because ``SLIDES`` is what a
    reader (and the catalog check) looks at and it must already carry its
    translations by then.
    """
    try:
        from .. import terms as terms_module

        terms_module.register_translations()
    except Exception:                                        # noqa: BLE001
        LOG.debug("the terms captions could not be catalogued", exc_info=True)
    try:
        from ..i18n import add_translation

        add_translation(ANIMATION_LABEL, (
            "Animation", "Animation", "Animación", "动画",
            "Animação", "एनिमेशन",
            "애니메이션", "Hreyfimynd", "Animation"))
    except Exception:                                        # noqa: BLE001
        LOG.debug("the animation caption could not be catalogued",
                  exc_info=True)


_catalogue_this_screen()


def open_setup_if_needed(parent=None) -> Optional[SetupSlides]:
    """Show the setup slides when the recorded setup state requires them.

    The centralized :func:`spacr.qt.setup_screen.should_open` check compares
    the answered setup version with the running version.
    :func:`spacr.qt.terms.needs_agreement` additionally requests setup when
    the terms have never been accepted or have changed since acceptance.
    """
    from ..setup_screen import should_open, skipped_on_purpose
    from ..terms import needs_agreement

    if skipped_on_purpose():
        return None
    if not should_open() and not needs_agreement():
        return None
    dialog = SetupSlides(parent)
    dialog.exec()
    return dialog
