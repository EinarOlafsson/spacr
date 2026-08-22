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
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import (QEasingCurve, QEvent, QPointF,
                            QPropertyAnimation, Qt, QTimer)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QComboBox, QDialog, QGraphicsOpacityEffect,
                               QHBoxLayout, QLabel, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

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
DONE_POINTS = 44

#: Point size of the greeting.
#:
#: BIG ENOUGH TO BE SEEN AT ALL. At body size it was one short word in the
#: corner of a card, at full opacity for about a third of a second -- which
#: is why it was reported as not appearing. It is the answer to the question
#: just asked, so it is the size of an answer.
GREETING_POINTS = 30

#: Milliseconds the greeting takes to fade up.
#:
#: IT ARRIVES, it does not appear. A word that is simply switched on reads
#: as a label that was always going to be there; one that fades up reads as
#: an answer to what was just chosen, which is what it is.
GREETING_FADE_MS = 420


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

        # THE GREETING HAS ITS OWN LINE. It used to be prepended to the
        # explanation, so choosing a language rewrote the paragraph under
        # the title and the one word that changed was buried in it.
        # THE GREETING IS THE ANSWER TO THE LANGUAGE QUESTION, so it comes
        # AFTER the question is answered rather than sitting under it while
        # it is still being decided. It is hidden until the first Next.
        self._greeting = QLabel("")
        self._greeting.setObjectName("CardTitle")
        self._greeting.setAlignment(Qt.AlignLeft)
        self._greeting.setVisible(False)
        column.addWidget(self._greeting)

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

        # NO BLACK BOXES INSIDE THE CARD (reported 2026-08-22). The card
        # paints itself translucent over the drifting backdrop, but every
        # plain QWidget between the two -- the page stack, each page, the
        # provider strip -- is caught by the blanket `QWidget` rule and
        # paints an opaque `bg`, which is a solid dark rectangle sitting on
        # top of the animation the dialog just installed.
        #
        # THE CONTAINERS ONLY. The combos and buttons stay opaque: they are
        # the readable surface, and a control you can see through is a
        # control you cannot read.
        self._clear_the_containers()
        self._use_the_light_face()

        self._fade = None
        self._hello = None
        self._pending = None
        #: Whether the greeting has already been waited for once.
        self._greeted = False
        self._show_slide(0)
        self.resize(720, 560)

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
                    if w.property("spacrProviderStrip")]
        try:
            make_transparent(*[w for w in holders if w is not None])
        except Exception:                                    # noqa: BLE001
            LOG.debug("a container would not go transparent", exc_info=True)

    # --------------------------------------------------------- the slides

    def _build_pages(self) -> None:
        """One page per slide, from the model's own question list."""
        from ..setup_screen import current, questions

        asked = {q[0]: q for q in questions()}
        answers = current()
        for index, (title, blurb, keys) in enumerate(SLIDES):
            if index == len(SLIDES) - 1 and not keys:
                # THE CLOSING SLIDE IS NOT A FORM, so it is not laid out
                # like one. It says one word, in the middle, with the
                # sentence that qualifies it underneath.
                self._pages.addWidget(self._closing_page(title, blurb))
                continue
            page = QWidget()
            form = QVBoxLayout(page)
            # A MARGIN ON THE RIGHT. The controls are right-aligned, so with
            # none they finish exactly on the card's content edge and their
            # drop-down arrow is drawn flush against it -- which reads as a
            # clipped control rather than as a control that fits.
            form.setContentsMargins(0, 8, 8, 0)
            form.setSpacing(14)
            signs_in = "issue_prompt" in keys
            for key in keys:
                if key not in asked:
                    # A QUESTION THAT REMOVED ITSELF LEAVES NO GAP. The
                    # provider question is absent when no CLI is installed,
                    # and an empty labelled row would read as a broken
                    # control rather than as a question that does not apply.
                    continue
                form.addLayout(self._row(asked[key], answers.get(key)))
            if signs_in:
                form.addWidget(self._github_row())
            self._pages.addWidget(page)

    def _github_row(self) -> QWidget:
        """Sign in to GitHub, on the slide that decides about issues.

        THE CLI OWNS THE CREDENTIAL. spaCR never captures or stores a token
        -- `gh` puts it in the platform credential manager, which is the one
        place a user can revoke it from and the one place that is not a
        second copy of a secret. This row says whether a token is reachable
        and starts `gh auth login` when it is not; it never asks for one.

        Filing an issue works WITHOUT it, through the browser, which is why
        this is a row on a slide and not a gate in front of one.
        """
        holder = QWidget()
        holder.setProperty("spacrProviderStrip", True)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        label = QLabel("GitHub")
        row.addWidget(label)
        row.addStretch(1)

        self._gh_status = QLabel("")
        self._gh_status.setObjectName("Muted")
        row.addWidget(self._gh_status)

        self._gh_button = QPushButton("Sign in")
        self._gh_button.setToolTip(
            "Runs `gh auth login`, the GitHub CLI's own browser sign-in. "
            "GitHub stores the credential; spaCR never sees it. Without it, "
            "reports open in whichever browser you are already signed in to.")
        self._gh_button.clicked.connect(self._sign_in_to_github)
        row.addWidget(self._gh_button)

        self._refresh_github()
        return holder

    #: What each token source is called on screen.
    GITHUB_SOURCES = {
        "gh": "signed in through the GitHub CLI",
        "env": "signed in through GITHUB_TOKEN",
        "token": "signed in with a stored token",
    }

    def _refresh_github(self) -> None:
        """Say whether a token is reachable, and from where."""
        try:
            from ..ai import github_auth

            source = github_auth.auth_source()
        except Exception:                                    # noqa: BLE001
            LOG.debug("GitHub auth is not readable here", exc_info=True)
            source = None
        if source:
            self._gh_status.setText(
                self.GITHUB_SOURCES.get(source, "signed in"))
            self._gh_button.setText("Signed in")
            self._gh_button.setEnabled(False)
            return
        import shutil

        if shutil.which("gh") is None:
            # NAMED, not "sign-in failed". The CLI being absent and the CLI
            # being logged out need different things from the user.
            self._gh_status.setText(
                "the GitHub CLI is not installed — reports open in your "
                "browser")
            self._gh_button.setText("Sign in")
            self._gh_button.setEnabled(False)
            self._gh_button.setToolTip(
                "Install the GitHub CLI (`gh`) and this signs you in. "
                "Without it, filing an issue still works -- it opens in "
                "whichever browser you are already signed in to.")
            return
        self._gh_status.setText("not signed in — reports open in your browser")
        self._gh_button.setText("Sign in")
        self._gh_button.setEnabled(True)

    def _sign_in_to_github(self) -> bool:
        """Start `gh auth login`. True when it was started.

        DETACHED, and the dialog does not wait on it: `gh` runs its own
        browser flow and can take as long as the user takes, and a modal
        setup screen frozen behind it would be a setup screen that looks
        crashed. The status re-reads when the process ends.
        """
        from PySide6.QtCore import QProcess

        process = QProcess(self)
        process.finished.connect(lambda *_a: self._refresh_github())
        try:
            process.start("gh", ["auth", "login", "--web"])
            started = process.waitForStarted(3000)
        except Exception:                                    # noqa: BLE001
            LOG.debug("gh auth login would not start", exc_info=True)
            started = False
        if not started:
            self._gh_status.setText(
                "`gh auth login` would not start — run it in a terminal")
            return False
        self._gh_process = process
        self._gh_status.setText("waiting for GitHub in your browser…")
        self._gh_button.setEnabled(False)
        return True

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

        # AS IT IS WRITTEN, not shouted: "Done", not "DONE".
        self._done_word = QLabel(str(title))
        self._done_word.setAlignment(Qt.AlignCenter)
        # THE SIZE GOES IN A STYLESHEET, not through setFont. The
        # application sheet already gives every QLabel a font-size, and QSS
        # beats a font set on the widget -- so setPointSize was overruled
        # and the word came out the size of the sentence beneath it.
        #
        # THE ACCENT BLUE, which is the blue the wordmark uses and the same
        # one the greeting arrives in -- one blue for the things this screen
        # is saying, rather than a second one invented for the last slide.
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
        """Claude, GPT and Gemini as their MARKS rather than as three words.

        "the claude gpt and gemeni buttons should be the logos for each not
        just buttons" -- a mark is recognised before it is read, and this is
        the one question where the user already knows the answer and only
        has to point at it. The marks are drawn, not shipped: see
        :mod:`spacr.qt.widgets.provider_marks`.
        """
        from .provider_marks import ProviderMark

        holder = QWidget()
        # Tagged so `_clear_the_containers` can find it without knowing the
        # shape of the page it ended up on.
        holder.setProperty("spacrProviderStrip", True)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        holder._chosen = str(value or "")
        holder._buttons = {}
        for code, label, command in PROVIDERS:
            ready = self._provider_is_installed(command)
            mark = ProviderMark(code, label, ready, holder)
            mark.set_chosen(holder._chosen == code)
            mark.setToolTip(
                f"Use {label}." if ready else
                f"Use {label}. Its `{command}` command is not on this "
                f"machine yet -- spaCR drives the vendor's own CLI, so "
                f"installing it is all that is needed. You can choose "
                f"{label} now and install it later.")
            mark.chosen.connect(
                lambda picked, h=holder: self._choose_provider(h, picked))
            row.addWidget(mark)
            holder._buttons[code] = mark
        row.addStretch(1)
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
        for name, mark in holder._buttons.items():
            mark.set_chosen(name == code)

    # ------------------------------------------------------- what it shows

    def _say_hello(self, *_args) -> None:
        """Set the greeting text for the language currently chosen.

        Setting it is not showing it: :meth:`_show_the_greeting` is what
        puts it on screen, and only the first Next calls that.
        """
        box = self._editors.get("language")
        if box is None:
            return
        # AS THE WORD IS WRITTEN in each language -- "Hello", "Hej", "Hallo",
        # 你好 -- rather than shouted. GREETINGS already holds each in its
        # own conventional form, so there is nothing to do to it.
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
            # THE SIZE GOES IN THE SHEET TOO. The application stylesheet
            # gives every QLabel a font-size and QSS beats a font set on the
            # widget, so setPointSize here would be overruled and the word
            # would come out the size of the prose above it.
            self._greeting.setStyleSheet(
                f"color: {accent}; font-family: '{SLIDE_FONT}'; "
                f"font-weight: 300; font-size: {GREETING_POINTS}pt;")
        except Exception:                                    # noqa: BLE001
            LOG.debug("no palette for the greeting", exc_info=True)
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
            # INVARIANTS 10: without an animation it is simply there.
            LOG.debug("no fade for the greeting", exc_info=True)
            self._hello = None

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

    def _show_slide(self, index: int, *, fade: bool = False) -> None:
        index = max(0, min(int(index), len(SLIDES) - 1))
        self._index = index
        title, blurb, _keys = SLIDES[index]
        self._pages.setCurrentIndex(index)
        closing = index == len(SLIDES) - 1
        self._title.setVisible(not closing)
        self._blurb.setVisible(not closing)
        self._title.setText(f"<b>{title}</b>")
        self._blurb.setText(blurb)
        # THE GREETING BELONGS TO THE LANGUAGE SLIDE and nowhere else: a
        # "Hello" left standing over the theme question is a word with no
        # job on that page.
        # THE GREETING BELONGS TO THE MOMENT THE LANGUAGE IS CONFIRMED, not
        # to a slide. It is shown by the first Next and hidden again by
        # anything that leaves that moment behind.
        if index != 0:
            self._greeting.setVisible(False)
        self._where.setText(f"{index + 1} of {len(SLIDES)}")
        self._back.setEnabled(index > 0)
        self._next.setText("Start spaCR" if index == len(SLIDES) - 1
                           else "Next ›")
        # A LEFTOVER FADE IS DROPPED WHETHER OR NOT A NEW ONE STARTS. The
        # effect belongs to the page STACK, not to a page, so one still
        # running when the slide changes again leaves the new page wearing
        # an opacity that stopped part-way -- which draws an empty card and
        # is indistinguishable from a page that failed to build.
        self._drop_the_fade()
        if fade:
            self._fade_in()

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
            # `setGraphicsEffect` takes ownership and deletes whatever was
            # there, so the old animation is now driving a dead object.
            # `_drop_the_fade` above has already stopped it; this is the
            # note for anyone who moves that call.
            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(FADE_MS)
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.OutCubic)
            animation.finished.connect(self._drop_the_fade)
            self._fade = animation
            animation.start()
        except Exception:                                    # noqa: BLE001
            # INVARIANTS 10: the slide is shown either way.
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

    # ------------------------------------------------------------ moving

    def next(self) -> int:
        """Forward one slide, and one CLOCKWISE circuit of the rim.

        LEAVING THE LANGUAGE SLIDE WAITS. "there should be a lag after the
        first next click to make time for Hello in the chosen language" --
        the greeting is the only proof the choice took, and it lives on the
        page being left, so without a pause it is on screen for one frame of
        a fade. The rim starts its circuit immediately, so the click is
        answered at once and only the page change is held.

        The wait happens ONCE. A pause on every return to the first slide
        would be a delay the user has already sat through.
        """
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
            # INVARIANTS 10: without a timer the slides still advance, they
            # just do not wait.
            LOG.debug("no timer for the greeting pause", exc_info=True)
            self._finish_the_greeting()
        return self._index

    def _finish_the_greeting(self) -> None:
        """The pause is over: move to the next slide."""
        self._pending = None
        self._next.setEnabled(True)
        self._show_slide(min(self._index + 1, len(SLIDES) - 1), fade=True)

    def previous(self) -> int:
        """Back one slide, and one ANTICLOCKWISE circuit.

        THE DIRECTION IS THE MESSAGE: it tells the user which way they went,
        which is worth more than the animation.
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
    from ..setup_screen import should_open, skipped_on_purpose

    # WHETHER THIS PROFILE IS DUE and whether THIS LAUNCH CAN ASK are two
    # different questions. `should_open` answers the first; a batch job on a
    # server can be due and still have nobody to answer.
    if skipped_on_purpose() or not should_open():
        return None
    dialog = SetupSlides(parent)
    dialog.exec()
    return dialog
