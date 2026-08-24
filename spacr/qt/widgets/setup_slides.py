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
import re
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
     "The mode decides how much of this machine spaCR keeps for itself: how "
     "many background processes it starts and holds, and whether it hands "
     "back its caches and GPU memory between runs. Balanced keeps them, "
     "which is fastest; the other two give them up so the rest of your "
     "machine has more to work with. The reproducibility hash records what "
     "went into a run, so a result can be traced back to the exact inputs "
     "that produced it.",
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
GREETING_BAND = 0.72

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
    """"Hello" in ``code``, falling back to English."""
    return GREETINGS.get(str(code or ""), GREETINGS["en"])


def _let_go_of(process) -> None:
    """Detach a still-running `gh` from a dialog that is being destroyed.

    Its signals are disconnected first: they point at widgets that are
    about to stop existing, and a `finished` delivered after that is the
    `libshiboken: Internal C++ object already deleted` crash.
    """
    # PER SIGNAL, not `QObject.disconnect()`: the argument-less form is
    # about connections FROM this object made through the QObject
    # overload, and it left the `finished` lambda connected -- measured,
    # not assumed.
    for signal in ("finished", "readyReadStandardOutput", "errorOccurred",
                   "readyReadStandardError"):
        try:
            getattr(process, signal).disconnect()
        except (RuntimeError, TypeError, AttributeError):
            # RuntimeError is Qt's "nothing was connected", which is the
            # ordinary case for a process that never emitted.
            pass
    try:
        process.setParent(None)
    except Exception:                                        # noqa: BLE001
        LOG.debug("gh process could not be detached", exc_info=True)


class SetupSlides(QDialog):
    """The setup screen: one question per slide, over a moving backdrop."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Set spaCR up")
        self.setModal(True)
        self._editors: Dict[str, QWidget] = {}
        self._index = 0

        # NO TITLE BAR HERE EITHER. The card this screen builds has rounded
        # corners, and a square window frame around it -- with a close and a
        # minimise button on top -- is the box the settings dialogs had
        # until they went frameless. This screen builds its own card, so
        # the glass filter deliberately leaves it alone, and leaving it
        # alone left it with its frame.
        #
        # Same order as `glass.make_frameless`: the attribute BEFORE the
        # flags, because the flags recreate the native window and a
        # translucency asked for afterwards applies to one that no longer
        # exists.
        self._go_frameless()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._backdrop = self._install_backdrop()

        from .setup_card import SetupCard

        self.card = SetupCard(self, radius=CARD_RADIUS)
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
        # NOT IN THE COLUMN. The greeting used to be a row in the layout, so
        # it took space on the language slide and had to be switched off the
        # moment the next slide arrived -- and switched off is what it looked
        # like: "the transition away from Hello is abrupt and bad".
        #
        # It floats over the card instead, low and centred, in a band the
        # question rows never reach on any slide. Nothing has to move out of
        # its way, so it can take its time leaving.
        self._greeting = QLabel("", self.card)
        self._greeting.setObjectName("CardTitle")
        self._greeting.setAlignment(Qt.AlignCenter)
        self._greeting.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._greeting.setVisible(False)

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
            # The window's own body paints nothing: `WA_TranslucentBackground`
            # stops Qt filling it from the palette, and this stops the
            # application stylesheet's `QDialog` rule doing it anyway.
            from .glass import _DragByBackground, _paint_nothing_behind_the_card

            _paint_nothing_behind_the_card(self)
            _DragByBackground(self)
            return True
        except Exception:                                    # noqa: BLE001
            LOG.debug("the setup screen would not go frameless",
                      exc_info=True)
            return False

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

        # THE MARK, WHICH IS THE STATE. "if spacr detects the login there
        # should also be a github logo that gains colour" -- signed out it
        # is drawn in the muted ink, signed in in GitHub's own black. The
        # mark is the same widget the AI providers use, so one rule covers
        # every sign-in on this screen.
        from .provider_marks import ProviderMark

        # THE LOGO IS THE BUTTON, the way the three AI marks are. It used
        # to be an indicator beside a "Sign in" push button, which the
        # user asked to collapse into one thing on 2026-08-23: "i want the
        # github button to also be a github logo just like the AI icons
        # work". One control, so there is nothing for a second one to
        # disagree with; what the click will DO is in the tooltip and
        # spelled out in the status line beside it.
        self._gh_mark = ProviderMark("github", "GitHub", False, holder)
        self._gh_mark.chosen.connect(lambda *_a: self._on_github_mark())
        row.addWidget(self._gh_mark)
        row.addStretch(1)

        self._gh_status = QLabel("")
        self._gh_status.setObjectName("Muted")
        self._gh_status.setWordWrap(True)
        row.addWidget(self._gh_status)

        self._refresh_github()
        return holder

    def _on_github_mark(self) -> bool:
        """What clicking the logo does, which `_sign_in_to_github` decides.

        Signed out or signed in, it is `gh auth login` -- signing in again
        is how you switch account or replace an expired token. With no
        `gh` on PATH there is nothing to log into, and that method opens
        the page that installs one.
        """
        return self._sign_in_to_github()

    def _still_on_screen(self) -> bool:
        """Whether this dialog's widgets are still real C++ objects.

        ``shiboken6.isValid`` is the only way to ask: a deleted QWidget
        keeps its Python wrapper, so ``is not None`` says yes right up
        until the attribute access raises.
        """
        try:
            from shiboken6 import isValid
        except Exception:                                    # noqa: BLE001
            return True
        for name in ("_gh_status", "_gh_mark"):
            widget = getattr(self, name, None)
            if widget is not None and not isValid(widget):
                return False
        return isValid(self)

    #: What each token source is called on screen.
    GITHUB_SOURCES = {
        "gh": "signed in through the GitHub CLI",
        "env": "signed in through GITHUB_TOKEN",
        "token": "signed in with a stored token",
    }

    def _refresh_github(self) -> None:
        """Say whether a token is reachable, and from where.

        SAFE AFTER THE SLIDES ARE GONE. `gh auth login` outlives this
        dialog -- the user finishes in a browser at their own pace -- and
        its `finished` signal lands here whenever that happens. If the
        setup screen has been closed by then, its child widgets are
        deleted C++ objects and touching one raises. Reported on
        2026-08-23 as `libshiboken: Internal C++ object (QPushButton)
        already deleted` after the sign-in returned.
        """
        if not self._still_on_screen():
            return
        try:
            from ..ai import github_auth

            source = github_auth.auth_source()
        except Exception:                                    # noqa: BLE001
            LOG.debug("GitHub auth is not readable here", exc_info=True)
            source = None
        import shutil

        # THE LOGO IS NEVER DEAD. All three states have something to do --
        # sign in, sign in again, or install `gh` -- and two of them used
        # to be greyed out, so on a machine where `gh` is already signed in
        # the row read "Signed in" beside a control nothing happened on.
        # Reported 2026-08-22 as "i cant click the github sign in", which
        # is exactly what a disabled control looks like from the outside.
        mark = getattr(self, "_gh_mark", None)
        if source:
            if mark is not None:
                self._light_the_github_mark(mark, mark.READY)
            self._gh_status.setText(
                self.GITHUB_SOURCES.get(source, "signed in"))
            if mark is not None:
                mark.setToolTip(
                    f"You are {self.GITHUB_SOURCES.get(source, 'signed in')}."
                    f" Clicking runs `gh auth login` again, which is how you"
                    f" switch to another account or replace a token that has"
                    f" expired.")
            self._gh_action = "login"
            return

        if shutil.which("gh") is None:
            # NAMED, not "sign-in failed". The CLI being absent and the CLI
            # being logged out need different things from the user -- and
            # what the absent one needs is the install page, which is
            # something this button can actually do.
            if mark is not None:
                self._light_the_github_mark(mark, mark.NOT_INSTALLED)
                mark.setToolTip(
                    "Opens the GitHub CLI's install page. With `gh` "
                    "installed, clicking this signs you in. Without it, "
                    "filing an issue still works -- it opens in whichever "
                    "browser you are already signed in to.")
            self._gh_status.setText(
                "the GitHub CLI is not installed — reports open in your "
                "browser")
            self._gh_action = "install"
            return

        if mark is not None:
            self._light_the_github_mark(mark, mark.SIGNED_OUT)
            mark.setToolTip(
                "Runs `gh auth login`, the GitHub CLI's own browser "
                "sign-in. GitHub stores the credential; spaCR never sees "
                "it.")
        self._gh_status.setText("not signed in — reports open in your browser")
        self._gh_action = "login"

    @staticmethod
    def _light_the_github_mark(mark, status: str) -> None:
        """Put the mark in ``status`` and repaint it if anything moved.

        `available` is what decides the brand fill, so it follows READY --
        the same rule the AI marks use, where a filled mark means the tool
        is there and usable and a muted one means it is not.
        """
        available = status == mark.READY
        if mark.status != status or bool(mark.available) != available:
            mark.status = status
            mark.available = available
            mark.update()

    #: Where the GitHub CLI is installed from. Opened when `gh` is absent,
    #: because "install this" is a thing a button can do and "the CLI is not
    #: installed" beside a dead button is not.
    GITHUB_CLI_PAGE = "https://cli.github.com/"

    #: What the button does next: 'login' or 'install'. Set by
    #: `_refresh_github`, which is the only thing that knows which state the
    #: machine is in.
    _gh_action = "login"

    #: The greeting's fade-out, held so it is not collected mid-animation.
    _goodbye = None

    #: Whether the browser has already been opened for THIS sign-in.
    #:
    #: `gh` reprints its code as it polls, and opening a tab per line would
    #: bury the one the user is typing into.
    _gh_opened = False

    #: GitHub's device-code page, which is where `gh auth login --web` sends
    #: you. Opened by spaCR rather than by `gh` -- see `_sign_in_to_github`.
    GITHUB_DEVICE_PAGE = "https://github.com/login/device"

    #: The shape of the one-time code `gh` prints before it opens a browser.
    GH_CODE = re.compile(r"\b([A-Z0-9]{4}-[A-Z0-9]{4})\b")

    def _sign_in_to_github(self) -> bool:
        """Start `gh auth login`, or open the install page. True if started.

        THE BROWSER IS OPENED HERE, and that is the whole point of this
        method. `gh auth login --web` does open one -- after printing a
        one-time code and waiting for Enter ON A TERMINAL. Started from a
        GUI there is no terminal, so `gh` sat forever on a prompt nobody
        could answer while the dialog said "waiting for GitHub in your
        browser…" about a browser that never opened. Reported exactly that
        way.

        So spaCR reads the code out of `gh`'s output, shows it, opens
        GitHub's device page itself, and then answers the prompt. The user
        sees the code they have to type and the page to type it into.

        DETACHED, and the dialog does not wait: the flow takes as long as
        the user takes, and a modal setup screen frozen behind it would look
        crashed. The status re-reads when the process ends.
        """
        from PySide6.QtCore import QProcess

        if self._gh_action == "install":
            return self._open_in_the_browser(self.GITHUB_CLI_PAGE)

        self._gh_opened = False
        process = QProcess(self)
        # ONE STREAM. `gh` prints the code on stderr and the prompt on
        # stdout, and reading only one of them loses half the exchange.
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda: self._read_github_output(process))
        process.finished.connect(lambda *_a: self._refresh_github())
        # AND IT IS CLEANED UP IF THE DIALOG GOES FIRST. A QProcess
        # destroyed with its child still running prints "QProcess:
        # Destroyed while process is still running" and leaves `gh`
        # parented to nothing -- so the dialog's destruction detaches it
        # rather than taking it down mid-login.
        self.destroyed.connect(lambda *_a: _let_go_of(process))
        try:
            process.start("gh", ["auth", "login", "--web",
                                 "--hostname", "github.com"])
            started = process.waitForStarted(3000)
        except Exception:                                    # noqa: BLE001
            LOG.debug("gh auth login would not start", exc_info=True)
            started = False
        if not started:
            self._gh_status.setText(
                "`gh auth login` would not start — run it in a terminal")
            return False
        self._gh_process = process
        self._gh_status.setText("starting GitHub sign-in…")
        # The logo stays live while `gh` runs. There is no second control to
        # disable now, and disabling the only one would leave a user whose
        # browser never opened with nothing to click.
        return True

    def _open_in_the_browser(self, url: str) -> bool:
        """Open ``url`` in the user's default browser. True if it opened."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        try:
            return bool(QDesktopServices.openUrl(QUrl(str(url))))
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not open %r", url, exc_info=True)
            return False

    def _read_github_output(self, process) -> str:
        """Show `gh`'s one-time code, open the device page, answer the prompt.

        :returns: the code found this time, or "".
        """
        try:
            chunk = bytes(process.readAllStandardOutput()).decode(
                "utf-8", "replace")
        except Exception:                                    # noqa: BLE001
            return ""
        found = self.GH_CODE.search(chunk or "")
        if found and not self._gh_opened:
            self._gh_opened = True
            code = found.group(1)
            opened = self._open_in_the_browser(self.GITHUB_DEVICE_PAGE)
            where = ("your browser" if opened
                     else self.GITHUB_DEVICE_PAGE)
            self._gh_status.setText(f"enter {code} in {where}")
            # AND ANSWER THE PROMPT `gh` is sitting on, so it proceeds to
            # poll GitHub. Without this it waits on Enter forever.
            try:
                process.write(b"\n")
            except Exception:                                # noqa: BLE001
                LOG.debug("could not answer the gh prompt", exc_info=True)
            return code
        return found.group(1) if found else ""

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
        # A COLUMN: the marks on one line, and under them a note, so that
        # choosing a provider can say what it started. Without somewhere to
        # say it, the sign-in would begin with no sign of it.
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        column.addLayout(row)
        holder._chosen = str(value or "")
        holder._buttons = {}
        holder._note = QLabel("")
        holder._note.setObjectName("Muted")
        holder._note.setWordWrap(True)
        column.addWidget(holder._note)
        for code, label, command in PROVIDERS:
            # SIGNED IN, not merely installed -- that is what the colour is
            # for. An installed-but-signed-out provider used to look exactly
            # like one that was ready to answer.
            state = self.provider_status(code, command)
            ready = state == ProviderMark.READY
            mark = ProviderMark(code, label, ready, holder, status=state)
            mark.set_chosen(holder._chosen == code)
            mark.setToolTip(
                f"Use {label}. You are signed in." if ready else
                f"Use {label}. Choosing it starts the sign-in; spaCR drives "
                f"the vendor's own CLI and never sees the credential.")
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
    def provider_status(code: str, command: str) -> str:
        """``ready`` / ``signed out`` / ``not installed`` for one provider.

        THREE STATES, because they need three different things from the
        user. `available` was one boolean covering "the CLI is missing" and
        "the CLI is there and signed out", so a mark could not say which,
        and both were drawn as a ghost -- "GPT brings no text and no color
        just a rim".
        """
        from .provider_marks import ProviderMark

        try:
            provider = SetupSlides._provider_object(code, command)
            if provider is not None:
                if not provider.is_installed():
                    return ProviderMark.NOT_INSTALLED
                return (ProviderMark.READY if provider.is_logged_in()
                        else ProviderMark.SIGNED_OUT)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not ask %r what state it is in", code,
                      exc_info=True)
        return (ProviderMark.READY
                if SetupSlides._provider_is_installed(command)
                else ProviderMark.NOT_INSTALLED)

    @staticmethod
    def _provider_is_signed_in(code: str, command: str) -> bool:
        """Whether this provider is installed AND logged in.

        WHAT THE COLOUR MEANS. The mark is drawn in the brand colour when
        this is true and in muted ink when it is not -- "the icon should be
        coloured if they are logged in". It used to mean only "the CLI is on
        PATH", so a provider that was installed and signed out looked
        identical to one that was ready to answer.
        """
        try:
            from ..ai.providers import get_provider

            provider = get_provider(str(code))
            if provider is not None:
                return bool(provider.is_configured())
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not ask %r whether it is signed in", code,
                      exc_info=True)
        return SetupSlides._provider_is_installed(command)

    @staticmethod
    def _provider_object(code: str, command: str = ""):
        """The registry entry for ``code``, whichever name it is filed under.

        THE TWO VOCABULARIES DID NOT MEET. This screen calls OpenAI's
        provider `gpt` -- which is what a user recognises -- and
        `ai.providers` files it under `codex`, which is its CLI. So
        `get_provider('gpt')` found nothing, GPT alone fell through every
        state check to a bare PATH test, and it was the one mark that
        rendered as neither installed nor signed in nor anything else.

        Tried by code first and by CLI name second, so both spellings
        resolve and neither module has to rename anything.
        """
        try:
            from ..ai.providers import get_provider
        except Exception:                                    # noqa: BLE001
            return None
        for name in (str(code), str(command)):
            if not name:
                continue
            try:
                found = get_provider(name)
            except Exception:                                # noqa: BLE001
                found = None
            if found is not None:
                return found
        return None

    #: Where each provider's CLI is installed from.
    #:
    #: A PAGE, NOT A COMMAND. The install one-liner differs by operating
    #: system and by package manager, and running the wrong one is worse
    #: than opening the page that lists them all -- which is also the only
    #: step that behaves the same on Linux, macOS and Windows.
    PROVIDER_PAGES = {
        "claude": "https://docs.anthropic.com/en/docs/claude-code/setup",
        "gpt": "https://github.com/openai/codex",
        "gemini": "https://github.com/google-gemini/gemini-cli",
    }

    #: Terminal emulators tried, in order, for an interactive CLI login.
    #:
    #: These logins are conversations -- a code to copy, a key to paste --
    #: and a GUI child process has no terminal to have them in. Started
    #: without one they hang on a prompt nobody can see, which is exactly
    #: how the GitHub button used to fail.
    TERMINALS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("x-terminal-emulator", ("-e",)),
        ("gnome-terminal", ("--",)),
        ("konsole", ("-e",)),
        ("xfce4-terminal", ("-e",)),
        ("xterm", ("-e",)),
    )

    def _run_in_a_terminal(self, command: str) -> bool:
        """Run ``command`` in the user's terminal. True if one was found."""
        import shutil
        import sys

        from PySide6.QtCore import QProcess

        parts = str(command).split()
        if not parts:
            return False
        if sys.platform == "darwin":
            return QProcess.startDetached(
                "osascript", ["-e", f'tell app "Terminal" to do script '
                                    f'"{command}"'])
        if sys.platform.startswith("win"):
            return QProcess.startDetached("cmd", ["/c", "start", *parts])
        for terminal, flag in self.TERMINALS:
            if shutil.which(terminal) is None:
                continue
            return bool(QProcess.startDetached(terminal,
                                               [*flag, *parts]))
        return False

    def _start_provider_login(self, code: str) -> str:
        """Begin ``code``'s sign-in. Returns what the user should be told.

        "If the user clicks an AI provider they should get prompted to login
        right away." Choosing one used to do nothing but tick it, and the
        login instructions lived in another screen entirely.
        """
        command = next((c for k, _l, c in PROVIDERS if k == code), "")
        provider = self._provider_object(code, command)
        if provider is None:
            return ""
        if provider.is_installed() and provider.is_logged_in():
            return ""
        return self._prompt_to_set_up(provider, code)

    def _prompt_to_set_up(self, provider, code: str) -> str:
        """Ask what to do about a provider that is not ready, and do it.

        A NOTE UNDER THE ROW WAS NOT ENOUGH -- "there is no popup no prompt
        for installing or any guidance". Choosing a provider you have not
        set up is the moment to say what it needs and offer to start it, so
        this is a dialog with the command in it and buttons that act.

        Every button works on every operating system: opening a page goes
        through QDesktopServices, and the terminal launcher knows macOS,
        Windows and the usual Linux emulators.

        :returns: a short line for the row's note, or "".
        """
        from PySide6.QtWidgets import QMessageBox

        installed = provider.is_installed()
        page = self.PROVIDER_PAGES.get(str(code), "")
        hint = str(getattr(provider, "install_hint", "") or "")
        login = str(getattr(provider, "login_command", "") or "")

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(f"Set up {provider.label}")
        if not installed:
            box.setText(
                f"{provider.label} is not installed yet.\n\n"
                f"spaCR drives the vendor's own command-line tool, so "
                f"installing `{provider.cli_name}` is all that is needed — "
                f"spaCR never sees your credentials.")
            box.setInformativeText(f"Install with:\n    {hint}"
                                   if hint else "")
        else:
            box.setText(
                f"{provider.label} is installed but not signed in.\n\n"
                f"Signing in happens in the vendor's own tool; spaCR never "
                f"sees the credential.")
            box.setInformativeText(f"Sign in with:\n    {login}"
                                   if login else "")

        act_open = (box.addButton("Open the page",
                                  QMessageBox.ButtonRole.AcceptRole)
                    if (page and not installed) else None)
        act_run = (box.addButton("Sign in now",
                                 QMessageBox.ButtonRole.AcceptRole)
                   if (installed and login) else None)
        act_copy = box.addButton("Copy the command",
                                 QMessageBox.ButtonRole.ActionRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.exec()

        clicked = box.clickedButton()
        if act_open is not None and clicked is act_open:
            self._open_in_the_browser(page)
            return f"{provider.label}: its install page is open in your browser."
        if act_run is not None and clicked is act_run:
            if self._run_in_a_terminal(login):
                return f"Signing in to {provider.label} in a terminal…"
            return f"Run `{login}` to sign in to {provider.label}."
        if clicked is act_copy:
            command = hint if not installed else login
            try:
                from PySide6.QtWidgets import QApplication

                QApplication.clipboard().setText(command)
            except Exception:                                # noqa: BLE001
                LOG.debug("could not copy the command", exc_info=True)
            return f"Copied: {command}"
        return (f"{provider.label} is not set up yet."
                if not installed else
                f"{provider.label} is installed but not signed in.")

    def _choose_provider(self, holder, code: str) -> None:
        """Select a provider AND start its login if it needs one."""
        holder._chosen = str(code)
        for name, mark in holder._buttons.items():
            mark.set_chosen(name == code)
        note = self._start_provider_login(code)
        status = getattr(holder, "_note", None)
        if status is not None:
            # SET EVEN WHEN EMPTY. A provider that is ready has nothing to
            # say, and leaving the previous provider's note standing under
            # it says something untrue about the one now selected.
            status.setText(note)
        # The mark's colour is the login state, so it has to be re-asked.
        self._refresh_provider_marks(holder)

    def _refresh_provider_marks(self, holder) -> None:
        """Re-colour every mark from its CURRENT sign-in state."""
        from .provider_marks import ProviderMark

        for code, label, command in PROVIDERS:
            mark = getattr(holder, "_buttons", {}).get(code)
            if mark is None:
                continue
            state = self.provider_status(code, command)
            signed_in = state == ProviderMark.READY
            if mark.status != state or bool(mark.available) != signed_in:
                mark.status = state
                mark.available = signed_in
                mark.update()
            mark.setToolTip(
                f"Use {label}. You are signed in." if signed_in else
                f"Use {label}. Choosing it starts the sign-in; spaCR drives "
                f"the vendor's own CLI and never sees the credential.")

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
            # INVARIANTS 10: without an animation it is simply there.
            LOG.debug("no fade for the greeting", exc_info=True)
            self._hello = None

    def _fade_the_greeting_away(self) -> None:
        """Take the greeting off gently, and only if it is on.

        The abrupt version was ``setVisible(False)`` the instant the slide
        changed. Nothing depends on the word being gone -- it floats over a
        band no slide uses -- so it can be given the time to leave.
        """
        # `isVisibleTo`, NOT `isVisible`. The latter is False whenever an
        # ancestor is hidden -- a dialog built but not yet shown -- so the
        # guard skipped the fade and left the word marked visible for
        # whenever the dialog did appear. The question here is whether this
        # widget is marked visible, which is what isVisibleTo answers.
        if not self._greeting.isVisibleTo(self.card):
            return
        try:
            effect = QGraphicsOpacityEffect(self._greeting)
            self._greeting.setGraphicsEffect(effect)
            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(GREETING_LEAVE_MS)
            animation.setStartValue(1.0)
            animation.setEndValue(0.0)
            animation.setEasingCurve(QEasingCurve.InCubic)
            # HIDDEN AT THE END, not at the start: a hidden widget does not
            # animate, so hiding first is the abrupt cut with extra steps.
            animation.finished.connect(
                lambda: self._greeting.setVisible(False))
            self._goodbye = animation
            animation.start()
        except Exception:                                    # noqa: BLE001
            # INVARIANTS 10: without an animation it simply goes.
            LOG.debug("no fade for the greeting", exc_info=True)
            self._greeting.setVisible(False)
            self._goodbye = None

    def _place_the_greeting(self) -> None:
        """Put the greeting in its band, across the width of the card."""
        card = getattr(self, "card", None)
        if card is None or not self._greeting:
            return
        height = self._greeting.sizeHint().height()
        self._greeting.setGeometry(
            0, int(card.height() * GREETING_BAND), card.width(), height)
        self._greeting.raise_()

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
            self._fade_the_greeting_away()
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

            # ROUNDED TO THE CARD'S RADIUS. The dialog is frameless and
            # translucent and holds exactly one card; a square backdrop
            # behind a rounded card is a second surface, and looked like
            # one -- "there is a square window with the theme and in front
            # of that window is a dark square with rounded edges".
            return install_ambient(self, theme=BACKDROP_THEME,
                                   speed=BACKDROP_SPEED,
                                   corner_radius=CARD_RADIUS)
        except Exception:                                    # noqa: BLE001
            LOG.debug("no ambient backdrop on this platform", exc_info=True)
            return None

    def resizeEvent(self, event):               # noqa: N802 - Qt naming
        super().resizeEvent(event)
        # NO MARGIN. The card used to be inset by 44px inside the ambient
        # backdrop, which put a themed square around a rounded card and made
        # the dialog read as two windows. The card now IS the window: same
        # rectangle as the backdrop, same corner radius, so there is one
        # rounded translucent surface and the settings sit on it rather
        # than in a container floating over it.
        self.card.setGeometry(self.rect())
        self.card.raise_()
        self._place_the_greeting()


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
