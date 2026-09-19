"""The search field beside the Help menu, and what each result opens.

Instruction 422. :mod:`spacr.qt.help_index` decides WHAT is addressable by
name; this module is where a user types a name and where the answer takes
them. The two halves are apart because only this one needs a display.

WHAT A RESULT DOES IS REGISTERED, NOT SWITCHED ON. :func:`register_opener`
binds a kind to ``(window, entry) -> str``; :func:`open_entry` looks the kind
up. A new kind of thing therefore needs a provider in the index module and an
opener here, and no existing function changes -- which is the instruction's
"and so on" clause taken literally.

The four that ship:

``module``
    :meth:`spacr.qt.app.MainWindow.open_module`, which resolves a folded
    module to the host that took it over.

``setting``
    Opens the module and then :func:`reveal_setting`: every category
    collapsed except the one holding the setting, the row scrolled to and
    marked. NOT by filtering the panel -- the per-module strip already has a
    filter and it leaves the other categories hidden, so a user who wanted to
    look around next has to work out what happened to the form.

``preference``
    Opens Preferences on the tab that holds the row and marks the row.

``api``
    Opens the published page WHEN IT IS REACHABLE, and otherwise shows the
    docstring that is on this machine and says the page is not reachable.
    A search result is believed, so a result that opens a 404 is worse than
    no result at all.

NOTHING THIS MODULE DOES ON THE GUI THREAD BLOCKS IT. The index costs about a
second to build, so it is built on a worker the first time the field is typed
in (:class:`_IndexLoader`), and the field says so until it lands. Whether the
documentation site answers is a network question, so it is asked on a worker
too (:class:`_DocsReach`) and only ever when the user has opened an API
result -- spaCR does not reach the network because somebody typed.

OPENING A RESULT NEVER DISCARDS SETTINGS. ``MainWindow._on_nav_selected``
keeps every screen it has built in ``self._screens`` and switches the stack to
it, so a half-filled Mask form is still half-filled when the user comes back
from Measure. That was true before this field existed and it is now pinned by
``tests/qt/test_the_help_search_field_lands_where_it_says.py`` so that a
future rebuild-on-navigate cannot quietly take it away.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .help_index import HelpEntry, build_index, search
from .i18n import tr

LOG = logging.getLogger("spacr.qt.help_search")

#: objectNames, so the theme can reach these and tests can find them.
FIELD_NAME = "HelpSearchField"
POPUP_NAME = "HelpSearchResults"
LIST_NAME = "HelpSearchResultList"
NOTE_NAME = "HelpSearchNote"
DIALOG_NAME = "HelpSearchApiEntry"

#: How long the field waits after a keystroke before searching. One search
#: over eleven thousand entries measured 15 ms, which is a sixth of a frame
#: on every letter of a word; a pause this short is invisible to a typist and
#: turns eight searches into one.
DEBOUNCE_MS = 140

#: How many rows the popup shows.
RESULT_LIMIT = 24

#: Role the entry is stashed under on its list item.
ENTRY_ROLE = int(Qt.UserRole) + 1


def _localize(widget: QWidget, setter_name: str, property_name: str,
              text: str, **kwargs) -> None:
    """Set a caption now and leave behind what a language change re-reads.

    The same seam :mod:`spacr.qt.settings_search` uses: the English source is
    stored on the widget so ``retranslate_widget_tree`` can set it again in
    whatever language is current, instead of a literal that freezes the
    control in English the first time it is rebuilt.

    :param widget: the widget to caption.
    :param setter_name: the setter to call, e.g. ``"setToolTip"``.
    :param property_name: the attribute the English is remembered under.
    :param text: the English source string.
    :param kwargs: placeholders for :func:`spacr.qt.i18n.tr`.
    """
    setter = getattr(widget, setter_name, None)
    if setter is None:
        return
    setattr(widget, property_name, text)
    try:
        setter(tr(text, **kwargs) if kwargs else tr(text))
    except Exception:
        setter(text)


_OPENERS: Dict[str, Callable[[QMainWindow, HelpEntry], str]] = {}


def register_opener(kind: str,
                    opener: Callable[[QMainWindow, HelpEntry], str]) -> None:
    """Say what happens when a result of ``kind`` is chosen.

    :param kind: the entry kind, matching a provider in
        :mod:`spacr.qt.help_index`.
    :param opener: ``(window, entry) -> str``; the string is shown in the
        status bar.
    """
    _OPENERS[str(kind)] = opener


def opener_kinds() -> tuple:
    """The kinds that can be opened, in registration order."""
    return tuple(_OPENERS)


def open_entry(window: QMainWindow, entry: HelpEntry) -> str:
    """Take the user to ``entry``.

    :param window: the main window.
    :param entry: the chosen result.
    :returns: what to say in the status bar; ``""`` when nothing was done.
    """
    opener = _OPENERS.get(entry.kind)
    if opener is None:
        LOG.debug("no opener registered for %r", entry.kind)
        return ""
    try:
        return str(opener(window, entry) or "")
    except Exception:
        LOG.exception("could not open the %s result %r", entry.kind,
                      entry.title)
        return tr("Could not open {name}.", name=entry.title)


def reveal_setting(window: QMainWindow, app_key: str, key: str) -> bool:
    """Open ``app_key`` with only ``key``'s category expanded and ``key`` shown.

    Goes through the per-module search strip, which already owns the map from
    a setting key to the section and field widget rendering it. Building a
    second map here would be a second thing to keep in step with the form.

    :param window: the main window.
    :param app_key: the module to open.
    :param key: the setting to reveal.
    :returns: True when the row was found and revealed.
    """
    opened = app_key
    try:
        opened = window.open_module(app_key)
    except Exception:
        LOG.exception("could not open %r", app_key)
        return False
    screen = getattr(window, "_screens", {}).get(opened)
    if screen is None:
        return False
    bar = getattr(screen, "_settings_search", None)
    reveal = getattr(bar, "reveal", None)
    if not callable(reveal):
        return False
    try:
        return bool(reveal(key))
    except Exception:
        LOG.exception("could not reveal %r on %r", key, opened)
        return False


def _open_module(window: QMainWindow, entry: HelpEntry) -> str:
    """Opener for ``kind="module"``."""
    key = entry.payload.get("app", "")
    window.open_module(key)
    return tr("Opened {name}", name=entry.title)


def _open_setting(window: QMainWindow, entry: HelpEntry) -> str:
    """Opener for ``kind="setting"``."""
    app_key = entry.payload.get("app", "")
    key = entry.payload.get("key", "")
    if reveal_setting(window, app_key, key):
        return tr("{key} in {where}", key=key, where=entry.subtitle)
    return tr("{key} is not on this module's form.", key=key)


def _open_preference(window: QMainWindow, entry: HelpEntry) -> str:
    """Opener for ``kind="preference"``."""
    tab = entry.payload.get("tab", "")
    label = entry.payload.get("label", entry.title)
    show = getattr(window, "show_preferences_on", None)
    if callable(show):
        show(tab, label)
    return tr("{name} in {where}", name=label, where=entry.subtitle)


def _open_api(window: QMainWindow, entry: HelpEntry) -> str:
    """Opener for ``kind="api"``: the page when it answers, the docstring when
    it does not."""
    symbol = entry.payload.get("symbol", entry.title)
    url = api_url(symbol)
    state = docs_reach().state()
    if state == "reachable":
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            LOG.exception("could not open %s", url)
            return tr("Could not open {url}", url=url)
        return tr("Opened {name} in the browser", name=symbol)
    docs_reach().start()
    show_api_entry(window, symbol, url, entry.description)
    return str(symbol)


register_opener("module", _open_module)
register_opener("setting", _open_setting)
register_opener("preference", _open_preference)
register_opener("api", _open_api)


def api_url(symbol: str, language: Optional[str] = None) -> str:
    """The published page for a dotted symbol.

    The same shape ``spacr.qt.screens.settings_model.api_docs_url`` builds
    for a settings row -- ``<base>/spacr/core/index.html#spacr.core.f``, with
    ``?lang=`` on a page the reader wants in another language -- from the
    same ``DOCS_API_BASE``, so the search field and a settings row cannot
    disagree about where the documentation lives.

    WHICH PREFIX IS THE MODULE is answered from the disk rather than assumed:
    ``spacr.qt.screens.mask`` is a module and ``spacr.layers.ShapesLayer.mask``
    is an attribute two levels inside one, and only the file layout knows
    which of the dots is the last one in the path.

    :param symbol: a dotted symbol name.
    :param language: a language code; the current one when ``None``.
    :returns: an absolute URL.
    """
    from .screens.settings_model import DOCS_API_BASE, _language_code

    parts = str(symbol).split(".")
    url = f"{DOCS_API_BASE}/index.html"
    for cut in range(len(parts), 0, -1):
        if _module_file(".".join(parts[:cut])) is None:
            continue
        page = f"{DOCS_API_BASE}/{'/'.join(parts[:cut])}/index.html"
        url = page if cut == len(parts) else f"{page}#{'.'.join(parts)}"
        break
    try:
        code = _language_code(language)
    except Exception:
        code = "en"
    if code == "en":
        return url
    base, _, fragment = url.partition("#")
    return f"{base}?lang={code}" + (f"#{fragment}" if fragment else "")


def _module_file(dotted: str):
    """The source file of ``dotted`` without importing it.

    An import would run module-level code for ten thousand candidate
    prefixes; the package layout answers the same question from the disk.

    :param dotted: a dotted module name inside ``spacr``.
    :returns: a :class:`pathlib.Path`, or ``None``.
    """
    from pathlib import Path

    import spacr

    parts = str(dotted).split(".")
    if not parts or parts[0] != "spacr":
        return None
    root = Path(spacr.__file__).resolve().parent
    rest = parts[1:]
    base = root.joinpath(*rest) if rest else root
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            return None
    return None


def local_docstring(symbol: str) -> str:
    """The docstring of ``symbol`` as it is on this machine.

    Read from the source with :mod:`ast` rather than by importing: the
    offline path must not be the one that drags ``spacr.core`` and its
    scientific stack into the process, and a docstring is in the syntax tree.

    :param symbol: a dotted symbol name.
    :returns: the docstring, or ``""`` when it cannot be found.
    """
    import ast

    parts = str(symbol).split(".")
    for cut in range(len(parts), 0, -1):
        path = _module_file(".".join(parts[:cut]))
        if path is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        node = tree
        for name in parts[cut:]:
            found = None
            for child in getattr(node, "body", []):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)) and child.name == name:
                    found = child
                    break
            if found is None:
                return ""
            node = found
        try:
            return ast.get_docstring(node) or ""
        except Exception:
            return ""
    return ""


class _DocsReach(QObject):
    """Whether the published documentation answers, asked off the GUI thread.

    Three states rather than two. ``"unknown"`` is the honest answer before
    anybody has asked, and it is treated as "do not claim the link works":
    the offline view opens, and its button to the web page becomes live if
    and when the probe says the site is there. A two-state version has to
    guess, and guessing "reachable" is how a search result becomes a 404.

    :ivar settled: emitted with the new state when the probe answers.
    """

    settled = Signal(str)

    #: What the probe fetches, and how long it waits. Short: the answer is
    #: only used to decide between a browser and a dialog, and a user who
    #: waited five seconds for that has been let down either way.
    TIMEOUT_S = 4.0

    def __init__(self) -> None:
        """Start unasked."""
        super().__init__()
        self._state = "unknown"
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def state(self) -> str:
        """``"reachable"``, ``"unreachable"`` or ``"unknown"``."""
        return self._state

    def set_state(self, state: str) -> None:
        """Record an answer, whoever found it out.

        Public so a test can state what the network is doing instead of
        depending on it.

        :param state: one of the three states.
        """
        self._state = str(state)
        self.settled.emit(self._state)

    def start(self) -> None:
        """Ask, once, on a daemon thread; do nothing if already asked."""
        with self._lock:
            if self._state != "unknown" or self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._probe, name="spacr-docs-reach", daemon=True)
            self._thread.start()

    def _probe(self) -> None:
        """Fetch the documentation root and record whether it answered."""
        from .screens.settings_model import DOCS_SITE_BASE

        answer = "unreachable"
        try:
            from urllib.request import urlopen

            with urlopen(DOCS_SITE_BASE, timeout=self.TIMEOUT_S) as response:
                if int(getattr(response, "status", 200) or 200) < 400:
                    answer = "reachable"
        except Exception:
            LOG.debug("the documentation site did not answer", exc_info=True)
        with self._lock:
            self._thread = None
        self.set_state(answer)


_REACH: Optional[_DocsReach] = None


def docs_reach() -> _DocsReach:
    """The one reachability probe, made on first use."""
    global _REACH
    if _REACH is None:
        _REACH = _DocsReach()
    return _REACH


class ApiEntryDialog(QDialog):
    """A symbol's local docstring, shown when its web page cannot be reached.

    :param symbol: the dotted symbol.
    :param url: where the page would be.
    :param fallback: the indexed summary, used when the source has no
        docstring to read.
    :param parent: parent widget.
    """

    def __init__(self, symbol: str, url: str, fallback: str = "",
                 parent: Optional[QWidget] = None):
        """Build the offline view of one API entry."""
        super().__init__(parent)
        self.setObjectName(DIALOG_NAME)
        self.setWindowTitle(symbol)
        self._url = url
        column = QVBoxLayout(self)

        self._note = QLabel(self)
        self._note.setObjectName(NOTE_NAME)
        self._note.setWordWrap(True)
        column.addWidget(self._note)

        body = QTextBrowser(self)
        body.setOpenExternalLinks(False)
        body.setPlainText(local_docstring(symbol) or fallback
                          or tr("No description is stored for this entry."))
        column.addWidget(body, 1)

        address = QLabel(url, self)
        address.setWordWrap(True)
        address.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(address)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        self._open = buttons.addButton(
            tr("Open the web page"), QDialogButtonBox.ActionRole)
        self._open.clicked.connect(self._open_the_page)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)

        reach = docs_reach()
        reach.settled.connect(self._on_settled)
        self._on_settled(reach.state())
        reach.start()

    def _on_settled(self, state: str) -> None:
        """Say what is known about the site, and enable the button if it is up.

        :param state: one of the three states of :class:`_DocsReach`.
        """
        if state == "reachable":
            _localize(self._note, "setText", "_spacr_i18n_text",
                      "This is the copy on this machine. The published page "
                      "is reachable — open it for the rendered version.")
        elif state == "unreachable":
            _localize(self._note, "setText", "_spacr_i18n_text",
                      "The published documentation is not reachable from "
                      "here, so this is the description stored with the code "
                      "on this machine.")
        else:
            _localize(self._note, "setText", "_spacr_i18n_text",
                      "This is the description stored with the code on this "
                      "machine. Checking whether the published page is "
                      "reachable…")
        self._open.setEnabled(state == "reachable")

    def _open_the_page(self) -> None:
        """Open the published page in the system browser."""
        import webbrowser

        try:
            webbrowser.open(self._url)
        except Exception:
            LOG.exception("could not open %s", self._url)
        self.accept()


def show_api_entry(window: Optional[QWidget], symbol: str, url: str,
                   fallback: str = "") -> ApiEntryDialog:
    """Show one API entry's local docstring.

    :param window: parent widget.
    :param symbol: the dotted symbol.
    :param url: where the published page would be.
    :param fallback: the indexed summary.
    :returns: the dialog, already shown.
    """
    dialog = ApiEntryDialog(symbol, url, fallback, window)
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    dialog.resize(560, 420)
    dialog.show()
    return dialog


class _IndexLoader(QObject):
    """Builds the index on a daemon thread and hands it back on the GUI one.

    :ivar ready: emitted with the list of entries.
    """

    ready = Signal(object)

    def __init__(self, parent: Optional[QObject] = None):
        """Start idle; :meth:`start` is what costs anything."""
        super().__init__(parent)
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def started(self) -> bool:
        """Whether the build has been asked for."""
        return self._started

    def start(self) -> None:
        """Build the index once, off the GUI thread."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._work, name="spacr-help-index", daemon=True)
        self._thread.start()

    def _work(self) -> None:
        """Build the index and emit it; an empty list on failure."""
        try:
            entries = build_index()
        except Exception:
            LOG.exception("the help index could not be built")
            entries = []
        self.ready.emit(entries)


class HelpSearchField(QLineEdit):
    """The box beside the Help menu.

    Owns the popup, the index and the keyboard. Arrow keys and Return are
    forwarded to the list while it is open, which is what lets the whole
    feature be used without the mouse leaving the keyboard: focus, type,
    Down, Return.

    :param window: the main window results act on.
    :param parent: parent widget.
    """

    #: Emitted after a result has been opened, with the entry.
    opened = Signal(object)

    def __init__(self, window: QMainWindow, parent: Optional[QWidget] = None):
        """Build the field, its popup and the debounce timer."""
        super().__init__(parent)
        self.setObjectName(FIELD_NAME)
        self.setClearButtonEnabled(True)
        self._window = window
        self._index: Optional[List[HelpEntry]] = None
        self._results: List[HelpEntry] = []

        _localize(self, "setPlaceholderText", "_spacr_i18n_placeholder",
                  "Search spaCR…")
        _localize(self, "setToolTip", "_spacr_i18n_tooltip",
                  "Find a module, a setting, a preference or an API entry by "
                  "name or by what it does. Opening a result never discards "
                  "the settings you have already typed.")
        _localize(self, "setAccessibleName", "_spacr_i18n_accessible_name",
                  "Search spaCR")

        self._popup = QFrame(window)
        self._popup.setObjectName(POPUP_NAME)
        self._popup.setFrameShape(QFrame.StyledPanel)
        self._popup.setAutoFillBackground(True)
        self._popup.setAttribute(Qt.WA_StyledBackground, True)
        self._popup.setFocusPolicy(Qt.NoFocus)
        self._popup.hide()
        popup_column = QVBoxLayout(self._popup)
        popup_column.setContentsMargins(0, 0, 0, 0)
        popup_column.setSpacing(0)
        popup_column.setSizeConstraint(QLayout.SetNoConstraint)

        self._note = QLabel(self._popup)
        self._note.setObjectName(NOTE_NAME)
        self._note.setWordWrap(True)
        self._note.setVisible(False)
        popup_column.addWidget(self._note)

        self._list = QListWidget(self._popup)
        self._list.setObjectName(LIST_NAME)
        self._list.setUniformItemSizes(False)
        self._list.setFocusPolicy(Qt.NoFocus)
        self._list.itemActivated.connect(self._on_activated)
        self._list.itemClicked.connect(self._on_activated)
        popup_column.addWidget(self._list, 1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._refresh)

        self._loader = _IndexLoader(self)
        self._loader.ready.connect(self._on_index_ready)

        self.textEdited.connect(self._on_text_edited)
        self.returnPressed.connect(self._activate_current)

    def set_index(self, entries: List[HelpEntry]) -> None:
        """Use ``entries`` instead of building one.

        :param entries: what :func:`spacr.qt.help_index.build_index` returned.
        """
        self._index = list(entries)
        if self.text().strip():
            self._refresh()

    def index(self) -> Optional[List[HelpEntry]]:
        """The index, or ``None`` while it is still being built."""
        return self._index

    def results(self) -> List[HelpEntry]:
        """What the popup is showing, best first."""
        return list(self._results)

    def popup(self) -> QFrame:
        """The results popup, so a test can ask whether it is up."""
        return self._popup

    def note(self) -> str:
        """The line above the list: what is being built, or what matched."""
        return self._note.text()

    def type_and_search(self, text: str) -> List[HelpEntry]:
        """Put ``text`` in the box and search now, skipping the debounce.

        The seam tests drive: typing is the user's action, and waiting 140 ms
        for a timer in every assertion would make the suite slower and no
        more honest about what the field does.

        :param text: what the user typed.
        :returns: the results now on screen.
        """
        self.setText(text)
        self._on_text_edited(text)
        self._debounce.stop()
        self._refresh()
        return self.results()

    def _on_text_edited(self, text: str) -> None:
        """Start the index if it is wanted, and debounce the search.

        :param text: the new contents of the box.
        """
        if str(text).strip() and self._index is None:
            self._loader.start()
            self._show_note(tr("Building the index…"))
        self._debounce.start()

    def _on_index_ready(self, entries) -> None:
        """Take the index the worker built and search with it.

        :param entries: the list of entries, possibly empty.
        """
        self._index = list(entries or [])
        if self.text().strip():
            self._refresh()

    def _show_note(self, text: str) -> None:
        """Show ``text`` above the list and make sure the popup is up.

        :param text: the line to show.
        """
        self._note.setText(text)
        self._note.setVisible(bool(text))
        self._place_popup()

    def _refresh(self) -> None:
        """Recompute the results for whatever is in the box."""
        query = self.text().strip()
        if not query:
            self._results = []
            self._list.clear()
            self.hide_popup()
            return
        if self._index is None:
            self._show_note(tr("Building the index…"))
            return
        self._results = search(self._index, query, limit=RESULT_LIMIT)
        self._list.clear()
        for entry in self._results:
            item = QListWidgetItem(self._row_text(entry))
            item.setData(ENTRY_ROLE, entry)
            if entry.description:
                item.setToolTip(entry.description)
            self._list.addItem(item)
        if self._results:
            self._list.setCurrentRow(0)
            self._note.setVisible(False)
            self._note.setText("")
        else:
            self._show_note(tr("Nothing called “{query}”.", query=query))
        self._place_popup()

    def _row_text(self, entry: HelpEntry) -> str:
        """One line for the list.

        :param entry: the result.
        :returns: the title, then where it lives.
        """
        return f"{entry.title}    {entry.subtitle}".rstrip()

    def _place_popup(self) -> None:
        """Size the list to the results and put it under the field.

        A CHILD OF THE WINDOW, not a ``Qt.Popup``. A popup window grabs the
        keyboard as soon as it is shown, so the next letter typed would go to
        the list instead of to the box the user is typing in -- and the list
        and the box are the same interaction. Staying inside the window keeps
        focus where the caret is, and both the frame and the list take
        ``NoFocus`` so that clicking a row does not move it either.

        The field is at the top of the window, so dropping the list down
        into the window costs nothing: there is always room below it.
        """
        if not self.isVisible():
            return
        window = self._popup.parentWidget()
        if window is None:
            return
        rows = min(max(self._list.count(), 1), 10)
        height = rows * max(self._list.sizeHintForRow(0), 18) + 8
        if self._note.isVisible():
            height += self._note.sizeHint().height() + 6
        width = max(self.width() * 3, 360)
        corner = self.mapTo(window, self.rect().bottomLeft())
        left = max(0, min(int(corner.x()), window.width() - int(width)))
        self._popup.setGeometry(left, int(corner.y()), int(width),
                                int(height))
        if not self._popup.isVisible():
            self._popup.show()
        self._popup.raise_()

    def hide_popup(self) -> None:
        """Put the popup away."""
        self._popup.hide()

    def _activate_current(self) -> None:
        """Open whatever row is selected."""
        item = self._list.currentItem()
        if item is not None:
            self._on_activated(item)

    def _on_activated(self, item: QListWidgetItem) -> None:
        """Open the entry behind ``item`` and put the field away.

        :param item: the row the user chose.
        """
        entry = item.data(ENTRY_ROLE) if item is not None else None
        if entry is None:
            return
        self.hide_popup()
        message = open_entry(self._window, entry)
        if message:
            try:
                self._window.statusBar().showMessage(message, 4000)
            except Exception:
                LOG.debug("no status bar to report to", exc_info=True)
        self.opened.emit(entry)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Steer the list from the box, and let Esc give the window back.

        :param event: the key press.
        """
        key = event.key()
        if key in (Qt.Key_Down, Qt.Key_Up, Qt.Key_PageDown, Qt.Key_PageUp):
            if self._list.count():
                self._place_popup()
                self._list.keyPressEvent(event)
                return
        if key == Qt.Key_Escape:
            if self._popup.isVisible():
                self.hide_popup()
            else:
                self.clear()
                self._window.setFocus()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        """Close the list when the caret leaves the box.

        Safe to do unconditionally now that the list takes ``NoFocus``:
        clicking a row cannot move focus, so a focus-out means the user went
        somewhere else and the list is in the way of whatever that was.
        """
        self.hide_popup()
        super().focusOutEvent(event)


def field_of(window: QMainWindow) -> Optional[HelpSearchField]:
    """The field installed on ``window``, if there is one.

    :param window: the main window.
    :returns: the field, or ``None``.
    """
    field = getattr(window, "_help_search", None)
    return field if isinstance(field, HelpSearchField) else None


def focus_field(window: QMainWindow) -> bool:
    """Put the caret in the help search box.

    :param window: the main window.
    :returns: True when there was a field to focus.
    """
    field = field_of(window)
    if field is None:
        return False
    field.setFocus(Qt.ShortcutFocusReason)
    field.selectAll()
    return True


def install(window: QMainWindow) -> Optional[HelpSearchField]:
    """Put the search field beside the Help menu.

    BESIDE, not inside. The menu bar's top-right corner widget is the strip
    that follows the last menu, and Help is the last menu, so the field goes
    at the left end of that strip -- in front of the minimise, full screen
    and close marks, which is where the eye already expects a search box on
    a window with no title bar.

    Idempotent: a second call hands back the field the first one installed.

    :param window: the main window.
    :returns: the field, or ``None`` when there is no menu bar to hang it on.
    """
    existing = field_of(window)
    if existing is not None:
        return existing
    try:
        bar = window.menuBar()
    except Exception:
        LOG.debug("no menu bar to hang the help search on", exc_info=True)
        return None
    if bar is None:
        return None

    field = HelpSearchField(window)
    field.setMinimumWidth(160)
    field.setMaximumWidth(280)

    corner = bar.cornerWidget(Qt.Corner.TopRightCorner)
    layout = corner.layout() if corner is not None else None
    if layout is not None:
        layout.insertWidget(0, field)
        field.setParent(corner)
    else:
        holder = QWidget(bar)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 6, 0)
        row.addWidget(field)
        bar.setCornerWidget(holder, Qt.Corner.TopRightCorner)
    window._help_search = field
    return field


def install_window_hooks(window: QMainWindow) -> Optional[HelpSearchField]:
    """Install the field from :func:`spacr.qt.shortcuts.install`.

    Named for the convention the other window-scoped installers follow, so
    that ``_install_window_hooks`` reads as one list of the same thing.

    :param window: the main window.
    :returns: the field, or ``None``.
    """
    return install(window)
