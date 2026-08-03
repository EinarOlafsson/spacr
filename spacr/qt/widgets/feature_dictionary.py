"""The in-app feature dictionary: look a measurement up without leaving spaCR.

A finished run writes hundreds of columns per object with names like
``cell_channel_1_percentile_75``. :mod:`spacr.feature_dict` has known what
those mean for a while, but only as an *export*: you could write a markdown
file describing a database, and that was the whole interface. So the moment a
user actually needs the answer — reading a results table, staring at a
regression coefficient, hovering a UMAP axis — they have to leave the app.

This module is the missing half:

:class:`FeatureDictionaryPanel`
    the searchable panel. Search by column name, by substring, or by *idea*
    ("intensity", "texture", "shape", "distance", "how big", "blurry"), filter
    by object type and concept, and read the definition, the unit, which
    objects the feature exists for, which channel it applies to and which
    module computes it.

:class:`FeatureDictionaryDialog` / :func:`open_feature_dictionary`
    the same panel as a non-modal window, optionally opened straight onto one
    column.

:func:`register`
    puts the panel in the app registry (Explore section) and its QSS in the
    theme, through the ``register_app`` / ``register_widget_qss`` seams —
    this module owns its own registration and edits neither ``app.py`` nor
    ``theme.py``.

:func:`install_window_hooks`
    the two reach-me-from-where-I-am routes: a **Help ▸ Feature Dictionary…**
    action, and a **"What is this?"** item on the context menu of any results
    table in the app.

The context-menu route is deliberately an application-level event filter
rather than an edit to each table screen. There are eleven table-bearing
screens and none of them claims a context menu today, so a filter reaches all
of them — including the ones built lazily, long after this hook ran — and
reaches any table added later for free. It stands aside for any widget that
has claimed its own context menu (``CustomContextMenu`` / ``ActionsContextMenu``),
so adopting one later silently takes precedence over this.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ...feature_dict import (
    CHANNEL_PAIR,
    CHANNEL_SINGLE,
    CONCEPTS,
    FEATURE_FAMILIES,
    OBJECT_TYPES,
    FeatureDoc,
    doc_for,
    parse_column,
    search_features,
)

LOG = logging.getLogger("spacr.qt.feature_dictionary")

#: App registry key. Load-bearing once shipped — saved user state keys off it.
APP_KEY = "feature_dict"
APP_NAME = "Feature Dictionary"
APP_DESC = ("What does cell_channel_1_percentile_75 mean? Search every "
            "measured feature by name or by idea")

#: ``objectName`` of the panel, and the name its QSS block registers under.
OBJECT_NAME = "FeatureDictionary"

#: Label of the Help menu action. Kept verbatim: `spacr/qt/i18n.py` keys its
#: catalog on the English string.
HELP_ACTION_TEXT = "Feature Dictionary…"
#: Label of the table context-menu action.
CONTEXT_ACTION_TEXT = "What is this?"

_PLACEHOLDER = ("Search a column name, or an idea: intensity, texture, "
                "shape, distance…")

_ANY_CONCEPT = "Any concept"
_ANY_OBJECT = "Any object"

#: How many columns of a table have to be recognisable before the context
#: menu offers to explain an *unrecognised* one. Below this the table is not a
#: measurements table and the menu would be noise on somebody else's grid.
_MEASUREMENT_TABLE_THRESHOLD = 3


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _escape(text: object) -> str:
    """Minimal HTML escape for the detail pane."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _channel_sentence(scope: str) -> str:
    """Say, in words, how channels enter a feature."""
    if scope == CHANNEL_SINGLE:
        return ("One column per channel — the <code>channel_&lt;i&gt;</code> "
                "in the name is which channel this number was measured in.")
    if scope == CHANNEL_PAIR:
        return ("One column per channel PAIR — the two "
                "<code>channel_&lt;i&gt;</code> infixes are the two channels "
                "being compared.")
    return "No channel: this number does not depend on what was imaged."


def _objects_sentence(doc: FeatureDoc) -> str:
    """Say which object types actually have this feature."""
    if not doc.object_types:
        if doc.kind == "metadata":
            return "Not a per-object measurement."
        return ("Not written for any object type by a standard run — see the "
                "note below.")
    listed = ", ".join(doc.object_types)
    missing = [o for o in OBJECT_TYPES if o not in doc.object_types]
    if not missing:
        return f"Written for every object type ({listed})."
    return (f"Written for {listed} — and NOT for "
            f"{', '.join(missing)}.")


def _doc_html(doc: FeatureDoc, entry=None) -> str:
    """Render one feature as the detail pane's HTML.

    :param doc: the feature.
    :param entry: optional :class:`spacr.feature_dict.FeatureEntry` for one
        concrete column, which pins the object type, the channel and the unit
        to what that column actually is.
    """
    rows: list[str] = []

    def field(name: str, value: object) -> None:
        if value in (None, "", ()):
            return
        rows.append(
            f"<tr><td style='padding-right:12px; vertical-align:top;'>"
            f"<b>{_escape(name)}</b></td><td>{value}</td></tr>")

    heading = _escape(entry.column if entry is not None else doc.title)
    parts = [f"<h3 style='margin-bottom:2px;'>{heading}</h3>"]
    if entry is not None and entry.key:
        parts.append(
            f"<p style='margin-top:0;'><i>an instance of "
            f"<b>{_escape(doc.title)}</b></i></p>")

    description = doc.description
    if entry is not None and entry.description:
        description = entry.description
    if description:
        parts.append(f"<p>{_escape(description)}</p>")
    else:
        parts.append("<p><i>No definition — see the note below.</i></p>")

    unit = doc.unit
    if entry is not None and entry.unit:
        unit = entry.unit
    field("Unit", _escape(unit) if unit else "<i>none (an identifier)</i>")
    field("Family", f"{_escape(doc.family)} — "
                    f"{_escape(FEATURE_FAMILIES.get(doc.family, ''))}")
    field("Objects", _escape(_objects_sentence(doc)))
    # Only for a genuine per-object-table feature. `cell_before_filtration`
    # carries a `cell_` prefix and lives in pivoted_counts, and an
    # `organelle_summary_*` column lives in `<parent>_organelle_summary` — for
    # either of them "the cell table" would be a confident lie, and each says
    # where it really lives in its own note.
    if (entry is not None and entry.object_type and doc.family != "meta"
            and not doc.key.startswith("organelle_summary_")):
        field("This column", f"the {_escape(entry.object_type)} table")
    channel_text = _channel_sentence(doc.channel_scope)
    if entry is not None and entry.channel is not None:
        which = f"channel {entry.channel}"
        if entry.channel_2 is not None:
            which = f"channels {entry.channel} and {entry.channel_2}"
        channel_text = f"This column is {which}. " + channel_text
    field("Channel", channel_text)
    field("Module", f"<code>{_escape(doc.module)}</code>")
    field("Computed by", f"<code>{_escape(doc.computed_by)}</code>")
    if doc.written_when:
        field("Written when", _escape(doc.written_when))
    if doc.concepts:
        field("Concepts", _escape(", ".join(doc.concepts)))
    if doc.examples:
        field("Example column",
              "<br>".join(f"<code>{_escape(x)}</code>" for x in doc.examples))

    notes = doc.notes
    if entry is not None and entry.notes:
        notes = entry.notes
    if notes:
        field("Note", _escape(notes))

    parts.append("<table>" + "".join(rows) + "</table>")
    return "".join(parts)


def _unknown_html(column: str) -> str:
    """The honest answer for a column the dictionary cannot explain."""
    return (
        f"<h3 style='margin-bottom:2px;'>{_escape(column)}</h3>"
        "<p><b>Not in the dictionary.</b> This column's name does not parse "
        "as a spaCR measurement and no curated entry matches it, so spaCR "
        "does not know what it means. It was <i>not</i> guessed at from the "
        "name.</p>"
        "<p>Likely explanations: a column added by hand or by another tool, "
        "a user-supplied custom feature (see <code>spacr.custom_features"
        "</code>), an annotation column named by whoever ran the annotation "
        "app, or a column from a spaCR version older than this one.</p>")


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

class FeatureDictionaryPanel(QWidget):
    """Searchable dictionary of every measurement spaCR writes.

    Constructing it costs one :func:`spacr.feature_dict.search_features` call
    and touches no database, so it is cheap to embed and testable without an
    event loop.

    :param parent: optional Qt parent.
    :param column: optional column name to open on.
    """

    #: Emitted with the curated key whenever the selection changes.
    feature_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None,
                 column: Optional[str] = None):
        super().__init__(parent)
        self.setObjectName(OBJECT_NAME)

        self._hits: list = []
        #: The concrete column the detail pane is pinned to, if any.
        self._column: Optional[str] = None
        #: The feature the detail pane is SHOWING. Not derived from the list
        #: selection: a column always resolves, even to a feature the free-text
        #: search did not surface, and the pane shows it either way.
        self._doc: Optional[FeatureDoc] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        blurb = QLabel(
            "Every number spaCR measures, and what it means. Search a column "
            "name you are looking at, or just say what you are after.")
        blurb.setObjectName("FeatureDictionaryBlurb")
        blurb.setWordWrap(True)
        outer.addWidget(blurb)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self._search = QLineEdit()
        self._search.setObjectName("FeatureDictionarySearch")
        self._search.setPlaceholderText(_PLACEHOLDER)
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._refresh)
        controls.addWidget(self._search, 1)

        self._concept = QComboBox()
        self._concept.setObjectName("FeatureDictionaryConcept")
        self._concept.addItem(_ANY_CONCEPT, None)
        for name, concept in CONCEPTS.items():
            self._concept.addItem(name, name)
            self._concept.setItemData(self._concept.count() - 1,
                                      concept.gloss, Qt.ToolTipRole)
        self._concept.currentIndexChanged.connect(self._refresh)
        controls.addWidget(self._concept)

        self._object = QComboBox()
        self._object.setObjectName("FeatureDictionaryObject")
        self._object.addItem(_ANY_OBJECT, None)
        for obj in OBJECT_TYPES:
            self._object.addItem(obj, obj)
        self._object.currentIndexChanged.connect(self._refresh)
        controls.addWidget(self._object)
        outer.addLayout(controls)

        body = QHBoxLayout()
        body.setSpacing(8)
        self._list = QListWidget()
        self._list.setObjectName("FeatureDictionaryList")
        self._list.currentRowChanged.connect(self._on_row_changed)
        body.addWidget(self._list, 2)

        self._detail = QTextBrowser()
        self._detail.setObjectName("FeatureDictionaryDetail")
        self._detail.setOpenExternalLinks(False)
        body.addWidget(self._detail, 3)
        outer.addLayout(body, 1)

        self._status = QLabel("")
        self._status.setObjectName("FeatureDictionaryStatus")
        outer.addWidget(self._status)

        if column:
            self.show_column(column)
        else:
            self._refresh()

    # -- public API ------------------------------------------------------

    def set_query(self, text: str) -> None:
        """Type ``text`` into the search box and re-run the search."""
        self._search.setText(str(text or ""))

    def show_column(self, column: str) -> None:
        """Explain one concrete column name.

        Searches for it (so the list shows the feature and its neighbours)
        and pins the detail pane to *that column* — its object type, its
        channel, its resolved unit — rather than to the generic feature.

        A name the dictionary cannot explain is reported as unknown. It is
        never approximated to the nearest-looking entry.
        """
        column = str(column or "").strip()
        self._column = column or None
        entry = parse_column(column) if column else None
        self._search.blockSignals(True)
        self._search.setText(column)
        self._search.blockSignals(False)
        # A column name is a specific question; the filters would only
        # narrow it away.
        self._concept.setCurrentIndex(0)
        self._object.setCurrentIndex(0)
        self._refresh()
        if entry is None or entry.family == "unknown" or not entry.key:
            self._list.setCurrentRow(-1)
            self._doc = None
            self._detail.setHtml(_unknown_html(column))
            self._status.setText(
                f"{column or '(no column)'} — not in the dictionary")
            return
        self._select_key(entry.key)

    def current_doc(self) -> Optional[FeatureDoc]:
        """The feature the detail pane is showing, or ``None``."""
        return self._doc

    def result_keys(self) -> list[str]:
        """Curated keys currently listed, best match first."""
        return [hit.doc.key for hit in self._hits]

    def detail_text(self) -> str:
        """The detail pane's rendered text — what the user actually reads."""
        return self._detail.toPlainText()

    # -- internals -------------------------------------------------------

    def _select_key(self, key: str) -> None:
        """Select the row holding ``key``, adding it if the search missed."""
        for row, hit in enumerate(self._hits):
            if hit.doc.key == key:
                self._list.setCurrentRow(row)
                return
        # A concrete column always resolves even when the free-text search
        # would not have surfaced its feature; show it anyway.
        doc = doc_for(key)
        if doc is not None:
            self._render(doc)

    def _refresh(self, *_args) -> None:
        """Re-run the search and repopulate the list."""
        query = self._search.text()
        concept = self._concept.currentData()
        obj = self._object.currentData()
        try:
            self._hits = search_features(
                query, concept=concept, object_type=obj, limit=300)
        except Exception:
            LOG.exception("Feature search failed for %r", query)
            self._hits = []

        self._list.blockSignals(True)
        self._list.clear()
        for hit in self._hits:
            doc = hit.doc
            where = ", ".join(doc.object_types) if doc.object_types else doc.kind
            item = QListWidgetItem(f"{doc.title}\n{doc.family} · {where}")
            item.setData(Qt.UserRole, doc.key)
            item.setToolTip(doc.description or "No definition.")
            self._list.addItem(item)
        self._list.blockSignals(False)

        if self._hits:
            self._status.setText(
                f"{len(self._hits)} feature(s)"
                + (f" matching “{query}”" if query.strip() else ""))
            if self._column is None:
                self._list.setCurrentRow(0)
        else:
            self._status.setText(
                f"Nothing matches “{query}”. Try an idea instead of a name: "
                "intensity, texture, shape, distance, size.")
            self._doc = None
            self._detail.setHtml(
                "<p><i>No feature matches that search.</i></p>")

    def _on_row_changed(self, row: int) -> None:
        if not (0 <= row < len(self._hits)):
            return
        doc = self._hits[row].doc
        # Once the user moves off the column they asked about, stop pinning
        # the detail pane to it.
        entry = None
        if self._column:
            candidate = parse_column(self._column)
            if candidate.key == doc.key:
                entry = candidate
            else:
                self._column = None
        self._render(doc, entry)

    def _render(self, doc: FeatureDoc, entry=None) -> None:
        self._doc = doc
        self._detail.setHtml(_doc_html(doc, entry))
        self.feature_selected.emit(doc.key)


class FeatureDictionaryDialog(QDialog):
    """:class:`FeatureDictionaryPanel` in a non-modal window."""

    def __init__(self, parent: Optional[QWidget] = None,
                 column: Optional[str] = None):
        super().__init__(parent)
        self.setObjectName("FeatureDictionaryDialog")
        self.setWindowTitle(APP_NAME)
        self.setModal(False)
        self.resize(940, 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.panel = FeatureDictionaryPanel(self, column=column)
        layout.addWidget(self.panel, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(12, 0, 12, 12)
        footer.addStretch(1)
        close = QPushButton("Close")
        close.setObjectName("GhostButton")
        close.clicked.connect(self.close)
        footer.addWidget(close)
        layout.addLayout(footer)

    def show_column(self, column: str) -> None:
        """Forward to the panel."""
        self.panel.show_column(column)


#: The one dialog, reused so repeated lookups do not stack windows.
_DIALOG: Optional[FeatureDictionaryDialog] = None


def open_feature_dictionary(parent: Optional[QWidget] = None,
                            column: Optional[str] = None
                            ) -> FeatureDictionaryDialog:
    """Show the dictionary, optionally opened onto ``column``.

    Reuses a single window: looking up six columns in a row should leave one
    dictionary open, not six.
    """
    global _DIALOG
    if _DIALOG is None:
        _DIALOG = FeatureDictionaryDialog(parent)
        _DIALOG.destroyed.connect(_forget_dialog)
    if column:
        _DIALOG.show_column(column)
    _DIALOG.show()
    _DIALOG.raise_()
    _DIALOG.activateWindow()
    return _DIALOG


def _forget_dialog(*_args) -> None:
    """Drop the cached dialog once Qt has destroyed it."""
    global _DIALOG
    _DIALOG = None


def close_feature_dictionary() -> None:
    """Close and forget the shared dialog. Used by tests and by teardown."""
    global _DIALOG
    if _DIALOG is not None:
        dialog, _DIALOG = _DIALOG, None
        dialog.close()
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# hook 1 — the app registry and the theme, through their seams
# ---------------------------------------------------------------------------

def make_screen(host=None) -> QWidget:
    """Screen factory for :func:`spacr.qt.app.register_app`."""
    return FeatureDictionaryPanel()


def _panel_qss(palette: dict, opacity) -> str:
    """QSS block for the panel, rendered against the live theme palette."""
    return f"""
QWidget#{OBJECT_NAME} QLabel#FeatureDictionaryBlurb,
QWidget#{OBJECT_NAME} QLabel#FeatureDictionaryStatus {{
    color: {palette['fg_muted']};
}}
QWidget#{OBJECT_NAME} QListWidget#FeatureDictionaryList {{
    background: {palette['surface_alt']};
    border: 1px solid {palette['border_soft']};
    border-radius: 8px;
}}
QWidget#{OBJECT_NAME} QTextBrowser#FeatureDictionaryDetail {{
    background: {palette['surface_alt']};
    border: 1px solid {palette['border_soft']};
    border-radius: 8px;
    padding: 8px;
}}
"""


def register() -> bool:
    """Register the app row and the QSS block. Idempotent.

    Called from :func:`spacr.qt.run` before the main window is built, because
    the sidebar, the menu bar and Home all read the registry during
    ``MainWindow.__init__``.

    :returns: ``True`` when the app row is in the registry afterwards.
    """
    ok = True
    try:
        from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
        if not any(row[0] == APP_KEY for row in APPS):
            register_app(APP_KEY, APP_NAME, APP_DESC, SECTION_EXPLORE,
                         factory=make_screen, stage=STAGE_ALPHA)
    except Exception:
        # A registry that cannot take one more app is not a reason for the
        # GUI to refuse to start; the Help menu route still works.
        LOG.exception("Could not register the Feature Dictionary app")
        ok = False
    try:
        from ..theme import register_widget_qss, widget_qss_names
        if OBJECT_NAME not in widget_qss_names():
            register_widget_qss(OBJECT_NAME, _panel_qss)
    except Exception:
        LOG.exception("Could not register the Feature Dictionary QSS")
    return ok


# ---------------------------------------------------------------------------
# hook 2 — Help menu + "What is this?" on every results table
# ---------------------------------------------------------------------------

def _find_menu(window: QMainWindow, title: str) -> Optional[QMenu]:
    """The window's menu-bar menu titled ``title``, ignoring ``&``.

    Found through ``findChildren`` rather than by walking the menu bar's
    actions and calling ``QAction.menu()``. That reading is the obvious one
    and it does not survive: the QMenu wrapper it returns is only valid while
    the QAction wrapper it came off is alive, so the menu went stale the
    moment the action list fell out of scope — "Internal C++ object
    (PySide6.QtWidgets.QMenu) already deleted" on the very next line, and,
    when the wrappers were kept alive to work around it, a segfault during
    the next event dispatch. ``findChildren`` hands back children the menu
    bar owns in C++, which stay valid for as long as the window does.
    """
    try:
        bar = window.menuBar()
        if bar is None:
            return None
        menus = bar.findChildren(QMenu)
    except Exception:
        return None
    for menu in menus:
        try:
            if menu.title().replace("&", "") == title:
                return menu
        except RuntimeError:
            continue
    return None


def install_help_action(window: QMainWindow) -> Optional[QAction]:
    """Add **Feature Dictionary…** to the window's Help menu.

    Returns the action, or ``None`` when there is no Help menu (a bare
    QMainWindow in a test) or one is already installed.
    """
    menu = _find_menu(window, "Help")
    if menu is None:
        return None
    for act in menu.actions():
        if act.text() == HELP_ACTION_TEXT:
            return None
    action = QAction(HELP_ACTION_TEXT, window)
    action.setStatusTip(
        "Look up what any measured feature means — its definition, its unit, "
        "which objects it exists for and which module computes it.")
    action.triggered.connect(
        lambda checked=False: open_feature_dictionary(window))
    # Above the separator that precedes "Check for updates…", so it sits with
    # the other "explain something" entries rather than with the tools.
    before = None
    for act in menu.actions():
        if act.isSeparator():
            before = act
            break
    if before is not None:
        menu.insertAction(before, action)
    else:
        menu.addAction(action)
    return action


def column_name_at(widget: QObject, pos) -> Optional[str]:
    """The name of the column under ``pos`` in ``widget``, or ``None``.

    Handles both halves of the gesture: a right-click on a header section and
    a right-click on a cell.
    """
    if isinstance(widget, QHeaderView):
        if widget.orientation() != Qt.Horizontal:
            return None
        index = widget.logicalIndexAt(pos)
        model = widget.model()
        if index < 0 or model is None:
            return None
        value = model.headerData(index, Qt.Horizontal, Qt.DisplayRole)
        return None if value is None else str(value)

    view = widget.parent() if not isinstance(widget, QAbstractItemView) else widget
    if not isinstance(view, QAbstractItemView):
        return None
    model = view.model()
    if model is None:
        return None
    index = view.indexAt(pos)
    if not index.isValid():
        return None
    value = model.headerData(index.column(), Qt.Horizontal, Qt.DisplayRole)
    return None if value is None else str(value)


def _table_looks_measured(model) -> bool:
    """Whether enough of a model's columns are spaCR measurements."""
    if model is None:
        return False
    recognised = 0
    for col in range(min(model.columnCount(), 40)):
        value = model.headerData(col, Qt.Horizontal, Qt.DisplayRole)
        if value is None:
            continue
        if parse_column(str(value)).family != "unknown":
            recognised += 1
            if recognised >= _MEASUREMENT_TABLE_THRESHOLD:
                return True
    return False


def _model_of(widget: QObject):
    """The item model behind a header, a view or a viewport."""
    if isinstance(widget, (QHeaderView, QAbstractItemView)):
        return widget.model()
    parent = widget.parent()
    return parent.model() if isinstance(parent, QAbstractItemView) else None


def _menu_family(widget: QObject) -> list[QObject]:
    """``widget`` plus everything a context-menu event passes through.

    A QContextMenuEvent that the header does not accept is PROPAGATED to the
    table behind it, so checking only the object the filter was handed lets a
    header with its own menu be answered by this one on the second pass —
    which is precisely the case the guard exists to prevent.
    """
    family = [widget, widget.parent()]
    view = widget if isinstance(widget, QAbstractItemView) else widget.parent()
    if isinstance(view, QAbstractItemView):
        family.append(view)
        for getter in ("horizontalHeader", "viewport"):
            member = getattr(view, getter, None)
            if callable(member):
                try:
                    family.append(member())
                except Exception:
                    pass
    return [obj for obj in family if obj is not None]


def _claims_own_menu(widget: QObject) -> bool:
    """Whether this table, or any part of it, has claimed its own menu."""
    claimed = (Qt.ContextMenuPolicy.CustomContextMenu,
               Qt.ContextMenuPolicy.ActionsContextMenu)
    for candidate in _menu_family(widget):
        policy = getattr(candidate, "contextMenuPolicy", None)
        if callable(policy) and policy() in claimed:
            return True
    return False


def _default_menu_runner(menu: QMenu, global_pos) -> None:
    """Show a context menu at ``global_pos`` and block until it closes."""
    menu.exec(global_pos)


#: How the context menu is shown. Replaced by :func:`set_menu_runner` so a
#: headless test can drive the whole gesture without entering a modal event
#: loop. Injected rather than monkey-patched: ``QMenu.exec`` is a C++ slot, a
#: test that rebinds it on the class does not intercept the call from inside
#: the filter, and the run simply hangs on a menu nobody can click.
_MENU_RUNNER = _default_menu_runner


def set_menu_runner(runner) -> None:
    """Replace the context-menu runner. ``None`` restores the default."""
    global _MENU_RUNNER
    _MENU_RUNNER = runner or _default_menu_runner


class FeatureHelpFilter(QObject):
    """Adds **What is this?** to the context menu of every results table.

    Installed on the :class:`QApplication`, so it covers the screens that do
    not exist yet when it is installed — every app screen in spaCR is built
    lazily on first navigation.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Type.ContextMenu:
            return False
        try:
            if not isinstance(obj, (QHeaderView, QAbstractItemView, QWidget)):
                return False
            if _claims_own_menu(obj):
                return False
            column = column_name_at(obj, event.pos())
            if column is None:
                return False
            entry = parse_column(column)
            if (entry.family == "unknown"
                    and not _table_looks_measured(_model_of(obj))):
                # Somebody else's grid: not a measurements table, so an item
                # about spaCR features would be noise on it.
                return False
            menu = QMenu(obj if isinstance(obj, QWidget) else None)
            action = menu.addAction(CONTEXT_ACTION_TEXT)
            action.setStatusTip(f"Explain the column “{column}”")
            action.triggered.connect(
                lambda checked=False, name=column:
                    open_feature_dictionary(None, name))
            _MENU_RUNNER(menu, event.globalPos())
            event.accept()
            return True
        except Exception:
            # A context menu is help, not function: never let it take a
            # right-click (or the app) down with it.
            LOG.debug("Feature help context menu failed", exc_info=True)
            return False


_FILTER: Optional[FeatureHelpFilter] = None


def install_context_menu_filter(app: Optional[QApplication] = None
                                ) -> Optional[FeatureHelpFilter]:
    """Install the table context-menu filter on the QApplication. Idempotent."""
    global _FILTER
    app = app or QApplication.instance()
    if app is None:
        return None
    if _FILTER is None:
        _FILTER = FeatureHelpFilter()
        app.installEventFilter(_FILTER)
    return _FILTER


def remove_context_menu_filter(app: Optional[QApplication] = None) -> bool:
    """Remove the filter again. ``True`` if there was one."""
    global _FILTER
    app = app or QApplication.instance()
    if _FILTER is None:
        return False
    if app is not None:
        app.removeEventFilter(_FILTER)
    _FILTER = None
    return True


def install_window_hooks(window: QMainWindow) -> None:
    """Wire the dictionary into a live main window.

    Called from :func:`spacr.qt.shortcuts.install`, which runs once from
    ``MainWindow.__init__`` after the menu bar exists. Every failure is
    logged and swallowed: a missing help entry must not cost anyone a window.
    """
    try:
        install_help_action(window)
    except Exception:
        LOG.exception("Could not add the Feature Dictionary help action")
    try:
        install_context_menu_filter()
    except Exception:
        LOG.exception("Could not install the feature help context menu")
