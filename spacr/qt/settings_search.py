"""Find a setting, and meet a module's settings a few at a time.

A spaCR module can render a lot of settings. Mask alone renders 190 of them
under thirteen collapsed headings, and across the shell there are 1,022. Until
now the only way to reach one was to guess which heading somebody filed it
under and open headings until it appeared — and the only thing a first-time
user saw was those thirteen headings, with nothing to say which two of them
they actually had to touch.

This module adds one strip above the settings form:

* **a search box** that matches the setting's key, its label *and* its
  description. The description is the only part written in the language a
  user thinks in, so "touching" finds ``merge_edge_pathogen_cells`` and
  "gpu" finds ``n_jobs`` — neither word appears in either name.
* **a Modified filter** that shows only what differs from the module's
  defaults. It is the fastest possible answer to "what did I change?", and
  it shares :func:`spacr.qt.settings_diff._values_equal` with the diff
  dialog and the run journal so all three agree about what an edit is.
* **an Essentials / All switch** — the progressive disclosure. Essentials
  shows the module's inputs plus the handful of decisions
  :func:`spacr.qt.screens.settings_model.essential_keys` derives from its
  curated layout, expanded and ready; All restores every heading, collapsed
  as before. Essentials is the default on a module's first visit and the
  choice is remembered per module thereafter, so a returning expert never
  meets the training wheels twice.

The three compose: with a query typed, Essentials narrows the search rather
than fighting it, and the count line always says exactly what is being shown
out of how many.

Installation is from outside the screen, deliberately::

    from spacr.qt.settings_search import install_window_hooks
    install_window_hooks(window)

:mod:`spacr.qt.shortcuts` calls that once from ``MainWindow.__init__``. The
installer then follows the screen stack, so a module built later in the
session gets its strip when it is first shown rather than needing a line
inside the shared screen.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .widgets.toggle import Toggle

LOG = logging.getLogger("spacr.qt.settings_search")

#: objectNames, so the theme can reach the strip and tests can find it.
BAR_NAME = "SettingsSearchBar"
INPUT_NAME = "SettingsSearchInput"
COUNT_NAME = "SettingsSearchCount"
MODIFIED_NAME = "SettingsSearchModified"
DISCLOSURE_NAME = "SettingsSearchDisclosure"
#: The wrapper `install` puts around the strip AND the settings scroll area,
#: so the two occupy one splitter slot. It spans the whole settings column,
#: which is what made it the single most damaging unstyled widget on the
#: page — see `_bar_qss`.
PANE_NAME = "SettingsSearchPane"

#: Where the per-module Essentials/All choice is remembered.
_QSETTINGS_ORG = "spacr"
_QSETTINGS_APP = "qt"
_KEY_DISCLOSURE = "settings/disclosure"

#: The two disclosure levels. Strings rather than a bool because they are
#: persisted, and a persisted bool named ``expanded`` is unreadable the day
#: a third level is wanted.
ESSENTIALS = "essentials"
ALL = "all"


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings(_QSETTINGS_ORG, _QSETTINGS_APP)


def disclosure_for(app_key: str) -> str:
    """The remembered disclosure level for ``app_key``.

    Defaults to :data:`ESSENTIALS`, which is the whole point: a module is
    met a few settings at a time until its user says otherwise.

    :param app_key: the module's app key.
    """
    raw = _settings().value(f"{_KEY_DISCLOSURE}/{app_key}", ESSENTIALS)
    return ALL if str(raw) == ALL else ESSENTIALS


def remember_disclosure(app_key: str, level: str) -> None:
    """Persist the disclosure level chosen for ``app_key``."""
    _settings().setValue(f"{_KEY_DISCLOSURE}/{app_key}",
                         ALL if level == ALL else ESSENTIALS)


def forget_disclosure(app_key: Optional[str] = None) -> None:
    """Forget one module's disclosure choice, or every module's.

    :param app_key: the module to forget, or ``None`` for all of them.
    """
    store = _settings()
    if app_key is None:
        store.remove(_KEY_DISCLOSURE)
    else:
        store.remove(f"{_KEY_DISCLOSURE}/{app_key}")


# ---------------------------------------------------------------------------
# The strip
# ---------------------------------------------------------------------------

class SettingsSearchBar(QWidget):
    """Search box, Modified filter, Essentials/All switch, and a count line.

    Owns no settings state of its own — it reads the screen's
    ``SettingsWidgets`` model and shows or hides rows that already exist.
    Hiding rather than rebuilding is what keeps a half-typed value alive
    across a filter change, which a rebuild would silently discard.
    """

    def __init__(self, screen: QWidget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName(BAR_NAME)
        # Fixed height, explicitly. The strip is two rows tall and the scroll
        # area under it wants everything else; without this the two share the
        # pane by their stretch factors and the search box ends up 800 pixels
        # high on first layout.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._screen = screen
        self._app_key = str(getattr(screen, "app_key", "") or "")
        self._model = getattr(screen, "_settings_model", None)
        # key -> (section, field widget). Built once, from the rendered form,
        # so the filter never has to guess which section a key ended up in.
        self._index: Dict[str, Tuple[QWidget, QWidget]] = {}
        self._sections: List[QWidget] = list(
            getattr(screen, "_settings_sections", []) or [])
        # Which sections the user had open before a filter took over, so
        # clearing the box puts the form back rather than leaving it splayed.
        self._restore_expanded: Optional[Dict[int, bool]] = None
        self._level = disclosure_for(self._app_key)

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 4)
        column.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        column.addLayout(row)

        self._input = QLineEdit(self)
        self._input.setObjectName(INPUT_NAME)
        self._input.setClearButtonEnabled(True)
        self._input.setPlaceholderText("Search settings…")
        self._input.setToolTip(
            "Search every setting in this module by name, by label, or by "
            "what its description says it does.")
        self._input.setAccessibleName("Search settings")
        self._input.textChanged.connect(self._on_query_changed)
        row.addWidget(self._input, 1)

        # A `Toggle`, not a QCheckBox: every boolean control in the shell is
        # a switch, and `tests/qt/test_widgets.py` bans the plain checkbox
        # outright so one panel cannot quietly reintroduce it.
        self._modified_label = QLabel("Modified", self)
        self._modified_label.setObjectName(MODIFIED_NAME + "Label")
        row.addWidget(self._modified_label, 0)

        self._modified = Toggle(parent=self)
        self._modified.setObjectName(MODIFIED_NAME)
        self._modified.setToolTip(
            "Show only the settings that no longer hold this module's "
            "default value.")
        self._modified.setAccessibleName("Show modified settings only")
        self._modified.toggled.connect(self._on_modified_toggled)
        row.addWidget(self._modified, 0)

        self._disclosure = QToolButton(self)
        self._disclosure.setObjectName(DISCLOSURE_NAME)
        self._disclosure.setCheckable(True)
        self._disclosure.setCursor(Qt.PointingHandCursor)
        self._disclosure.setChecked(self._level == ALL)
        self._disclosure.toggled.connect(self._on_disclosure_toggled)
        row.addWidget(self._disclosure, 0)
        # Where other modules hang their own controls — see
        # `spacr.qt.recipes.install`. Kept as the row itself rather than a
        # nested container so a trailing button lines up with the ones
        # above it rather than being visibly bolted on.
        self._controls_row = row

        # The count line sits under the controls, so the whole strip is one
        # widget the host inserts in one place.
        self._count = QLabel(self)
        self._count.setObjectName(COUNT_NAME)
        self._count.setWordWrap(True)
        column.addWidget(self._count)

        self._refresh_disclosure_text()
        self._build_index()
        self.apply()

    # -- public -------------------------------------------------------
    def query(self) -> str:
        """The current search text."""
        return self._input.text()

    def set_query(self, text: str) -> None:
        """Type ``text`` into the search box, filtering as it goes."""
        self._input.setText(str(text or ""))

    def level(self) -> str:
        """:data:`ESSENTIALS` or :data:`ALL`."""
        return self._level

    def set_level(self, level: str) -> None:
        """Switch disclosure level and remember the choice."""
        self._disclosure.setChecked(level == ALL)

    def modified_only(self) -> bool:
        """True when the Modified filter is on."""
        return self._modified.isChecked()

    def set_modified_only(self, on: bool) -> None:
        """Turn the Modified filter on or off."""
        self._modified.setChecked(bool(on))

    def visible_keys(self) -> List[str]:
        """Setting keys whose form row is currently shown.

        Row visibility, not widget visibility: a collapsed section makes
        every widget inside it invisible, and "you collapsed that heading"
        is a different statement from "the filter excluded that setting".
        """
        return [key for key, (section, field) in self._index.items()
                if _row_is_visible(section, field)]

    def indexed_keys(self) -> List[str]:
        """Every setting key the strip can show or hide."""
        return list(self._index)

    def add_trailing_widget(self, widget: QWidget) -> None:
        """Add ``widget`` to the right-hand end of the control row.

        The seam other modules use to put a settings-scoped control where
        the settings are, instead of inventing a second strip. Reparents
        ``widget`` onto the bar.

        :param widget: any widget; it keeps its own size policy.
        """
        widget.setParent(self)
        self._controls_row.addWidget(widget, 0)

    def count_text(self) -> str:
        """The sentence under the controls. Public so tests read what users
        read rather than recomputing it."""
        return self._count.text()

    def apply(self) -> None:
        """Recompute which rows and sections are shown.

        Called on every change to the query, the Modified filter or the
        disclosure level — one path, so the three can never disagree about
        what should be on screen.
        """
        model = self._model
        if model is None or not self._index:
            self._count.setText("")
            return

        total = len(self._index)
        wanted = set(self._index)

        query = self._input.text().strip()
        if query:
            try:
                wanted &= set(model.keys_matching(query))
            except Exception:
                LOG.debug("settings search failed for %r", query, exc_info=True)

        if self._modified.isChecked():
            try:
                wanted &= set(model.modified_keys())
            except Exception:
                LOG.debug("modified-only filter failed", exc_info=True)

        essentials: List[str] = []
        if self._level == ESSENTIALS:
            try:
                essentials = [k for k in model.essential_keys()
                              if k in self._index]
            except Exception:
                LOG.debug("essential keys unavailable", exc_info=True)
            if essentials:
                wanted &= set(essentials)

        for key, (section, field) in self._index.items():
            _set_row_visible(section, field, key in wanted)

        shown_per_section: Dict[int, int] = {}
        for key, (section, _field) in self._index.items():
            if key in wanted:
                shown_per_section[id(section)] = (
                    shown_per_section.get(id(section), 0) + 1)

        narrowing = bool(query) or self._modified.isChecked() \
            or (self._level == ESSENTIALS and bool(essentials))
        self._apply_section_state(shown_per_section, narrowing)
        self._count.setText(
            self._compose_count(len(wanted), total, len(essentials)))

    # -- wiring -------------------------------------------------------
    def _on_query_changed(self, _text: str) -> None:
        self.apply()

    def _on_modified_toggled(self, _on: bool) -> None:
        self.apply()

    def _on_disclosure_toggled(self, on: bool) -> None:
        self._level = ALL if on else ESSENTIALS
        remember_disclosure(self._app_key, self._level)
        self._refresh_disclosure_text()
        self.apply()

    # -- internals ----------------------------------------------------
    def _refresh_disclosure_text(self) -> None:
        if self._level == ALL:
            self._disclosure.setText("All settings")
            self._disclosure.setToolTip(
                "Showing every setting. Click for the essentials only.")
        else:
            self._disclosure.setText("Essentials")
            self._disclosure.setToolTip(
                "Showing the settings this module cannot run without. "
                "Click for all of them.")
        self._disclosure.setAccessibleName(self._disclosure.text())

    def _build_index(self) -> None:
        """Map each setting key to the section and field widget showing it.

        Built from the model's own ``key -> widget`` map and the sections the
        screen kept, rather than by re-deriving the layout: the screen has
        already decided which key went where, and a second opinion here would
        be a second thing to keep in sync.
        """
        widgets = getattr(self._model, "_widgets", {}) or {}
        by_widget = {id(w): key for key, w in widgets.items()}
        for section in self._sections:
            form = _form_of(section)
            if form is None:
                continue
            for i in range(form.rowCount()):
                item = form.itemAt(i, QFormLayout.FieldRole)
                field = item.widget() if item is not None else None
                if field is None:
                    continue
                key = by_widget.get(id(field))
                if key is not None:
                    self._index[key] = (section, field)

    def _apply_section_state(self, shown: Dict[int, int],
                             narrowing: bool) -> None:
        """Hide emptied sections; open the surviving ones while narrowing.

        A filter that leaves every section collapsed has told the user how
        many settings match and then hidden all of them, which is worse than
        not filtering. So a narrowing view expands what it kept — and
        remembers what was open beforehand, so releasing the filter restores
        the form the user had rather than one it invented.
        """
        if narrowing and self._restore_expanded is None:
            self._restore_expanded = {
                id(s): bool(s.is_expanded()) for s in self._sections
                if hasattr(s, "is_expanded")
            }
        for section in self._sections:
            count = shown.get(id(section), 0)
            visible = count > 0
            if not visible and not narrowing:
                # Not narrowing means nothing is filtered, and a section
                # with no rows was already invisible for its own reasons
                # (maturity). Leave that judgement alone.
                continue
            section.setVisible(visible)
            if not hasattr(section, "set_expanded"):
                continue
            if narrowing:
                if visible:
                    section.set_expanded(True)
            elif self._restore_expanded is not None:
                section.set_expanded(
                    self._restore_expanded.get(id(section), False))
        if not narrowing:
            self._restore_expanded = None
            # Hand maturity visibility back to the screen, which is the only
            # thing that knows why a section was hidden in the first place.
            refresh = getattr(self._screen, "refresh_maturity_visibility", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception:
                    LOG.debug("could not restore maturity visibility",
                              exc_info=True)

    def _compose_count(self, shown: int, total: int,
                       essentials: int) -> str:
        if shown == total:
            if self._level == ESSENTIALS and essentials:
                return f"Showing all {total} settings."
            return f"{total} settings."
        parts = [f"Showing {shown} of {total} settings"]
        if self._level == ESSENTIALS and essentials:
            parts.append(
                f"{total - essentials} more under All settings")
        if self._modified.isChecked():
            parts.append("modified only")
        if shown == 0:
            return ("No setting matches. Clear the search box, or switch to "
                    "All settings.")
        return " — ".join(parts) + "."


# ---------------------------------------------------------------------------
# Row visibility
# ---------------------------------------------------------------------------
#
# `Section` builds its rows with `QFormLayout.addRow`, and the label side is
# a wrapper widget it builds itself and does not hand back. `setRowVisible`
# keyed on the FIELD widget therefore reaches both halves, which nothing
# outside the section can do by hand.

def _form_of(section: QWidget) -> Optional[QFormLayout]:
    form = getattr(section, "_form", None)
    if isinstance(form, QFormLayout):
        return form
    return section.findChild(QFormLayout)


def _set_row_visible(section: QWidget, field: QWidget, visible: bool) -> None:
    form = _form_of(section)
    if form is None:
        field.setVisible(visible)
        return
    try:
        form.setRowVisible(field, visible)
    except (AttributeError, RuntimeError):
        # Qt < 6.4 has no setRowVisible. Hiding the field alone leaves an
        # orphaned label, but a stranded label is a far smaller problem than
        # a settings panel that will not draw.
        field.setVisible(visible)


def _row_is_visible(section: QWidget, field: QWidget) -> bool:
    form = _form_of(section)
    if form is None:
        return field.isVisible()
    try:
        return bool(form.isRowVisible(field))
    except (AttributeError, RuntimeError):
        return field.isVisible()


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install(screen: QWidget) -> Optional[SettingsSearchBar]:
    """Put a search strip above ``screen``'s settings form.

    The form is a ``QScrollArea`` sitting directly in the screen's splitter.
    The strip goes *outside* the scroll area, in a container that takes its
    place: a search box that scrolls away with the results it is filtering is
    a search box you have to scroll back up to reach.

    Returns the strip, or ``None`` when the screen has no settings form
    (a bespoke screen), the form failed to build, or one is already
    installed. Never raises — a missing search box must not cost anyone a
    module.

    :param screen: an ``AppScreen``.
    """
    existing = getattr(screen, "_settings_search", None)
    if existing is not None:
        return existing
    scroll = getattr(screen, "_settings_scroll", None)
    model = getattr(screen, "_settings_model", None)
    sections = getattr(screen, "_settings_sections", None)
    if not isinstance(scroll, QScrollArea) or model is None or not sections:
        return None
    parent = scroll.parentWidget()
    if not isinstance(parent, QSplitter):
        return None
    try:
        index = parent.indexOf(scroll)
        sizes = list(parent.sizes())
        bar = SettingsSearchBar(screen)
        container = QWidget(parent)
        container.setObjectName(PANE_NAME)
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(bar)
        # addWidget re-parents the scroll area out of the splitter, which is
        # what frees the slot the container then takes.
        column.addWidget(scroll, 1)
        parent.insertWidget(index, container)
        # `setParent` hides a widget, and a hidden widget is one a layout
        # skips. Without these two the QVBoxLayout saw no visible children,
        # left the scroll area at the geometry it had as a splitter pane,
        # and centred the strip on top of the settings form -- both of them
        # drawing over each other at full pane height. Nothing that reads
        # form-row visibility notices, which is why it has a geometry test.
        container.show()
        scroll.show()
        bar.show()
        if len(sizes) == parent.count():
            parent.setSizes(sizes)
    except Exception:
        LOG.debug("could not install the settings search strip", exc_info=True)
        return None
    screen._settings_search = bar
    return bar


class _StackWatcher(QObject):
    """Installs the strip on each settings screen as it is first shown.

    A ``QObject`` parented to the window rather than a closure, so the
    connection dies with the window and the handler is a bound method — a
    lambda here would keep the window alive for as long as the stack lived.
    """

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self._window = window

    def on_current_changed(self, _index: int) -> None:
        """Install into whatever screen the stack just switched to."""
        self.install_current()

    def install_current(self) -> Optional[SettingsSearchBar]:
        """Install into the stack's current widget, if it has a form."""
        try:
            screen = self._window._stack.currentWidget()
        except Exception:
            return None
        if screen is None:
            return None
        return install(screen)


def install_window_hooks(window: QMainWindow) -> Optional[_StackWatcher]:
    """Follow ``window``'s screen stack, adding the strip to each module.

    Called once from :func:`spacr.qt.shortcuts.install`. Screens are built
    lazily on first navigation, so this cannot be a one-shot sweep; it
    connects to the stack and also installs into anything already built.

    :returns: the watcher, kept alive by the window, or ``None``.
    """
    stack = getattr(window, "_stack", None)
    if stack is None:
        return None
    if getattr(window, "_settings_search_watcher", None) is not None:
        return window._settings_search_watcher
    watcher = _StackWatcher(window)
    try:
        stack.currentChanged.connect(watcher.on_current_changed)
    except Exception:
        LOG.debug("could not follow the screen stack", exc_info=True)
        return None
    window._settings_search_watcher = watcher
    # The first module may already be on screen by the time hooks run.
    QTimer.singleShot(0, watcher.install_current)
    return watcher


def _bar_qss(palette: dict, opacity) -> str:
    """QSS for the strip, registered through the theme seam.

    The first four rules are the important ones and they all say the same
    thing: **paint nothing**.

    The strip is not a card. It is type and controls sitting on the page,
    the way the module masthead is, and what belongs behind it is the
    theme. But every widget here is *named*, and a named widget is exactly
    what :func:`spacr.qt.theme.clear_container_surfaces` leaves alone — it
    tags only anonymous ``QWidget`` scaffolding, on the reasonable
    assumption that a name means somebody styled it on purpose. Nobody had
    styled these, so they fell through to the blanket
    ``QWidget {{ background-color: bg }}``, and ``bg`` is the WINDOW
    colour: near-black, and not a surface, so no page-opacity setting can
    reach it.

    :data:`PANE_NAME` is the one that did the damage. It is the wrapper
    :func:`install` puts around the strip *and* the settings scroll area,
    so it spans the entire settings column — an opaque black rectangle
    behind the whole thing. Everything in front of it was translucent and
    correct, and every one of them still measured 0.000 at every position
    of the slider, because what showed through was the black pane rather
    than the page. That is the "the container is not subject to the
    opacity setting" report, and the categories inside it with it: neither
    was broken, both were composited onto a black rectangle.

    The Recipes button is the same fault and lives in
    :mod:`spacr.qt.recipes`, which styles it there.
    """
    from .theme import font_px, pane_surface
    surface = pane_surface("surface_alt", palette["theme"], opacity)
    return f"""
QWidget#{PANE_NAME}, QWidget#{BAR_NAME},
QLabel#{MODIFIED_NAME}Label, QCheckBox#{MODIFIED_NAME} {{
    background: transparent;
    border: none;
}}
QLineEdit#{INPUT_NAME} {{
    background: {surface};
    border: 1px solid {palette["border_soft"]};
    border-radius: 6px;
    padding: 4px 8px;
}}
QLineEdit#{INPUT_NAME}:focus {{
    border-color: {palette["accent"]};
}}
QLabel#{COUNT_NAME} {{
    color: {palette["fg_dim"]};
    font-size: {font_px(11)}px;
}}
QToolButton#{DISCLOSURE_NAME} {{
    background: transparent;
    color: {palette["fg_dim"]};
    border: 1px solid {palette["border_soft"]};
    border-radius: 6px;
    padding: 3px 10px;
}}
QToolButton#{DISCLOSURE_NAME}:checked {{
    color: {palette["fg"]};
    border-color: {palette["accent"]};
}}
"""


try:  # pragma: no cover - present in every real launch
    from .theme import register_widget_qss as _register_widget_qss
    _register_widget_qss(BAR_NAME, _bar_qss, replace=True)
except Exception:  # pragma: no cover
    LOG.debug("could not register the settings-search QSS", exc_info=True)
