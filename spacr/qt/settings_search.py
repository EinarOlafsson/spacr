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

from .i18n import tr
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

#: How long a row revealed from the Help search box stays outlined. Long
#: enough to find with the eye after the page has settled, short enough that
#: the form is not left permanently marked up.
_MARK_MS = 4000

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
    """Open spaCR's ``QSettings``.

    :returns: the settings store.
    """
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


def _localize(widget: QWidget, setter_name: str, property_name: str,
              source: str) -> None:
    """Apply ``source`` in the user's language, keeping the English behind it.

    The strip is built from ``stack.currentChanged``, which fires after the
    window has run its one language pass over the screen, so every caption
    here has to translate itself. That alone is not enough:
    :func:`spacr.qt.i18n.retranslate_widget_tree` reads a widget's English
    back out of two properties — the source it should translate, and the
    rendering it last put on screen. Text written straight out in the user's
    language leaves both unset, so the *next* language switch would translate
    a translation; text re-applied by a handler looks instead like live data
    the translator must leave alone. Writing both makes either safe.

    :param widget: the widget to caption.
    :param setter_name: the Qt setter, e.g. ``"setToolTip"``.
    :param property_name: the i18n source property that setter reads, e.g.
        ``"_spacr_i18n_tooltip"``.
    :param source: the English string, exactly as the catalog keys it.
    """
    rendered = tr(source)
    widget.setProperty(property_name, source)
    widget.setProperty(f"{property_name}_last_rendered", rendered)
    getattr(widget, setter_name)(rendered)



class SettingsSearchBar(QWidget):
    """Search box, Modified filter, Essentials/All switch, and a count line.

    Owns no settings state of its own — it reads the screen's
    ``SettingsWidgets`` model and shows or hides rows that already exist.
    Hiding rather than rebuilding is what keeps a half-typed value alive
    across a filter change, which a rebuild would silently discard.

    :param screen: the module screen to filter. The bar owns no settings
        state -- it reads that screen's `SettingsWidgets` model and shows or
        hides rows that already exist, which is what keeps a half-typed value
        alive across a filter change.
    :param parent: parent widget.
    """

    def __init__(self, screen: QWidget, parent: Optional[QWidget] = None):
        """Build the settings search strip above a module's form.

        Fixed height, explicitly: the strip is two rows tall and the scroll area
        under it wants everything else, so without a policy the two share the
        pane by stretch factor and the search box lands 800 pixels high on the
        first layout.

        The key-to-section index is built once from the rendered form, so
        filtering never has to guess which section a setting ended up in, and
        which sections the user had open is remembered -- clearing the box puts
        the form back rather than leaving it splayed.

        :param screen: the module screen whose settings this filters.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setObjectName(BAR_NAME)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._screen = screen
        self._app_key = str(getattr(screen, "app_key", "") or "")
        self._model = getattr(screen, "_settings_model", None)
        self._index: Dict[str, Tuple[QWidget, QWidget]] = {}
        self._sections: List[QWidget] = list(
            getattr(screen, "_settings_sections", []) or [])
        self._restore_expanded: Optional[Dict[int, bool]] = None
        self._level = disclosure_for(self._app_key)
        self._grid_section_counted: Optional[QWidget] = None
        self._sections_kept: Optional[set] = None

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
        _localize(self._input, "setPlaceholderText",
                  "_spacr_i18n_placeholder", "Search settings…")
        _localize(
            self._input, "setToolTip", "_spacr_i18n_tooltip",
            "Search every setting in this module by name, by label, or by "
            "what its description says it does.")
        _localize(self._input, "setAccessibleName",
                  "_spacr_i18n_accessible_name", "Search settings")
        self._input.textChanged.connect(self._on_query_changed)
        row.addWidget(self._input, 1)

        self._modified_label = QLabel(self)
        _localize(self._modified_label, "setText", "_spacr_i18n_text",
                  "Modified")
        self._modified_label.setObjectName(MODIFIED_NAME + "Label")
        row.addWidget(self._modified_label, 0)

        self._modified = Toggle(parent=self)
        self._modified.setObjectName(MODIFIED_NAME)
        _localize(
            self._modified, "setToolTip", "_spacr_i18n_tooltip",
            "Show only the settings that no longer hold this module's "
            "default value.")
        _localize(self._modified, "setAccessibleName",
                  "_spacr_i18n_accessible_name",
                  "Show modified settings only")
        self._modified.toggled.connect(self._on_modified_toggled)
        row.addWidget(self._modified, 0)

        self._disclosure = QToolButton(self)
        self._disclosure.setObjectName(DISCLOSURE_NAME)
        self._disclosure.setCheckable(True)
        self._disclosure.setCursor(Qt.PointingHandCursor)
        self._disclosure.setChecked(self._level == ALL)
        self._disclosure.toggled.connect(self._on_disclosure_toggled)
        row.addWidget(self._disclosure, 0)
        self._controls_row = row

        self._count = QLabel(self)
        self._count.setObjectName(COUNT_NAME)
        self._count.setWordWrap(True)
        column.addWidget(self._count)

        self._refresh_disclosure_text()
        self._build_index()
        self.apply()

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

    def _show_all_without_remembering(self) -> None:
        """Put every setting on the form without writing that to the store.

        The user's Essentials/All choice is a choice, and arriving from
        somewhere else is not the user making it again. Showing a row that
        Essentials hides needs the level raised on the form; it does not
        need that raise written to ``QSettings``, and writing it means a
        lookup permanently moves a module out of Essentials -- on Mask, 190
        rendered rows against a handful of essentials, so the common case.

        Blocking the toggle's own signal is what separates the two: the
        button, the level and the caption all move, and
        :meth:`_on_disclosure_toggled` -- which is the only caller of
        :func:`remember_disclosure` -- does not run. Clicking the button
        still remembers, because that is the user choosing.

        One direction only, deliberately: showing a hidden row is the one
        reason to move the level behind the user's back, and there is no
        reason to lower it behind their back at all.
        """
        blocked = self._disclosure.blockSignals(True)
        try:
            self._disclosure.setChecked(True)
        finally:
            self._disclosure.blockSignals(blocked)
        self._level = ALL
        self._refresh_disclosure_text()

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

    def section_of(self, key: str) -> Optional[QWidget]:
        """The section widget holding ``key``'s row, or ``None``.

        :param key: a setting key.
        :returns: the collapsible section, or ``None`` when this module does
            not render that setting.
        """
        row = self._index.get(str(key))
        return row[0] if row else None

    def reveal(self, key: str) -> bool:
        """Show one setting with every other category shut.

        WHAT INSTRUCTION 422 ASKS FOR, and it is deliberately NOT the search
        filter. Typing the key into the box above would hide every other
        setting as well, so a user who arrived from the Help search and then
        wanted to look at the neighbouring rows would first have to work out
        what had happened to the form. Revealing instead leaves the module
        whole and only decides which heading is open.

        Nothing is rebuilt and no value is read or written: the row was
        already on the form, and this shows its section and scrolls to it.
        That is what makes arriving here from a search safe for a half-typed
        value -- the same property the filter has, for the same reason.

        THE DISCLOSURE LEVEL IS CHANGED ONLY IF IT HAS TO BE, AND THE CHANGE
        IS NEVER REMEMBERED. Switching to All settings unconditionally would
        work, and it would also rewrite this module's remembered
        Essentials/All choice every time anybody arrived here -- a setting
        the user chose, changed as a side effect of looking something up. So
        the filter is cleared first and the level is raised only when the row
        is still not on the form afterwards, which is exactly the case where
        Essentials is what is hiding it; and the raise goes through
        :meth:`_show_all_without_remembering`, so the form shows the row
        while the store still holds the level the user picked. Most settings
        are not essentials, so a lookup that persisted the raise would move
        almost every module out of Essentials for good.

        :param key: the setting to reveal.
        :returns: True when the module renders ``key`` and it was revealed.
        """
        row = self._index.get(str(key))
        if row is None:
            return False
        section, field = row
        self._input.clear()
        self._modified.setChecked(False)
        self.apply()
        if not _row_is_visible(section, field) and self._level != ALL:
            self._show_all_without_remembering()
            self.apply()
        for other in self._sections:
            if not hasattr(other, "set_expanded"):
                continue
            try:
                other.set_expanded(other is section)
            except Exception:
                LOG.debug("could not collapse a section", exc_info=True)
        self._restore_expanded = None
        _set_row_visible(section, field, True)
        section.setVisible(True)
        self._revealed = str(key)
        self._mark(field)
        QTimer.singleShot(0, lambda: self._scroll_to(field))
        return True

    def revealed_key(self) -> str:
        """The setting :meth:`reveal` last showed, or ``""``."""
        return getattr(self, "_revealed", "")

    def _mark(self, field: QWidget) -> None:
        """Outline ``field`` for a few seconds so the eye finds it.

        A STATIC MARK, not a flash: anything that moves has to answer to the
        Animation preferences and to the reduced-motion equivalents, and a
        border that simply appears and then goes away needs neither and is
        not lost on anybody who turned motion off.

        The previous stylesheet is put back rather than cleared, so a field
        that carried one of its own -- a validation warning, say -- still
        carries it afterwards.

        :param field: the field widget to outline.
        """
        previous = field.styleSheet()
        field.setProperty("spacrRevealed", True)
        field.setStyleSheet(
            previous + "\nQWidget { border: 1px solid palette(highlight); }")

        def _unmark() -> None:
            """Put the field back the way it was found."""
            try:
                field.setProperty("spacrRevealed", False)
                field.setStyleSheet(previous)
            except RuntimeError:
                LOG.debug("the marked row went away before the mark did")

        QTimer.singleShot(_MARK_MS, _unmark)

    def _scroll_to(self, field: QWidget) -> None:
        """Bring ``field`` into view and put the caret in it.

        Deferred by one event-loop turn from :meth:`reveal`, because a
        section that has just been expanded has no geometry yet and
        ``ensureWidgetVisible`` on a widget with none scrolls to the top of
        the form -- which looks exactly like the failure this is here to
        prevent.

        :param field: the field widget to show.
        """
        try:
            scroll = self._screen.findChild(QScrollArea)
            if scroll is not None:
                scroll.ensureWidgetVisible(field, 0, 40)
            field.setFocus(Qt.ShortcutFocusReason)
        except RuntimeError:
            LOG.debug("the row went away before it could be shown")

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

    def apply(self, reopen: bool = True) -> None:
        """Recompute which rows and sections are shown.

        Called on every change to the query, the Modified filter or the
        disclosure level — one path, so the three can never disagree about
        what should be on screen. The screen also calls it after each pass of
        the object rule, so a channel the user commits is judged by the same
        filter as every other row.

        The per-object table has no form rows of its own to count. Its
        section is counted by the settings it answers for instead, so under
        Essentials the table stays on screen while it holds the channels,
        which the flat form no longer shows while the table is on.

        A heading that holds only sub-headings has no rows of its own
        either, so its matches are rolled up out of the headings below it
        before any section is hidden — see
        :meth:`_counting_the_sub_headings`.

        :param reopen: while the filter narrows, open every section it
            keeps. The screen passes ``False`` when it re-applies the filter
            after the object rule or after laying out rows: a section the
            user shut then stays shut, and only a section this call brings
            back onto the form is opened.
        """
        model = self._model
        if model is None or not self._index:
            self._count.setText("")
            return

        total = len(self._index)
        hidden: set = set()
        hidden_by_run = getattr(model, "keys_hidden_by_the_run", None)
        if callable(hidden_by_run):
            try:
                hidden = set(hidden_by_run())
            except Exception:                                # noqa: BLE001
                hidden = set()
        by_grid = set(getattr(model, "_hidden_by_the_grid", ()) or ())
        lacking = getattr(model, "_hidden_by_their_object", None)
        lacking = set(lacking) if lacking is not None else hidden - by_grid
        grid_section, grid_keys = self._grid_section()

        query = self._input.text().strip()
        matching: Optional[set] = None
        if query:
            try:
                matching = set(model.keys_matching(query))
            except Exception:
                LOG.debug("settings search failed for %r", query, exc_info=True)

        modified: Optional[set] = None
        if self._modified.isChecked():
            try:
                modified = set(model.modified_keys())
            except Exception:
                LOG.debug("modified-only filter failed", exc_info=True)

        essentials: List[str] = []
        essential_set: set = set()
        if self._level == ESSENTIALS:
            try:
                essential_set = set(model.essential_keys())
            except Exception:
                LOG.debug("essential keys unavailable", exc_info=True)
            essentials = [k for k in self._index if k in essential_set]

        def narrowed(keys: set) -> set:
            """``keys`` less whatever the query, Modified and level exclude."""
            out = set(keys)
            if matching is not None:
                out &= matching
            if modified is not None:
                out &= modified
            if essentials:
                out &= essential_set
            return out

        wanted = narrowed(set(self._index) - hidden)
        in_the_grid = (narrowed(set(grid_keys) - lacking)
                       if grid_section is not None else set())

        for key, (section, field) in self._index.items():
            _set_row_visible(section, field, key in wanted)

        shown_per_section: Dict[int, int] = {}
        for key, (section, _field) in self._index.items():
            if key in wanted:
                shown_per_section[id(section)] = (
                    shown_per_section.get(id(section), 0) + 1)
        if grid_section is not None:
            shown_per_section[id(grid_section)] = len(in_the_grid)

        narrowing = bool(query) or self._modified.isChecked() \
            or (self._level == ESSENTIALS and bool(essentials))
        self._apply_section_state(
            self._counting_the_sub_headings(shown_per_section),
            narrowing, reopen)
        self._count.setText(
            self._compose_count(len(wanted), total, len(essentials)))

    def _counting_the_sub_headings(
            self, shown: Dict[int, int]) -> Dict[int, int]:
        """Add what each heading's sub-headings keep to the heading's count.

        A heading that owns no form rows -- ``Advanced settings``, and the
        object families nested under it -- counts zero however many rows
        match below it, and :meth:`_apply_section_state` hides whatever
        counts zero while the view narrows. That took the matches off screen
        with the umbrella: on a built Mask screen under All settings,
        searching ``remove border objects`` reported one match and left
        ``cell_remove_border_objects`` visible on the ``Cell`` form, while
        ``Object Filtration (all objects)`` and ``Advanced settings`` above
        it were both hidden, so the match the count line promised was
        nowhere.

        Counted upwards rather than down: every heading this strip decides is
        already in ``_sections``, and its ancestors are read off the widget
        tree, so a heading nested at any depth reaches each umbrella above it
        without a second description of the layout to keep in step.

        :param shown: how many rows each section keeps, by ``id()``.
        :returns: a new mapping — each heading's own count plus every count
            below it. The count line is composed from the matching keys and
            is not affected, so a rolled-up heading adds nothing to it.
        """
        known = {id(section): section for section in self._sections}
        rolled = dict(shown)
        for section in self._sections:
            count = shown.get(id(section), 0)
            if not count:
                continue
            reached = {id(section)}
            try:
                node = section.parentWidget()
            except RuntimeError:
                continue
            while node is not None:
                marker = id(node)
                if marker in known and marker not in reached:
                    rolled[marker] = rolled.get(marker, 0) + count
                    reached.add(marker)
                try:
                    node = node.parentWidget()
                except RuntimeError:
                    break
        return rolled

    def _grid_section(self) -> Tuple[Optional[QWidget], frozenset]:
        """The per-object table's section and the settings it answers for.

        The table can be mounted or taken down by Preferences after this
        strip was built, so the section is looked up on every call and the
        list of sections this strip decides is kept in step with it: a
        section that was taken down is dropped before it can be touched.

        :returns: ``(section, keys)``, or ``(None, frozenset())`` when the
            screen shows no table.
        """
        screen = self._screen
        grid = getattr(screen, "_object_grid", None)
        binding = getattr(screen, "_object_grid_binding", None)
        section: Optional[QWidget] = None
        keys: frozenset = frozenset()
        if grid is not None and binding is not None:
            try:
                node = grid.parentWidget()
                while node is not None and not hasattr(node, "add_prose_row"):
                    node = node.parentWidget()
                section = node
                keys = (frozenset(binding.owned_keys()) if node is not None
                        else frozenset())
            except RuntimeError:
                section, keys = None, frozenset()
        previous = self._grid_section_counted
        if previous is not section:
            self._sections = [s for s in self._sections if s is not previous]
            if section is not None and not any(
                    s is section for s in self._sections):
                self._sections.append(section)
            self._grid_section_counted = section
        return section, keys

    def _on_query_changed(self, _text: str) -> None:
        """Re-apply the filter after the search text changed.

        :param _text: the new text; re-read from the box, so it is not used.
        """
        self.apply()

    def _on_modified_toggled(self, _on: bool) -> None:
        """Re-apply the filter after the modified-only switch changed.

        :param _on: the switch's new state; re-read, so it is not used.
        """
        self.apply()

    def _on_disclosure_toggled(self, on: bool) -> None:
        """Switch between essential and all settings, and remember the choice.

        :param on: ``True`` for all settings, ``False`` for the essentials.
        """
        self._level = ALL if on else ESSENTIALS
        remember_disclosure(self._app_key, self._level)
        self._refresh_disclosure_text()
        self.apply()

    def _refresh_disclosure_text(self) -> None:
        """Caption the switch for the level it is now on.

        Re-applied on every level change, which is why the captions go
        through :func:`_localize`: a raw English literal here would put the
        button back into English the first time somebody switched between
        Essentials and All, undoing an otherwise successful language pass.
        """
        if self._level == ALL:
            caption = "All settings"
            hint = "Showing every setting. Click for the essentials only."
        else:
            caption = "Essentials"
            hint = ("Showing the settings this module cannot run without. "
                    "Click for all of them.")
        _localize(self._disclosure, "setText", "_spacr_i18n_text", caption)
        _localize(self._disclosure, "setToolTip", "_spacr_i18n_tooltip", hint)
        _localize(self._disclosure, "setAccessibleName",
                  "_spacr_i18n_accessible_name", caption)

    def _build_index(self) -> None:
        """Map each setting key to the section and field widget showing it.

        Built from the model's own ``key -> widget`` map and the sections the
        screen kept, rather than by re-deriving the layout: the screen has
        already decided which key went where, and a second opinion here would
        be a second thing to keep in sync.

        A SUB-HEADING IS A ROW OF ITS PARENT'S FORM and is skipped here. A
        nested :class:`~spacr.qt.widgets.section.Section` is added with
        ``add_prose``, which spans the form, and PySide hands a spanning
        widget back for the field role — so the search below it walked into
        the sub-heading and claimed the first setting it found there for the
        parent. Measured on Mask and on Timelapse: six keys each,
        ``cell_min_area`` among them, recorded against a heading two levels
        above the form that draws them, because the sections are indexed
        deepest first and the parent's pass overwrote the right answer. The
        row it recorded was the sub-heading itself, so hiding that "row"
        hid the whole sub-heading and everything under it.
        """
        widgets = getattr(self._model, "_widgets", {}) or {}
        by_widget = {id(w): key for key, w in widgets.items()}
        headings = {id(section) for section in self._sections}
        for section in self._sections:
            form = _form_of(section)
            if form is None:
                continue
            for i in range(form.rowCount()):
                item = form.itemAt(i, QFormLayout.FieldRole)
                field = item.widget() if item is not None else None
                if field is None or id(field) in headings:
                    continue
                key = by_widget.get(id(field))
                if key is None:
                    for child in field.findChildren(QWidget):
                        key = by_widget.get(id(child))
                        if key is not None:
                            break
                if key is not None:
                    self._index[key] = (section, field)

    def _apply_section_state(self, shown: Dict[int, int],
                             narrowing: bool, reopen: bool = True) -> None:
        """Hide emptied sections; open the surviving ones while narrowing.

        A filter that leaves every section collapsed has told the user how
        many settings match and then hidden all of them, which is worse than
        not filtering. So a narrowing view expands what it kept — and
        remembers what was open beforehand, so releasing the filter restores
        the form the user had rather than one it invented.

        :param shown: how many rows each section keeps, by ``id()``.
        :param narrowing: whether a query, Modified or Essentials narrows.
        :param reopen: open every kept section; ``False`` opens only the
            sections the previous call did not keep.
        """
        if narrowing and self._restore_expanded is None:
            self._restore_expanded = {
                id(s): bool(s.is_expanded()) for s in self._sections
                if hasattr(s, "is_expanded")
            }
        kept_before = self._sections_kept
        self._sections_kept = {
            id(s) for s in self._sections if shown.get(id(s), 0) > 0}
        for section in self._sections:
            count = shown.get(id(section), 0)
            visible = count > 0
            if not visible and not narrowing:
                continue
            section.setVisible(visible)
            if not hasattr(section, "set_expanded"):
                continue
            if narrowing:
                if visible and (reopen or kept_before is None
                                or id(section) not in kept_before):
                    section.set_expanded(True)
            elif self._restore_expanded is not None:
                section.set_expanded(
                    self._restore_expanded.get(id(section), False))
        if not narrowing:
            self._restore_expanded = None
            refresh = getattr(self._screen, "refresh_maturity_visibility", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception:
                    LOG.debug("could not restore maturity visibility",
                              exc_info=True)

    def _compose_count(self, shown: int, total: int,
                       essentials: int) -> str:
        """Build the line under the form saying how much of it is showing.

        Composed from translated parts rather than assembled and then looked up:
        the catalogue is keyed on the sentence with its numbers as placeholders,
        so an f-string built first matches nothing -- and this line sits under
        every settings panel in the program.

        :param shown: settings currently visible.
        :param total: settings this module has.
        :param essentials: how many are marked essential.
        :returns: the line, ending in a full stop.
        """
        if shown == total:
            if self._level == ESSENTIALS and essentials:
                return tr("Showing all {total} settings.", total=total)
            return tr("{total} settings.", total=total)
        parts = [tr("Showing {shown} of {total} settings",
                    shown=shown, total=total)]
        if self._level == ESSENTIALS and essentials:
            parts.append(tr("{n} more under All settings",
                            n=total - essentials))
        if self._modified.isChecked():
            parts.append(tr("modified only"))
        if shown == 0:
            return tr("No setting matches. Clear the search box, or switch "
                      "to All settings.")
        return " — ".join(parts) + "."



def _form_of(section: QWidget) -> Optional[QFormLayout]:
    """Find the form layout a settings section lays its rows out with.

    :param section: the section.
    :returns: the layout, found by attribute first and by search second, or
        ``None`` when the section has none.
    """
    form = getattr(section, "_form", None)
    if isinstance(form, QFormLayout):
        return form
    return section.findChild(QFormLayout)


def _set_row_visible(section: QWidget, field: QWidget, visible: bool) -> None:
    """Show or hide a settings row, label and all.

    Qt before 6.4 has no ``setRowVisible``; there the field alone is hidden,
    which leaves an orphaned label -- a far smaller problem than a settings
    panel that will not draw.

    :param section: the section holding the row.
    :param field: the row's field widget.
    :param visible: whether to show it.
    """
    form = _form_of(section)
    if form is None:
        field.setVisible(visible)
        return
    try:
        form.setRowVisible(field, visible)
    except (AttributeError, RuntimeError):
        field.setVisible(visible)


def _row_is_visible(section: QWidget, field: QWidget) -> bool:
    """Report whether a settings row is showing.

    :param section: the section holding the row.
    :param field: the row's field widget.
    :returns: the row's visibility, falling back to the field's own on a Qt
        that cannot answer for the row.
    """
    form = _form_of(section)
    if form is None:
        return field.isVisible()
    try:
        return bool(form.isRowVisible(field))
    except (AttributeError, RuntimeError):
        return field.isVisible()



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
        # THE SIZES ARE ONLY WORTH KEEPING ONCE THERE ARE SOME. A splitter
        # that has never been laid out answers `sizes()` with pre-layout
        # defaults, so restoring them after the insert would WRITE those
        # defaults over the layout the first show is about to compute --
        # this file's own WATCH list warns about reading geometry before
        # the layout settles. Installing before the screen is shown is the
        # point of doing it early (item 380), so the unlaid case is the
        # common one now rather than the exception.
        sizes = list(parent.sizes()) if scroll.isVisible() else []
        bar = SettingsSearchBar(screen)
        container = QWidget()
        container.setObjectName(PANE_NAME)
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(bar)
        column.addWidget(scroll, 1)
        parent.insertWidget(index, container)
        container.show()
        scroll.show()
        bar.show()
        if sizes and len(sizes) == parent.count():
            parent.setSizes(sizes)
    except Exception:
        LOG.debug("could not install the settings search strip", exc_info=True)
        return None
    screen._settings_search = bar
    try:
        from .i18n import retranslate_widget_tree
        retranslate_widget_tree(bar)
    except Exception:
        LOG.debug("could not translate the settings search strip",
                  exc_info=True)
    return bar


class _StackWatcher(QObject):
    """Installs the strip on each settings screen as it is first shown.

    A ``QObject`` parented to the window rather than a closure, so the
    connection dies with the window and the handler is a bound method — a
    lambda here would keep the window alive for as long as the stack lived.
    """

    def __init__(self, window: QMainWindow):
        """Watch a window's stack and install into each screen as it is shown.

        :param window: the main window. Its stack is read at install time,
            not here, so this works for screens created after the watcher --
            and it is the QObject PARENT, so a currentChanged arriving during
            teardown cannot reach a watcher holding a deleted stack.
        """
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
    from .theme import block_surface, font_px
    surface = block_surface("surface_alt", palette["theme"], opacity)
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


try:
    from .theme import register_widget_qss as _register_widget_qss
    _register_widget_qss(BAR_NAME, _bar_qss, replace=True)
except Exception:
    LOG.debug("could not register the settings-search QSS", exc_info=True)
