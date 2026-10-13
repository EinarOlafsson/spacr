"""
Command palette — Ctrl-K searchable action list.

Every navigable app, every preference toggle, every menu action gets
registered as a :class:`Command` and the palette lets users jump to
it by typing a few characters. Modelled on VS Code / Slack / Linear.

Public API::

    from spacr.qt.command_palette import CommandPalette
    CommandPalette(window).exec()

The window is inspected on show — no need to preregister anything.
Commands are:

* every entry in :data:`spacr.qt.app.APPS`
* every recent run journal entry (jump straight into its app + load
  the settings via ``AppScreen.apply_settings_dict``)
* every action in the menu bar
* an "Open Preferences…" shortcut
* an "Open Providers…" shortcut

Matching is a case-insensitive substring search over each command's
label and keywords; there is no fuzzy or ranked matching.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from PySide6.QtCore import (
    QAbstractListModel, QItemSelectionModel, QModelIndex, QSignalBlocker, Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QKeyEvent
from PySide6.QtWidgets import (
    QDialog, QLineEdit, QListView, QMainWindow, QVBoxLayout,
)
from shiboken6 import isValid

LOG = logging.getLogger("spacr.qt.command_palette")


@dataclass
class Command:
    """One entry in the palette.

    :ivar label: human-readable text shown in the list.
    :ivar section: category badge (``"Apps"`` / ``"Recent"`` / …).
    :ivar action: callable invoked when the user hits Enter.
    :ivar keywords: extra strings the fuzzy match will search
        alongside the label (e.g. "mask cellpose" so typing "cellpose"
        finds the Mask app).
    """
    label:    str
    section:  str
    action:   Callable[[], None]
    keywords: List[str] = field(default_factory=list)


class _Row:
    """One row of the palette: a section heading or a command.

    Read back the way a list-widget item is, so callers can ask a row for its
    text, its flags and the command it runs.

    :param text: the text the row shows.
    :param command: the command the row runs; ``None`` for a heading.
    """

    __slots__ = ("_text", "_command")

    def __init__(self, text: str, command: Optional[Command] = None):
        """Hold the row's text and command.

        :param text: the text the row shows.
        :param command: the command the row runs; ``None`` for a heading.
        """
        self._text = text
        self._command = command

    def text(self) -> str:
        """The text the row shows."""
        return self._text

    def flags(self):
        """``NoItemFlags`` for a heading; enabled and selectable otherwise."""
        if self._command is None:
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def data(self, role: int = Qt.UserRole):
        """The command for ``Qt.UserRole``, the text for the display role.

        :param role: the item data role asked for.
        """
        if role == Qt.UserRole:
            return self._command
        if role == Qt.DisplayRole:
            return self._text
        return None


class _RowModel(QAbstractListModel):
    """A fixed list of palette rows.

    A model is never changed once a view shows it. A new filter builds a new
    model and the view is pointed at it in one step, so nothing a view is
    showing is ever cleared or deleted while the view still reads it, and no
    row is a C++ object that Python has to keep track of.

    :param rows: the rows, in display order.
    :param parent: the view the model belongs to.
    """

    def __init__(self, rows, parent=None):
        """Hold ``rows``; they never change afterwards.

        :param rows: the rows, in display order.
        :param parent: the view the model belongs to.
        """
        super().__init__(parent)
        self._rows = tuple(rows)
        self._heading = QBrush(Qt.gray)

    def rowCount(self, parent=QModelIndex()) -> int:
        """How many rows there are; a row has no children.

        :param parent: the parent index; only the root has rows.
        """
        return 0 if parent.isValid() else len(self._rows)

    def row(self, index: int) -> Optional[_Row]:
        """The row at ``index``, or ``None`` past either end.

        :param index: the row number.
        """
        if 0 <= index < len(self._rows):
            return self._rows[index]
        return None

    def data(self, index, role=Qt.DisplayRole):
        """The text of a row, and grey for a heading.

        :param index: the model index asked about.
        :param role: the item data role asked for.
        """
        row = self.row(index.row()) if index.isValid() else None
        if row is None:
            return None
        if role == Qt.DisplayRole:
            return row.text()
        if role == Qt.ForegroundRole and row.data() is None:
            return self._heading
        return None

    def flags(self, index):
        """The row's flags: a heading cannot be selected.

        :param index: the model index asked about.
        """
        row = self.row(index.row()) if index.isValid() else None
        return Qt.NoItemFlags if row is None else row.flags()


class _RowList(QListView):
    """The palette's result list.

    A list view that answers the way the list widget it replaced did --
    ``count``, ``item``, ``currentRow``, ``setCurrentRow``, ``currentItem``,
    ``currentRowChanged`` and ``itemActivated`` -- and whose rows are only
    ever replaced whole by :meth:`show_rows`.
    """

    currentRowChanged = Signal(int)
    itemActivated = Signal(object)

    def __init__(self, parent=None):
        """Start empty.

        :param parent: the widget the list sits in.
        """
        super().__init__(parent)
        self.setModel(_RowModel((), self))
        self._watch_current()
        self.activated.connect(self._on_activated)

    def show_rows(self, rows) -> None:
        """Show ``rows`` in place of every row shown now.

        The rows go into a new model that no view has seen, and the view is
        pointed at it in one call. The old model and its selection are
        released only after the view has let go of them.

        :param rows: the rows, in display order.
        """
        model = _RowModel(rows, self)
        old_model, old_selection = self.model(), self.selectionModel()
        self.setModel(model)
        self._watch_current()
        old_selection.deleteLater()
        old_model.deleteLater()

    def _watch_current(self) -> None:
        """Report the current row of the selection the view has now."""
        self.selectionModel().currentRowChanged.connect(self._on_current)

    def _on_current(self, current, _previous) -> None:
        """Pass a new current row on as ``currentRowChanged``.

        :param current: the new current index.
        :param _previous: the index that was current; unused.
        """
        self.currentRowChanged.emit(current.row())

    def _on_activated(self, index) -> None:
        """Pass an activated row on as ``itemActivated``.

        :param index: the index activated.
        """
        self.itemActivated.emit(self.item(index.row()))

    def count(self) -> int:
        """How many rows the list shows."""
        return self.model().rowCount()

    def item(self, index: int) -> Optional[_Row]:
        """The row at ``index``, or ``None`` past either end.

        :param index: the row number.
        """
        return self.model().row(index)

    def currentRow(self) -> int:
        """The current row's number, or -1 when there is none."""
        return self.currentIndex().row()

    def currentItem(self) -> Optional[_Row]:
        """The current row, or ``None`` when there is none."""
        return self.item(self.currentRow())

    def setCurrentRow(self, index: int) -> None:
        """Make row ``index`` current and selected, as a list widget does.

        A heading can be made current this way, as it can on a list widget;
        the arrow keys and the first selection never put it there.

        :param index: the row number.
        """
        self.selectionModel().setCurrentIndex(
            self.model().index(index, 0),
            QItemSelectionModel.ClearAndSelect)

    def setCurrentItem(self, row: _Row) -> None:
        """Make ``row`` current; a row the list does not show is ignored.

        :param row: a row the list shows.
        """
        model = self.model()
        for index in range(model.rowCount()):
            if model.row(index) is row:
                self.setCurrentRow(index)
                return


class CommandPalette(QDialog):
    """Modal dialog with a live-filtering command list.

    :param window: the MainWindow the palette should operate on.
    """

    def __init__(self, window: QMainWindow):
        """Build the command palette over the main window.

        :param window: the window every command acts on.
        """
        from .i18n import tr
        super().__init__(window)
        self._window = window
        self.setWindowTitle(tr("spaCR — Command palette"))
        self.setModal(True)
        from .preferences import scaled_px
        
        self.setMinimumWidth(scaled_px(560))
        self.setMinimumHeight(420)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._input = QLineEdit()
        self._input.setPlaceholderText(tr(
            "Type to filter — Enter to run, Esc to cancel"))
        self._input.setObjectName("CommandInput")
        from .theme import font_px
        self._input.setStyleSheet(
            "QLineEdit#CommandInput {"
            "  border: none;"
            "  padding: 14px 18px;"
            "  font-family: 'Open Sans', sans-serif;"
            f"  font-size: {font_px(15)}px;"
            "}"
        )
        outer.addWidget(self._input)

        self._list = _RowList()
        self._list.setStyleSheet(
            "QListView { border: none; padding: 6px 0; }"
            "QListView::item { padding: 8px 18px; }"
        )
        outer.addWidget(self._list, 1)

        self._commands: List[Command] = []
        self._rendering = False
        self._pending: Optional[List[Command]] = None
        self._closed = False
        self._collect_commands()
        self._render(self._commands)

        self._input.textChanged.connect(self._on_filter)
        self._input.returnPressed.connect(self._on_activate)
        self.finished.connect(self._stop_rendering)
        self._list.itemActivated.connect(lambda _i: self._on_activate())

    def _collect_commands(self) -> None:
        """Gather everything the palette can run: modules, recent runs and actions.

        The app registry is imported here rather than at module level and its
        failure is tolerated: with no registry the app loop simply does not run,
        and only the visibility check needs a stand-in.
        """
        from .i18n import tr
        try:
            from .app import app_is_visible, app_stage, visible_apps
            apps = visible_apps()
        except Exception:
            apps = []

            def app_is_visible(_key):
                """Fallback that treats every app as visible."""
                return True

        for key, name, desc, section in apps:
            localized_name = tr(name)
            localized_section = tr(section)
            words = [
                key, desc, section, name.lower(), localized_name.lower(),
                localized_section.lower(), app_stage(key),
            ]
            self._commands.append(Command(
                label=tr("Go to  {name}", name=localized_name),
                section=tr("Apps · {section}", section=localized_section),
                action=lambda k=key: self._nav(k),
                keywords=words,
            ))

        self._commands.append(Command(
            label=tr("Go to  {name}", name=tr("Home")),
            section=tr("Navigation"),
            action=lambda: self._nav("__home__"),
            keywords=["home", "start", "landing"],
        ))

        self._commands.append(Command(
            label=tr("Open Preferences…"),
            section=tr("Actions"),
            action=self._open_preferences,
            keywords=["preferences", "settings", "theme", "font",
                      "colour", "color", "accessibility"],
        ))

        self._commands.append(Command(
            label=tr("Open AI Providers…"),
            section=tr("Actions"),
            action=self._open_providers,
            keywords=["providers", "ai", "claude", "chatgpt",
                      "gemini", "llm"],
        ))

        self._commands.append(Command(
            label=tr("Keyboard shortcuts…"),
            section=tr("Help"),
            action=self._open_shortcuts,
            keywords=["shortcuts", "keyboard", "help", "cheat",
                      "hotkeys"],
        ))

        try:
            from ..run_journal import recent_runs
            for r in recent_runs(limit=8):
                if not app_is_visible(str(r.get("app_key", ""))):
                    continue
                dur = f"{r.get('elapsed_s', 0) or 0:.1f}s"
                status = r.get("status", "?")
                dir_name = r["dir"].name
                self._commands.append(Command(
                    label=f"Recent · {r['app_key']}  ({status}, {dur})",
                    section="Recent runs",
                    action=lambda run=r: self._open_run(run),
                    keywords=[r["app_key"], dir_name, status,
                              "recent", "run", "journal"],
                ))
        except Exception as e:
            LOG.debug("recent_runs unavailable: %s", e)

        self._collect_settings_commands()

        try:
            from PySide6.QtWidgets import QMenu
            bar = self._window.menuBar()
            for menu in (bar.findChildren(QMenu) if bar is not None else []):
                menu_title = menu.title().replace("&", "")
                if not menu_title:
                    continue
                for act in menu.actions():
                    if act.isSeparator() or act.menu() is not None:
                        continue
                    label = act.text().replace("&", "")
                    if not label:
                        continue
                    self._commands.append(Command(
                        label=f"{menu_title} → {label}",
                        section="Menu",
                        action=lambda a=act: a.trigger(),
                        keywords=[label.lower(), menu_title.lower()],
                    ))
        except Exception:
            LOG.debug("menu actions unavailable for the palette",
                      exc_info=True)

    def _collect_settings_commands(self, limit: int = 400) -> None:
        """One command per setting of the module currently on screen.

        Activating it opens that setting: the module's search strip is
        filtered to the key and the section holding it is expanded, so the
        palette lands the user *on* the control rather than merely naming
        it.

        Scoped to the current module on purpose. Every setting of every
        module is 1,022 rows, and a palette in which "diameter" returns
        eleven identically-named entries from six modules is a worse answer
        than no entry at all.

        :param limit: safety cap, so an unusually large module cannot make
            opening the palette feel slow.
        """
        try:
            screen = self._window._stack.currentWidget()
        except Exception:
            return
        model = getattr(screen, "_settings_model", None)
        widgets = getattr(model, "_widgets", None) if model is not None else None
        if not widgets:
            return
        app_key = str(getattr(screen, "app_key", "") or "")
        section = f"Settings · {app_key}" if app_key else "Settings"
        for key in list(widgets)[:limit]:
            try:
                label = model._label_for(key)
                hint = model.plain_tooltip_for(key)
            except Exception:
                label, hint = key, ""
            self._commands.append(Command(
                label=f"{label}  ({key})",
                section=section,
                action=lambda k=key: self._reveal_setting(k),
                keywords=[key.lower(), label.lower(), hint.lower(),
                          "setting", app_key.lower()],
            ))

    def _reveal_setting(self, key: str) -> None:
        """Filter the current module's settings strip down to ``key``.

        Falls back to expanding the section and focusing the widget when the
        strip is not installed, so the command still lands somewhere useful
        on a screen the search bar could not reach.
        """
        try:
            screen = self._window._stack.currentWidget()
        except Exception:
            return
        bar = getattr(screen, "_settings_search", None)
        if bar is not None:
            try:
                bar.set_modified_only(False)
                bar.set_level("all")
                bar.set_query(key)
            except Exception:
                LOG.debug("could not reveal %r through the strip", key,
                          exc_info=True)
        opener = getattr(screen, "_open_the_heading_of", None)
        if callable(opener):
            try:
                opener(key)
            except Exception:
                LOG.debug("could not open the category of %r", key,
                          exc_info=True)
        widget = (getattr(screen, "_settings_model", None)
                  and screen._settings_model._widgets.get(key))
        if widget is None:
            return
        for section in getattr(screen, "_settings_sections", []) or []:
            try:
                holds = getattr(section, "_holds", None)
                if (holds(widget) if callable(holds)
                        else section.isAncestorOf(widget)):
                    section.set_expanded(True)
                    break
            except (AttributeError, RuntimeError):
                continue
        try:
            widget.setFocus()
        except Exception:
            pass

    def _render(self, cmds: List[Command]) -> None:
        """Show ``cmds``, grouped under section headings.

        The headings are not selectable, and the first selection skips past
        the first one -- otherwise Return on a freshly opened palette would
        activate a heading. Every command follows a heading, so the first
        command is always the second row.

        A render never runs inside another one. A render asked for while one
        is running only records its commands, and the running render shows
        them once it has finished. Text typed while the rows are being built
        reaches no filter, because the input is silent then; the running
        render sees that the text changed and shows what it now selects. A
        palette that has closed renders nothing.

        :param cmds: the commands to show, in section order.
        """
        if self._closed:
            return
        if self._rendering:
            self._pending = list(cmds)
            return
        self._rendering = True
        try:
            while True:
                typed = self._input.text()
                self._fill(cmds)
                if self._closed:
                    break
                if self._input.text() != typed:
                    self._pending = self._matching(self._input.text())
                if self._pending is None:
                    break
                cmds, self._pending = self._pending, None
        finally:
            self._pending = None
            self._rendering = False

    def _rows_for(self, cmds: List[Command]) -> List[_Row]:
        """The rows that show ``cmds``: a heading before each new section.

        Plain Python objects; building them touches no widget.

        :param cmds: the commands to show, in section order.
        """
        rows: List[_Row] = []
        current_section = None
        for cmd in cmds:
            if cmd.section != current_section:
                rows.append(_Row(cmd.section.upper()))
                current_section = cmd.section
            rows.append(_Row(cmd.label, cmd))
        return rows

    def _fill(self, cmds: List[Command]) -> None:
        """Replace the rows with the rows for ``cmds``, in one step.

        The rows are built first, off screen, as plain Python objects. The
        input and the list are silent while that happens and while the list
        takes the new rows, so a key typed meanwhile only changes the text and
        cannot start a filter. A palette destroyed while the rows were being
        built counts as closed and touches nothing it held.

        :param cmds: the commands to show, in section order.
        """
        quiet = QSignalBlocker(self._input)
        try:
            rows = self._rows_for(cmds)
            if not isValid(self._list):
                self._closed = True
                return
            hushed = QSignalBlocker(self._list)
            try:
                self._list.show_rows(rows)
                if self._list.count() > 1:
                    self._list.setCurrentRow(1)
            finally:
                hushed.unblock()
        finally:
            if isValid(self._input):
                quiet.unblock()
            else:
                quiet.dismiss()

    def _matching(self, needle: str) -> List[Command]:
        """The commands whose label or keywords hold ``needle``.

        Matched case-insensitively, so a command can be found by what it does
        as well as by what it is called.

        :param needle: the typed text; empty matches every command.
        """
        needle = (needle or "").strip().lower()
        if not needle:
            return list(self._commands)
        return [
            c for c in self._commands
            if needle in c.label.lower()
               or any(needle in k.lower() for k in c.keywords)
        ]

    def _stop_rendering(self, _result: int = 0) -> None:
        """Stop the palette from rendering once it has closed.

        Typing still queued behind the closing key would otherwise rebuild
        the list of a dialog that is going away.

        :param _result: the dialog result ``finished`` carries; unused.
        """
        self._closed = True
        self._pending = None

    def _on_filter(self, needle: str) -> None:
        """Narrow the list to commands matching the typed text.

        :param needle: the typed text; empty shows everything.
        """
        self._render(self._matching(needle))

    def _on_activate(self) -> None:
        """Run the selected command and close the palette.

        The palette is dismissed before the command runs, so a command that
        opens a dialog does not open it behind this one. A command that raises
        is logged rather than propagated -- the palette has already closed, and
        there is nothing left to show an exception on.
        """
        item = self._list.currentItem()
        if item is None:
            return
        cmd: Optional[Command] = item.data(Qt.UserRole)
        if cmd is None:
            return
        self.accept()
        try:
            cmd.action()
        except Exception as e:
            LOG.warning("command failed: %s (%s)", cmd.label, e)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Move through the results, or run the highlighted command.

        :param event: the Qt key event.
        """
        if event.key() == Qt.Key_Down:
            row = self._list.currentRow()
            for i in range(row + 1, self._list.count()):
                if self._list.item(i).flags() != Qt.NoItemFlags:
                    self._list.setCurrentRow(i); break
            return
        if event.key() == Qt.Key_Up:
            row = self._list.currentRow()
            for i in range(row - 1, -1, -1):
                if self._list.item(i).flags() != Qt.NoItemFlags:
                    self._list.setCurrentRow(i); break
            return
        super().keyPressEvent(event)

    def _nav(self, key: str) -> None:
        """Navigate the window to a module.

        :param key: the module to open. A window without the navigation slot --
            a test, or a bare dialog -- is tolerated.
        """
        try:
            self._window._on_nav_selected(key)
        except Exception:
            pass

    def _open_preferences(self) -> None:
        """Open the Preferences dialog."""
        try:
            from .preferences import PreferencesDialog
            PreferencesDialog(self._window).exec()
        except Exception:
            pass

    def _open_providers(self) -> None:
        """Open the AI providers dialog."""
        try:
            from .widgets.ai_chat_panel import _ProvidersDialog
            _ProvidersDialog(self._window).exec()
        except Exception:
            pass

    def _open_shortcuts(self) -> None:
        """Open the keyboard shortcut cheat sheet."""
        try:
            from .shortcuts import show_cheat_sheet
            show_cheat_sheet(self._window)
        except Exception:
            pass

    def _open_run(self, run: dict) -> None:
        """Navigate to the run's app + load its settings CSV."""
        try:
            app_key = run["app_key"]
            self._nav(app_key)
            screen = self._window._screens.get(app_key)
            if screen is None or not hasattr(screen, "apply_settings_dict"):
                return
            from ..run_journal import load_run_settings
            settings = load_run_settings(run["dir"])
            screen.apply_settings_dict(settings)
        except Exception as e:
            LOG.warning("failed to open run %s: %s", run.get("dir"), e)
