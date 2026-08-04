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

Fuzzy match is a plain substring case-insensitive scan — good
enough for the ~30 commands the palette will ever hold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QDialog, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QVBoxLayout,
)

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


class CommandPalette(QDialog):
    """Modal dialog with a live-filtering command list.

    :param window: the MainWindow the palette should operate on.
    """

    def __init__(self, window: QMainWindow):
        from .i18n import tr
        super().__init__(window)
        self._window = window
        self.setWindowTitle(tr("spaCR — Command palette"))
        self.setModal(True)
        # Frameless-ish look — big centred dialog on top of the app.
        self.setMinimumWidth(560)
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

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget { border: none; padding: 6px 0; }"
            "QListWidget::item { padding: 8px 18px; }"
        )
        outer.addWidget(self._list, 1)

        self._commands: List[Command] = []
        self._collect_commands()
        self._render(self._commands)

        self._input.textChanged.connect(self._on_filter)
        self._input.returnPressed.connect(self._on_activate)
        self._list.itemActivated.connect(lambda _i: self._on_activate())

    # -- collection --------------------------------------------------------
    def _collect_commands(self) -> None:
        from .i18n import tr
        try:
            from .app import app_is_visible, app_stage, visible_apps
            apps = visible_apps()
        except Exception:
            apps = []

            def app_stage(_key):
                return "stable"

            def app_is_visible(_key):
                return True

        # Apps
        for key, name, desc, section in apps:
            # The badge is the app's category — the same single grouping
            # Home and the sidebar use, now that maturity is a colour
            # rather than a second set of sections. How finished an app
            # is is a KEYWORD instead: "alpha" is a useful thing to be
            # able to type, and a useless thing to sort a list by.
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

        # Home
        self._commands.append(Command(
            label=tr("Go to  {name}", name=tr("Home")),
            section=tr("Navigation"),
            action=lambda: self._nav("__home__"),
            keywords=["home", "start", "landing"],
        ))

        # Preferences
        self._commands.append(Command(
            label=tr("Open Preferences…"),
            section=tr("Actions"),
            action=self._open_preferences,
            keywords=["preferences", "settings", "theme", "font",
                      "colour", "color", "accessibility"],
        ))

        # Providers dialog
        self._commands.append(Command(
            label=tr("Open AI Providers…"),
            section=tr("Actions"),
            action=self._open_providers,
            keywords=["providers", "ai", "claude", "chatgpt",
                      "gemini", "llm"],
        ))

        # Cheat sheet
        self._commands.append(Command(
            label=tr("Keyboard shortcuts…"),
            section=tr("Help"),
            action=self._open_shortcuts,
            keywords=["shortcuts", "keyboard", "help", "cheat",
                      "hotkeys"],
        ))

        # Recent runs
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

        # Settings of the module on screen. Without these the palette
        # answered "which app?" and nothing else, while the thing a user is
        # most often hunting for is one setting among a hundred and ninety.
        self._collect_settings_commands()

        # Menu bar actions.
        #
        # Menus are reached through `bar.findChildren(QMenu)`, not by
        # walking `menuBar().actions()` and calling `QAction.menu()`: on
        # PySide6 6.11 the QMenu wrapper the latter returns is only valid
        # while the QAction wrapper it came off is alive, so it went stale
        # as soon as this loop moved on — and every menu command in the
        # palette raised "Internal C++ object already deleted" when
        # triggered. `findChildren` hands back children the bar owns in C++.
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
        widget = (getattr(screen, "_settings_model", None)
                  and screen._settings_model._widgets.get(key))
        if widget is None:
            return
        for section in getattr(screen, "_settings_sections", []) or []:
            try:
                if section.isAncestorOf(widget):
                    section.set_expanded(True)
                    break
            except (AttributeError, RuntimeError):
                continue
        try:
            widget.setFocus()
        except Exception:
            pass

    # -- rendering ---------------------------------------------------------
    def _render(self, cmds: List[Command]) -> None:
        self._list.clear()
        current_section = None
        for cmd in cmds:
            if cmd.section != current_section:
                header = QListWidgetItem(cmd.section.upper())
                header.setFlags(Qt.NoItemFlags)
                header.setForeground(Qt.gray)
                self._list.addItem(header)
                current_section = cmd.section
            item = QListWidgetItem(cmd.label)
            item.setData(Qt.UserRole, cmd)
            self._list.addItem(item)
        if self._list.count() > 1:
            # Skip the first section header when auto-selecting
            for i in range(self._list.count()):
                if self._list.item(i).flags() != Qt.NoItemFlags:
                    self._list.setCurrentRow(i); break

    def _on_filter(self, needle: str) -> None:
        needle = (needle or "").strip().lower()
        if not needle:
            self._render(self._commands)
            return
        filtered = [
            c for c in self._commands
            if needle in c.label.lower()
               or any(needle in k.lower() for k in c.keywords)
        ]
        self._render(filtered)

    # -- activation --------------------------------------------------------
    def _on_activate(self) -> None:
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
        # Arrow keys → move selection; everything else falls through
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

    # -- actions -----------------------------------------------------------
    def _nav(self, key: str) -> None:
        try:
            self._window._on_nav_selected(key)
        except Exception:
            pass

    def _open_preferences(self) -> None:
        try:
            from .preferences import PreferencesDialog
            PreferencesDialog(self._window).exec()
        except Exception:
            pass

    def _open_providers(self) -> None:
        try:
            from .widgets.ai_chat_panel import _ProvidersDialog
            _ProvidersDialog(self._window).exec()
        except Exception:
            pass

    def _open_shortcuts(self) -> None:
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
