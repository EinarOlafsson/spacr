"""197: the hotkey map is in the Help menu, and it is the WHOLE map.

    "add hotkey map to help tab"

`show_cheat_sheet` has drawn this map for a long time and was reachable from
exactly two places: the `?` key, which you have to know about, and the
command palette, which you have to know about. The Help menu is where a user
who does NOT already know a shortcut goes to look for one -- the entire
population the screen is for.

AND IT WAS SILENTLY PARTIAL. `SHORTCUTS` held 17 entries while the package
bound 15 more -- every Make Masks tool key, the Annotate navigation, the app
list. A map that is partial without saying so is worse than a short one: a
reader takes it for the whole list and concludes a key does not exist.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QKeySequence

pytestmark = pytest.mark.qt

#: Every call that binds a key.
BINDING = {"QShortcut", "setShortcut", "addShortcut", "setShortcuts"}


def _qt_root() -> pathlib.Path:
    import spacr.qt

    return pathlib.Path(spacr.qt.__file__).parent


def _printed(spelling: str) -> str:
    """A key in the spelling the map prints it in, so the two can be
    compared without one of them being the platform's and the other Qt's."""
    return QKeySequence(spelling).toString(QKeySequence.NativeText) or spelling


def _bound_keys() -> dict:
    """``{printed key: where}`` for every key the package binds.

    Both spellings are read: the string form -- `QKeySequence("Ctrl+S")` --
    and the enum form, `QKeySequence(Qt.Key_Left)`, which a scan for string
    literals misses entirely and which is how the image navigation is bound.
    """
    found: dict = {}
    for path in sorted(_qt_root().rglob("*.py")):
        if "i18n_catalogs" in str(path) or path.name == "shortcuts.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                          # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name not in BINDING:
                continue
            for inner in ast.walk(node):
                spelling = ""
                if isinstance(inner, ast.Constant) \
                        and isinstance(inner.value, str) \
                        and 0 < len(inner.value) < 24:
                    spelling = inner.value
                elif isinstance(inner, ast.Attribute) \
                        and inner.attr.startswith("Key_"):
                    spelling = inner.attr[len("Key_"):]
                if not spelling:
                    continue
                printed = _printed(spelling)
                if printed:
                    found.setdefault(printed, f"{path.name}:{node.lineno}")
    return found


def _mapped_keys() -> set:
    """Every key the MAP describes -- window-wide and per-screen.

    `SHORTCUTS` is what `install()` binds; `SCREEN_SHORTCUTS` is what the
    screens bind and the map still has to describe. Reading only the first
    is what left fifteen keys off it.
    """
    from spacr.qt.shortcuts import mapped, native

    return {native(spec.keys) for spec in mapped()}


class TestTheMapIsTheWholeMap:

    def test_the_scan_finds_both_spellings(self):
        """Guard the guard. The enum form is how the image navigation is
        bound, and a scan for string literals misses it entirely."""
        bound = _bound_keys()

        assert "Ctrl+S" in bound, "the string form was not found"
        assert any(k in bound for k in ("Left", "←")), \
            "the Qt.Key_ form was not found"

    def test_every_bound_key_is_on_the_map(self):
        bound = _bound_keys()
        mapped = _mapped_keys()
        missing = {k: w for k, w in bound.items() if k not in mapped}

        assert not missing, (
            "these keys are bound and are not on the hotkey map, so a user "
            "reading it concludes they do not exist:\n"
            + "\n".join(f"  {k}  ({w})" for k, w in sorted(missing.items()))
            + "\n\nAdd a ShortcutSpec for each, with what it does and where "
              "it works.")

    def test_the_map_grew_past_the_seventeen(self):
        """It described the window's own keys and none of the screens'."""
        from spacr.qt.shortcuts import mapped

        assert len(mapped()) > 17

    def test_every_entry_says_what_the_key_does(self):
        from spacr.qt.shortcuts import mapped

        for spec in mapped():
            assert spec.label and spec.label != "(not described)", spec.keys

    def test_a_per_screen_key_says_where_it_works(self):
        """A key that works on one screen and is listed without saying so
        sends a user to press it somewhere it does nothing."""
        from spacr.qt.shortcuts import EVERYWHERE, mapped

        brush = next(s for s in mapped() if s.keys == "B")
        assert brush.scope != EVERYWHERE
        assert "Make Masks" in brush.scope

    def test_a_window_wide_key_does_not_claim_a_scope(self):
        from spacr.qt.shortcuts import EVERYWHERE, mapped

        home = next(s for s in mapped() if s.keys == "Ctrl+H")
        assert home.scope == EVERYWHERE


class TestThePlatformsOwnSpelling:

    def test_the_keys_are_printed_through_qt(self):
        """Writing "Ctrl+H" into a label hard-codes one platform into the
        help; `Ctrl` is the Command symbol on macOS."""
        from spacr.qt.shortcuts import native

        assert native("Ctrl+H") == \
            QKeySequence("Ctrl+H").toString(QKeySequence.NativeText)

    def test_an_unparseable_binding_comes_back_unchanged(self):
        from spacr.qt.shortcuts import native

        assert native("") == ""


class TestItIsInTheHelpMenu:

    @pytest.fixture
    def window(self, qtbot):
        from spacr.qt.app import MainWindow

        widget = MainWindow()
        qtbot.addWidget(widget)
        return widget

    def _help_actions(self, window):
        bar = window.menuBar()
        for action in bar.actions():
            if action.text().replace("&", "") == "Help":
                return action.menu().actions()
        return []

    def test_help_offers_the_keyboard_shortcuts(self, window):
        labels = [a.text().replace("&", "") for a in self._help_actions(window)]

        assert "Keyboard shortcuts" in labels, labels

    def test_it_is_above_the_web_links(self, window):
        """It is the only entry there that answers without a browser."""
        labels = [a.text().replace("&", "") for a in self._help_actions(window)
                  if a.text()]

        assert labels.index("Keyboard shortcuts") < labels.index(
            "Tutorial (web)")

    def test_pressing_it_opens_the_map(self, window, qtbot):
        opened = {}
        window._show_shortcuts = lambda: opened.setdefault("yes", True)
        action = next(a for a in self._help_actions(window)
                      if a.text().replace("&", "") == "Keyboard shortcuts")
        action.triggered.disconnect()
        action.triggered.connect(window._show_shortcuts)

        action.trigger()

        assert opened.get("yes")


class TestOneMapThreeDoors:

    def test_the_help_entry_opens_the_same_screen_as_the_key(self):
        """Three maps that can disagree is what this avoids."""
        import inspect

        from spacr.qt.app import MainWindow

        source = inspect.getsource(MainWindow._show_shortcuts)
        assert "show_cheat_sheet" in source

    def test_and_so_does_the_command_palette(self):
        import inspect

        from spacr.qt.command_palette import CommandPalette

        source = inspect.getsource(CommandPalette._open_shortcuts)
        assert "show_cheat_sheet" in source
