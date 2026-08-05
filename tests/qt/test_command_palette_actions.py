"""Ctrl-K: what each command actually does when you press Enter.

The palette is the one surface that reaches everything -- every app, every
menu action, every setting of the module on screen, every recent run. It was
63% covered, and the missing 81 statements were, almost exactly, the half
that *acts*: ``_on_activate``, the arrow keys, ``_nav``, ``_open_preferences``,
``_open_providers``, ``_open_shortcuts``, ``_open_run`` and
``_reveal_setting``.

Those eight methods share a shape that makes the gap dangerous rather than
merely untidy: every one of them ends in ``except Exception: pass`` or a
``LOG.warning``. A command that has silently stopped working is
indistinguishable from one that works -- same absence of exception, same
closed dialog, nothing in the list to look at afterwards. So every test here
asserts the *effect*: the key that reached the window's navigation, the
settings dict that reached the screen, the query that reached the search
strip, the row the arrow key landed on. "It did not raise" is precisely the
assertion that cannot tell the two apart.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget

from spacr.qt import command_palette as CP


class _Window(QMainWindow):
    """A main window with only the attributes the palette reaches for."""

    def __init__(self):
        super().__init__()
        self._stack = QStackedWidget(self)
        self.setCentralWidget(self._stack)
        self._screens = {}
        self.navigated = []

    def _on_nav_selected(self, key):
        self.navigated.append(key)


@pytest.fixture
def window(qtbot):
    win = _Window()
    qtbot.addWidget(win)
    return win


@pytest.fixture
def palette(window, qtbot):
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)
    return pal


def _command_labelled(palette, fragment):
    for cmd in palette._commands:
        if fragment in cmd.label:
            return cmd
    raise AssertionError(
        f"no command labelled like {fragment!r}; have "
        f"{[c.label for c in palette._commands][:12]}")


def _select_label(palette, fragment):
    """Select the visible row whose label contains ``fragment``."""
    for i in range(palette._list.count()):
        item = palette._list.item(i)
        if item.flags() != Qt.NoItemFlags and fragment in item.text():
            palette._list.setCurrentRow(i)
            return item
    raise AssertionError(f"no visible row containing {fragment!r}")


# ---------------------------------------------------------------------------
# 1. Enter runs the selected command
# ---------------------------------------------------------------------------

def test_enter_runs_the_selected_command_and_closes_the_palette(palette,
                                                               window):
    _select_label(palette, "Home")

    palette._on_activate()

    assert window.navigated == ["__home__"]
    assert not palette.isVisible()
    assert palette.result() == CP.CommandPalette.Accepted


def test_enter_on_a_section_header_does_nothing(palette, window):
    """Headers are ``NoItemFlags`` and carry no command; activating one must
    not close the palette on an action that does not exist."""
    header = next(i for i in range(palette._list.count())
                  if palette._list.item(i).flags() == Qt.NoItemFlags)
    palette._list.setCurrentRow(header)
    # setCurrentRow on a disabled row leaves currentItem on it
    palette._list.setCurrentItem(palette._list.item(header))

    palette._on_activate()

    assert window.navigated == []


def test_enter_with_an_empty_list_does_nothing(palette, window):
    palette._input.setText("zzzz-no-such-command-zzzz")
    assert palette._list.count() == 0

    palette._on_activate()

    assert window.navigated == []
    assert palette.result() != CP.CommandPalette.Accepted


def test_a_command_that_raises_is_logged_and_does_not_escape(palette, caplog):
    """One broken command may not take the palette -- or the window -- down."""
    boom = CP.Command(label="Explode", section="Test",
                      action=lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    palette._commands = [boom]
    palette._render(palette._commands)
    _select_label(palette, "Explode")

    with caplog.at_level("WARNING", logger="spacr.qt.command_palette"):
        palette._on_activate()

    assert "Explode" in caplog.text
    assert "nope" in caplog.text


# ---------------------------------------------------------------------------
# 2. The arrow keys skip section headers
# ---------------------------------------------------------------------------

def _key(code):
    return QKeyEvent(QKeyEvent.KeyPress, code, Qt.NoModifier)


def test_down_skips_the_section_header_between_two_groups(palette):
    """Selection must never land on a header: Enter there does nothing, so a
    user pressing Down-Down-Enter would silently get no command."""
    rows = []
    for _ in range(6):
        palette.keyPressEvent(_key(Qt.Key_Down))
        rows.append(palette._list.currentRow())

    for row in rows:
        assert palette._list.item(row).flags() != Qt.NoItemFlags
    assert rows == sorted(rows)
    assert len(set(rows)) > 1, "Down never moved the selection"


def test_up_also_skips_headers_and_stops_at_the_first_command(palette):
    for _ in range(6):
        palette.keyPressEvent(_key(Qt.Key_Down))
    high = palette._list.currentRow()
    for _ in range(10):
        palette.keyPressEvent(_key(Qt.Key_Up))
    low = palette._list.currentRow()

    assert low < high
    assert palette._list.item(low).flags() != Qt.NoItemFlags


def test_other_keys_fall_through_to_the_dialog(palette):
    """Escape has to keep reaching QDialog, or the palette cannot be
    cancelled."""
    assert palette._list.currentRow() >= 0
    palette.keyPressEvent(_key(Qt.Key_Escape))
    assert palette.result() == CP.CommandPalette.Rejected


# ---------------------------------------------------------------------------
# 3. Filtering
# ---------------------------------------------------------------------------

def test_a_keyword_finds_a_command_its_label_does_not_name(palette):
    """"cellpose" has to find Mask; that is what the keyword list is for."""
    palette._on_filter("landing")          # a Home keyword, not in its label
    labels = [palette._list.item(i).text() for i in range(palette._list.count())]
    assert any("Home" in text for text in labels)


def test_clearing_the_filter_restores_every_command(palette):
    full = palette._list.count()
    palette._on_filter("preferences")
    assert palette._list.count() < full
    palette._on_filter("")
    assert palette._list.count() == full


def test_filtering_is_case_insensitive_and_ignores_padding(palette):
    palette._on_filter("  PREFERENCES  ")
    labels = [palette._list.item(i).text() for i in range(palette._list.count())]
    assert any("Preferences" in text for text in labels)


# ---------------------------------------------------------------------------
# 4. Navigation and the dialog-opening commands
# ---------------------------------------------------------------------------

def test_nav_hands_the_key_to_the_window(palette, window):
    palette._nav("mask")
    assert window.navigated == ["mask"]


def test_nav_survives_a_window_that_cannot_navigate(palette, window):
    """A navigation slot that is broken must cost that one command and not
    the palette: the next command still gets through."""
    def _boom(_key):
        raise RuntimeError("navigation is down")

    window._on_nav_selected = _boom
    palette._nav("mask")
    assert window.navigated == []

    window._on_nav_selected = window.navigated.append
    palette._nav("measure")
    assert window.navigated == ["measure"]


@pytest.mark.parametrize("method,module,attr", [
    ("_open_preferences", "spacr.qt.preferences", "PreferencesDialog"),
    ("_open_providers", "spacr.qt.widgets.ai_chat_panel", "_ProvidersDialog"),
])
def test_the_dialog_commands_construct_their_dialog_on_the_window(
        palette, window, monkeypatch, method, module, attr):
    """Asserting the dialog was built for *this* window is the difference
    between the command working and the command silently passing."""
    import importlib
    mod = importlib.import_module(module)
    built = []

    class _Fake:
        def __init__(self, parent):
            built.append(parent)

        def exec(self):
            built.append("exec")

    monkeypatch.setattr(mod, attr, _Fake)
    getattr(palette, method)()
    assert built == [window, "exec"]


def test_the_shortcut_command_shows_the_cheat_sheet_for_the_window(
        palette, window, monkeypatch):
    from spacr.qt import shortcuts
    seen = []
    monkeypatch.setattr(shortcuts, "show_cheat_sheet", seen.append)
    palette._open_shortcuts()
    assert seen == [window]


@pytest.mark.parametrize("method", [
    "_open_preferences", "_open_providers", "_open_shortcuts",
])
def test_a_dialog_command_that_blows_up_is_contained(palette, monkeypatch,
                                                     method):
    """These deliberately swallow: a broken optional dialog may not take the
    palette with it. Pinned so the swallow stays a swallow and not a crash."""
    import importlib
    target = {
        "_open_preferences": ("spacr.qt.preferences", "PreferencesDialog"),
        "_open_providers": ("spacr.qt.widgets.ai_chat_panel", "_ProvidersDialog"),
        "_open_shortcuts": ("spacr.qt.shortcuts", "show_cheat_sheet"),
    }[method]

    reached = []

    def _boom(*args, **kwargs):
        reached.append(args[:1])
        raise RuntimeError("this dialog is broken today")

    monkeypatch.setattr(importlib.import_module(target[0]), target[1], _boom)
    getattr(palette, method)()

    # The broken collaborator was really reached -- "nothing raised" is also
    # true of a command that never called anything at all.
    assert len(reached) == 1


# ---------------------------------------------------------------------------
# 5. Recent runs
# ---------------------------------------------------------------------------

def test_opening_a_recent_run_navigates_and_loads_its_settings(
        palette, window, monkeypatch, tmp_path):
    """The whole point of the Recent entry: land in the app *and* bring the
    run's settings with you."""
    from spacr import run_journal

    class _Screen:
        def __init__(self):
            self.applied = None

        def apply_settings_dict(self, settings):
            self.applied = dict(settings)
            return len(settings)

    screen = _Screen()
    window._screens["mask"] = screen
    monkeypatch.setattr(run_journal, "load_run_settings",
                        lambda _d: {"n_jobs": 4, "cell_channel": 1})

    palette._open_run({"app_key": "mask", "dir": tmp_path})

    assert window.navigated == ["mask"]
    assert screen.applied == {"n_jobs": 4, "cell_channel": 1}


def test_a_recent_run_whose_screen_cannot_take_settings_still_navigates(
        palette, window, tmp_path):
    window._screens["mask"] = QWidget()      # no apply_settings_dict
    palette._open_run({"app_key": "mask", "dir": tmp_path})
    assert window.navigated == ["mask"]


def test_a_recent_run_with_no_screen_yet_still_navigates(palette, window,
                                                         tmp_path):
    palette._open_run({"app_key": "mask", "dir": tmp_path})
    assert window.navigated == ["mask"]


def test_a_run_whose_settings_will_not_load_is_logged_not_raised(
        palette, window, monkeypatch, tmp_path, caplog):
    from spacr import run_journal

    class _Screen:
        def apply_settings_dict(self, settings):
            return 0

    window._screens["mask"] = _Screen()

    def _boom(_dir):
        raise OSError("the settings file is gone")

    monkeypatch.setattr(run_journal, "load_run_settings", _boom)

    with caplog.at_level("WARNING", logger="spacr.qt.command_palette"):
        palette._open_run({"app_key": "mask", "dir": tmp_path})

    assert "failed to open run" in caplog.text
    assert window.navigated == ["mask"]


def test_a_malformed_run_entry_is_logged_not_raised(palette, caplog):
    with caplog.at_level("WARNING", logger="spacr.qt.command_palette"):
        palette._open_run({})           # no app_key at all
    assert "failed to open run" in caplog.text


# ---------------------------------------------------------------------------
# 6. Revealing a setting on the module that is on screen
# ---------------------------------------------------------------------------

class _Bar:
    def __init__(self):
        self.calls = []

    def set_modified_only(self, value):
        self.calls.append(("modified_only", value))

    def set_level(self, value):
        self.calls.append(("level", value))

    def set_query(self, value):
        self.calls.append(("query", value))


class _Section(QWidget):
    def __init__(self, child=None):
        super().__init__()
        self.expanded = None
        self._child = child
        if child is not None:
            child.setParent(self)

    def set_expanded(self, value):
        self.expanded = value


class _Model:
    def __init__(self, widgets):
        self._widgets = widgets

    def _label_for(self, key):
        return key.replace("_", " ").title()

    def plain_tooltip_for(self, key):
        return f"what {key} does"

    def collect(self):
        return {k: 0 for k in self._widgets}


@pytest.fixture
def screen_with_settings(window, qtbot):
    """A screen on the stack carrying one settings widget and a strip."""
    screen = QWidget()
    qtbot.addWidget(screen)
    field = QWidget()
    screen.app_key = "mask"
    screen._settings_model = _Model({"cell_channel": field})
    screen._settings_search = _Bar()
    screen._settings_sections = [_Section(field)]
    window._stack.addWidget(screen)
    window._stack.setCurrentWidget(screen)
    return screen


def test_the_palette_offers_the_settings_of_the_module_on_screen(
        window, screen_with_settings, qtbot):
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)
    cmd = _command_labelled(pal, "(cell_channel)")
    assert cmd.section == "Settings · mask"
    assert "what cell_channel does" in cmd.keywords


def test_activating_a_setting_filters_the_strip_and_opens_its_section(
        window, screen_with_settings, qtbot):
    """The command lands the user *on* the control: the strip is cleared of
    other filters, set to show everything, then narrowed to the key."""
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    pal._reveal_setting("cell_channel")

    assert screen_with_settings._settings_search.calls == [
        ("modified_only", False), ("level", "all"), ("query", "cell_channel"),
    ]
    assert screen_with_settings._settings_sections[0].expanded is True


def test_revealing_a_setting_the_module_does_not_have_expands_nothing(
        window, screen_with_settings, qtbot):
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    pal._reveal_setting("not_a_setting")

    assert screen_with_settings._settings_sections[0].expanded is None
    assert ("query", "not_a_setting") in screen_with_settings._settings_search.calls


def test_a_strip_that_refuses_the_query_still_expands_the_section(
        window, screen_with_settings, qtbot):
    """The fallback the docstring promises: the command still has to land
    somewhere useful when the search bar cannot be driven."""
    class _AngryBar:
        def set_modified_only(self, _v):
            raise RuntimeError("strip is gone")

    screen_with_settings._settings_search = _AngryBar()
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    pal._reveal_setting("cell_channel")

    assert screen_with_settings._settings_sections[0].expanded is True


def test_a_screen_with_no_strip_still_expands_and_focuses(
        window, screen_with_settings, qtbot):
    screen_with_settings._settings_search = None
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    pal._reveal_setting("cell_channel")

    assert screen_with_settings._settings_sections[0].expanded is True


def test_a_deleted_section_is_skipped_rather_than_fatal(
        window, screen_with_settings, qtbot):
    class _Dead:
        def isAncestorOf(self, _w):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    good = screen_with_settings._settings_sections[0]
    screen_with_settings._settings_sections = [_Dead(), good]
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    pal._reveal_setting("cell_channel")

    assert good.expanded is True


def test_a_setting_widget_that_cannot_take_focus_is_not_fatal(
        window, screen_with_settings, qtbot):
    """The widget can be gone by the time the command runs; the strip has
    already been filtered by then and that half must still stand."""
    class _Gone(QWidget):
        def setFocus(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    gone = _Gone()
    qtbot.addWidget(gone)
    screen_with_settings._settings_model._widgets = {"cell_channel": gone}
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    pal._reveal_setting("cell_channel")

    assert ("query", "cell_channel") in screen_with_settings._settings_search.calls


def test_revealing_needs_a_stack_and_shrugs_without_one(qtbot):
    """A window with no screen stack offers no settings commands in the
    first place, and revealing one is a no-op rather than an error."""
    bare = QMainWindow()
    qtbot.addWidget(bare)
    pal = CP.CommandPalette(bare)
    qtbot.addWidget(pal)

    assert not [c for c in pal._commands if c.section.startswith("Settings")]
    assert pal._reveal_setting("cell_channel") is None
    # ...and the palette is still the working list it was.
    assert any("Preferences" in c.label for c in pal._commands)


def test_a_setting_whose_label_cannot_be_read_falls_back_to_its_key(
        window, qtbot):
    """A model that raises on one key must not cost the whole palette."""
    class _AngryModel:
        _widgets = {"cell_channel": QWidget()}

        def _label_for(self, key):
            raise KeyError(key)

        def plain_tooltip_for(self, key):
            return ""

    screen = QWidget()
    qtbot.addWidget(screen)
    screen.app_key = "mask"
    screen._settings_model = _AngryModel()
    window._stack.addWidget(screen)
    window._stack.setCurrentWidget(screen)

    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)
    assert _command_labelled(pal, "cell_channel").label.startswith(
        "cell_channel  (")


def test_a_screen_with_no_settings_contributes_none(window, qtbot):
    plain = QWidget()
    qtbot.addWidget(plain)
    window._stack.addWidget(plain)
    window._stack.setCurrentWidget(plain)

    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)
    assert not [c for c in pal._commands if c.section.startswith("Settings")]


# ---------------------------------------------------------------------------
# 7. Collection stays up when its sources do not
# ---------------------------------------------------------------------------

def test_the_palette_still_opens_when_the_app_registry_is_unreadable(
        window, monkeypatch, qtbot):
    """No apps is a usable palette; no palette is not."""
    from spacr.qt import app as qt_app

    def _boom():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(qt_app, "visible_apps", _boom)
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    labels = [c.label for c in pal._commands]
    assert not [text for text in labels if text.startswith("Go to  ")
                and "Home" not in text]
    assert any("Preferences" in text for text in labels)


def test_the_palette_still_opens_when_the_run_journal_is_unreadable(
        window, monkeypatch, qtbot):
    from spacr import run_journal

    def _boom(limit=8):
        raise OSError("journal unreadable")

    monkeypatch.setattr(run_journal, "recent_runs", _boom)
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    assert not [c for c in pal._commands if c.section == "Recent runs"]
    assert any("Preferences" in c.label for c in pal._commands)


def test_a_recent_run_for_a_hidden_app_is_not_offered(window, monkeypatch,
                                                      qtbot, tmp_path):
    """A run of an app the user has hidden must not reappear in Ctrl-K."""
    from spacr import run_journal
    from spacr.qt import app as qt_app

    monkeypatch.setattr(run_journal, "recent_runs", lambda limit=8: [
        {"app_key": "mask", "dir": tmp_path, "status": "ok", "elapsed_s": 1.0},
        {"app_key": "hidden_app", "dir": tmp_path, "status": "ok",
         "elapsed_s": 2.0},
    ])
    monkeypatch.setattr(qt_app, "app_is_visible",
                        lambda key: key != "hidden_app")

    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)
    recent = [c.label for c in pal._commands if c.section == "Recent runs"]
    assert any("mask" in text for text in recent)
    assert not any("hidden_app" in text for text in recent)


def test_menu_actions_become_commands_that_trigger_them(window, qtbot):
    menu = window.menuBar().addMenu("&File")
    fired = []
    act = menu.addAction("&Export report")
    act.triggered.connect(lambda: fired.append(1))
    menu.addSeparator()
    menu.addMenu("&Recent")             # a submenu is not a command
    menu.addAction("")                  # an unlabelled action is not either

    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    labels = [c.label for c in pal._commands if c.section == "Menu"]
    assert "File → Export report" in labels
    assert not any("Recent" == text.split("→ ")[-1] for text in labels)

    _command_labelled(pal, "Export report").action()
    assert fired == [1]


def test_the_palette_opens_even_if_the_menu_bar_raises(window, monkeypatch,
                                                       qtbot):
    monkeypatch.setattr(type(window), "menuBar",
                        lambda _self: (_ for _ in ()).throw(
                            RuntimeError("no menu bar here")))
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)
    assert not [c for c in pal._commands if c.section == "Menu"]
    assert any("Preferences" in c.label for c in pal._commands)
