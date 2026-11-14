"""Preference readers and repairs over a hand-edited or half-written store.

``~/.config/spacr/qt.conf`` is a plain INI a user can open and an older or
newer spaCR may have written, so every reader in :mod:`spacr.qt.preferences`
answers a value it cannot use with its default instead of raising, and every
repair (migrations, frozen figure colours, the run notifier) never takes the
window down over something cosmetic.  Each test walks a table of stored
values through the public getters and pins exactly what comes back.
"""
from __future__ import annotations

import builtins
import json
import logging
import os
import sys

import pytest

pytest.importorskip("PySide6")

import PySide6.QtCore as QtCore                                  # noqa: E402
import PySide6.QtWidgets as QtWidgets                            # noqa: E402
from PySide6.QtCore import QSettings                             # noqa: E402
from PySide6.QtGui import QColor, QIcon, QPalette, QPixmap       # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox          # noqa: E402

from spacr import run_journal                                    # noqa: E402
from spacr.qt import preferences as prefs                        # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def store(qapp, tmp_path, monkeypatch):
    """A throwaway INI store, so no test touches the real preferences."""
    settings = QSettings(str(tmp_path / "spacr-qt.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(prefs, "_SAFE_MODE", False)
    assert str(tmp_path) in os.path.normpath(prefs._settings().fileName())
    return settings


def _fresh(store, values):
    """Clear the store and write ``values`` (``{key_attr: value}``) into it."""
    store.clear()
    for key, value in values.items():
        store.setValue(getattr(prefs, key), value)
    store.sync()


# ---------------------------------------------------------------------------
# JSON-valued preferences
# ---------------------------------------------------------------------------

def test_json_valued_preferences_read_damage_as_empty(store, subtests):
    """Folded panels and figure styles read truncated or mis-shaped JSON as ``{}``."""
    cases = (
        # (case, key attr, stored text, getter, expected)
        ("folded unset", "_KEY_FOLDED", "", "get_folded_panels", {}),
        ("folded truncated", "_KEY_FOLDED", '{"mask/console": true',
         "get_folded_panels", {}),
        ("folded half word", "_KEY_FOLDED", '{"console": tru',
         "get_folded_panels", {}),
        ("folded list", "_KEY_FOLDED", '["mask/console"]',
         "get_folded_panels", {}),
        ("folded numbers", "_KEY_FOLDED", "[1, 2, 3]", "get_folded_panels", {}),
        ("folded valid", "_KEY_FOLDED", '{"mask/console": 1, "a/b": 0}',
         "get_folded_panels", {"mask/console": True, "a/b": False}),
        ("style truncated", "_KEY_FIG_STYLE", '{"font.size": 9',
         "get_figure_style", {}),
        ("style not json", "_KEY_FIG_STYLE", "{not json", "get_figure_style", {}),
        ("style list", "_KEY_FIG_STYLE", "[1]", "get_figure_style", {}),
        ("style valid", "_KEY_FIG_STYLE", '{"font.size": 9}',
         "get_figure_style", {"font.size": 9}),
        ("per graph truncated", "_KEY_FIG_STYLE_PER_GRAPH", '{"volcano": ',
         "get_figure_style_per_graph", {}),
        ("per graph not json", "_KEY_FIG_STYLE_PER_GRAPH", "{not json",
         "get_figure_style_per_graph", {}),
        ("per graph a string", "_KEY_FIG_STYLE_PER_GRAPH", '"volcano"',
         "get_figure_style_per_graph", {}),
        ("per graph drops non-dicts", "_KEY_FIG_STYLE_PER_GRAPH",
         '{"volcano": {"dpi": 300}, "bar": 4}', "get_figure_style_per_graph",
         {"volcano": {"dpi": 300}}),
    )
    for case, key, text, getter, expected in cases:
        with subtests.test(msg=case):
            _fresh(store, {key: text} if text else {})
            assert getattr(prefs, getter)() == expected


def test_set_folded_panel_stores_only_named_panels_off_their_default(
        store, subtests):
    """A blank key records nothing; a panel back at its default is forgotten."""
    cases = (
        # (case, calls [(key, shut, default_shut)], expected record)
        ("folded open panel", [("mask/console", True, False)],
         {"mask/console": True}),
        ("blank key ignored", [("mask/console", True, False), ("   ", True, False),
                               ("", True, False)],
         {"mask/console": True}),
        ("key is stripped", [("  mask/console ", True, False)],
         {"mask/console": True}),
        ("reopened default-open panel forgotten",
         [("mask/console", True, False), ("mask/console", False, False)], {}),
        ("default-shut panel stores being opened",
         [("make_masks/psf_details", False, True)],
         {"make_masks/psf_details": False}),
        ("default-shut panel forgets being shut",
         [("make_masks/psf_details", False, True),
          ("make_masks/psf_details", True, True)], {}),
    )
    for case, calls, expected in cases:
        with subtests.test(msg=case):
            store.clear()
            for key, shut, default_shut in calls:
                prefs.set_folded_panel(key, shut, default_shut=default_shut)
            assert prefs.get_folded_panels() == expected
            assert json.loads(store.value(prefs._KEY_FOLDED, "{}")) == expected


# ---------------------------------------------------------------------------
# Numbers and level lists typed by hand
# ---------------------------------------------------------------------------

def test_a_numeric_setting_typed_as_a_word_reads_as_its_default(store, subtests):
    """Every number a user can mistype answers with its published default."""
    cases = (
        ("_KEY_FIG_LIVE_CACHE", "get_figure_live_cache",
         prefs.DEFAULT_FIG_LIVE_CACHE),
        ("_KEY_FIG_PNG_DPI", "get_figure_png_dpi", prefs.DEFAULT_PNG_DPI),
        ("_KEY_MONTAGE_COLUMNS", "get_montage_columns",
         prefs.DEFAULT_MONTAGE_COLUMNS),
        ("_KEY_PANE_OPACITY", "get_pane_opacity",
         prefs.DEFAULT_PANE_OPACITY_PCT / 100.0),
        ("_KEY_RIM_PERIOD", "get_rim_period", prefs.DEFAULT_RIM_PERIOD),
    )
    for key, getter, expected in cases:
        with subtests.test(msg=getter):
            _fresh(store, {key: "lots"})
            assert getattr(prefs, getter)() == pytest.approx(expected)


def test_log_file_levels_parse_lists_strings_and_stray_commas(store, subtests):
    """List and comma-string spellings name the same levels; blanks add nothing."""
    expected = frozenset(prefs._parse_levels("INFO,WARNING", ()))
    assert logging.INFO in expected and logging.WARNING in expected
    assert logging.DEBUG not in expected
    cases = (
        ("list", ["INFO", "WARNING"]),
        ("string", "INFO,WARNING"),
        ("doubled comma", "INFO,,WARNING"),
        ("trailing comma", "INFO,,WARNING,"),
        ("lower case and spaces", " info , warning "),
        ("unknown token dropped", "INFO,WARNING,NOPE"),
    )
    for case, raw in cases:
        with subtests.test(msg=case):
            _fresh(store, {"_KEY_VERBOSE_LOG": False,
                           "_KEY_LOG_FILE_LEVELS": raw})
            assert prefs.get_log_file_levels() == expected
    with subtests.test(msg="parse fallback on blank and garbage"):
        assert prefs._parse_levels("", (logging.ERROR,)) == frozenset(
            {logging.ERROR})
        assert prefs._parse_levels(None, ()) == frozenset()
        assert prefs._parse_levels("NOPE", (logging.ERROR,)) == frozenset(
            {logging.ERROR})


def test_fractal_settings_parse_words_nan_booleans_and_ranges(store, subtests):
    """A word or NaN reads as the default, booleans parse words, steering is clamped."""
    store.clear()
    default_speed = prefs.get_fractal_settings()["speed"]
    cases = (
        # (case, {key attr: stored}, {field: expected})
        ("speed word", {"_KEY_FRACTAL_SPEED": "quickly"},
         {"speed": default_speed}),
        ("speed nan", {"_KEY_FRACTAL_SPEED": float("nan")},
         {"speed": default_speed}),
        ("speed number", {"_KEY_FRACTAL_SPEED": 1.5}, {"speed": 1.5}),
        ("switches off as words", {"_KEY_FRACTAL_VARIABLE_SPEED": "false",
                                   "_KEY_FRACTAL_POINTER": "false"},
         {"variable_speed": False, "pointer_gravity": False}),
        ("switches on as words", {"_KEY_FRACTAL_VARIABLE_SPEED": "on",
                                  "_KEY_FRACTAL_POINTER": "yes"},
         {"variable_speed": True, "pointer_gravity": True}),
        ("steering above ceiling", {"_KEY_FRACTAL_STEERING": 5.0},
         {"steering": 1.0}),
        ("steering below floor", {"_KEY_FRACTAL_STEERING": -2.0},
         {"steering": 0.0}),
        ("steering in range", {"_KEY_FRACTAL_STEERING": 0.4},
         {"steering": 0.4}),
    )
    for case, stored, expected in cases:
        with subtests.test(msg=case):
            _fresh(store, stored)
            values = prefs.get_fractal_settings()
            for field, value in expected.items():
                if isinstance(value, bool):
                    assert values[field] is value, field
                else:
                    assert values[field] == pytest.approx(value), field


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

def test_theme_choice_tokens_round_trip_and_unknown_ones_are_refused(
        store, subtests):
    """Every offered token reads back as itself; a token nobody offers is refused."""
    from spacr.qt.imagery import CELL_VARIANTS

    tokens = [token for _label, token in prefs.theme_choices()]
    assert tokens[:5] == ["dark", "light", "glass", "high_contrast", "system"]
    cells = [token for token in tokens if token.startswith("cell:")]
    assert cells == [f"cell:{variant}" for variant in CELL_VARIANTS]
    for token in tokens:
        with subtests.test(msg=f"round trip {token}"):
            prefs.set_theme_choice(token)
            assert prefs.get_theme_choice() == token
    for token in (cells[0], cells[-1]):
        with subtests.test(msg=f"cell family {token}"):
            prefs.set_theme_choice(token)
            assert prefs.get_theme() == "cell"
            assert prefs.get_cell_variant() == token.split(":", 1)[1]
    for bad in ("neon", "space:not_a_real_variant", "space:nebula"):
        with subtests.test(msg=f"refused {bad}"):
            before = prefs.get_theme_choice()
            with pytest.raises(ValueError) as excinfo:
                prefs.set_theme_choice(bad)
            assert f"unknown theme choice {bad!r}" in str(excinfo.value)
            assert "'dark'" in str(excinfo.value)
            assert prefs.get_theme_choice() == before
    for stored in ("aurora", "space", ""):
        with subtests.test(msg=f"stored {stored!r} reads as the default"):
            store.setValue(prefs._KEY_THEME, stored)
            assert prefs.get_theme() == prefs.DEFAULT_THEME


@pytest.fixture
def light_application_palette(qapp, monkeypatch):
    """A desktop whose own colour scheme is light (offscreen reports none)."""
    from spacr.qt import theme

    monkeypatch.setattr(theme, "system_colour_scheme", lambda app=None: "light")
    original = qapp.palette()
    palette = QPalette(original)
    palette.setColor(QPalette.ColorRole.Window, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#000000"))
    qapp.setPalette(palette)
    try:
        yield qapp
    finally:
        qapp.setPalette(original)


def test_follow_system_resolves_to_a_palette_theme(store, qapp, monkeypatch,
                                                   subtests):
    """'system' becomes the desktop's scheme, or dark when it cannot be asked."""
    from spacr.qt import theme

    def _raises(app=None):
        raise AttributeError("no colour scheme hint on this platform")

    cases = (
        ("light desktop", lambda app=None: "light", "light"),
        ("dark desktop", lambda app=None: "dark", "dark"),
        ("unknown scheme", lambda app=None: None, "dark"),
        ("scheme read raises", _raises, "dark"),
    )
    for case, scheme, expected in cases:
        with subtests.test(msg=case):
            with monkeypatch.context() as patch:
                patch.setattr(theme, "system_colour_scheme", scheme)
                prefs.set_theme("system")
                assert prefs.get_theme() == "system"
                assert prefs.resolve_effective_theme() == expected
                assert expected in prefs.PALETTE_THEMES
    with subtests.test(msg="no application"):
        with monkeypatch.context() as patch:
            patch.setattr(QApplication, "instance", staticmethod(lambda: None))
            prefs.set_theme("system")
            assert prefs.resolve_effective_theme() == "dark"
    with subtests.test(msg="palette that raises"):
        with monkeypatch.context() as patch:
            patch.setattr(QApplication, "palette",
                          lambda _self: (_ for _ in ()).throw(
                              AttributeError("no palette")))
            prefs.set_theme("system")
            assert prefs.resolve_effective_theme() == "dark"
    with subtests.test(msg="an explicit palette theme passes through"):
        prefs.set_theme("light")
        assert prefs.resolve_effective_theme() == "light"


def test_follow_system_on_a_light_desktop_is_light(store,
                                                   light_application_palette):
    """A light desktop with a light palette resolves 'Follow system' to light."""
    prefs.set_theme("system")
    assert prefs.resolve_effective_theme() == "light"


# ---------------------------------------------------------------------------
# One-time migrations and the frozen-colour repair
# ---------------------------------------------------------------------------

def test_store_migrations_run_once_past_a_damaged_marker(store, monkeypatch,
                                                         subtests):
    """Old ambient blur and frozen figure colours convert; a word marker does not block."""
    auto = prefs.AUTO_FIGURE_COLOR
    cases = (
        # (case, migration, stored, expected {key attr: value})
        ("blur 4 converts", "_migrate_ambient_motion",
         {"_KEY_AMBIENT_BLUR": 4.0},
         {"_KEY_AMBIENT_SCALE": prefs.AMBIENT_MOTION_SCALE,
          "_KEY_AMBIENT_RESOLUTION": 0.25, "_KEY_AMBIENT_BLUR": 3.0}),
        ("blur 2 converts", "_migrate_ambient_motion",
         {"_KEY_AMBIENT_BLUR": 2.0},
         {"_KEY_AMBIENT_SCALE": prefs.AMBIENT_MOTION_SCALE,
          "_KEY_AMBIENT_RESOLUTION": 0.5, "_KEY_AMBIENT_BLUR": 1.0}),
        ("blur 0 only marks", "_migrate_ambient_motion",
         {"_KEY_AMBIENT_BLUR": 0.0},
         {"_KEY_AMBIENT_SCALE": prefs.AMBIENT_MOTION_SCALE,
          "_KEY_AMBIENT_RESOLUTION": None, "_KEY_AMBIENT_BLUR": 0.0}),
        ("word marker ambient", "_migrate_ambient_motion",
         {"_KEY_AMBIENT_SCALE": "done", "_KEY_AMBIENT_BLUR": 2.0},
         {"_KEY_AMBIENT_SCALE": prefs.AMBIENT_MOTION_SCALE,
          "_KEY_AMBIENT_BLUR": 1.0}),
        ("already migrated ambient", "_migrate_ambient_motion",
         {"_KEY_AMBIENT_SCALE": prefs.AMBIENT_MOTION_SCALE,
          "_KEY_AMBIENT_BLUR": 4.0},
         {"_KEY_AMBIENT_BLUR": 4.0, "_KEY_AMBIENT_RESOLUTION": None}),
        ("word marker figure", "_migrate_frozen_figure_colors",
         {"_KEY_FIG_COLOR_SCALE": "done", "_KEY_FIG_BG": "#000000",
          "_KEY_FIG_FG": "#ffffff"},
         {"_KEY_FIG_BG": auto, "_KEY_FIG_FG": auto,
          "_KEY_FIG_COLOR_SCALE": prefs.FIGURE_COLOR_SCALE}),
        ("number-less marker figure", "_migrate_frozen_figure_colors",
         {"_KEY_FIG_COLOR_SCALE": "not a number", "_KEY_FIG_BG": "#000000",
          "_KEY_FIG_FG": "#ffffff"},
         {"_KEY_FIG_BG": auto, "_KEY_FIG_FG": auto}),
        ("custom colour kept", "_migrate_frozen_figure_colors",
         {"_KEY_FIG_BG": "#123456"},
         {"_KEY_FIG_BG": "#123456",
          "_KEY_FIG_COLOR_SCALE": prefs.FIGURE_COLOR_SCALE}),
        ("already migrated figure", "_migrate_frozen_figure_colors",
         {"_KEY_FIG_COLOR_SCALE": prefs.FIGURE_COLOR_SCALE,
          "_KEY_FIG_BG": "#000000"},
         {"_KEY_FIG_BG": "#000000"}),
    )
    for case, migration, stored, expected in cases:
        with subtests.test(msg=case):
            _fresh(store, stored)
            getattr(prefs, migration)()
            for key, value in expected.items():
                raw = store.value(getattr(prefs, key), None)
                if value is None:
                    assert raw is None, key
                elif isinstance(value, float):
                    assert float(raw) == pytest.approx(value), key
                elif isinstance(value, int):
                    assert int(raw) == value, key
                else:
                    assert str(raw) == value, key
    with subtests.test(msg="ranges unavailable leaves the store alone"):
        _fresh(store, {"_KEY_AMBIENT_BLUR": 4.0})
        with monkeypatch.context() as patch:
            patch.setattr(prefs, "_ambient_ranges", lambda: (_ for _ in ()).throw(
                RuntimeError("the ranges are unavailable")))
            prefs._migrate_ambient_motion()
        assert store.value(prefs._KEY_AMBIENT_SCALE) is None
        assert float(store.value(prefs._KEY_AMBIENT_BLUR)) == 4.0


def test_unfreezing_figure_colours_undoes_only_spacrs_own_pairs(
        store, monkeypatch, capsys, subtests):
    """Only an implicit frozen pair fighting the theme goes back to 'auto'."""
    auto = prefs.AUTO_FIGURE_COLOR
    light_pair = tuple(str(v).lower() for v in prefs.auto_figure_colors())
    fighting = ("#000000", "#ffffff") if light_pair != (
        "#000000", "#ffffff") else ("#ffffff", "#000000")
    cases = (
        # (case, stored, expected (bg, fg), prints)
        ("custom text colour", {"_KEY_FIG_BG": "#000000",
                                "_KEY_FIG_FG": "#3366cc"},
         ("#000000", "#3366cc"), False),
        ("red text explicit off", {"_KEY_FIG_COLORS_EXPLICIT": False,
                                   "_KEY_FIG_BG": "#000000",
                                   "_KEY_FIG_FG": "#ff0000"},
         ("#000000", "#ff0000"), False),
        ("explicit pair kept", {"_KEY_FIG_COLORS_EXPLICIT": True,
                                "_KEY_FIG_BG": fighting[0],
                                "_KEY_FIG_FG": fighting[1]},
         fighting, False),
        ("one half auto", {"_KEY_FIG_BG": auto, "_KEY_FIG_FG": fighting[1]},
         (auto, fighting[1]), False),
        ("custom background", {"_KEY_FIG_BG": "#123456",
                               "_KEY_FIG_FG": fighting[1]},
         ("#123456", fighting[1]), False),
        ("pair matching the theme", {"_KEY_FIG_BG": light_pair[0],
                                     "_KEY_FIG_FG": light_pair[1]},
         light_pair, False),
        ("frozen pair fighting the theme", {"_KEY_FIG_BG": fighting[0],
                                            "_KEY_FIG_FG": fighting[1]},
         (auto, auto), True),
    )
    for case, stored, expected, prints in cases:
        with subtests.test(msg=case):
            _fresh(store, stored)
            prefs._unfreeze_figure_colors_that_fight_the_theme()
            assert (str(store.value(prefs._KEY_FIG_BG, auto)).lower(),
                    str(store.value(prefs._KEY_FIG_FG, auto)).lower()) == expected
            out = capsys.readouterr().out
            assert ("Figure colours were pinned to" in out) is prints
    with subtests.test(msg="auto check raises"):
        _fresh(store, {"_KEY_FIG_BG": "#000000"})
        with monkeypatch.context() as patch:
            patch.setattr(prefs, "figure_color_is_auto",
                          lambda _v: (_ for _ in ()).throw(
                              RuntimeError("the store is unreadable")))
            prefs._unfreeze_figure_colors_that_fight_the_theme()
        assert str(store.value(prefs._KEY_FIG_BG)) == "#000000"
    with subtests.test(msg="settings read raises"):
        with monkeypatch.context() as patch:
            patch.setattr(prefs, "_settings", lambda: (_ for _ in ()).throw(
                RuntimeError("the settings backend is gone")))
            prefs._unfreeze_figure_colors_that_fight_the_theme()
        assert capsys.readouterr().out == ""


def test_restore_visuals_reports_whether_the_stash_came_back(store, monkeypatch,
                                                             subtests):
    """A stash is consumed either way; refused or unreadable ones answer False."""
    def _refuses(_value):
        raise ValueError("that animation is not installed")

    cases = (
        ("no stash", None, None, False),
        ("not json", "{oops", None, False),
        ("not a mapping", "[1, 2]", None, False),
        ("refused animation", json.dumps({"ambient_animation": "aurora"}),
         _refuses, False),
        ("density restored", json.dumps({"ambient_density": 0.3}), None, True),
    )
    for case, stash, setter, expected in cases:
        with subtests.test(msg=case):
            store.clear()
            if stash is not None:
                store.setValue(prefs._KEY_MODE_VISUAL_STASH, stash)
            with monkeypatch.context() as patch:
                if setter is not None:
                    patch.setattr(prefs, "set_ambient_animation", setter)
                assert prefs._restore_visuals() is expected
            assert not store.value(prefs._KEY_MODE_VISUAL_STASH, "")
            if case == "density restored":
                assert prefs.get_ambient_density() == pytest.approx(0.3)


def test_cards_are_told_about_the_rim_only_when_they_can_be(store, monkeypatch,
                                                            subtests):
    """No application or no card module means nobody is told, and nothing raises."""
    with subtests.test(msg="no application"):
        with monkeypatch.context() as patch:
            patch.setattr(QApplication, "instance", staticmethod(lambda: None))
            assert prefs._tell_the_cards_the_rim_changed() == 0
    with subtests.test(msg="module set to None"):
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "spacr.qt.widgets.setup_card", None)
            assert prefs._tell_the_cards_the_rim_changed() == 0
    with subtests.test(msg="import raises"):
        real_import = builtins.__import__

        def _no_setup_card(name, globals=None, locals=None, fromlist=(),
                           level=0):
            if "setup_card" in name or "setup_card" in (fromlist or ()):
                raise ImportError("no setup card in this build")
            return real_import(name, globals, locals, fromlist, level)

        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", _no_setup_card)
            assert prefs._tell_the_cards_the_rim_changed() == 0
        assert builtins.__import__ is real_import


def test_a_resource_action_runs_only_after_its_named_button(store, monkeypatch,
                                                            subtests):
    """The accept button names the action, Cancel is default, dismissal is refusal."""
    from spacr.qt import resource_cleanup

    seen = {}

    def _press(role):
        def _exec(box):
            seen.update(object=box.objectName(), title=box.windowTitle(),
                        text=box.text(), informative=box.informativeText(),
                        default=box.defaultButton().text(),
                        labels=[b.text() for b in box.buttons()])
            for button in box.buttons():
                if role is not None and box.buttonRole(button) == role:
                    button.click()
            return 0
        return _exec

    title = resource_cleanup.confirmation_title("ram")
    cases = (("accepted", QMessageBox.AcceptRole, True),
             ("rejected", QMessageBox.RejectRole, False),
             ("dismissed", None, False))
    for case, role, expected in cases:
        with subtests.test(msg=case):
            seen.clear()
            with monkeypatch.context() as patch:
                patch.setattr(QMessageBox, "exec", _press(role))
                assert prefs.confirm_resource_action("ram") is expected
            assert seen == {
                "object": "ResourceActionConfirm", "title": title,
                "text": title,
                "informative": resource_cleanup.confirmation_text("ram"),
                "default": "Cancel", "labels": [title, "Cancel"]}


# ---------------------------------------------------------------------------
# The run-finished desktop notifier
# ---------------------------------------------------------------------------

class _FakeTray:
    """A system tray icon double that records what it was asked to show."""

    available = True
    made = []

    class MessageIcon:
        Critical = "critical"
        Information = "information"

    @staticmethod
    def isSystemTrayAvailable():
        return _FakeTray.available

    def __init__(self, icon, parent):
        self.icon = icon
        self.parent = parent
        self.tooltip = None
        self.shown = 0
        self.messages = []
        self.hidden = 0
        _FakeTray.made.append(self)

    def setToolTip(self, text):
        self.tooltip = text

    def show(self):
        self.shown += 1

    def showMessage(self, title, body, icon, msecs):
        if title == "refused":
            raise RuntimeError("tray gone")
        self.messages.append((title, body, icon, msecs))

    def hide(self):
        self.hidden += 1


@pytest.fixture
def relay_env(qapp, monkeypatch):
    """A fake tray and recorded timers; every relay made is torn down."""
    monkeypatch.setattr(run_journal, "_DESKTOP_NOTIFIER", [None])
    monkeypatch.setattr(QtWidgets, "QSystemTrayIcon", _FakeTray)
    _FakeTray.available = True
    _FakeTray.made = []
    timers = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot",
                        staticmethod(lambda msecs, func: timers.append(
                            (msecs, func))))
    made = []
    previous = qapp.windowIcon()
    yield made, timers
    qapp.setWindowIcon(previous)
    for relay in made:
        relay.setParent(None)
        relay.deleteLater()


def test_the_run_notifier_relays_to_one_tray(relay_env, qapp, caplog):
    """One tray, critical for failures, hidden later; a refused message is logged."""
    made, timers = relay_env
    qapp.setWindowIcon(QIcon())
    relay = prefs._install_run_notifier(qapp)
    made.append(relay)
    assert relay.objectName() == "RunFinishedNotifier"
    assert run_journal._DESKTOP_NOTIFIER[0] is not None

    relay.arrived.emit("Run failed", "mask stopped", True)
    relay.arrived.emit("Run done", "measure finished", False)
    with caplog.at_level(logging.DEBUG, logger=prefs.LOG.name):
        relay.arrived.emit("refused", "b", True)

    assert len(_FakeTray.made) == 1
    tray = _FakeTray.made[0]
    assert tray.parent is relay and tray.tooltip == "spaCR"
    assert not tray.icon.isNull()
    assert tray.shown == 3
    assert tray.messages == [
        ("Run failed", "mask stopped", "critical", 15000),
        ("Run done", "measure finished", "information", 15000)]
    assert [msecs for msecs, _func in timers] == [20000, 20000]
    assert any("could not show the run notification" in r.getMessage()
               for r in caplog.records)
    timers[0][1]()
    assert tray.hidden == 1

    # A second install reuses the relay and rewires the journal to it.
    run_journal._DESKTOP_NOTIFIER[0] = None
    assert prefs._install_run_notifier() is relay
    run_journal._DESKTOP_NOTIFIER[0]("again", "body", False)
    assert tray.messages[-1] == ("again", "body", "information", 15000)
    assert len(qapp.findChildren(QtCore.QObject, "RunFinishedNotifier")) == 1


def test_the_run_notifier_uses_the_app_icon_or_the_os_or_nothing(
        relay_env, qapp, qtbot, monkeypatch, caplog, subtests):
    """App icon on the tray; no tray falls back to the OS; no app installs nothing."""
    made, _timers = relay_env
    with subtests.test(msg="application icon"):
        pixmap = QPixmap(8, 8)
        pixmap.fill(QtCore.Qt.red)
        qapp.setWindowIcon(QIcon(pixmap))
        expected = qapp.windowIcon().cacheKey()
        relay = prefs._install_run_notifier(qapp)
        made.append(relay)
        relay.arrived.emit("t", "b", False)
        assert _FakeTray.made[0].icon.cacheKey() == expected
        assert _FakeTray.made[0].messages == [("t", "b", "information", 15000)]
        relay.setParent(None)
    with subtests.test(msg="no tray, OS notify fails"):
        _FakeTray.available = False
        _FakeTray.made = []

        def refuse(title, body):
            raise OSError("no notify-send")

        monkeypatch.setattr(run_journal, "_desktop_os_notify", refuse)
        relay = prefs._install_run_notifier(qapp)
        made.append(relay)
        with caplog.at_level(logging.INFO, logger=prefs.LOG.name):
            relay.arrived.emit("t", "b", True)
            qtbot.waitUntil(lambda: any(
                "no desktop notification shown (no notify-send)"
                in r.getMessage() for r in caplog.records), timeout=5000)
        assert _FakeTray.made == []
        relay.setParent(None)
    with subtests.test(msg="no application"):
        run_journal._DESKTOP_NOTIFIER[0] = None
        with monkeypatch.context() as patch:
            patch.setattr(QApplication, "instance", staticmethod(lambda: None))
            assert prefs._install_run_notifier() is None
        assert run_journal._DESKTOP_NOTIFIER == [None]


def test_the_store_watch_is_queued_once_and_only_with_an_application(
        qapp, monkeypatch, subtests):
    """The watcher start is queued for the event loop once, never without an application."""
    queued = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot",
                        lambda delay, callback: queued.append((delay, callback)))
    for case, app, thread, state, calls in (
            ("no application queues nothing", None, None, None, []),
            ("an application queues one start", qapp, None, "queued",
             [(0, prefs._start_the_gui_store_watch)]),
            ("an already queued start is not queued again", qapp, "queued",
             "queued", [])):
        with subtests.test(msg=case), monkeypatch.context() as patch:
            watch = {"seen": None, "path": None, "thread": thread}
            patch.setattr(prefs, "_GUI_WATCH", watch)
            patch.setattr(QtCore.QCoreApplication, "instance",
                          staticmethod(lambda: app))
            queued.clear()
            prefs._watch_the_gui_store("/profile/qt.conf")
            assert watch == {"seen": None, "path": "/profile/qt.conf",
                             "thread": state}
            assert queued == calls
