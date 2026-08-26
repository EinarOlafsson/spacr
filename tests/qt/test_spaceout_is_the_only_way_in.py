"""``spaceout`` starts the same application, and it is the only way in.

Three claims, each asserted through the route somebody actually takes rather
than by reading a flag:

* **the same application.** :func:`spacr.qt.spaceout.main` hands its argv
  straight to :func:`spacr.qt.run` and changes nothing else, and the
  stylesheet it produces differs from an ordinary one only in colours — the
  selectors, the geometry and the rules are the same string.
* **nothing in Preferences offers it.** The real
  :class:`spacr.qt.preferences.PreferencesDialog` is built and every
  dropdown in it is read.
* **nothing persists it.** A launch writes no settings key, and a settings
  file that names the fractal cannot put an ordinary ``spacr`` start into
  it — a fresh interpreter that never calls the launcher comes up undressed.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from spacr.qt import theme
from spacr.qt.widgets import ambient

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PY = REPO_ROOT / "setup.py"


@pytest.fixture
def undressed():
    """Leave the process in the dressing the test found it in.

    Process state plus a randomly ordered suite: a leak here would re-colour
    every later test's palette.
    """
    was = theme.spaceout_enabled()
    yield
    if was:
        theme.enable_spaceout()
    else:
        theme.disable_spaceout()


def _console_scripts() -> list:
    """The ``console_scripts`` list, read out of ``setup.py`` as data.

    Parsed rather than imported for the reason
    ``tests/test_packaging_metadata.py`` gives: executing ``setup.py`` is
    what this repository spent a release learning not to do in a test.
    """
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) \
                and getattr(node.func, "id", "") == "setup":
            for keyword in node.keywords:
                if keyword.arg == "entry_points":
                    return list(ast.literal_eval(keyword.value)
                                .get("console_scripts", []))
    pytest.fail("no setup(entry_points=...) literal found in setup.py")


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def test_spaceout_is_a_console_script_beside_spacr_and_spacr_qt():
    scripts = _console_scripts()
    assert "spaceout=spacr.qt.spaceout:main" in scripts
    # Beside them, not instead of them.
    assert "spacr=spacr.qt:run" in scripts
    assert "spacr-qt=spacr.qt:run" in scripts


def test_the_entry_point_resolves_to_something_callable():
    """The half of an entry point a packaging test cannot see: whether the
    module and attribute it names exist."""
    module, _, attribute = "spacr.qt.spaceout:main".partition(":")
    import importlib
    assert callable(getattr(importlib.import_module(module), attribute))


def test_it_is_the_ordinary_launcher_with_the_dressing_already_on(
        monkeypatch, undressed):
    """The same application: one call to :func:`spacr.qt.run`, the argv
    untouched, its return value passed back — and the dressing already in
    force by the time the application starts, because the stylesheet is
    built from the palette during it."""
    import spacr.qt as qt
    from spacr.qt import spaceout

    theme.disable_spaceout()
    seen = {}

    def fake_run(argv=None):
        seen["argv"] = argv
        seen["dressed"] = theme.spaceout_enabled()
        seen["calls"] = seen.get("calls", 0) + 1
        return 7

    monkeypatch.setattr(qt, "run", fake_run)
    assert spaceout.main(["mask", "--no-setup"]) == 7
    assert seen == {"argv": ["mask", "--no-setup"], "dressed": True,
                    "calls": 1}


def test_it_passes_none_through_so_the_console_script_reads_sys_argv(
        monkeypatch, undressed):
    import spacr.qt as qt
    from spacr.qt import spaceout

    seen = {}
    monkeypatch.setattr(qt, "run", lambda argv=None: seen.setdefault(
        "argv", argv) or 0)
    spaceout.main()
    assert seen["argv"] is None


def test_only_the_dressing_changes(undressed):
    """The stylesheet is the whole application's chrome, and under the
    dressing it must differ from an ordinary one in COLOURS AND NOTHING
    ELSE. Strip every ``#rrggbb`` and ``rgba(...)`` out of both and what is
    left — every selector, every size, every radius, every image path —
    has to match character for character.
    """
    import re

    theme.disable_spaceout()
    plain = theme.stylesheet("dark")
    theme.enable_spaceout()
    rainbow = theme.stylesheet("dark")

    assert rainbow != plain, "the dressing did not change the stylesheet"
    colour = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)")
    assert colour.sub("<colour>", rainbow) == colour.sub("<colour>", plain)


def test_every_module_is_still_registered(undressed):
    """"Every module reachable from ``spacr`` is reachable from it": the
    registry the sidebar, Home and the command palette are all built from is
    the same registry."""
    from spacr.qt import app as qt_app

    theme.disable_spaceout()
    plain = list(qt_app.APPS)
    names = qt_app.registered_metadata("name")
    theme.enable_spaceout()
    assert list(qt_app.APPS) == plain
    assert qt_app.registered_metadata("name") == names


# ---------------------------------------------------------------------------
# It is not a theme menu entry
# ---------------------------------------------------------------------------

def test_the_fractal_is_in_no_list_a_menu_is_built_from():
    assert ambient.SPACEOUT_THEME not in ambient.AMBIENT_THEMES
    assert ambient.SPACEOUT_THEME not in ambient.ANIMATION_CHOICES
    assert not ambient.is_valid_theme(ambient.SPACEOUT_THEME)
    assert not ambient.is_animation_choice(ambient.SPACEOUT_THEME)
    for name in ambient.AMBIENT_THEMES:
        assert ambient.SPACEOUT_PALETTE not in ambient.palettes_for(name)
    assert ambient.SPACEOUT_THEME not in theme.THEMES


def test_no_dropdown_in_the_real_preferences_dialog_offers_it(
        qtbot, qt_theme_applied, undressed):
    """Driven through the dialog, not through the lists it is built from.

    Every combo box on every page, in both dressings — because a control
    that only appears once the mode is on would be just as much of a menu
    entry as one that is always there.
    """
    from PySide6.QtWidgets import QComboBox
    from spacr.qt.preferences import PreferencesDialog

    # The animation is never on offer under any name, anywhere in the
    # dialog. The PALETTE is checked against the two controls that could
    # actually select it, because "Rainbow" is a word another preference
    # already uses for something unrelated — the RimMode dropdown offers
    # Glow / Rainbow / Beat, and banning the string outright would be
    # asserting about that control instead of this one.
    banned_everywhere = {ambient.SPACEOUT_THEME,
                         ambient.theme_label(ambient.SPACEOUT_THEME)}
    banned_in_ambient = banned_everywhere | {
        ambient.SPACEOUT_PALETTE,
        ambient.PALETTE_SETS[ambient.SPACEOUT_PALETTE].label}
    for dressed in (False, True):
        theme.enable_spaceout() if dressed else theme.disable_spaceout()
        dialog = PreferencesDialog()
        qtbot.addWidget(dialog)
        # The panel fills itself through QTimer.singleShot; a probe that
        # never spins the loop reads an empty dialog and proves nothing.
        for _ in range(5):
            qtbot.wait(10)
        combos = dialog.findChildren(QComboBox)
        assert combos, "the dialog came up with no dropdowns at all"

        def offered(boxes):
            return ({str(box.itemData(i)) for box in boxes
                     for i in range(box.count())}
                    | {box.itemText(i) for box in boxes
                       for i in range(box.count())})

        assert not (offered(combos) & banned_everywhere), \
            f"Preferences offers the fractal (dressed={dressed})"
        ambient_combos = [box for box in combos
                          if box.objectName() in ("AmbientTheme",
                                                  "AmbientPalette")]
        assert len(ambient_combos) == 2, \
            "the Animation controls moved; this test is looking at nothing"
        leaked = offered(ambient_combos) & banned_in_ambient
        assert not leaked, \
            f"the Animation controls offer {sorted(leaked)} " \
            f"(dressed={dressed})"
        dialog.deleteLater()


def test_the_preference_writer_refuses_the_fractal():
    from spacr.qt.preferences import set_ambient_animation, set_ambient_theme

    with pytest.raises(ValueError):
        set_ambient_theme(ambient.SPACEOUT_THEME)
    with pytest.raises(ValueError):
        set_ambient_animation(ambient.SPACEOUT_THEME)


def test_a_settings_file_that_says_fractal_does_not_dress_an_ordinary_start(
        undressed):
    """The leak the request rules out, written by hand into the store the
    way a downgrade or a text editor would."""
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    theme.disable_spaceout()
    settings = QSettings("spacr", "qt")
    settings.setValue("prefs/ambient_theme", ambient.SPACEOUT_THEME)
    settings.setValue("prefs/ambient_palette", ambient.SPACEOUT_PALETTE)
    settings.sync()

    assert preferences.get_ambient_theme() != ambient.SPACEOUT_THEME
    assert preferences.get_ambient_animation() != ambient.SPACEOUT_THEME
    assert preferences.get_ambient_palette() != ambient.SPACEOUT_PALETTE
    assert not theme.spaceout_enabled()


def test_launching_writes_nothing_to_the_settings_store(monkeypatch,
                                                        undressed):
    """A stored preference would survive a restart and leak the dressing
    into an ordinary start, which is the one thing that is ruled out. So the
    whole store is snapshotted around a launch."""
    from PySide6.QtCore import QSettings
    import spacr.qt as qt
    from spacr.qt import spaceout

    theme.disable_spaceout()
    settings = QSettings("spacr", "qt")
    settings.setValue("prefs/ambient_theme", "aurora")
    settings.sync()
    before = {key: settings.value(key) for key in settings.allKeys()}

    monkeypatch.setattr(qt, "run", lambda argv=None: 0)
    spaceout.main([])

    after = QSettings("spacr", "qt")
    assert {key: after.value(key) for key in after.allKeys()} == before


def test_a_fresh_interpreter_that_never_calls_the_launcher_is_undressed():
    """The claim an in-process test cannot make: what a plain ``spacr``
    start looks like. A subprocess, because the only honest way to ask
    whether the dressing is off *by default* is to start a process that has
    never turned it on.
    """
    source = ("import spacr.qt.theme as t;"
              "print(t.spaceout_enabled(), t.palette_for('dark')['page'])")
    result = subprocess.run([sys.executable, "-c", source],
                            capture_output=True, text=True,
                            cwd=str(REPO_ROOT), timeout=180)
    assert result.returncode == 0, result.stderr
    flag, page = result.stdout.split()
    assert flag == "False"
    assert page == theme.DARK_PALETTE["page"]
