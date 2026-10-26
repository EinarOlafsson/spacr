"""No colour picker in spaCR may ask the platform for its dialog.

Instruction 151. The maintainer reported that changing a line width "takes
like 1 minut". The restyle work itself is free — instruction 151 timed
``set_line_style`` at 0.000 s on a 1,200-point volcano — and the control
chained a colour dialog onto it. Every ``QColorDialog.getColor`` in the tree
(six of them) asked for the NATIVE dialog, which on this machine means Qt
hands the request to ``xdg-desktop-portal`` and waits for GTK over D-Bus.

HONEST ABOUT WHAT IS PROVEN HERE. The portal round trip cannot be reproduced
headless: offscreen Qt never asks the portal, so no test in this suite can
watch the stall appear or disappear. "The portal is the minute" stays a named
hypothesis to be confirmed on the real display. What these tests DO prove is
the part that can be proven anywhere and is the part that rots: no call site
can ask for the platform dialog, and a seventh one cannot appear.

Structural, not behavioural, on purpose. An option that has to be remembered
at each call site is an option that gets forgotten at the next one, so the
rule enforced is "go through the helper" rather than "pass the flag".
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialog

_REAL_DIALOG_EXEC = QDialog.exec

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "spacr"

#: The module allowed to name ``QColorDialog.getColor`` at all.
HELPER = "spacr/qt/widgets/colour_picker.py"

#: Files still calling ``getColor`` directly, waiting on their owner.
#:
#: EMPTY, AND IT IS MEANT TO STAY EMPTY. ``fast_plots.py`` was the one entry:
#: instruction 151's other half, owned by a different agent in this wave. Its
#: colour dialogs now go through :func:`pick_colour` -- all FOUR of them, not
#: the three the instruction counted, because ``_ask_style_value`` (the
#: figure style's own colour fields) was a seventh call site nobody had
#: listed. THIS SET IS A DEBT, NOT A DESIGN: never add to it.
PENDING: set = set()


def _get_color_calls(path: Path):
    """Every ``*.getColor(...)`` call in ``path``, as (lineno, source)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source = path.read_text(encoding="utf-8")
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "getColor":
            found.append((node.lineno,
                          ast.get_source_segment(source, node) or ""))
    return found


def _tree_calls():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT.parent).as_posix()
        for lineno, segment in _get_color_calls(path):
            yield relative, lineno, segment


def test_no_call_site_asks_for_the_platform_colour_dialog():
    """A ``getColor`` without ``DontUseNativeDialog`` is the reported stall."""
    offenders = [f"{rel}:{line}" for rel, line, segment in _tree_calls()
                 if "DontUseNativeDialog" not in segment
                 and rel not in PENDING]
    assert offenders == [], (
        "these call QColorDialog.getColor with the platform dialog left on; "
        "use spacr.qt.widgets.colour_picker.pick_colour instead: "
        + ", ".join(offenders))


def test_only_the_helper_names_get_color_at_all():
    """One helper, so the option cannot be forgotten at a seventh site."""
    strays = sorted({rel for rel, _line, _seg in _tree_calls()
                     if rel != HELPER and rel not in PENDING})
    assert strays == [], (
        "call pick_colour() rather than QColorDialog.getColor(): "
        + ", ".join(strays))


def test_the_helper_itself_turns_the_native_dialog_off():
    path = PACKAGE_ROOT.parent / HELPER
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    options = [ast.get_source_segment(source, node) for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "setOption"]
    assert options, "the owned colour dialog must explicitly disable the platform picker"
    assert any("DontUseNativeDialog" in option and "True" in option for option in options)


# ---------------------------------------------------------------------------
# behaviour — what the helper actually passes to Qt
# ---------------------------------------------------------------------------

@pytest.fixture
def recorded(monkeypatch, qapp):
    """Capture the arguments ``pick_colour`` hands to Qt."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    seen = {}

    def _fake(dialog):
        seen.update(initial=QColor(dialog.currentColor()), parent=dialog.parentWidget(),
                    title=dialog.windowTitle(), options=dialog.options())
        dialog.setCurrentColor(QColor("#123456"))
        dialog.accept()
        return QColorDialog.DialogCode.Accepted

    monkeypatch.setattr(QColorDialog, "exec", _fake)
    return seen


def test_the_option_reaches_qt(recorded):
    from PySide6.QtWidgets import QColorDialog
    from spacr.qt.widgets.colour_picker import pick_colour

    assert pick_colour(None, "#ff0000", "Line colour").name() == "#123456"
    assert recorded["title"] == "Line colour"
    assert recorded["initial"].name() == "#ff0000"
    assert (recorded["options"]
            & QColorDialog.ColorDialogOption.DontUseNativeDialog)


def test_an_unusable_initial_colour_does_not_open_on_transparent_black(
        recorded):
    """A stored preference can hold "auto" or "none"; neither is a colour."""
    from spacr.qt.widgets.colour_picker import pick_colour

    for junk in ("auto", "none", "", None):
        pick_colour(None, junk)
        assert recorded["initial"].name() == "#ffffff", junk


def test_a_cancel_comes_back_invalid(monkeypatch, qapp):
    """Every call site tests ``isValid()``; a cancel must keep failing it."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog
    from spacr.qt.widgets.colour_picker import pick_colour

    monkeypatch.setattr(QColorDialog, "exec", lambda dialog: QColorDialog.DialogCode.Rejected)
    assert not pick_colour(None, "#ff0000").isValid()


# ---------------------------------------------------------------------------
# the three sites in this territory actually go through it
# ---------------------------------------------------------------------------

def test_the_figure_settings_swatch_uses_the_helper(qtbot, monkeypatch):
    from PySide6.QtGui import QColor
    from spacr.qt.widgets import figure_settings

    picked = []
    monkeypatch.setattr(figure_settings, "pick_colour",
                        lambda *a, **k: QColor("#00ff00"))
    button = figure_settings._colour_button("#ff0000", picked.append)
    qtbot.addWidget(button)
    button.click()
    assert picked == ["#00ff00"]


def test_the_dna_rain_swatch_uses_the_helper(qtbot, monkeypatch):
    from PySide6.QtGui import QColor
    from spacr.qt.widgets import dna_rain

    bar = dna_rain.DnaRainSettingsBar()
    qtbot.addWidget(bar)
    monkeypatch.setattr(dna_rain, "pick_colour",
                        lambda *a, **k: QColor("#00ff00"))
    bar.pick_color()
    assert bar.color().name() == "#00ff00"


def test_the_owned_picker_centers_after_detachment_and_preserves_parent(qtbot, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QColorDialog, QWidget
    from spacr.qt.widgets.colour_picker import pick_colour

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.resize(1100, 800)
    parent.move(180, 120)
    parent.show()
    qtbot.waitExposed(parent)
    observed = []

    def inspect_and_cancel(dialog):
        observed.append(dialog)
        assert dialog.parentWidget() is parent
        assert dialog.windowType() == Qt.Window
        assert dialog.frameGeometry().center() == parent.frameGeometry().center()
        assert dialog.testOption(QColorDialog.DontUseNativeDialog)
        return QColorDialog.Rejected

    monkeypatch.setattr(QColorDialog, "exec", inspect_and_cancel)
    assert not pick_colour(parent, "#55aa77").isValid()
    assert len(observed) == 1
    assert parent.isVisible()


@pytest.mark.parametrize("theme", ("light", "dark"))
@pytest.mark.parametrize("accept", (False, True), ids=("cancel", "accept"))
def test_the_shown_picker_preserves_initial_colour_and_real_button_result(qtbot, monkeypatch, theme, accept):
    import os
    from pathlib import Path
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QColor
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QColorDialog, QDialogButtonBox, QWidget
    from spacr.qt import preferences
    from spacr.qt.app import install_the_dialog_filters
    from spacr.qt.widgets.colour_picker import pick_colour

    app = QApplication.instance()
    monkeypatch.setattr(QColorDialog, "exec", _REAL_DIALOG_EXEC)
    install_the_dialog_filters(app)
    old_theme = preferences.get_theme()
    parent = QWidget()
    qtbot.addWidget(parent)
    observed = []

    def inspect_and_choose():
        dialog = app.activeModalWidget()
        try:
            assert isinstance(dialog, QColorDialog) and dialog.isVisible()
            qtbot.waitExposed(dialog)
            assert dialog.parentWidget() is parent
            assert dialog.testOption(QColorDialog.DontUseNativeDialog)
            assert dialog.currentColor() == QColor("#55aa77")
            assert (dialog.frameGeometry().center() - parent.frameGeometry().center()).manhattanLength() <= 2
            observed.append(dialog.currentColor().name())
            output = os.environ.get("SPACR_PICKER_NATIVE_ARTIFACTS")
            if output:
                app.primaryScreen().grabWindow(0).save(str(Path(output) / (
                    "picker-initial-" + theme + ("-accept-after.png" if accept else "-cancel-after.png"))))
            dialog.setCurrentColor(QColor("#123456"))
            buttons = dialog.findChild(QDialogButtonBox)
            selected = QDialogButtonBox.Ok if accept else QDialogButtonBox.Cancel
            QTest.mouseClick(buttons.button(selected), Qt.LeftButton)
        finally:
            if isinstance(dialog, QColorDialog) and dialog.isVisible():
                dialog.reject()

    timer = QTimer(parent)
    timer.setSingleShot(True)
    timer.timeout.connect(inspect_and_choose)
    try:
        preferences.set_theme(theme)
        preferences.apply_preferences_to_app(app)
        parent.resize(1100, 800)
        parent.move(180, 120)
        parent.show()
        qtbot.waitExposed(parent)
        timer.start(0)
        result = pick_colour(parent, "#55aa77", "Line colour")
        assert observed == ["#55aa77"]
        if accept:
            assert result.name() == "#123456"
        else:
            assert not result.isValid()
        assert parent.isVisible()
    finally:
        timer.stop()
        preferences.set_theme(old_theme)
        preferences.apply_preferences_to_app(app)
