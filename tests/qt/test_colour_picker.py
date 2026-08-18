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

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "spacr"

#: The module allowed to name ``QColorDialog.getColor`` at all.
HELPER = "spacr/qt/widgets/colour_picker.py"

#: Files still calling ``getColor`` directly, waiting on their owner.
#:
#: ``fast_plots.py`` is instruction 151's other half and belongs to a
#: different agent in this wave; its sites are the three in ``_ask_*``. THIS
#: SET IS A DEBT, NOT A DESIGN — delete the entry once those calls go through
#: :func:`spacr.qt.widgets.colour_picker.pick_colour`, and never add to it.
PENDING = {"spacr/qt/widgets/fast_plots.py"}


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
    segments = [segment for _line, segment
                in _get_color_calls(PACKAGE_ROOT.parent / HELPER)]
    assert segments, "the helper stopped calling getColor"
    for segment in segments:
        assert "DontUseNativeDialog" in segment


# ---------------------------------------------------------------------------
# behaviour — what the helper actually passes to Qt
# ---------------------------------------------------------------------------

@pytest.fixture
def recorded(monkeypatch):
    """Capture the arguments ``pick_colour`` hands to Qt."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    seen = {}

    def _fake(initial, parent=None, title="", options=None):
        seen.update(initial=initial, parent=parent, title=title,
                    options=options)
        return QColor("#123456")

    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(_fake))
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


def test_a_cancel_comes_back_invalid(monkeypatch):
    """Every call site tests ``isValid()``; a cancel must keep failing it."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog
    from spacr.qt.widgets.colour_picker import pick_colour

    monkeypatch.setattr(QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: QColor()))
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
