"""FlowView's surfaces are transparent, and nothing draws a yellow rim.

Reported 2026-09-01 against Classify: "there is a black box it is in this
should be a transparent box with rounded edges ... and i do not like the
yellow rim, remove the rim entirely".

The rim was a colour-format bug rather than a colour choice. ``#FFFFFF1A``
means white at 10% alpha in CSS, which reads ``#RRGGBBAA`` -- but a QT
STYLESHEET reads eight hex digits as ``#AARRGGBB``, making that literal opaque
``rgb(255, 255, 26)``. Bright yellow. The identical literal in
``flowview/export.py`` is CORRECT, because that one really is CSS and goes to
a browser, which is why this test distinguishes the two rather than banning the
string.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def panel(qtbot):
    from spacr.flowview.panel import FlowViewPanel
    from spacr.flowview.trace import get_collector

    p = FlowViewPanel(get_collector(), embedded=True)
    qtbot.addWidget(p)
    p.resize(900, 560)
    return p


def test_the_graph_surface_is_transparent(panel):
    """Both brushes: the view's own, and the scene's underneath it.

    Clearing only the scene left the view still painting CANVAS over it, which
    is why the box stayed black through the first attempt at this.
    """
    assert panel.scene.backgroundBrush().color().alpha() == 0
    assert panel.view.backgroundBrush().color().alpha() == 0


def test_the_inspector_starts_tall_enough_to_read(panel):
    """118 px was about four lines, so every stage needed scrolling at once."""
    assert panel.inspector.minimumHeight() >= 200


def test_the_inspector_can_take_more_height(panel):
    """"expandable down" -- the splitter must give it a share, not a sliver."""
    splitter = panel._splitter
    assert not splitter.isCollapsible(1), (
        "a pane that collapses to nothing is not expandable in practice")
    # QSplitter exposes no stretchFactor getter, so the share is measured
    # rather than read back: give the panel real height and the inspector
    # must take more than its bare minimum.
    panel.resize(900, 900)
    panel.show()
    assert panel.inspector.height() > panel.inspector.minimumHeight(), (
        f"inspector stayed at its minimum "
        f"({panel.inspector.height()} px) instead of taking a share")


def test_no_qt_stylesheet_uses_the_eight_digit_hex_that_means_yellow():
    """The class of bug, over every Qt stylesheet in the package.

    Qt parses ``#AARRGGBB``; anyone writing CSS habits into ``setStyleSheet``
    gets a colour they did not choose. ``#FFFFFF1A`` is the specific one that
    produced the reported yellow, and it is indistinguishable from a typo at a
    glance, so it is checked mechanically.
    """
    offenders = []
    pattern = re.compile(r"#[0-9A-Fa-f]{8}\b")
    for path in sorted((ROOT / "spacr").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "setStyleSheet" not in text and "register_widget_qss" not in text:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if "#" not in line:
                continue
            # A comment is not a stylesheet. The explanation of this very bug
            # quotes the literal, and flagging prose would make the guard
            # unwritable.
            if line.lstrip().startswith("#"):
                continue
            # Only Qt stylesheet lines. export.py writes real CSS for a
            # browser and #RRGGBBAA is right there.
            if "export" in path.name:
                continue
            for hit in pattern.findall(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number} {hit}")
    assert not offenders, (
        "eight-digit hex in a Qt stylesheet is read as #AARRGGBB, not "
        "#RRGGBBAA -- use rgba() instead:\n  " + "\n  ".join(offenders))


def test_the_exporter_keeps_its_css_alpha():
    """Not vacuous in the other direction: the CSS instances must SURVIVE.

    Rewriting those would change a correct browser colour to fix a Qt bug that
    is not in that file.
    """
    css = (ROOT / "spacr" / "flowview" / "export.py").read_text(encoding="utf-8")
    assert "#FFFFFF1A" in css, (
        "export.py writes CSS for a browser, where #RRGGBBAA is correct")


def test_the_panel_draws_no_border_of_its_own(panel):
    sheet = panel.styleSheet()
    assert "border: none" in sheet
    assert "#FFFFFF1A" not in sheet
