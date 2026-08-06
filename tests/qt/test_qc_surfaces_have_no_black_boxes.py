"""No QC surface may paint the window colour behind its own text.

This is INVARIANTS 3, and it has now been reported twice. A ``QLabel`` is a
``QWidget``, so a label with no background of its own is matched by the
blanket ``QWidget { background-color: bg }`` rule and paints ``bg`` -- the
WINDOW colour, ``#000000`` on the dark theme -- as a solid rectangle behind
its own text, on top of a panel that does have a background. The same is
true of any plain ``QWidget`` used only to hold a layout.

The first fix tagged the two outer containers in ``spacr.qt.prerun`` and was
reported as not working, because the black box was coming from the widgets
INSIDE them: one anonymous ``QWidget`` per finding row, and every card label
on the QC dashboard, which had colour rules but no background rule.

So these tests do not check a widget. They walk the whole surface and assert
that nothing in it is left to fall through -- which is the only form of this
assertion that cannot be satisfied by fixing one screen and leaving the next.

Asserted on the stylesheet and the transparency property rather than on
pixels: ``QWidget.render()`` cannot reproduce paint-ordering bugs
(INVARIANTS 7) and reported ``0.0% black`` four times for a screen that was
solid black on the user's display.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QWidget       # noqa: E402


def _sheet():
    from spacr.qt import theme
    return theme.stylesheet()


def _paints_its_own_background(widget, sheet: str) -> bool:
    """Whether this widget is spared the blanket rule.

    Three ways, and any one is enough: it is tagged transparent, it carries
    an objectName the sheet gives a background, or it is a class the sheet
    styles by type.
    """
    from spacr.qt import theme

    if widget.property(theme.TRANSPARENT_PROPERTY) is True:
        return True
    name = widget.objectName()
    if name:
        for m in re.finditer(r"#" + re.escape(name) + r"\b[^{]*\{([^}]*)\}",
                             sheet):
            if "background" in m.group(1):
                return True
    return False


def _labels_covered_by_an_ancestor_rule(widget, sheet: str) -> bool:
    """A descendant rule like ``#panel QLabel { background: transparent }``."""
    node = widget.parent()
    while node is not None:
        name = node.objectName()
        if name and re.search(
                r"#" + re.escape(name) + r"[^{]*\bQLabel\b[^{]*\{[^}]*background",
                sheet):
            return True
        node = node.parent()
    return False


#: Qt's own private children. `qt_scrollarea_hcontainer` and
#: `qt_scrollarea_vcontainer` hold the scrollbars and are never behind text;
#: the viewport IS checked, because it is the one that paints.
_QT_INTERNALS = ("qt_scrollarea_hcontainer", "qt_scrollarea_vcontainer")


def _offenders(root, sheet):
    """Every descendant that would paint the window colour."""
    bad = []
    for child in root.findChildren(QWidget):
        if child.objectName() in _QT_INTERNALS:
            continue
        if isinstance(child, QLabel):
            if not (_paints_its_own_background(child, sheet)
                    or _labels_covered_by_an_ancestor_rule(child, sheet)):
                bad.append(f"QLabel#{child.objectName() or '(unnamed)'}")
        elif type(child) is QWidget:
            if not _paints_its_own_background(child, sheet):
                bad.append("anonymous QWidget")
    return bad


def test_the_seg_qc_banner_has_nothing_that_falls_through(qtbot,
                                                          qt_theme_applied):
    """With findings DRAWN. An empty banner has no per-finding rows, and the
    per-finding rows are exactly where the black box was the second time."""
    from types import SimpleNamespace

    import spacr.qt.prerun as prerun

    parent = QWidget()
    qtbot.addWidget(parent)
    banner = prerun.SegQCBanner(parent)
    banner._expanded = True
    banner._draw_findings(SimpleNamespace(findings=[
        SimpleNamespace(severity="fail", headline="A01 lost 40% of cells",
                        detail="detail", fix="Lower the threshold."),
        SimpleNamespace(severity="warn", headline="B02 borderline",
                        detail="detail", fix="Check it."),
    ]))

    # The fixture has to actually build rows, or this passes on an empty tree.
    anon = [c for c in banner.findChildren(QWidget) if type(c) is QWidget]
    assert len(anon) >= 3, "no per-finding rows were built"

    assert _offenders(banner, _sheet()) == []


def test_the_diameter_panel_has_nothing_that_falls_through(qtbot,
                                                           qt_theme_applied):
    import spacr.qt.prerun as prerun

    parent = QWidget()
    qtbot.addWidget(parent)
    panel = prerun.DiameterPanel(parent)
    assert _offenders(panel, _sheet()) == []


def test_the_qc_dashboard_has_nothing_that_falls_through(qtbot,
                                                         qt_theme_applied):
    """The cards' labels had colour rules but no background rule, so each one
    painted a black rectangle behind its own text on a panel that had a
    background of its own."""
    from spacr.qt.screens.qc_dashboard import QCDashboardScreen

    screen = QCDashboardScreen()
    qtbot.addWidget(screen)
    assert _offenders(screen, _sheet()) == []


def test_the_detector_can_actually_fail(qtbot, qt_theme_applied):
    """The control. Without it, every assertion above could be vacuous."""
    host = QWidget()
    qtbot.addWidget(host)
    QWidget(host)                      # anonymous container, untagged
    QLabel("unstyled", host)           # label with no background rule

    found = _offenders(host, _sheet())
    assert "anonymous QWidget" in found
    assert any(f.startswith("QLabel") for f in found)
