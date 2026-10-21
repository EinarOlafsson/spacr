"""Rows meant to be transparent never paint pure black over their card.

2026-10-20 macOS: the console's "N% of your share left" row (AiUsageRow),
the reserved status row under Run and Stop (ActionStatusRow) and, once,
the empty category hint strip under Actions painted #000000 where the card
behind them is translucent. Those rows inherit a pure-black Window colour
and the macOS style painted it; the theme's transparency rule now names
them. Each row is grabbed on a card styled by the real application sheet,
with the application palette's Window colour pure black as on that Mac,
and no pure-black pixel may show. Off macOS the style does not paint the
inherited colour, so there this guards the sheet; on the macOS runners it
guards the paint itself.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget  # noqa: E402

from spacr.qt import theme  # noqa: E402

pytestmark = pytest.mark.qt

ROWS = (("ConsoleBox", "QWidget", "AiUsageRow"),
        ("ConsoleBox", "QLabel", "AiUsageText"),
        ("ActionsCard", "QWidget", "ActionStatusRow"),
        ("ActionsCard", "QLabel", "CategoryHintStrip"),
        ("ConsoleBox", "QWidget", "UsageBarRow"))


@pytest.fixture
def black_window_palette(qapp):
    """The application palette with a pure-black Window colour, restored after."""
    before = QApplication.palette()
    palette = QPalette(before)
    palette.setColor(QPalette.Window, QColor("#000000"))
    QApplication.setPalette(palette)
    yield
    QApplication.setPalette(before)


def _opaque_black_pixels(image):
    """Count opaque pure-black pixels in ``image``."""
    black = 0
    for y in range(image.height()):
        for x in range(image.width()):
            colour = image.pixelColor(x, y)
            if colour.alpha() == 255 and colour.red() == colour.green() == colour.blue() == 0:
                black += 1
    return black


@pytest.mark.parametrize("parent_name,kind,name", ROWS)
def test_a_transparent_row_shows_its_card_not_black(qtbot, black_window_palette,
                                                    parent_name, kind, name):
    card = QFrame()
    card.setObjectName(parent_name)
    card.setStyleSheet(theme.stylesheet("dark") + f"\nQFrame#{parent_name} "
                       "{ background: rgb(27, 29, 32); }")
    qtbot.addWidget(card)
    row = QLabel("") if kind == "QLabel" else QWidget()
    row.setObjectName(name)
    row.setFixedSize(160, 24)
    QVBoxLayout(card).addWidget(row)
    card.resize(200, 60)
    card.show()
    qtbot.waitExposed(card)
    image = card.grab().toImage()
    assert image.pixelColor(row.geometry().center()) == QColor(27, 29, 32), name
    assert _opaque_black_pixels(image) == 0, name


def test_every_named_row_is_in_the_transparency_rule():
    sheet = theme.stylesheet("dark")
    rule = sheet[sheet.index("QWidget#CardBody, QWidget#UsageBarRow"):]
    rule = rule[:rule.index("}")]
    assert "background: transparent" in rule
    for _parent, kind, name in ROWS:
        assert f"{kind}#{name}" in rule, name
