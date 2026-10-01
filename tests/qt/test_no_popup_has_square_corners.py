"""Shared windows preserve antialiased alpha instead of a binary mask."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from spacr.qt.widgets import glass


@pytest.fixture
def glassed(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """A plain dialog, given the treatment every popup gets."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    dialog = QDialog()
    qtbot.addWidget(dialog)
    column = QVBoxLayout(dialog)
    column.addWidget(QLabel("something to read"))
    dialog.resize(420, 300)
    glass.glass(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    return dialog


def test_translucent_corners_have_alpha_instead_of_a_binary_cut(glassed):
    """The compositor receives partial coverage instead of a jagged region."""
    from PySide6.QtCore import Qt
    assert glassed.testAttribute(Qt.WA_TranslucentBackground)
    assert glassed.mask().isEmpty()
    image = glassed.grab().toImage()
    assert image.pixelColor(0, 0).alpha() == 0
    assert any(0 < image.pixelColor(x, y).alpha() < 200
               for x in range(18) for y in range(18))
    assert image.pixelColor(image.width() // 2, image.height() // 2).alpha() > 200


def test_alpha_corners_follow_a_resize(glassed, qtbot):
    glassed.resize(760, 520)
    qtbot.wait(20)
    test_translucent_corners_have_alpha_instead_of_a_binary_cut(glassed)


def test_setup_window_uses_the_same_alpha_corners(qtbot, qt_theme_applied):
    from spacr.qt.widgets.setup_slides import SetupSlides
    slides = SetupSlides()
    qtbot.addWidget(slides)
    slides.show()
    qtbot.waitExposed(slides)
    assert slides.mask().isEmpty()
    assert slides.grab().toImage().pixelColor(0, 0).alpha() == 0


def test_the_github_mark_is_the_octicons_path(qapp):
    """The real Octocat, not a circle with ears.

    Asserted against the SHAPE rather than against the string: a path
    that parsed to something lopsided would still match a string compare.
    A 16-unit box, the head wider than it is tall at the top, and the two
    legs leaving a gap between them at the bottom.
    """
    from PySide6.QtCore import QRectF

    from spacr.qt.widgets.provider_marks import GITHUB_MARK, github_path

    assert GITHUB_MARK.startswith("M8 0C")
    assert GITHUB_MARK.rstrip().endswith("Z")

    path = github_path(QRectF(0, 0, 64, 64))
    bounds = path.boundingRect()
    assert 60 <= bounds.width() <= 64
    assert bounds.height() <= bounds.width()

    # The gap between the cat's legs: a point low and centred is outside
    # the silhouette, while the same height to either side is inside.
    assert not path.contains(bounds.center() + _below(bounds, 0.0, 0.46))
    assert path.contains(bounds.center() + _below(bounds, -0.22, 0.40))
    assert path.contains(bounds.center() + _below(bounds, 0.22, 0.40))


def _below(bounds, across, down):
    from PySide6.QtCore import QPointF

    return QPointF(bounds.width() * across, bounds.height() * down)


def test_the_corner_arc_shows_no_more_of_the_backdrop_than_the_middle(qapp,
                                                                      monkeypatch):
    """No blue crescent on the rounded parts.

    The card's body used to be painted on a rect inset by a pixel while
    the backdrop behind it was not, so a ring of pure backdrop showed
    round the edge -- a hairline along the straight sides, half again as
    wide across each corner's diagonal, where it read as a blue crescent
    on every rounded part.

    Measured as a COMPARISON. The body is translucent on purpose, so an
    absolute reading cannot tell a gap from the glass; what matters is
    that a pixel on the corner arc lets no more through than one in the
    middle.
    """
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QDialog

    from spacr.qt import preferences as prefs, theme
    from spacr.qt.widgets.glass import CARD_RADIUS, round_the_corners
    from spacr.qt.widgets.setup_card import SetupCard

    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "dark")
    qapp.setStyleSheet(theme.stylesheet("dark"))
    card = SetupCard(radius=CARD_RADIUS)
    card.resize(400, 300)
    card.show()
    qapp.processEvents()

    def ground_showing(point):
        readings = []
        for ground in ("#000000", "#ffffff"):
            image = QImage(card.size(), QImage.Format_ARGB32_Premultiplied)
            image.fill(QColor(ground))
            card.render(image)
            readings.append(image.pixelColor(point).lightness())
        return (readings[1] - readings[0]) / 255.0

    holder = QDialog()
    holder.resize(400, 300)
    round_the_corners(holder, CARD_RADIUS)
    mask = holder.mask()

    middle = ground_showing(QPoint(200, 150))
    on_the_arc = [QPoint(x, y) for x, y in
                  ((4, 10), (10, 4), (389, 10), (10, 289), (5, 12))
                  if mask.contains(QPoint(x, y))]
    assert on_the_arc, "no sampled point is inside the mask"

    for point in on_the_arc:
        extra = ground_showing(point) - middle
        assert extra < 0.05, (
            f"{point} lets {extra:+.3f} more of the ground through than the "
            f"middle does -- that is the crescent")
    card.hide()
