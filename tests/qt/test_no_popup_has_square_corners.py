"""Every popup is cut to the card's shape, not merely painted over it.

"the rectangular non-rounded black corners are still visible around the
preferences and settings windows ... i want these windows to ALL mimic
the setup spacr window."

TRANSLUCENCY WAS NOT ENOUGH, and that is why this kept coming back.
`WA_TranslucentBackground` asks a compositor to throw the corner pixels
away; whether it does depends on the window manager, and on whether the
surface still had an alpha channel after its flags were rewritten. A mask
REMOVES those pixels from the window's shape, so there is nothing left to
composite and nothing left to paint black.

The mask is asserted through the widget's own `mask()`, which is what Qt
hands the platform -- a screenshot would only say what one compositor did
with it.
"""
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


def test_the_four_corners_are_cut_away(glassed):
    """The corner pixel is not part of the window at all."""
    mask = glassed.mask()
    assert not mask.isEmpty(), "the window carries no mask"

    width, height = glassed.width(), glassed.height()
    for corner in (QPoint(0, 0), QPoint(width - 1, 0),
                   QPoint(0, height - 1), QPoint(width - 1, height - 1)):
        assert not mask.contains(corner), f"{corner} is still square"


def test_the_middle_of_every_edge_is_kept(glassed):
    """A mask that cut more than the corners would clip the rim."""
    mask = glassed.mask()
    width, height = glassed.width(), glassed.height()

    assert mask.contains(QPoint(width // 2, 0))
    assert mask.contains(QPoint(width // 2, height - 1))
    assert mask.contains(QPoint(0, height // 2))
    assert mask.contains(QPoint(width - 1, height // 2))
    assert mask.contains(QPoint(width // 2, height // 2))


def test_the_cut_follows_the_card_radius(glassed):
    """Just inside the radius is kept; just outside it is gone."""
    mask = glassed.mask()
    radius = glass.CARD_RADIUS

    assert not mask.contains(QPoint(1, 1))
    assert mask.contains(QPoint(radius + 2, radius + 2))


def test_the_mask_follows_a_resize(glassed, qtbot):
    """A window resized after it opened must not go square again."""
    glassed.resize(760, 520)
    qtbot.wait(20)

    mask = glassed.mask()
    assert not mask.contains(QPoint(0, 0))
    assert not mask.contains(QPoint(glassed.width() - 1, 0))
    assert mask.contains(QPoint(glassed.width() // 2,
                                glassed.height() // 2))


def test_the_setup_window_is_cut_the_same_way(qtbot, qt_theme_applied,
                                              tmp_path, monkeypatch):
    """"ALL mimic the setup spacr window" -- including the setup window."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt.widgets.setup_slides import SetupSlides

    slides = SetupSlides()
    qtbot.addWidget(slides)
    slides.show()
    qtbot.waitExposed(slides)

    mask = slides.mask()
    assert not mask.isEmpty()
    assert not mask.contains(QPoint(0, 0))
    assert mask.contains(QPoint(slides.width() // 2, slides.height() // 2))


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
