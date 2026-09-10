"""The setup dialog was two stacked windows, and looked like it.

Reported three times, most plainly as: "there is a square window with the
theme and in front of that window is a dark square with rounded edges with
all the settings. I only want that dark square with rounded edges."

The dialog held two children. ``AmbientWidget`` covered the whole frameless
window and painted the theme in a SQUARE; ``SetupCard`` sat 44 pixels inside
it, rounded and translucent, and carried the settings. So the themed square
framed the rounded card on all four sides.

Earlier attempts went after the wrong rectangle -- the glass INSET was taken
to zero and a paint-nothing-behind-the-card pass was added, and the maintainer
still saw two surfaces, because neither of those is what drew this one.

Now they are ONE rectangle: the card fills the dialog, and the backdrop is
clipped to the card's own radius so its corners cannot show past. The proof
is the rendered pixels rather than the geometry -- geometry alone would pass
with a square backdrop exactly covered by a rounded card.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QWidget

from spacr.qt.widgets.setup_slides import CARD_RADIUS, SetupSlides


WIDTH, HEIGHT = 900, 620


@pytest.fixture
def rendered(qapp):
    """The setup dialog, drawn onto a transparent image."""
    dialog = SetupSlides()
    dialog.resize(WIDTH, HEIGHT)
    dialog.show()
    qapp.processEvents()
    qapp.processEvents()

    image = QImage(WIDTH, HEIGHT, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    dialog.render(image)
    try:
        yield dialog, image
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def _alpha(image, x, y):
    return QColor(image.pixelColor(x, y)).alpha()


@pytest.mark.parametrize("corner,x,y", [
    ("top-left", 1, 1),
    ("top-right", WIDTH - 2, 1),
    ("bottom-left", 1, HEIGHT - 2),
    ("bottom-right", WIDTH - 2, HEIGHT - 2),
])
def test_every_corner_is_see_through(rendered, corner, x, y):
    """A painted corner IS the square backdrop, whatever the geometry says."""
    _dialog, image = rendered
    assert _alpha(image, x, y) < 40, (
        f"the {corner} corner is painted, so something square is drawn behind "
        f"the rounded card")


def test_the_middle_of_each_edge_is_painted(rendered):
    """Rounding may not become a hole: only the corners come away."""
    _dialog, image = rendered
    assert _alpha(image, 1, HEIGHT // 2) > 100
    assert _alpha(image, WIDTH // 2, 1) > 100
    assert _alpha(image, WIDTH // 2, HEIGHT // 2) > 100


def test_the_card_and_the_backdrop_are_one_rectangle(rendered):
    """Same rect, so neither can frame the other."""
    dialog, _image = rendered
    children = [c for c in dialog.children() if isinstance(c, QWidget)]
    rects = {(c.geometry().x(), c.geometry().y(),
              c.geometry().width(), c.geometry().height()) for c in children}
    assert len(rects) == 1, f"the dialog's children occupy {len(rects)} rects"
    assert rects.pop() == (0, 0, WIDTH, HEIGHT)


def test_the_backdrop_rounds_by_the_same_amount_as_the_card(rendered):
    """Two radii would let the backdrop's corners show past the card's."""
    dialog, _image = rendered
    backdrops = [c for c in dialog.children()
                 if getattr(c, "_corner_radius", None) is not None]
    assert backdrops, "no backdrop with a corner radius was installed"
    for backdrop in backdrops:
        assert backdrop._corner_radius == CARD_RADIUS


def test_a_backdrop_is_square_unless_it_is_asked_not_to_be(qapp):
    """Every other screen is a rectangle and must not start rounding itself."""
    from spacr.qt.widgets.ambient import AmbientWidget

    host = QWidget()
    plain = AmbientWidget(host)
    try:
        assert plain._corner_radius == 0
    finally:
        host.deleteLater()
        qapp.processEvents()
