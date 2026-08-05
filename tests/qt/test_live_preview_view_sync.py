"""The two preview canvases must always show the same thing.

Zoom was mirrored; panning was not. `_ZoomView` uses `ScrollHandDrag`,
which moves the scroll bars rather than the view transform, so a drag never
reached `_apply_zoom` and the twin canvases stayed locked in scale while
drifting apart in position. Zoom in, drag the raw pane, and the mask beside
it no longer sits over the cell it was drawn from -- which is the one thing
a side-by-side preview exists to show.
"""

from __future__ import annotations

import pytest

from PySide6.QtGui import QPixmap


@pytest.fixture()
def paired_views(qapp):
    """Two peered views showing the same image, zoomed in enough to scroll."""
    lp = pytest.importorskip("spacr.qt.widgets.live_preview")
    view_cls = getattr(lp, "_ZoomView", None)
    if view_cls is None:
        pytest.skip("_ZoomView not available")

    left, right = view_cls(), view_cls()
    for view in (left, right):
        view.resize(200, 200)
        view.set_pixmap(QPixmap(1200, 1200))
        view.show()
    left.set_peer(right)
    right.set_peer(left)
    for _ in range(4):
        qapp.processEvents()
    # Zoom in so there is somewhere to scroll to.
    left._apply_zoom(4.0, broadcast=True)
    for _ in range(4):
        qapp.processEvents()
    yield left, right
    left.deleteLater()
    right.deleteLater()


def test_panning_one_canvas_pans_the_other(paired_views, qapp):
    left, right = paired_views
    bar = left.horizontalScrollBar()
    if bar.maximum() <= bar.minimum():
        pytest.skip("the view did not become scrollable")

    target = bar.minimum() + (bar.maximum() - bar.minimum()) // 2
    bar.setValue(target)
    for _ in range(4):
        qapp.processEvents()

    assert right.horizontalScrollBar().value() == left.horizontalScrollBar().value(), (
        "the peer did not follow the pan")


def test_panning_the_peer_pans_back(paired_views, qapp):
    """Both directions, and no ping-pong.

    The guard has to sit on the sender. Setting it on the peer would make
    the peer's own handler a no-op, and assigning to its scroll bars fires
    that handler, so the two would either fight or one direction would go
    dead.
    """
    left, right = paired_views
    bar = right.verticalScrollBar()
    if bar.maximum() <= bar.minimum():
        pytest.skip("the view did not become scrollable")

    target = bar.minimum() + (bar.maximum() - bar.minimum()) // 3
    bar.setValue(target)
    for _ in range(4):
        qapp.processEvents()

    assert left.verticalScrollBar().value() == right.verticalScrollBar().value()
    assert not left._syncing and not right._syncing, (
        "a guard was left set, which would deaden every later pan")


def test_random_is_the_default_outline_colour(qapp):
    """A fixed colour is a coin flip against the image.

    Green outlines on a green channel are invisible exactly when you most
    need to see whether the mask landed. `auto` picks per compartment, so
    two touching objects of the same type share an outline and read as one.
    """
    lp = pytest.importorskip("spacr.qt.widgets.live_preview")
    panel = lp.LivePreviewPanel()
    try:
        assert panel._outline_colour.currentText() == "color (random)"
    finally:
        panel.deleteLater()
