"""The Image UMAP explorer's fallbacks: a dead canvas, a thin theme, a blank key.

Each of these is a path the screen only takes when something else has already
gone wrong -- Qt has deleted the canvas underneath a deferred draw, the active
palette does not carry the role the divider asks for, a table names its objects
but leaves one of the key columns blank. The explorer has to keep drawing in
all three, because the alternative is a screen that dies while the run it is
showing is still going.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from spacr.qt.widgets.umap_explorer import ImageUmapExplorer  # noqa: E402
from spacr.selection import OBJECT_KEY_COLUMNS  # noqa: E402


@pytest.fixture
def explorer(qtbot):
    """An explorer on a private link, so a lasso here reaches nobody else."""
    from spacr.qt.linked_selection import LinkedSelection

    widget = ImageUmapExplorer()
    qtbot.addWidget(widget)
    widget.link_selection("umap", link=LinkedSelection())
    return widget


class _ArraySource:
    """An in-memory crop, the shape the montage sources hand over."""

    def __init__(self, array):
        self._array = array

    def array(self):
        return self._array


def _payload(records, *, frame=None, count=None):
    n = count if count is not None else len(records)
    embedding = np.column_stack([np.arange(n, dtype=float),
                                 np.arange(n, dtype=float) * 0.5])
    payload = {"embedding": embedding,
               "labels": np.zeros(n, dtype=int),
               "records": records}
    if frame is not None:
        payload["frame"] = frame
    return payload


# ---------------------------------------------------------------------------
# a canvas Qt has already deleted
# ---------------------------------------------------------------------------

def test_a_deferred_draw_onto_a_deleted_canvas_is_dropped_not_raised(explorer):
    """Qt can delete the canvas in the same turn the draw was queued for.

    The timer is owned by the canvas so the callback cannot outlive it, but
    the widget can still be torn down between the queue and the fire. What
    comes back from Matplotlib then is a RuntimeError about a deleted C++
    object, and a repaint that nobody can see is not worth a traceback.
    """
    canvas = explorer._canvas
    attempts = []

    def _deleted():
        attempts.append(1)
        raise RuntimeError(
            "Internal C++ object (FigureCanvasQTAgg) already deleted.")

    canvas.draw = _deleted
    canvas.draw_idle()
    assert canvas._draw_pending is True

    canvas._spacr_draw()                      # must not raise

    assert attempts == [1], "the draw was never attempted"
    # The pending flag is cleared BEFORE the draw, so a canvas that dies does
    # not leave the widget permanently believing a repaint is still queued.
    assert canvas._draw_pending is False


def test_a_deferred_draw_with_nothing_pending_does_not_repaint(explorer):
    """The timer fires once for however many draw_idle calls arrived."""
    canvas = explorer._canvas
    canvas._draw_pending = False
    attempts = []
    canvas.draw = lambda: attempts.append(1)

    canvas._spacr_draw()

    assert attempts == []


# ---------------------------------------------------------------------------
# a palette that does not carry the role the divider asks for
# ---------------------------------------------------------------------------

def test_a_palette_without_a_soft_border_still_paints_the_divider(qtbot,
                                                                  monkeypatch):
    """A theme missing one role must not cost the whole screen.

    The chart/sidebar handle asks the palette for ``border_soft`` and
    ``accent``. A palette that has neither is a broken theme, not a broken
    explorer, so the divider falls back to its own colours and the widget
    still builds.
    """
    from spacr.qt import theme

    thin = dict(theme.active_palette())
    thin.pop("border_soft", None)
    thin.pop("accent", None)
    monkeypatch.setattr(theme, "active_palette", lambda *a, **k: dict(thin))

    widget = ImageUmapExplorer()
    qtbot.addWidget(widget)

    qss = widget._body_splitter.styleSheet()
    assert "#3A3A3A" in qss, "the handle line lost its colour"
    assert "#4A9EFF" in qss, "the hover accent lost its colour"


# ---------------------------------------------------------------------------
# a table that names its objects but leaves a blank
# ---------------------------------------------------------------------------

def _key_frame(n=4, blank_row=None):
    """A frame carrying every object-key column, optionally with one gap."""
    columns = list(OBJECT_KEY_COLUMNS)
    frame = pd.DataFrame({
        "plateID": ["plate1"] * n,
        "rowID": ["A"] * n,
        "columnID": ["01"] * n,
        "fieldID": ["1"] * n,
        "object_label": [str(i + 1) for i in range(n)],
        "cell_area": np.arange(n, dtype=float) + 10.0,
    })
    if blank_row is not None:
        frame.loc[blank_row, "object_label"] = None
    assert set(columns) <= set(frame.columns)
    return frame


def test_a_complete_key_table_keys_every_point(explorer):
    """The control for the test below: no gap, so every point gets a key."""
    records = [{"display_name": f"o{i}"} for i in range(4)]
    explorer.set_payload(_payload(records, frame=_key_frame(4)))

    keys = explorer.point_keys()
    assert keys is not None
    assert len(keys) == 4
    assert len(set(keys)) == 4, "four objects must be four keys"


def test_one_blank_key_column_refuses_the_whole_set_of_keys(explorer):
    """Half a lasso published as the whole of it is the worse failure.

    A blank in a key column cannot be keyed, and keying only the rows that
    are complete would publish a selection that silently omits the rest.
    """
    records = [{"display_name": f"o{i}"} for i in range(4)]
    explorer.set_payload(_payload(records, frame=_key_frame(4, blank_row=2)))

    assert explorer.point_keys() is None
    # The frame itself is still held, so a measurement filter can be tested
    # against it even though a selection cannot be published.
    assert explorer._point_frame is not None
    assert "cell_area" in explorer._point_frame.columns


# ---------------------------------------------------------------------------
# a crop that never touched the disk
# ---------------------------------------------------------------------------

def test_a_point_backed_by_an_array_previews_without_a_file(explorer):
    """Streamed crops arrive as arrays; only exported ones are files."""
    crop = np.zeros((50, 80, 3), dtype=np.uint8)
    crop[:, :, 0] = 200
    records = [{"image": _ArraySource(crop), "display_name": "streamed"}]
    explorer.set_payload(_payload(records))

    explorer.show_point(0)

    pixmap = explorer._preview.source_pixmap()
    assert not pixmap.isNull(), explorer._preview.text()
    assert (pixmap.width(), pixmap.height()) == (80, 50)
    assert "Preview unavailable" not in explorer._point_label.text()


def test_a_point_with_no_image_says_so_rather_than_showing_the_last_one(
        explorer):
    """An empty preview beside a stale picture would misname the point."""
    records = [{"image": None, "display_name": "no source"}]
    explorer.set_payload(_payload(records))

    explorer.show_point(0)

    assert explorer._preview.source_pixmap().isNull()
    assert "Preview unavailable" in explorer._preview.text()
