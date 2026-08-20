"""Changing how a crop is DRAWN must not read it off disk again.

Reported 2026-08-19: "if they have been loaded it should take a verry shourt
amoutn of time to change nd reapply the settings ... i think the current
behaviour is that they are reloaded every time something changes." It was:
`_on_settings_changed` called `build()` unconditionally, which submits a
fresh load.

THE SPLIT ALREADY EXISTED. `spacr.picture_settings` separates the settings
that decide what is CUT from disk -- source, image type, size, channels, crop
shape -- from the ones that decide how an obtained crop is DRAWN: normalise,
outline, edge width, percentiles. Only the first kind can need new pixels,
and the second is most of what a user touches while looking at a montage.
"""
import sys

import pandas as pd
import pytest

sys.path.insert(0, "tests/qt")


@pytest.fixture()
def view(qtbot, tmp_path):
    import test_cells_behind_the_dot_tab as T

    root, db, csv = T._screen(tmp_path, with_png=True)
    widget = T.CellMontageView(
        frame_provider=lambda: pd.read_csv(csv),
        results_provider=lambda: csv,
        database_provider=lambda: T._rows(db), threaded=False)
    qtbot.addWidget(widget)
    widget.set_coefficient(T.GENE_KEY)
    widget.build()
    return widget


@pytest.fixture()
def loads(monkeypatch):
    """Count the reads that actually touch the object tables."""
    import spacr.cell_montage as CM

    seen = []
    real = CM.load_montage_objects

    def counted(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(CM, "load_montage_objects", counted)
    return seen


def _set(view, **picture):
    settings = dict(view.picture_settings())
    settings.update(picture)
    view._picture_settings = settings
    view._on_settings_changed()


def test_a_display_setting_redraws_without_reloading(view, loads):
    """THE WHOLE POINT: normalisation changes the picture, not the pixels
    that were read."""
    before = len(loads)

    _set(view, normalize_channels="r,g,b")

    assert len(loads) == before, "the crops were read off disk again"
    assert view.thumbnails(), "the montage went blank instead of redrawing"


def test_an_outline_setting_redraws_without_reloading(view, loads):
    before = len(loads)
    _set(view, outline="r")
    assert len(loads) == before


def test_a_cut_setting_does_reload(view, loads):
    """A different crop size is a different crop, and no cache can answer
    for it."""
    before = len(loads)

    _set(view, img_size=120)

    assert len(loads) > before, "a new cut was served from the cache"


def test_changing_the_channels_reloads(view, loads):
    """`channels` decides which planes are CUT, so it is not a display
    setting even though it changes how the picture looks."""
    before = len(loads)

    view._channels.setText("r,g")

    assert len(loads) > before


def test_the_signature_ignores_the_display_settings(view):
    """Two requests differing only in how the crop is drawn want the same
    crops and a different picture of them."""
    first = view._load_signature()
    _set(view, normalize_channels="r,g,b", outline="b", edge_thickness=0.4)

    assert view._load_signature() == first


def test_the_signature_notices_a_cut_setting(view):
    first = view._load_signature()
    _set(view, img_size=64)

    assert view._load_signature() != first


def test_nothing_loaded_means_nothing_to_redraw(qtbot):
    """A fresh tab has no crops, so the fast path must not claim it can
    answer -- it would draw an empty montage over a real request."""
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    fresh = CellMontageView(threaded=False)
    qtbot.addWidget(fresh)

    assert fresh._can_redraw_without_loading() is False


def test_the_cache_is_dropped_when_the_montage_is(view):
    """A stale signature would let a later change redraw crops that are no
    longer on screen."""
    view._drop_montage()

    assert view._loaded_signature is None
    assert view._can_redraw_without_loading() is False
