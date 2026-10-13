"""The timelapse movie: frames, the filmstrip, the toggles and the ceiling.

``TrackStats`` says a field produced 41 tracks with a median length of 6.
It cannot say why, and why is the only thing that changes what you do next.
This is the view that can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _field(offset: int = 0, frames: int = 6, size: int = 48) -> dict:
    """Two blobs moving in opposite directions, with a track for one."""
    images = np.zeros((frames, size, size), np.uint16)
    labels = np.zeros((frames, size, size), np.int32)
    for t in range(frames):
        for object_id, x0 in ((1, 6 + 3 * t), (2, 30 - 2 * t)):
            images[t, 10:20, x0:x0 + 10] = 800 + 100 * object_id
            labels[t, 10:20, x0:x0 + 10] = object_id
    tracks = pd.DataFrame([
        {"frame": t, "x": 10 + 3 * t + offset, "y": 15, "track_id": 1}
        for t in range(frames)
    ])
    return {"images": images, "labels": labels, "tracks": tracks}


@pytest.fixture()
def panel(qtbot):
    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel

    widget = TimelapseMoviePanel()
    qtbot.addWidget(widget)
    widget.resize(700, 700)
    widget.show()
    return widget


# ---------------------------------------------------------------------------
# 1. The movie
# ---------------------------------------------------------------------------

def test_a_field_becomes_a_playable_movie(panel, qtbot):
    panel.set_fields([dict(title="f1", **_field())])
    movie = panel.movies()[0]

    assert movie.frame_count() == 6
    assert not movie._timer.isActive()
    movie.play()
    assert movie._timer.isActive()
    movie.pause()
    assert not movie._timer.isActive()


def test_a_single_frame_field_does_not_play(panel, qtbot):
    """Nothing to animate, and a running timer would just burn a wakeup."""
    panel.set_fields([dict(title="f1", **_field(frames=1))])
    movie = panel.movies()[0]
    movie.play()
    assert not movie._timer.isActive()


def test_playing_wraps_at_the_end(panel, qtbot):
    panel.set_fields([dict(title="f1", **_field())])
    movie = panel.movies()[0]
    movie.show_frame(5)
    movie._advance()
    assert movie._frame == 0


def test_scrubbing_does_not_re_enter_through_the_slider(panel, qtbot):
    """`show_frame` sets the slider, whose signal calls `show_frame`."""
    panel.set_fields([dict(title="f1", **_field())])
    movie = panel.movies()[0]
    movie.show_frame(3)
    assert movie._frame == 3
    assert movie._scrub.value() == 3


# ---------------------------------------------------------------------------
# 2. The filmstrip
# ---------------------------------------------------------------------------

def test_the_strip_starts_closed_and_opens_on_a_click(panel, qtbot):
    panel.set_fields([dict(title="f1", **_field())])
    movie = panel.movies()[0]

    assert not movie.strip_is_open()
    movie.toggle_strip()
    assert movie.strip_is_open()


def test_the_strip_holds_one_thumbnail_per_frame(panel, qtbot):
    panel.set_fields([dict(title="f1", **_field(frames=6))])
    movie = panel.movies()[0]
    movie.toggle_strip()
    assert len(movie._strip._cells) == 6


def test_picking_a_thumbnail_moves_the_movie(panel, qtbot):
    panel.set_fields([dict(title="f1", **_field())])
    movie = panel.movies()[0]
    movie.toggle_strip()
    movie._strip.frame_picked.emit(4)
    assert movie._frame == 4


# ---------------------------------------------------------------------------
# 3. The toggles
# ---------------------------------------------------------------------------

def test_objects_and_tracks_toggle_independently(panel, qtbot):
    """Four combinations, and each one has to render.

    Cached per combination, so flipping a toggle on a long field does not
    re-render every frame twice -- once for the canvas and once for the
    strip -- on the GUI thread.
    """
    panel.set_fields([dict(title="f1", **_field())])
    movie = panel.movies()[0]

    for objects in (True, False):
        for tracks in (True, False):
            movie.set_overlays(objects=objects, tracks=tracks)
            assert movie._rendered(2) is not None, (
                f"objects={objects} tracks={tracks} rendered nothing")
    assert len({key[1:] for key in movie._cache}) == 4


def test_the_panel_toggles_reach_every_field(panel, qtbot):
    panel.set_fields([dict(title="f1", **_field()),
                      dict(title="f2", **_field(offset=5))])
    panel._objects_check.setChecked(False)
    assert all(not m._show_objects for m in panel.movies())
    panel._tracks_check.setChecked(False)
    assert all(not m._show_tracks for m in panel.movies())


def test_turning_objects_off_changes_the_picture(panel, qtbot):
    """Otherwise the toggle is a no-op that looks like it worked."""
    panel.set_fields([dict(title="f1", **_field())])
    movie = panel.movies()[0]

    movie.set_overlays(objects=True, tracks=False)
    with_objects = movie._rendered(2).copy()
    movie.set_overlays(objects=False, tracks=False)
    without = movie._rendered(2)
    assert not np.array_equal(with_objects, without)


# ---------------------------------------------------------------------------
# 4. Several fields, and the ceiling on them
# ---------------------------------------------------------------------------

def test_fields_stack(panel, qtbot):
    panel.set_fields([dict(title="f1", **_field()),
                      dict(title="f2", **_field(offset=5))])
    assert len(panel.movies()) == 2


def test_the_ceiling_drops_surplus_fields_immediately(panel, qtbot):
    """Not at the next preview.

    Somebody lowering this has just been told the machine is short of
    memory; a setting that gives it back "later" is not the setting they
    reached for.
    """
    panel.set_fields([dict(title="f1", **_field()),
                      dict(title="f2", **_field(offset=5))])
    panel.set_max_fields(1)
    assert len(panel.movies()) == 1


def test_more_fields_than_the_ceiling_are_dropped_not_queued(panel, qtbot):
    panel.set_max_fields(1)
    panel.set_fields([dict(title="f1", **_field()),
                      dict(title="f2", **_field(offset=5))])
    assert len(panel.movies()) == 1


def test_the_ceiling_is_bounded(panel, qtbot):
    from spacr.qt.widgets.timelapse_movie import MAX_FIELDS_CEILING

    panel.set_max_fields(999)
    assert panel.max_fields() == MAX_FIELDS_CEILING


# ---------------------------------------------------------------------------
# 5. The renderer it is built on
# ---------------------------------------------------------------------------

def test_render_frame_skips_tracks_it_cannot_read(qapp):
    """`particle` is what trackpy returns before spaCR renames it.

    The guard used to check only x and y while the code went on to group by
    `track_id`, so a frame carrying the wrong id column raised a KeyError
    out of a renderer instead of being skipped like any other unusable
    input.
    """
    from spacr.qt.widgets.timelapse_preview import render_frame

    image = np.zeros((32, 32), np.uint16)
    image[8:16, 8:16] = 900
    labels = np.zeros((32, 32), np.int32)
    labels[8:16, 8:16] = 1
    wrong = pd.DataFrame([{"frame": t, "x": 10 + t, "y": 12, "particle": 1}
                          for t in range(4)])

    out = render_frame(image, labels, wrong, frame=2)
    assert out.shape == (32, 32, 3)


def test_a_track_keeps_its_colour_across_frames(qapp):
    """The whole reason the movie is worth watching."""
    from spacr.qt.widgets.timelapse_preview import track_colour

    assert track_colour(7) == track_colour(7)
    assert track_colour(7) != track_colour(8)


# ---------------------------------------------------------------------------
# 6. The card wires the two together
# ---------------------------------------------------------------------------

def test_the_preview_card_carries_a_movie(qtbot):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel
    from spacr.qt.widgets.timelapse_preview import build_timelapse_preview_card

    host = QWidget()
    qtbot.addWidget(host)
    panel, card = build_timelapse_preview_card(host)
    qtbot.addWidget(card)

    movie = card.findChild(TimelapseMoviePanel)
    assert movie is not None
    assert panel._movie_panel is movie
