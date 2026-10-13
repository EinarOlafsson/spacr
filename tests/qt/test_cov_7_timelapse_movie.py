"""The timelapse movie when there is nothing to play, or nothing to draw.

The point of this view is watching a track break. Everything below is what
happens when it cannot: a field bound with no frames, a renderer that throws
on one frame, a single-frame series a play button must not spin on, and the
panel shrinking from two fields to one. Each of those has to leave the canvas
saying something and the timer stopped, because a movie widget that silently
plays an empty canvas looks identical to one whose data failed to load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.qt


def _field(frames: int = 4, size: int = 32) -> dict:
    """One blob crossing the field, with a track that follows it."""
    images = np.zeros((frames, size, size), np.uint16)
    labels = np.zeros((frames, size, size), np.int32)
    for t in range(frames):
        images[t, 8:16, 4 + 3 * t:12 + 3 * t] = 900
        labels[t, 8:16, 4 + 3 * t:12 + 3 * t] = 1
    tracks = pd.DataFrame([{"frame": t, "x": 8 + 3 * t, "y": 12,
                            "track_id": 1} for t in range(frames)])
    return {"images": images, "labels": labels, "tracks": tracks}


@pytest.fixture()
def movie(qtbot):
    from spacr.qt.widgets.timelapse_movie import FovMovie

    widget = FovMovie(title="field 1")
    qtbot.addWidget(widget)
    widget.resize(400, 320)
    return widget


@pytest.fixture()
def panel(qtbot):
    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel

    widget = TimelapseMoviePanel()
    qtbot.addWidget(widget)
    widget.resize(600, 600)
    return widget


# ---------------------------------------------------------------------------
# Nothing to show
# ---------------------------------------------------------------------------

def test_a_field_with_no_frames_says_so_on_the_canvas(movie):
    """An empty binding leaves a sentence, not a blank rectangle.

    A blank canvas is indistinguishable from a field whose frames are still
    loading, and the strip must be emptied with it rather than keeping the
    thumbnails of whatever was shown before.
    """
    movie.set_sequence(**_field())
    movie.set_sequence(None)

    assert movie.frame_count() == 0
    assert movie._rendered(0) is None
    assert movie._canvas.text() == "No frames loaded."
    assert movie._counter.text() == "0 / 0"
    assert movie._strip._cells == []


def test_a_frame_that_cannot_be_rendered_is_left_out_of_the_strip(
        movie, monkeypatch):
    """One unrenderable frame must not take the whole field down.

    The strip is built frame by frame; a renderer that throws on a frame -- a
    label array of the wrong shape, a track table missing a column -- leaves
    that thumbnail out and keeps the rest, and the canvas falls back to its
    sentence rather than showing a stale frame.
    """
    from spacr.qt.widgets import timelapse_preview

    def refuse(*args, **kwargs):
        raise ValueError("labels do not match the image")

    monkeypatch.setattr(timelapse_preview, "render_frame", refuse)
    movie.set_sequence(**_field())

    assert movie.frame_count() == 4
    assert movie._rendered(0) is None
    assert movie._strip._cells == []
    assert movie._canvas.text() == "No frames loaded."


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def test_the_play_button_is_the_same_switch_in_both_directions(movie):
    """One button, and its label has to follow the timer it controls."""
    movie.set_sequence(**_field())

    movie.toggle_play()
    assert movie._timer.isActive()
    assert movie._play.text() == "Pause"

    movie.toggle_play()
    assert not movie._timer.isActive()
    assert movie._play.text() == "Play"


def test_a_running_timer_stops_itself_when_the_frames_go_away(movie):
    """A timer left running on a single frame burns a wakeup forever.

    The frames can be replaced while the movie plays, so the tick itself has
    to notice that there is no longer anything to advance through.
    """
    movie.set_sequence(**_field())
    movie.play()
    assert movie._timer.isActive()

    movie.set_sequence(**_field(frames=1))
    movie._advance()

    assert not movie._timer.isActive()
    assert movie._play.text() == "Play"
    assert movie._frame == 0


def test_the_frame_rate_setting_reaches_the_timer(movie):
    """Frames per second is stored as the interval the timer actually uses."""
    movie.set_fps(10)
    assert movie._timer.interval() == 100

    # A rate of zero would be an infinite interval; it is floored instead.
    movie.set_fps(0)
    assert movie._timer.interval() == 2000


# ---------------------------------------------------------------------------
# The panel over several fields
# ---------------------------------------------------------------------------

def test_fewer_fields_drops_the_surplus_movies(panel):
    """A shorter list of fields releases the movies it no longer covers.

    Each movie holds its own rendered frames, so leaving a stale one in the
    stack keeps that memory and shows a field the current preview did not
    produce.
    """
    panel.set_fields([dict(title="a", **_field()),
                      dict(title="b", **_field())])
    assert len(panel.movies()) == 2
    panel.movies()[1].play()

    panel.set_fields([dict(title="a", **_field())])

    assert len(panel.movies()) == 1
    assert panel.movies()[0]._title.text() == "a"


def test_play_all_starts_every_field_and_then_stops_every_field(panel):
    """One control over several movies, and its label says which way it goes."""
    panel.set_fields([dict(title="a", **_field()),
                      dict(title="b", **_field())])

    panel._toggle_all()
    assert all(m._timer.isActive() for m in panel.movies())
    assert panel._play_all.text() == "Pause all"

    panel._toggle_all()
    assert not any(m._timer.isActive() for m in panel.movies())
    assert panel._play_all.text() == "Play all"


def test_the_frame_rate_reaches_every_field_at_once(panel):
    """Fields are compared against each other, so they must play in step."""
    panel.set_fields([dict(title="a", **_field()),
                      dict(title="b", **_field())])

    panel.set_fps(8)

    assert [m._timer.interval() for m in panel.movies()] == [125, 125]
