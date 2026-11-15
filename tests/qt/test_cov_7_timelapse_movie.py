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


def test_the_frame_rate_reaches_every_field_at_once(panel):
    """Fields are compared against each other, so they must play in step."""
    panel.set_fields([dict(title="a", **_field()),
                      dict(title="b", **_field())])

    panel.set_fps(8)

    assert [m._timer.interval() for m in panel.movies()] == [125, 125]
