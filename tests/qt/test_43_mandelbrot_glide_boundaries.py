"""Camera boundary behavior missing from the completed nightly CI coverage."""
import math

import numpy as np
import pytest

from spacr.qt.widgets.fractal_mandelbrot import _GlideCamera, _complexity_map


@pytest.mark.parametrize("shape", [(8, 20), (20, 8), (0, 5)])
def test_survey_smaller_than_the_scoring_window_offers_no_false_detail(shape):
    escaped = np.ones(shape, dtype=bool)
    iterations = np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
    scores = _complexity_map(escaped, iterations, window=9)
    assert scores.shape == shape
    assert np.all(scores == -1), "Incomplete neighbourhoods must remain unscored"
    np.testing.assert_array_equal(iterations, np.arange(math.prod(shape)).reshape(shape))
    assert escaped.all()


@pytest.mark.parametrize("target_distance", [0.25, 10.0])
def test_flat_survey_revokes_old_target_score_and_backs_out_without_jumping(target_distance):
    camera = _GlideCamera()
    camera.depth = 2.0
    span = camera.scale()
    camera.target = (target_distance * span, 0.0)
    camera.target_score = 3.0
    previous = camera.centre
    camera.consider({"scores": np.zeros((27, 48)), "centre": previous,
                     "scale": span, "aspect": 48 / 27})
    assert camera.flat
    assert camera.centre_score == 0.0
    assert camera.target_score == 0.0, "An out-of-view target has no current complexity evidence"
    assert camera.centre == previous, "Choosing from a survey must not move the camera"
    for _ in range(60):
        previous, span = camera.centre, camera.scale()
        camera.advance(1 / 30, 0.1)
        assert math.dist(previous, camera.centre) <= camera.speed_limit * span / 30 * (1 + 1e-12)
    assert camera.depth < 2.0
    assert camera.zoom_velocity < 0.0


@pytest.mark.parametrize("depth", [0.0, 12.0])
@pytest.mark.parametrize("direction", [(3.0, 4.0), (-3.0, -4.0)])
def test_large_target_and_inherited_momentum_cannot_exceed_screen_speed(depth, direction):
    camera = _GlideCamera()
    camera.depth = depth
    span = camera.scale()
    camera.target = tuple(component * 100 * span for component in direction)
    camera.velocity = tuple(component * 20 * span for component in direction)
    origin = camera.centre
    step = 1 / 30
    camera.advance(step, 0.0)
    travel = math.dist(origin, camera.centre)
    assert travel == pytest.approx(camera.speed_limit * span * step)
    assert camera.centre[0] * direction[0] > 0
    assert camera.centre[1] * direction[1] > 0
    assert math.hypot(*camera.velocity) < math.hypot(*(component * 20 * span for component in direction))
    assert camera.depth == depth
