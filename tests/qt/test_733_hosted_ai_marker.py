"""Hosted AI progress uses one black marker and the existing bounded icon timer."""
import math

import pytest
from PySide6.QtGui import QImage

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


class _Clock:
    now = 0

    def elapsed(self):
        return self.now


@pytest.fixture
def hosted(qtbot, private_ai_profile):
    from spacr.qt.widgets.home import _NavigatorAiButton

    preferences, _, _ = private_ai_profile
    preferences.set_reduce_motion(False)
    button = _NavigatorAiButton()
    qtbot.addWidget(button)
    button.setStyleSheet('background: rgb(25,25,25); border:none;')
    button._clock = _Clock()
    button._last_key = -10000
    button.show()
    button.setChecked(True)
    return button, preferences


def _frame(button, now):
    button._clock.now = now
    button._tick()
    return button.grab().toImage().convertToFormat(QImage.Format_RGBA8888)


def _marker(frame, radius=6):
    center = 17
    samples = []
    for y in range(7, 28):
        for x in range(7, 28):
            if math.hypot(x + .5 - center, y + .5 - center) >= radius - .6:
                continue
            weight = 255 - frame.pixelColor(x, y).red()
            if weight > 2:
                samples.append((x + .5 - center, y + .5 - center, weight))
    total = sum(sample[2] for sample in samples)
    if not total:
        return None
    return (sum(x * w for x, y, w in samples) / total,
            sum(y * w for x, y, w in samples) / total, total)


def test_actual_hosted_marker_enters_orbits_returns_and_blends_to_white(hosted):
    button, _ = hosted
    button._busy = True
    initial = _frame(button, 0)
    entered = _marker(initial)
    assert entered and math.hypot(*entered[:2]) < .5
    midpoint = _marker(_frame(button, 125))
    assert midpoint and 1.0 < midpoint[0] < 2.0
    edge = _marker(_frame(button, 250))
    assert edge and 2.5 < edge[0] < 3.5 and abs(edge[1]) < .5
    orbit_frame = _frame(button, 700)
    orbit = _marker(orbit_frame)
    assert orbit and abs(orbit[0]) < .5 and 2.5 < orbit[1] < 3.5
    assert initial.pixelColor(17, 3) == orbit_frame.pixelColor(17, 3)
    assert button._motion.isActive()
    button._busy = False
    _frame(button, 700)
    returning = _marker(_frame(button, 850))
    assert returning and .8 < math.hypot(*returning[:2]) < 2
    assert returning[2] < orbit[2]
    done = _frame(button, 1000)
    assert _marker(done) is None
    assert button._marker_phase is None and not button._motion.isActive()


def test_typing_scales_hosted_marker_position_and_size_with_white_fill(hosted):
    button, _ = hosted
    button._busy = True
    _frame(button, 0)
    small = _marker(_frame(button, 250))
    button._typing('a')
    button._typing('ab')
    radius = (8 + 5 * button._pulse) * .75
    large = _marker(_frame(button, 250), radius)
    assert small and large
    assert large[0] / small[0] == pytest.approx(radius / 6, abs=.2)
    assert large[2] > small[2] * 1.7
    assert button._motion.isActive()


def test_hosted_marker_cancel_hide_and_reduced_motion_stop_animation(hosted):
    button, preferences = hosted
    button._busy = True
    _frame(button, 0)
    button.hide()
    assert not button._motion.isActive()
    button.show()
    button._tick()
    assert button._motion.isActive()
    preferences.set_reduce_motion(True)
    first = _frame(button, 700)
    second = _frame(button, 1200)
    assert first == second
    assert _marker(first) and _marker(first)[0] > 2.5
    assert not button._motion.isActive()
    button._busy = False
    assert _marker(_frame(button, 1200)) is None
    preferences.set_reduce_motion(False)
    button._busy = True
    _frame(button, 1300)
    button.setChecked(False)
    assert button._marker_phase is None and not button._motion.isActive()
