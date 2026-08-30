"""What ``spacr.qt.hidpi`` does when the display will not answer properly.

The happy path -- a widget on a 2x screen, a pixmap rendered at 2x and
carrying its ratio -- is measured next door in
``test_every_picture_is_drawn_for_the_screen_it_is_on.py``. This file drives
the halves of each branch that only appear when the environment is odd, and
every one of them is a real environment:

* a paint device that answers ``devicePixelRatioF()`` with 0.0, NaN or
  infinity -- a null QPaintDevice, a broken ``QT_SCALE_FACTOR``, a stub in
  somebody else's test -- where the ratio has to come from the *other*
  accessor rather than being believed;
* a widget that has not been shown yet, whose ``screen()`` is not known but
  whose ``window()`` is, so the lookup must keep walking;
* a headless or half-built application whose primary screen is ``None``,
  where the answer must be a plain 1.0 and never a crash on the way to
  drawing the very first picture;
* a picture that scales to nothing, where stamping a device pixel ratio onto
  a null pixmap would be meaningless;
* a picture object older or simpler than a ``QPixmap``, with no
  ``deviceIndependentSize()``, whose occupied size still has to be right or
  every centring and hit test against it is off by the ratio.

Each test drives the branch BOTH ways in the same test: the odd input and,
beside it, an ordinary one that produces the opposite answer. An assertion
that "nothing went wrong" passes just as well against a test that exercised
nothing at all.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSize, Qt  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402

from spacr.qt import hidpi  # noqa: E402
from spacr.qt.hidpi import (  # noqa: E402
    MAX_RATIO,
    device_ratio,
    logical_size,
    scaled_for,
)

pytestmark = pytest.mark.qt


class _Ratios:
    """An object that answers the two ratio accessors with fixed numbers.

    Qt's ratio comes from the windowing system and cannot be talked into a
    value, so a stand-in is the only way to drive what the module does WITH
    a ratio it was given. Both accessors are recorded, because which one was
    consulted is part of what is under test.
    """

    def __init__(self, ratio_f, ratio=None):
        self._ratio_f = ratio_f
        self._ratio = ratio if ratio is not None else ratio_f
        self.asked = []

    def devicePixelRatioF(self):               # noqa: N802 - Qt spelling
        self.asked.append("devicePixelRatioF")
        return self._ratio_f

    def devicePixelRatio(self):                # noqa: N802 - Qt spelling
        self.asked.append("devicePixelRatio")
        return self._ratio


class _NoRatioAtAll:
    """A paint target that has never heard of a device pixel ratio."""


def _pixmap(width, height):
    """A drawable source picture of a known shape."""
    picture = QPixmap(width, height)
    picture.fill(Qt.white)
    return picture


# ---------------------------------------------------------------------------
# a first accessor whose answer is not a ratio
# ---------------------------------------------------------------------------

def test_a_ratio_of_zero_nan_or_infinity_is_not_believed(qapp):
    """A meaningless ratio must send the lookup on, not become the scale.

    ``devicePixelRatioF()`` is 0.0 on a null paint device and can come back
    NaN or infinity from a display whose ``QT_SCALE_FACTOR`` was set to
    something absurd. Taking any of those at face value multiplies every
    render by it: 0 collapses the picture to nothing, NaN makes ``round()``
    raise inside a paint path, and infinity asks Qt for an unbounded
    allocation. The rule is that a number that is not a ratio does not end
    the search -- the second accessor, and then the screen, still get asked.
    """
    # the odd half: the first accessor is junk, the second one is the truth
    for junk in (0.0, -1.0, float("nan"), float("inf")):
        target = _Ratios(junk, 2.0)
        assert device_ratio(target) == 2.0, f"{junk!r} was believed"
        assert target.asked == ["devicePixelRatioF", "devicePixelRatio"]

    # the ordinary half: a real first answer is taken and the second accessor
    # is never consulted, so the loop above really did fall through.
    good = _Ratios(1.5, 4.0)
    assert device_ratio(good) == 1.5
    assert good.asked == ["devicePixelRatioF"]

    # and a ratio nobody's hardware has is clamped rather than obeyed
    assert device_ratio(_Ratios(40.0)) == MAX_RATIO == 8.0


def test_a_ratio_accessor_that_raises_is_stepped_over(qapp):
    """A deleted C++ object must not take the render down with it.

    ``devicePixelRatioF()`` on a widget whose C++ half Qt has already freed
    raises ``RuntimeError: Internal C++ object already deleted``, and that
    happens routinely while a screen is being torn down or a dialog closed
    mid-render. The picture should still be drawn at whatever ratio the next
    source can supply.
    """
    class _Half(_Ratios):
        def devicePixelRatioF(self):           # noqa: N802 - Qt spelling
            self.asked.append("devicePixelRatioF")
            raise RuntimeError("Internal C++ object already deleted")

    target = _Half(0.0, 3.0)
    assert device_ratio(target) == 3.0
    assert target.asked == ["devicePixelRatioF", "devicePixelRatio"]

    # the opposite half: when nothing raises, the first accessor answers.
    assert device_ratio(_Ratios(3.0, 9.9)) == 3.0


# ---------------------------------------------------------------------------
# a widget whose screen is not known yet
# ---------------------------------------------------------------------------

def test_a_widget_with_no_screen_yet_is_asked_for_its_window(qapp):
    """A picture built before the widget is shown still gets the right ratio.

    Icons and logos are scaled while a page is being constructed, which is
    before the widget has been put on a screen -- ``screen()`` is ``None`` or
    the screen has no ratio at that moment. The top-level window does know,
    because it is the thing that was placed. If the lookup stopped at the
    screen, every picture built during construction would render at 1x and
    arrive on a retina panel already blurred.
    """
    window = _Ratios(2.0)

    class _NotShownYet:
        def screen(self):
            return None

        def window(self):
            return window

    assert device_ratio(_NotShownYet()) == 2.0
    assert window.asked == ["devicePixelRatioF"]

    # the opposite half: once the widget IS on a screen, the screen answers
    # and the window is never consulted.
    screen = _Ratios(3.0)
    later_window = _Ratios(2.0)

    class _Shown:
        def screen(self):
            return screen

        def window(self):
            return later_window

    assert device_ratio(_Shown()) == 3.0
    assert later_window.asked == []


def test_a_screen_accessor_that_raises_falls_through_to_the_window(qapp):
    """Losing the screen mid-teardown must not lose the ratio.

    ``widget.screen()`` raises once Qt has deleted the C++ widget, which is
    exactly when a queued redraw arrives. Asking the window next keeps that
    redraw at the right density instead of dropping to 1x.
    """
    window = _Ratios(2.5)

    class _Deleted:
        def screen(self):
            raise RuntimeError("Internal C++ object already deleted")

        def window(self):
            return window

    assert device_ratio(_Deleted()) == 2.5
    assert window.asked == ["devicePixelRatioF"]


# ---------------------------------------------------------------------------
# an application with no primary screen
# ---------------------------------------------------------------------------

def test_no_primary_screen_means_an_ordinary_display_not_a_crash(monkeypatch):
    """Headless and half-built applications must still get a number back.

    ``QGuiApplication.primaryScreen()`` is ``None`` before the platform
    plugin has finished starting and on a machine with no display at all --
    the batch/CLI runs that still build pixmaps for saved figures. The only
    safe answer is 1.0: the pre-existing behaviour of a bare ``.scaled()``.
    Returning ``None`` or raising here would abort a render in a code path
    that has no display to report it on.
    """
    class _App:
        def __init__(self, screen):
            self._screen = screen

        def primaryScreen(self):               # noqa: N802 - Qt spelling
            return self._screen

    class _Fake:
        def __init__(self, app):
            self._app = app

        def instance(self):
            return self._app

    monkeypatch.setattr(hidpi, "QGuiApplication", _Fake(_App(None)))
    assert device_ratio(None) == 1.0
    assert device_ratio(_NoRatioAtAll()) == 1.0

    # the opposite half: the SAME call with a primary screen that does have a
    # ratio hands that ratio back, so the 1.0 above is the fall-through and
    # not a lookup that never happened.
    monkeypatch.setattr(hidpi, "QGuiApplication", _Fake(_App(_Ratios(2.0))))
    assert device_ratio(None) == 2.0

    # and no application at all is 1.0 as well
    monkeypatch.setattr(hidpi, "QGuiApplication", _Fake(None))
    assert device_ratio(None) == 1.0


# ---------------------------------------------------------------------------
# a picture that scales to nothing
# ---------------------------------------------------------------------------

def test_a_picture_scaled_to_nothing_carries_no_device_pixel_ratio(qapp):
    """A zero-sized render must come back inert, not mislabelled.

    A widget that has not been laid out yet reports a size of 0, and the
    figure strip scales its thumbnails to whatever width it currently has.
    ``QPixmap.scaled(0, 0)`` gives a null pixmap; stamping a ratio of 2 onto
    it would claim it occupies half of nothing, and Qt warns when asked to
    set a device pixel ratio on a null pixmap. The picture stays null, and
    the caller's own ``isNull()`` check still means what it meant.
    """
    source = _pixmap(400, 300)
    target = _Ratios(2.0)

    empty = scaled_for(source, target, 0)
    assert empty.isNull() is True
    assert (empty.width(), empty.height()) == (0, 0)
    assert empty.devicePixelRatio() == 1.0
    assert logical_size(empty) == QSize(0, 0)

    # the opposite half: the same source and the same target at a real size
    # DOES get the ratio stamped on, which is what makes the branch above a
    # branch rather than dead code.
    real = scaled_for(source, target, 64)
    assert (real.width(), real.height()) == (128, 96)
    assert real.devicePixelRatio() == 2.0
    assert logical_size(real) == QSize(64, 48)


def test_a_source_that_cannot_carry_a_ratio_is_scaled_and_returned(qapp):
    """Anything with ``.scaled()`` may be handed in, ratio-capable or not.

    ``scaled_for`` is documented as taking a ``QPixmap`` or a ``QImage``, but
    it is called from drawing code that also passes icon renderers and test
    doubles. If it insisted on ``setDevicePixelRatio`` existing, those call
    sites would raise ``AttributeError`` inside a paint path instead of
    simply getting a scaled picture back.
    """
    class _Plain:
        """A picture-like object with no device pixel ratio to set."""

        def __init__(self):
            self.scaled_to = None

        def isNull(self):                      # noqa: N802 - Qt spelling
            return False

        def scaled(self, width, height, aspect, mode):
            self.scaled_to = (width, height, aspect, mode)
            return self

    plain = _Plain()
    out = scaled_for(plain, _Ratios(2.0), (30, 20))
    assert out is plain
    assert plain.scaled_to == (60, 40, Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)

    # the opposite half: a real QPixmap through the same call gets the ratio.
    stamped = scaled_for(_pixmap(80, 80), _Ratios(2.0), (30, 20))
    assert stamped.devicePixelRatio() == 2.0


# ---------------------------------------------------------------------------
# a picture with no deviceIndependentSize()
# ---------------------------------------------------------------------------

def test_a_picture_without_device_independent_size_is_still_measured(qapp):
    """The size a picture OCCUPIES must be right for every picture object.

    ``deviceIndependentSize()`` arrived in Qt 6.2; ``QImage`` and the icon
    and test-double objects the drawing code also handles do not all have it.
    Everything that centres a picture, hit-tests a click on it or sizes a
    label round it measures in widget coordinates, so a fallback that
    returned raw device pixels would put every one of those out by the ratio
    -- a 2x thumbnail would be centred as if it were twice its real size.
    """
    class _OldPicture:
        """A 2x picture from before ``deviceIndependentSize`` existed."""

        def isNull(self):                      # noqa: N802 - Qt spelling
            return False

        def width(self):
            return 200

        def height(self):
            return 100

        def devicePixelRatio(self):            # noqa: N802 - Qt spelling
            return 2.0

    assert logical_size(_OldPicture()) == QSize(100, 50)

    class _NoRatioPicture:
        """The same pixels from an object that reports no ratio: 1:1."""

        def isNull(self):                      # noqa: N802 - Qt spelling
            return False

        def width(self):
            return 200

        def height(self):
            return 100

    assert logical_size(_NoRatioPicture()) == QSize(200, 100)

    # the opposite half: a real pixmap DOES answer deviceIndependentSize, and
    # the two routes agree about the same 2x picture.
    modern = scaled_for(_pixmap(400, 400), _Ratios(2.0), 100)
    assert hasattr(modern, "deviceIndependentSize")
    assert logical_size(modern) == QSize(100, 100)
    assert (modern.width(), modern.height()) == (200, 200)


def test_a_broken_device_independent_size_falls_back_to_the_ratio(qapp):
    """A picture whose own measurement raises still reports a usable size.

    ``deviceIndependentSize()`` on a pixmap whose C++ half has been freed
    raises, and the layout asking for the size is mid-paint. Dividing the
    device pixels by the ratio gets the same answer without the exception,
    so the panel lays out instead of falling over.
    """
    class _Broken:
        def isNull(self):                      # noqa: N802 - Qt spelling
            return False

        def deviceIndependentSize(self):       # noqa: N802 - Qt spelling
            raise RuntimeError("Internal C++ object already deleted")

        def width(self):
            return 300

        def height(self):
            return 150

        def devicePixelRatio(self):            # noqa: N802 - Qt spelling
            return 3.0

    assert logical_size(_Broken()) == QSize(100, 50)

    # the opposite half: when the measurement works it is what is used, and
    # it is NOT the width/ratio arithmetic -- these numbers disagree on
    # purpose, so only the accessor's answer can produce 7 x 5.
    class _Working(_Broken):
        def deviceIndependentSize(self):       # noqa: N802 - Qt spelling
            return QSize(7, 5)

    assert logical_size(_Working()) == QSize(7, 5)
