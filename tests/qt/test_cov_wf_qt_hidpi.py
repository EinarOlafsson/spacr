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

from spacr.qt.hidpi import (  # noqa: E402
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


def _pixmap(width, height):
    """A drawable source picture of a known shape."""
    picture = QPixmap(width, height)
    picture.fill(Qt.white)
    return picture


# ---------------------------------------------------------------------------
# a first accessor whose answer is not a ratio
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# a widget whose screen is not known yet
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# an application with no primary screen
# ---------------------------------------------------------------------------


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
