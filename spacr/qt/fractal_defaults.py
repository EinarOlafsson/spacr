"""The fractal's numbers, importable without Qt, numba or vispy.

`preferences` needs the defaults and the clamp to read a stored value, and
`preferences` is imported before the first window exists. Reaching into
`widgets.fractal_travel` for them would pull numpy and numba onto the launch
path for every ordinary session -- a module nothing but spaceout uses.
"""
from __future__ import annotations

from typing import Final

#: The maintainer's own two command lines, as one set of numbers. `auto`
#: takes the GPU when vispy is importable and the CPU when it is not.
#: Which fractal family. Kept here with the rest so `preferences` can read
#: it without importing the widget, and therefore without numba.
#: What spaceout draws unless the user says otherwise.
#:
#: MANDELBROT, asked for 2026-08-28 -- but only where a GPU can draw it. It
#: is the one pattern with no CPU renderer, so `pattern_for_this_machine`
#: falls back to the orbit fold, which has one and is the cheapest of the
#: three that do.
DEFAULT_PATTERN: str = "mandelbrot"

#: What to draw when the Mandelbrot cannot be.
FALLBACK_PATTERN: str = "orbit"

DEFAULT_BACKEND: Final[str] = "auto"
DEFAULT_QUALITY: Final[str] = "auto"
DEFAULT_SCALE: Final[float] = 1.0
DEFAULT_SPEED: Final[float] = 4.0
DEFAULT_DREAM: Final[float] = 1.5
DEFAULT_VARIABLE_SPEED: Final[bool] = False
#: The pointer pulls the pattern toward it; a click shoves it away.
DEFAULT_FOLLOW_POINTER: Final[bool] = True
#: The bounds variable speed sweeps between. They bracket DEFAULT_SPEED, so
#: turning it on changes the RANGE and not the average pace.
DEFAULT_SPEED_MIN: Final[float] = 2.0
DEFAULT_SPEED_MAX: Final[float] = 6.0
#: Seconds for one full sweep, slow to fast and back -- the "how gradually"
#: control. A larger number is a slower change, not a slower fractal.
DEFAULT_SPEED_PERIOD: Final[float] = 41.0


def clamp(value: float, low: float, high: float) -> float:
    """``value`` held inside ``[low, high]``."""
    return low if value < low else high if value > high else value


#: How far that pull reaches, in the -1..1 coordinate space the patterns
#: work in: 1.0 reaches the widget's short edge.
DEFAULT_POINTER_SIZE: float = 1.0

#: How hard it pulls. 0 is off however the switch is set; above 1
#: exaggerates.
DEFAULT_POINTER_STRENGTH: float = 1.0
