"""The fractal's numbers, importable without Qt, numba or vispy.

`preferences` needs the defaults and the clamp to read a stored value, and
`preferences` is imported before the first window exists. Reaching into
`widgets.fractal_travel` for them would pull numpy and numba onto the launch
path for every ordinary session -- a module nothing but spaceout uses.
"""
from __future__ import annotations

from typing import Final

#: Reference defaults for both renderers. `auto` takes the GPU when vispy is
#: importable and the CPU when it is not.
#: Which fractal family. Kept here with the rest so `preferences` can read
#: it without importing the widget, and therefore without numba.
#: What spaceout draws unless the user says otherwise.
#:
#: THE ORBIT FOLD, asked for on 2026-09-01. It was the Mandelbrot, which
#: is the one pattern with no CPU renderer -- so on every machine without
#: a usable GPU the default was a pattern that could not be drawn, and
#: `pattern_for_this_machine` quietly substituted this one anyway. Naming
#: it here makes the default the same everywhere instead of depending on
#: what the machine turned out to have.
DEFAULT_PATTERN: str = "orbit"

#: What to draw when the Mandelbrot cannot be.
FALLBACK_PATTERN: str = "orbit"

#: EVERY PATTERN, in the order they are offered. Lives here rather than
#: in `widgets.fractal_travel` because `preferences` needs it at import
#: time and importing that widget pulls numba onto the launch path.
#:
#: There used to be a second copy in `preferences.FRACTAL_PATTERNS`, and
#: adding a pattern to one and not the other meant the new pattern was
#: selectable in code and absent from the Preferences combo -- which is
#: how it was found.
PATTERNS: Final[tuple] = ("orbit", "orbit_gpu", "cascade", "space",
                          "mandelbrot")

#: Patterns with NO CPU renderer. A machine that cannot give them a GL
#: context gets FALLBACK_PATTERN instead of a backdrop that draws
#: nothing -- which is how the Mandelbrot came to be a default that
#: could not be drawn.
GPU_ONLY_PATTERNS: Final[frozenset] = frozenset({"mandelbrot", "orbit_gpu"})

DEFAULT_BACKEND: Final[str] = "auto"
#: Samples per pixel per side.
#:
#: ONE, changed 2026-09-01: "default to computationally easy settings
#: like supersampling 1 and scale 0.5 and speed 1". Supersampling SQUARES
#: the cost -- 2 is four samples per pixel, not two -- so on a backdrop
#: it is the single most expensive setting to have on by default.
#:
#: The picture is softer. That is the right trade for something drawn
#: behind the interface: anyone who wants it sharper can raise it, and
#: the setting now says what it costs.
DEFAULT_SUPERSAMPLING: Final[int] = 1
DEFAULT_QUALITY: Final[str] = "auto"
#: Render scale, asked for on 2026-09-01. Half resolution: the backdrop is
#: behind the interface and a full-resolution one buys sharpness nobody
#: reads at the cost of frames everybody feels.
DEFAULT_SCALE: Final[float] = 0.5
DEFAULT_SPEED: Final[float] = 1.0
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
