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
DEFAULT_BACKEND: Final[str] = "auto"
DEFAULT_QUALITY: Final[str] = "auto"
DEFAULT_SCALE: Final[float] = 1.0
DEFAULT_SPEED: Final[float] = 4.0
DEFAULT_DREAM: Final[float] = 1.5
DEFAULT_VARIABLE_SPEED: Final[bool] = False


def clamp(value: float, low: float, high: float) -> float:
    """``value`` held inside ``[low, high]``."""
    return low if value < low else high if value > high else value
