"""The font figures are drawn in: the one that ships with spaCR.

Matplotlib's default is DejaVu Sans, and asking for "Helvetica" on a Linux
machine that has no Helvetica silently falls back to it too -- so figures came
out in a different face from the interface around them, and in a different
face on each contributor's machine.

spaCR ships Open Sans (``spacr/resources/font/open_sans``), so the face is
always present. REGISTERING IT IS THE POINT: setting the rcParam alone names a
family matplotlib may not have, and a name it cannot resolve is a silent
fallback rather than an error. Adding the bundled files to the font manager is
what makes the name resolve on a machine where Open Sans was never installed.
"""
from __future__ import annotations

import os
from typing import List

#: Body text is Light; titles are Regular. Asked for 2026-08-28.
BODY_WEIGHT = "light"
TITLE_WEIGHT = "regular"

#: The family name inside the bundled files.
FAMILY = "Open Sans"

_registered = False

#: What the last completed registration concluded, so a repeat call answers
#: without re-deriving it.
_resolved = False


def font_dir() -> str:
    """The directory holding the bundled static faces.

    :returns: an absolute path. It exists in an installed build too, because
        the fonts are package data.
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "resources", "font", "open_sans", "static")


def bundled_faces() -> List[str]:
    """Every bundled ``.ttf``.

    :returns: absolute paths, empty when the directory is missing rather
        than raising -- a figure drawn in the wrong font is a blemish, and
        never a reason for a plot not to appear.
    """
    directory = font_dir()
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if name.lower().endswith((".ttf", ".otf"))]


def use_open_sans_for_figures() -> bool:
    """Register the bundled faces and make them matplotlib's default.

    Idempotent, and safe to call before any figure is drawn.

    :returns: whether Open Sans is now resolvable by name.
    """
    global _registered
    if _registered:
        # ALREADY DONE THIS PROCESS. The check below is a set comprehension
        # over every font matplotlib knows, which is not free either, and
        # this function is called before every figure is styled.
        return _resolved

    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:
        return False

    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    # ASK BEFORE ADDING. `addfont` is expensive -- it reads the file, builds
    # a FontProperties and resolves alternative family names, and measured on
    # the Mask screen the eight bundled faces cost 23 SECONDS of a 13-second
    # module open, because this ran while the settings panel was being built.
    # A machine that already has Open Sans installed needs none of it.
    if FAMILY not in available:
        for path in bundled_faces():
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                # One unreadable face must not cost the other seven.
                continue
        try:
            available = {f.name for f in font_manager.fontManager.ttflist}
        except Exception:
            available = set()

    _registered = True
    if FAMILY not in available:
        return False

    # KEEP A FALLBACK CHAIN. The rcParam is a list for a reason: if a future
    # build ships without the font data, a figure should still be drawn.
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        FAMILY, "Helvetica", "Arial", "DejaVu Sans"]
    globals()["_resolved"] = True
    return True
