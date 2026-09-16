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

import contextlib
import os
from typing import List

#: Body text is Light; titles are Regular.
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
    """Every bundled TrueType or OpenType face.

    :returns: Absolute paths in filename order. A missing or unreadable font
        directory returns an empty list rather than raising -- a figure drawn
        in the wrong font is a blemish, and never a reason for a plot not to
        appear.
    """
    directory = font_dir()
    if not os.path.isdir(directory):
        return []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [os.path.join(directory, name)
            for name in names
            if name.lower().endswith((".ttf", ".otf"))]


def use_open_sans_for_figures() -> bool:
    """Register the bundled faces with Matplotlib's font manager.

    Idempotent and safe to call before any figure is drawn.  This deliberately
    does not alter process-wide ``rcParams``: the two house-style helpers put
    Open Sans in their scoped parameter dictionaries instead.

    :returns: whether Open Sans is now resolvable by name.
    """
    global _registered
    if _registered:
        return _resolved

    try:
        from matplotlib import font_manager
    except Exception:
        return False

    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    if FAMILY not in available:
        for path in bundled_faces():
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                continue
        try:
            available = {f.name for f in font_manager.fontManager.ttflist}
        except Exception:
            available = set()

    _registered = True
    if FAMILY not in available:
        return False

    globals()["_resolved"] = True
    return True


# ---------------------------------------------------------------------------
# Item 291: Open Sans as matplotlib's DEFAULT -- in the app and in pipeline
# runs, and nowhere else.
#
# Decided 2026-09-15, verbatim: "Global in the app only (Recommended)". The
# option read: "The spaCR GUI and spaCR's pipeline runs set Open Sans as
# matplotlib's default; plain `import spacr` in a notebook leaves the user's
# matplotlib alone."
#
# So nothing below runs on import (`use_open_sans_for_figures` above still
# changes no rcParam). The ENTRY POINTS call it: `spacr.qt.run`, which every
# GUI console script goes through; `spacr.cli.cmd_run`, which is `spacr-run`
# and also what every batch-queue job executes; `spacr-repro`;
# `spacr-tutorial`; and the parameter sweep's worker processes.
#
# HELD FOR THE RUN, NOT WRITTEN BARE. In those processes the run IS the
# process, so an `rc_context` held for the whole run is the process default.
# But `spacr.cli.main` and `spacr.batch.inprocess_runner` are also called
# in-process -- by the test suite, and by a frozen build -- and a bare
# `rcParams.update` there would restyle every later figure of whoever called
# it. The context hands the caller its matplotlib back when the run ends.
# ---------------------------------------------------------------------------

#: Present in the environment for the length of such a run. A worker process
#: the run starts -- a `spawn` pool, or a `python -m` child -- begins with
#: matplotlib's stock rcParams, and this is how it can tell that a run started
#: it, while a worker started from a notebook can tell that none did.
_RUN_MARKER = "SPACR_FIGURES_IN_OPEN_SANS"

#: What follows Open Sans in ``font.family``. DejaVu Sans ships inside
#: matplotlib, so it always resolves -- no "findfont: Font family ... not
#: found" line per drawn string -- and matplotlib's per-glyph fallback draws a
#: character Open Sans lacks (an arrow, a maths sign) from it instead of as an
#: empty box.
_FALLBACK_FAMILIES = ("DejaVu Sans",)


def _default_font_params() -> dict:
    """The rcParams that make the bundled Open Sans matplotlib's default.

    :returns: an empty dict when matplotlib is missing or the bundled faces
        cannot be registered. The stock default then stays: a figure in the
        wrong face is a blemish, a run that cannot draw at all is not.
    """
    try:
        import matplotlib
    except Exception:
        return {}
    if not use_open_sans_for_figures():
        return {}
    try:
        others = [name for name in matplotlib.rcParams["font.sans-serif"]
                  if name != FAMILY]
    except Exception:
        others = []
    return {
        "font.family": [FAMILY, *_FALLBACK_FAMILIES],
        # A caller that names the generic family itself,
        # `fontfamily="sans-serif"`, gets Open Sans as well.
        "font.sans-serif": [FAMILY, *others],
    }


@contextlib.contextmanager
def _open_sans_is_the_default():
    """Hold Open Sans as matplotlib's default for the length of a run.

    :returns: a context manager yielding whether Open Sans was applied. It
        never raises on its own account: when the face cannot be applied the
        run goes ahead in matplotlib's stock default.
    """
    params = _default_font_params()
    held = contextlib.ExitStack()
    applied = False
    if params:
        try:
            import matplotlib

            held.enter_context(matplotlib.rc_context(params))
            applied = True
        except Exception:
            held.close()
    if not applied:
        yield False
        return

    previous = os.environ.get(_RUN_MARKER)
    os.environ[_RUN_MARKER] = "1"
    try:
        with held:
            yield True
    finally:
        if previous is None:
            os.environ.pop(_RUN_MARKER, None)
        else:
            os.environ[_RUN_MARKER] = previous


def _open_sans_if_a_run_started_this():
    """In a worker process: the run's default if a run started it, else none.

    :returns: :func:`_open_sans_is_the_default` when :data:`_RUN_MARKER` says
        an app or pipeline run started this process, and a context manager
        that changes nothing otherwise.
    """
    if os.environ.get(_RUN_MARKER) != "1":
        return contextlib.nullcontext(False)
    return _open_sans_is_the_default()
