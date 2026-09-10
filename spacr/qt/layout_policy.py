"""The measured window width a module needs, and what to open on first run.

A window that opens too narrow cuts off the right-hand side of a module's
settings, and the width that avoids it is not something a resolution can
answer on its own: the same pixel dimensions describe a low-DPI monitor, a
Retina panel at 2x, and an accessibility-scaled desktop.

WHAT THIS MODULE IS. The reader of a GENERATED artifact,
``spacr/resources/layout_policy.json``, produced by
``tools/measure_the_layout_matrix.py``. Every number in it was measured by
building each module offscreen in a container that cannot grow and reading
what was clipped -- the same rule the package's text-fit sweep asserts,
not a second opinion about it.

WHAT IT IS NOT. Not telemetry. "Capture" here means reading Qt screen
metrics into local preferences, not sending hardware details anywhere.
Nothing this module reads or writes leaves the machine.

IT IMPORTS NOTHING HEAVY, which the item requires of the artifact's
loader: ``json`` and ``importlib.resources``, no Qt, no numpy. The Qt
metrics arrive as ARGUMENTS, so the policy can be read and tested on a
machine with no display at all.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

#: The schema this reader understands. A file claiming a higher number is
#: not read -- a policy from the future is more dangerous than no policy,
#: because it would be interpreted with rules that were not written yet.
SUPPORTED_SCHEMA = 1

#: What every caller gets when there is no usable artifact.
#:
#: 1200x850 is the size the text-fitting sweep builds every screen at, and
#: every declared app has come up clean there in 192 combinations. So it is
#: not a guess: it is the one width the package already has evidence for,
#: and it is the right thing to fall back to when the measured file is
#: missing from a wheel or is from a schema nobody here can read.
FALLBACK: Tuple[int, int] = (1200, 850)

_CACHE: Optional[dict] = None


def read_policy(refresh: bool = False) -> dict:
    """The bundled policy, or ``{}``.

    :param refresh: re-read the file instead of answering from the cache.
    :returns: the artifact as a dict, or an empty one on ANY failure.

    Degrades to empty rather than raising, for the reason the release-notes
    panel does: an application must not fail to open a window because a
    resource file is absent from a wheel.
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    try:
        from importlib.resources import files

        raw = (files("spacr.resources") / "layout_policy.json")
        data = json.loads(raw.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
        elif int(data.get("schema", 0)) > SUPPORTED_SCHEMA:
            data = {}
    except Exception:                                        # noqa: BLE001
        data = {}
    _CACHE = data
    return data


def minimum_width_for(font_scale: float, policy: Optional[dict] = None
                      ) -> Optional[int]:
    """The widest measured requirement at ``font_scale``.

    :param font_scale: the preference, 1.0 being 100 %.
    :param policy: an artifact to read instead of the bundled one.
    :returns: logical pixels, or None when nothing was measured.

    THE WORST CASE ACROSS MODULES, NOT THE MEDIAN. A window opens once and
    every module is reached from inside it, so a width that suits most of
    them still cuts off the widest one -- which is the report.

    BETWEEN TWO MEASURED SCALES IT TAKES THE HIGHER, because the artifact
    measures a ladder and the requirement rises with the scale. Rounding
    toward the smaller one would open a window that fits the scale below
    the user's.
    """
    table = (policy if policy is not None else read_policy()).get(
        "minimum_width") or {}
    if not table:
        return None
    try:
        wanted = float(font_scale)
    except (TypeError, ValueError):
        return None
    measured: Dict[float, int] = {}
    for key, value in table.items():
        try:
            measured[float(key)] = int(value)
        except (TypeError, ValueError):
            continue
    if not measured:
        return None
    at_or_above = [s for s in measured if s >= wanted]
    if at_or_above:
        return measured[min(at_or_above)]
    # Above every measured scale: the largest requirement there is, which
    # is the honest extrapolation -- it is a floor, and a floor from the
    # widest thing measured is better than one invented above it.
    return measured[max(measured)]


def recommended_window_size(available: Tuple[int, int], font_scale: float = 1.0,
                            policy: Optional[dict] = None
                            ) -> Tuple[int, int]:
    """What to open the main window at, given what the screen has.

    :param available: the screen's AVAILABLE geometry as ``(w, h)`` --
        available, not total, because a dock or a menu bar owns the
        difference and a window sized to the total opens partly underneath
        one.
    :param font_scale: the font-scale preference in force.
    :param policy: an artifact to read instead of the bundled one.
    :returns: ``(width, height)`` in logical pixels.

    CLAMPED TO THE SCREEN, ALWAYS, and that ordering is the whole contract:
    a measured requirement wider than the display is a statement that this
    module cannot be shown whole here, not permission to open a window
    off the edge of it. The user can still scroll; a window they cannot
    reach the corner of to resize is a worse answer.
    """
    try:
        screen_w, screen_h = int(available[0]), int(available[1])
    except Exception:                                        # noqa: BLE001
        return FALLBACK
    if screen_w <= 0 or screen_h <= 0:
        return FALLBACK
    wanted = minimum_width_for(font_scale, policy) or FALLBACK[0]
    height = FALLBACK[1]
    try:
        height = int((policy if policy is not None
                      else read_policy()).get("matrix", {}).get(
                          "height", FALLBACK[1]))
    except Exception:                                        # noqa: BLE001
        height = FALLBACK[1]
    return (max(640, min(int(wanted), screen_w)),
            max(480, min(int(height), screen_h)))


def why(font_scale: float = 1.0, policy: Optional[dict] = None) -> str:
    """One sentence naming the evidence, for a log line or a hint.

    A recommendation a user cannot trace is one they cannot argue with,
    and this item's whole history is a layout decision nobody could check.
    """
    data = policy if policy is not None else read_policy()
    if not data:
        return ("no measured layout policy is bundled; using the size the "
                "text-fit sweep builds every screen at")
    wanted = minimum_width_for(font_scale, data)
    matrix = data.get("matrix", {})
    return (f"{wanted} px is the widest any of "
            f"{len(matrix.get('apps', []))} modules needed at font scale "
            f"{font_scale}, measured over "
            f"{len(matrix.get('locales', []))} locales")
