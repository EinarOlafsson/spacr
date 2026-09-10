"""Measure the width every module needs. Skipped unless asked (359, part 3).

INSTRUCTION 359 asks for "a deterministic offscreen layout harness" run
over a matrix of geometries, scales and locales, whose results are
generated into a versioned policy artifact that ships in the wheel -- so
first launch can open a window wide enough that "the right side of its
settings is not cut off" instead of guessing.

WHY THE MEASUREMENT LIVES IN A TEST FILE AND NOT IN `tools/`. It was
written as a standalone script first, with a hand-rolled qtbot, and it
DISAGREED WITH THE SWEEP: `tests/qt/test_the_text_fits_sweep.py` reports
zero clipped captions for Regression in English at 1200 px, and the script
reported two -- on every module, including ones that do not have the
button it named. The difference was the fixtures. `tests/qt/conftest.py`
carries fifteen autouse fixtures and a session one that fills
`theme._WIDGET_QSS`, and a harness that reproduces some of them measures a
screen no user has.

A GENERATOR THAT DISAGREES WITH THE SUITE PRODUCES A POLICY THE SUITE
CANNOT DEFEND, and the disagreement is invisible until a user meets it. So
the measurement runs where the fixtures are, and the number written to the
artifact is the number this suite asserts.

It is SKIPPED unless ``SPACR_MEASURE_LAYOUT`` names an output path, so an
ordinary run never pays for it. ``tools/measure_the_layout_matrix.py`` is
the driver that sets it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QAbstractScrollArea  # noqa: E402

from .test_the_text_fits_sweep import (LOCALES, SCALES,  # noqa: E402
                                       SCREENS, _offenders,
                                       at_font_scale,
                                       screen_in_a_container)

#: RE-EXPORTED SO PYTEST CAN SEE IT. `at_font_scale` is the sweep's own
#: fixture, and the pair it applies -- the font-scale PREFERENCE, which
#: sizes every widget through `scaled_px`, and the STYLESHEET, which sizes
#: every glyph -- is the difference between measuring a screen a user has
#: and one nobody does. Importing it is deliberate: a second copy here
#: could drift from the one the sweep asserts under.
at_font_scale = at_font_scale

#: Where to write, or nothing and the whole file is skipped.
OUT = os.environ.get("SPACR_MEASURE_LAYOUT", "")

#: The artifact's schema. Bumped when a READER would have to change; adding
#: a row or a column of the matrix does not bump it.
SCHEMA = 1

#: Rungs, in logical pixels.
#:
#: A LADDER, NOT A BISECTION. Bisecting assumes the predicate is monotonic
#: in width and layout is not: a form that reflows at one width can be
#: clean above and below it and clipped between. A ladder reports the first
#: clean rung and how many it tried, which is a claim a reader can check.
#:
#: It starts well below any laptop because the number wanted is where a
#: module STOPS fitting, and a ladder whose first rung always passes
#: measures nothing. 2560 is the last because a module needing more than
#: that has a defect the policy cannot paper over, and the artifact should
#: say so rather than round up.
WIDTHS = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
          1600, 1800, 2100, 2560)

#: Held fixed. The report was about the RIGHT side being cut off, and one
#: measured axis is worth more than two entangled ones.
HEIGHT = 850


def _sideways_scrollers(screen):
    """Whether the SETTINGS column is holding more than it can show.

    THE SIGNAL THIS ITEM IS ACTUALLY ABOUT, and it is not the one
    instruction 350 measures. Asked with clipped captions alone, every
    module came up clean at the narrowest rung tried -- 700 px -- because
    a settings form does not clip its captions when it runs out of room:
    it puts the right-hand side past the edge and offers a horizontal
    scrollbar. That IS the report -- "widening some settings containers did
    not widen their fields", and "every module must initially open wide
    enough that the right side of its settings is not cut off" -- and 359's
    part 3 names it in the list of things to record: "off-window geometry
    ... scrollbar visibility".

    ONE NAMED SCROLL AREA, NOT EVERY SCROLL AREA ON THE SCREEN, and this
    is the false positive that would otherwise sink the number. A table or
    a plot canvas has an enormous horizontal `sizeHint` BY DESIGN and
    scrolling it sideways is correct behaviour; counting those would
    report every module as needing a display nobody sells. The settings
    column is the one whose right edge the user was talking about, and it
    carries `SETTINGS_PANEL_NAME` precisely so it can be addressed.

    Asked as CONTENT AGAINST VIEWPORT rather than as
    ``horizontalScrollBar().isVisible()``, because a scrollbar with an
    always-off policy hides the symptom without fixing it: the content is
    still wider than the column and the right edge is still gone.

    :returns: sentences naming the content width and the viewport's.
    """
    from spacr.qt.screens.app_screen import SETTINGS_PANEL_NAME

    found = []
    for area in screen.findChildren(QAbstractScrollArea):
        if area.objectName() != SETTINGS_PANEL_NAME or not area.isVisible():
            continue
        inner = area.widget()
        if inner is None:
            continue
        wanted = max(inner.sizeHint().width(),
                     inner.minimumSizeHint().width())
        have = area.viewport().width()
        if have >= LAID_OUT_PX and wanted > have:
            found.append(f"the settings column holds {wanted} px of "
                         f"content in a {have} px viewport")
    return found


#: A viewport narrower than this has not been laid out, whatever it says.
LAID_OUT_PX = 8

pytestmark = pytest.mark.skipif(
    not OUT, reason="set SPACR_MEASURE_LAYOUT=<path> to measure the matrix")


def _narrowest_clean(qtbot, app_key: str) -> dict:
    """The first rung at which nothing on ``app_key`` is clipped."""
    worst: list = []
    for index, width in enumerate(WIDTHS, start=1):
        host, screen = screen_in_a_container(qtbot, app_key, width=width,
                                             height=HEIGHT)
        try:
            offenders = _offenders(screen) + _sideways_scrollers(screen)
        finally:
            _discard(host)
        if not offenders:
            return {"width": width, "probes": index, "worst": []}
        worst = offenders
    return {"width": None, "probes": len(WIDTHS),
            "worst": [str(line) for line in worst[:4]]}


def _discard(host) -> None:
    """Close a probed screen and let its C++ half go, now.

    THE MEASUREMENT IS ONE TEST AND IT BUILDS THOUSANDS OF SCREENS -- up
    to fifteen rungs for each of 24 modules, four locales and two scales.
    pytest-qt closes and `deleteLater`s the widgets it was given, but only
    at TEARDOWN, and `deleteLater` merely posts an event: inside a single
    long test nothing ever spins the loop far enough to deliver it. The
    first full run reached the 12 GB test-guard ceiling three quarters of
    the way through and was ended with no artifact written.

    So each probe is disposed of as it finishes, and the loop is spun
    afterwards so the deletion is actually delivered rather than queued.
    """
    from PySide6.QtWidgets import QApplication

    try:
        host.close()
        host.setParent(None)
        host.deleteLater()
    except Exception:                                        # noqa: BLE001
        pass
    app = QApplication.instance()
    if app is not None:
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def _worst_per_scale(rows: dict) -> dict:
    """The widest requirement at each scale, as ``{scale: width}``.

    A case that never came up clean contributes the widest rung tried
    rather than being dropped, so an unmeasurable module RAISES the
    recommendation instead of quietly lowering it.
    """
    worst: dict = {}
    for key, row in rows.items():
        scale = key.rsplit("|", 1)[-1]
        width = row["width"] or max(WIDTHS)
        worst[scale] = max(worst.get(scale, 0), int(width))
    return worst


def test_the_matrix_is_measured_and_written(qtbot, qapp, at_font_scale,
                                            monkeypatch):
    """Walk the matrix and write the policy artifact.

    ONE TEST, NOT A PARAMETRISATION, and deliberately: the artifact is a
    single document about a whole matrix, and a parametrised run would
    have to accumulate through module state that a `-k` selection could
    silently leave half full. A partial artifact that looks whole is the
    failure this item is most exposed to, because nobody reads a policy
    file -- they read the window it opens.
    """
    started = time.perf_counter()
    rows: dict = {}
    for scale in SCALES:
        at_font_scale(scale)
        for locale in LOCALES:
            monkeypatch.setenv(_language_env(), locale)
            qapp.processEvents()
            for app_key in SCREENS:
                row = _narrowest_clean(qtbot, app_key)
                rows[f"{app_key}|{locale}|{scale}"] = row
                print(f"{app_key:20} {locale} {scale:>4}  "
                      f"{row['width'] or 'NONE'}", flush=True)

    clean = [row["width"] for row in rows.values() if row["width"]]
    artifact = {
        "schema": SCHEMA,
        "measured_seconds": round(time.perf_counter() - started, 1),
        "matrix": {"apps": list(SCREENS), "locales": list(LOCALES),
                   "scales": list(SCALES), "widths": list(WIDTHS),
                   "height": HEIGHT},
        # THE WORST CASE PER SCALE, not per module. A window opens once and
        # every module is reached from inside it, so a width that suits the
        # median module still cuts off the widest one -- which is the
        # report this item exists to answer.
        "minimum_width": _worst_per_scale(rows),
        "rows": rows,
    }
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")

    # A MEASUREMENT THAT CANNOT REACH AN ANSWER IS NOT ONE. A case that
    # never came up clean means the ladder does not reach far enough, or
    # that module cannot be shown whole on any display this policy could
    # recommend -- either way the artifact must not pretend otherwise, and
    # `_worst_per_scale` has already raised the recommendation to the top
    # rung for it.
    unfixable = {key: row["worst"][:1] for key, row in rows.items()
                 if row["width"] is None}
    assert not unfixable, (
        "no rung up to 2560 px showed these whole, so the policy cannot "
        f"recommend a width that does: {unfixable}")
    assert len(clean) == len(rows)


def _language_env() -> str:
    """The variable the package reads a locale from."""
    from spacr.qt import i18n

    return i18n.ENV_LANGUAGE
