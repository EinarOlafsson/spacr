"""The ``spaceout`` backdrop is alive rather than looping, and still legible.

Six claims, and every one of them is measured on FRAMES AND PIXELS rather
than on the constants that produced them:

* the buffer this engine shades into is larger than the diffuse themes',
  and the guard trims it on a machine that cannot pay for it;
* the number of forms on screen CHANGES over a run, and a bud's pixels are
  the parent's pixels — it left the parent's rim carrying its constant;
* a form turns: the ring of pixels around its centre at one clock is the
  ring at another, rotated;
* the state sequence does not repeat on any period, measured against a
  deliberately periodic control that the same measurement catches at once;
* the palette's hue moves, and EVERY offset it can reach clears the contrast
  rules, the page separation and the scrims;
* the mark that follows the cursor oscillates under the dressing and is
  constant without it.

WHICH MARK. ``SetupCard`` paints a run of rim that is aimed by
:meth:`~spacr.qt.widgets.setup_card.SetupCard._aim_at_the_cursor`, which
reads the GLOBAL cursor position rather than waiting for mouse events — so
it tracks a pointer that has left the window, which is what makes it "the
little blue window thing that follows the mouse". It is the only mark in the
application that chases the pointer: ``availability_panel`` and
``hover_tooltip`` read the cursor too, but to decide whether it is inside a
region, not to draw anything at it. That is the one this suite measures and
the one that changed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spacr.qt import theme
from spacr.qt.widgets import ambient as amb
from spacr.qt.widgets.setup_card import (SPACEOUT_RIM_PERIOD, SetupCard)

#: The canvas every frame in this file is measured on.
W, H = 1280, 720

#: Samples round the ring the rotation is measured on, and where the ring
#: sits as a share of the buffer's short edge.
RING_SAMPLES = 180
RING_RADIUS = 0.30


@pytest.fixture
def dressed():
    """Run the test in the spaceout dressing, with the drift at zero.

    The mode and the drift are both process state and this suite is randomly
    ordered, so a leaked dressing would re-colour every test that ran after
    it and a leaked drift would move a palette another test was asserting on.
    """
    was = theme.spaceout_enabled()
    before = theme.spaceout_drift_seconds()
    theme.enable_spaceout()
    theme.set_spaceout_drift_seconds(0.0)
    yield
    theme.set_spaceout_drift_seconds(before)
    if not was:
        theme.disable_spaceout()


def _engine(seed: int = 5, **kwargs):
    return amb.make_engine(amb.SPACEOUT_THEME, amb.SPACEOUT_PALETTE,
                           "#101010", seed=seed, **kwargs)


def _frame(engine, width: int = W, height: int = H) -> np.ndarray:
    """One shaded frame as ``(h, w, 3)`` of 0..255.

    The buffer itself, at whatever size the guard has settled on — which is
    the picture the blit stretches onto the canvas, so it is the thing to
    measure.
    """
    image = engine.shade(width, height)
    raw = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4)
    return raw[:, :image.width(), :3].astype(np.int16).copy()


def _ring(field: np.ndarray, cx: float, cy: float, radius: float,
          samples: int = RING_SAMPLES) -> np.ndarray:
    """The mean brightness round a circle, as one sample per step of angle.

    A ring is the right shape to ask "did it turn" of: a rotation of the
    picture about that centre is a CIRCULAR SHIFT of this array and nothing
    else, so the question becomes "is some shift of it a better match than
    no shift", which is a number.
    """
    angles = np.arange(samples) * (2.0 * math.pi / samples)
    xs = np.clip((cx + radius * np.cos(angles)).astype(int),
                 0, field.shape[1] - 1)
    ys = np.clip((cy + radius * np.sin(angles)).astype(int),
                 0, field.shape[0] - 1)
    return field[ys, xs].astype(float).mean(axis=1)


def _turn_ratio(before: np.ndarray, after: np.ndarray) -> float:
    """How much better the best rotation matches than no rotation at all.

    Below 1.0 means the second ring is closer to a TURNED copy of the first
    than to the first; at 1.0 nothing is gained by turning it, which is what
    a field that drifts rather than rotates gives.
    """
    errors = np.array([np.abs(before - np.roll(after, shift)).mean()
                       for shift in range(len(before))])
    return float(errors.min() / max(1e-9, errors[0]))


# ---------------------------------------------------------------------------
# 1. Higher resolution, and the guard still decides
# ---------------------------------------------------------------------------

def test_the_fractal_shades_into_a_bigger_buffer_than_the_blobs(dressed):
    """"The pattern needs to be higher resolution."

    Asserted on the buffer that is actually allocated, not on the constant:
    a blob is a gradient and loses nothing to a small buffer, and a fractal
    carries structure at every scale, so this is the one theme whose buffer
    edge is visible in the picture.
    """
    fractal = _engine()
    blobs = amb.make_engine("blobs", "spacr", "#101010", seed=5)
    assert amb.FRACTAL_BUFFER_EDGE > amb.BUFFER_MAX_EDGE
    assert fractal.resolution_edge() > blobs.resolution_edge()
    wide, tall = fractal.buffer_size(W, H)
    flat_wide, flat_tall = blobs.buffer_size(W, H)
    assert wide * tall > 2 * flat_wide * flat_tall, (
        f"fractal {wide}x{tall} against blobs {flat_wide}x{flat_tall}")


def test_a_machine_that_cannot_afford_it_gets_less_of_it(dressed):
    """The guard, driven by making the budget unreachable.

    Not a mock of the measurement — the engine really does shade, really
    does time itself, and really does give ground. What the test supplies is
    a machine it cannot satisfy, by asking for a frame budget no shading
    pass can meet.
    """
    engine = _engine()
    engine.shade(W, H)
    full_edge, full_buds = engine.resolution_edge(), engine.bud_slots()
    engine.frame_budget = lambda: 0.001
    for _ in range(40):
        engine.shade(W, H)
        engine.advance(1.0 / 24.0)
    assert engine.afford() == pytest.approx(amb.FRACTAL_AFFORD_FLOOR,
                                            abs=1e-3)
    assert engine.resolution_edge() < full_edge
    assert engine.bud_slots() < max(1, full_buds), \
        "the population is not what gives way first"
    # And it stops there rather than shrinking to nothing.
    assert engine.resolution_edge() >= amb.BUFFER_MIN_EDGE


def test_a_machine_that_can_afford_it_keeps_all_of_it(dressed):
    """The other half: a budget nothing can exceed leaves the buffer alone,
    so the guard is a guard and not a permanent tax."""
    engine = _engine()
    engine.frame_budget = lambda: 10_000.0
    asked = engine.resolution_edge()
    for _ in range(20):
        engine.shade(W, H)
        engine.advance(1.0 / 24.0)
    assert engine.afford() == 1.0
    assert engine.resolution_edge() == asked


def test_two_paints_of_one_clock_are_the_same_frame(dressed):
    """The guard measures time, and time is not a pure function of the seed
    — so the one piece of state it owns is moved by :meth:`advance` and
    never by the paint. Without that split this assertion fails whenever a
    shading pass happens to straddle a step of the guard."""
    engine = _engine()
    engine.set_time(180.0)
    first = _frame(engine)
    second = _frame(engine)
    assert np.array_equal(first, second)


def test_the_buffer_ceiling_is_the_screen_rather_than_a_guess(dressed, qapp):
    """"The buffer's ceiling is the screen", measured by moving it.

    A 4K panel is entitled to four times the pixels of a 1080p one, and the
    engine has to be told rather than assume — so the number is settable and
    the buffer follows it.
    """
    engine = _engine(resolution=2.0)
    assert engine.max_pixels == amb.BUFFER_MAX_PIXELS
    engine.set_max_pixels(3840 * 2160)
    roomy = engine.buffer_size(7680, 4320)
    engine.set_max_pixels(640 * 360)
    cramped = engine.buffer_size(7680, 4320)
    assert roomy[0] * roomy[1] > cramped[0] * cramped[1]
    assert cramped[0] * cramped[1] <= 640 * 360
    # And it reads a real screen when there is one to read.
    assert amb.screen_pixels() >= amb.BUFFER_MIN_EDGE ** 2


def test_a_backdrop_takes_the_ceiling_from_the_screen_it_is_on(qtbot, dressed):
    """The widget half of it: shown, the backdrop asks the screen it is
    actually on rather than keeping the constant."""
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(640, 480)
    backdrop = amb.AmbientWidget(host)
    host.show()
    qtbot.waitExposed(host)
    assert backdrop.engine.max_pixels == amb.screen_pixels(backdrop)


# ---------------------------------------------------------------------------
# 2. Budding: more than one thing on screen, and not always
# ---------------------------------------------------------------------------

def test_the_number_of_forms_on_screen_changes_over_a_run(dressed):
    """"Sometimes there is more happening than the thing in the middle. Not
    always."

    A measurement that finds exactly one form every time has not got
    budding, and one that never finds exactly one has not got the variation
    — so both are asserted.
    """
    engine = _engine()
    counts = []
    for step in range(1200):
        engine.set_time(step * 0.5)
        counts.append(len(engine.geometry(W, H)))
    seen = set(counts)
    assert 1 in seen, "there is never a screen with only the middle form"
    assert max(seen) >= 3, f"never more than {max(seen)} forms at once"
    assert len(seen) >= 3, f"only {len(seen)} populations in ten minutes"


def test_a_bud_leaves_the_parent_rim_carrying_the_parent_constant(dressed):
    """"They separate from it rather than fading in somewhere else."

    Two things make a bud the parent's: it is BORN ON the parent's rim, and
    it iterates the parent's constant — so what is inside it is the parent's
    own field at the bud's scale, not a second picture that happens to be
    nearby.
    """
    engine = _engine()
    wide, tall = engine.buffer_size(W, H)
    born = None
    for step in range(4000):
        engine.set_time(step * 0.25)
        forms = engine.geometry(wide, tall)
        fresh = [form for form in forms[1:] if form.age < 0.03]
        if fresh:
            born = (step * 0.25, forms[0], fresh[0])
            break
    assert born is not None, "no bud was ever born"
    when, parent, bud = born
    away = math.hypot(bud.cx - parent.cx, bud.cy - parent.cy)
    assert away == pytest.approx(parent.radius, rel=0.05), \
        f"born {away:.1f} px out from a rim at {parent.radius:.1f} px"
    assert (bud.c_re, bud.c_im) == (parent.c_re, parent.c_im)
    assert bud.bud and not parent.bud
    # And it goes on leaving.
    engine.set_time(when + 4.0)
    later = [form for form in engine.geometry(wide, tall)[1:] if form.bud]
    assert later, "the bud vanished within four seconds of being born"
    assert max(math.hypot(form.cx - parent.cx, form.cy - parent.cy)
               for form in later) > away


def test_a_buds_pixels_are_the_pixels_a_bud_accounts_for(dressed):
    """"A bud's pixels are traceable back to the parent it left."

    Measured by taking the buds away and diffing: every pixel that changes
    has to be inside a disc some bud reported, and there have to BE changed
    pixels or the population is decoration that draws nothing.
    """
    engine = _engine()
    wide, tall = engine.buffer_size(W, H)
    buds = []
    for step in range(4000):
        engine.set_time(step * 0.25)
        # Mid-life, where a bud is at its widest and there are pixels of it
        # to account for.
        buds = [form for form in engine.geometry(wide, tall)[1:]
                if form.bud and 0.35 < form.age < 0.65]
        if buds:
            break
    assert buds, "no bud reached the middle of its life"
    with_buds = _frame(engine)
    engine.bud_slots = lambda: 0
    try:
        without = _frame(engine)
    finally:
        del engine.bud_slots
    moved = np.abs(with_buds - without).sum(axis=2) > 0
    assert moved.sum() > 200, "taking the buds away changed nothing"
    rows, cols = np.nonzero(moved)
    inside = np.zeros(len(rows), dtype=bool)
    for form in buds:
        reach = form.radius * (1.0 + amb.FRACTAL_BUD_FEATHER) + 1.5
        inside |= ((cols + 0.5 - form.cx) ** 2
                   + (rows + 0.5 - form.cy) ** 2) <= reach ** 2
    assert int((~inside).sum()) == 0, \
        f"{int((~inside).sum())} changed pixels lie outside every bud"


# ---------------------------------------------------------------------------
# 3. The vortex: looking INTO a form, not at it
# ---------------------------------------------------------------------------

def test_the_middle_of_a_form_turns_rather_than_merely_changing(dressed):
    """"Sampling the same form's centre over time shows it turning."

    The ring around the centre at one clock against the ring at another: if
    the form turned, some circular shift of the second matches the first
    better than no shift does. Forty-nine samples across seven seeds and
    seven moments, because a single pair could be a coincidence and the
    field is morphing at the same time as it turns.

    THE CONTROL IS THE POINT. The same measurement is run on ``blobs``,
    which drifts and breathes but does not rotate, and it comes back at
    0.93 — turning its ring buys almost nothing. The fractal comes back at
    a median of 0.45.
    """
    ratios = []
    for seed in (1, 2, 5, 7, 9, 13, 21):
        engine = _engine(seed)
        wide, tall = engine.buffer_size(W, H)
        for base in (0.0, 40.0, 90.0, 200.0, 350.0, 500.0, 700.0):
            engine.set_time(base)
            before = _frame(engine)
            start = engine.geometry(wide, tall)[0]
            engine.set_time(base + 6.0)
            after = _frame(engine)
            end = engine.geometry(wide, tall)[0]
            radius = min(wide, tall) * RING_RADIUS
            ratios.append(_turn_ratio(
                _ring(before, start.cx, start.cy, radius),
                _ring(after, end.cx, end.cy, radius)))
    turning = float(np.median(ratios))

    still = amb.make_engine("blobs", "spacr", "#101010", seed=1)
    wide, tall = still.buffer_size(W, H)
    still.set_time(0.0)
    before = _frame(still)
    still.set_time(6.0)
    after = _frame(still)
    radius = min(wide, tall) * RING_RADIUS
    drifting = _turn_ratio(_ring(before, wide / 2, tall / 2, radius),
                           _ring(after, wide / 2, tall / 2, radius))

    assert turning < 0.65, f"the forms do not turn: median ratio {turning:.3f}"
    assert drifting > 0.80, (
        "the control rotated too, so this measurement is not measuring "
        f"rotation: {drifting:.3f}")


def test_the_swirl_winds_tighter_toward_the_middle(dressed):
    """What makes it a tunnel rather than a spiral drawn on a wall.

    The sampling is turned by an amount that GROWS with the log of the
    radius, so the twist between two radii is the swirl times the distance
    between them in e-folds — measured here on the engine's own sampling,
    which is the array ``_paint_field`` iterates.
    """
    engine = _engine()
    engine.set_time(300.0)
    wide, tall = engine.buffer_size(W, H)
    radius, log_radius, angle = engine._grid(wide, tall)
    form = engine.geometry(wide, tall)[0]._replace(angle=0.0, tunnel=0.0)
    assert form.swirl > 0.2, "this clock is not a deep enough state to check"
    flat = form._replace(swirl=0.0)
    twisted_re, twisted_im, _ = engine._sample(form, radius, log_radius, angle)
    flat_re, flat_im, _ = engine._sample(flat, radius, log_radius, angle)
    middle_row = tall // 2
    inner, outer = int(wide * 0.62), int(wide * 0.95)
    turned = []
    for column in (inner, outer):
        # The ORIGIN COMES OFF FIRST: `_sample` returns a point of the
        # complex plane, and the sampling angle is the angle of the offset
        # from the form's own centre, not of the point.
        near = math.atan2(flat_im[middle_row, column] - form.origin_im,
                          flat_re[middle_row, column] - form.origin_re)
        far = math.atan2(twisted_im[middle_row, column] - form.origin_im,
                         twisted_re[middle_row, column] - form.origin_re)
        turned.append(((far - near + math.pi) % (2.0 * math.pi)) - math.pi)
    gap = (float(log_radius[middle_row, outer])
           - float(log_radius[middle_row, inner]))
    measured = ((turned[1] - turned[0] + math.pi) % (2.0 * math.pi)) - math.pi
    assert measured == pytest.approx(form.swirl * gap, abs=0.02), \
        "the twist does not grow with the log of the radius"


def test_flat_is_the_picture_the_engine_drew_before(dressed):
    """"Sometimes deep 3D vortex and sometimes not" costs nothing, because
    at depth zero both vortex terms are zero and the sampling is the plain
    rotation this engine always had."""
    engine = _engine()
    engine.set_time(300.0)
    wide, tall = engine.buffer_size(W, H)
    radius, log_radius, angle = engine._grid(wide, tall)
    flat = engine.geometry(wide, tall)[0]._replace(swirl=0.0, tunnel=0.0)
    got_re, got_im, bands = engine._sample(flat, radius, log_radius, angle)
    assert bands is None
    want_re = radius / flat.scale * np.cos(angle + flat.angle) + flat.origin_re
    assert np.allclose(got_re, want_re, atol=1e-4)


def test_the_depth_really_reaches_both_ends(dressed):
    """A state that hovers at its own average is not a state.

    The raw wander is a sum of smoothed noise and is therefore a bell; the
    depth stretches its middle over the whole range so the engine spends
    real time flat and real time deep.
    """
    engine = _engine()
    depths = []
    for step in range(1800):
        engine.set_time(float(step))
        depths.append(engine.depth())
    depths = np.array(depths)
    assert depths.min() < 0.02 and depths.max() > 0.98
    assert (depths < 0.15).mean() > 0.10, "it is never really flat"
    assert (depths > 0.85).mean() > 0.10, "it is never really deep"


# ---------------------------------------------------------------------------
# 4. It beats, and the order does not repeat
# ---------------------------------------------------------------------------

def test_it_beats_and_the_beat_is_not_a_metronome(dressed):
    """A pulse, and one whose rate wanders.

    Measured as the gaps between thumps over ten minutes. A fixed rate would
    give one gap; what this asks for is a rate that speeds and slows without
    the phase ever jumping, which is why the phase is an integral rather
    than a sample.
    """
    engine = _engine()
    clocks = np.arange(0.0, 600.0, 0.02)
    values = []
    for clock in clocks:
        engine.set_time(float(clock))
        values.append(engine.beat())
    values = np.array(values)
    assert values.max() > 0.95 and values.min() < 0.05, "it does not thump"
    peaks = [clocks[i] for i in range(1, len(values) - 1)
             if values[i] > 0.8 and values[i] >= values[i - 1]
             and values[i] > values[i + 1]]
    gaps = np.diff(peaks)
    assert len(gaps) > 100
    assert gaps.max() > 1.8 * gaps.min(), (
        f"the rate never changes: gaps {gaps.min():.2f}-{gaps.max():.2f} s")
    # Continuous through every change of speed: no gap is a skipped beat.
    assert gaps.max() < 4.0 * gaps.min(), "the phase jumped"


def test_the_state_sequence_never_repeats_on_any_period(dressed):
    """"The sequence is not a fixed cycle."

    Half an hour of the depth channel, against every lag from five seconds
    to fifteen minutes: the closest thing to a repeat still differs by more
    than a tenth of the range. The same measurement run on a sine finds an
    exact repeat, which is what makes this an assertion about the engine
    rather than about the measurement.
    """
    engine = _engine()
    clocks = np.arange(0.0, 1800.0, 1.0)
    series = np.array([engine.wander(amb.FRACTAL_STATE_DEPTH, at=float(clock))
                       for clock in clocks])

    def closest_repeat(values: np.ndarray) -> float:
        return min(float(np.abs(values[:-lag] - values[lag:]).max())
                   for lag in range(5, 900))

    looping = 0.5 + 0.5 * np.sin(2.0 * math.pi * clocks / 97.0)
    assert closest_repeat(looping) < 1e-9, \
        "the measurement cannot even find a loop in a sine"
    assert closest_repeat(series) > 0.10, "the states repeat on a period"
    assert series.max() - series.min() > 0.5, "the state barely moves"


def test_the_states_disagree_with_each_other(dressed):
    """Four channels rather than one number, so the picture can be deep and
    quiet or flat and busy — and does not pass through the same handful of
    looks."""
    engine = _engine()
    clocks = np.arange(0.0, 1800.0, 2.0)
    channels = [np.array([engine.wander(which, at=float(clock))
                          for clock in clocks])
                for which in (amb.FRACTAL_STATE_DEPTH, amb.FRACTAL_STATE_BUSY,
                              amb.FRACTAL_STATE_PACE,
                              amb.FRACTAL_STATE_CROWD)]
    for first in range(len(channels)):
        for second in range(first + 1, len(channels)):
            pair = np.corrcoef(channels[first], channels[second])[0, 1]
            assert abs(pair) < 0.4, \
                f"channels {first} and {second} move together: {pair:.2f}"


# ---------------------------------------------------------------------------
# 5. The palette drifts, and the check drifts with it
# ---------------------------------------------------------------------------

def test_the_hue_moves_and_keeps_moving(dressed):
    """"The palette moves through hue space over time."

    Measured on the colour a window is actually painted, at frame 1 and at
    frame N, and then over twenty minutes to show it does not simply step
    once and stop.
    """
    theme.set_spaceout_drift_seconds(0.0)
    first = theme.palette_for("dark")["page"]
    seen = set()
    for second in range(0, 1200, 7):
        theme.set_spaceout_drift_seconds(float(second))
        seen.add(theme.palette_for("dark")["page"])
    assert first in seen
    assert len(seen) > 30, f"only {len(seen)} page colours in twenty minutes"
    theme.set_spaceout_drift_seconds(120.0)
    assert theme.palette_for("dark")["page"] != first


def test_the_drift_is_not_a_loop_either(dressed):
    """The hue travels AND wanders, so the order it visits the spectrum in
    is not a cycle a watcher can learn — the same claim as the engine's
    states, measured the same way and against the same control."""
    clocks = np.arange(0.0, 3600.0, 2.0)
    series = np.unwrap(np.array(
        [math.radians(theme.spaceout_drift(at=float(clock)))
         for clock in clocks]))
    rate = np.diff(series)
    assert rate.max() > 2.5 * rate.min() or rate.min() < 0, \
        "the spectrum turns at one unvarying rate"

    def closest_repeat(values: np.ndarray) -> float:
        span = values.max() - values.min()
        return min(float(np.abs(values[:-lag] - values[lag:]).max()) / span
                   for lag in range(5, 600))

    assert closest_repeat(rate) > 0.10


@pytest.mark.parametrize("name", theme.THEMES)
def test_every_offset_the_drift_can_reach_is_still_readable(name, dressed):
    """THE CHECK IS NOT SUSPENDED, and a drifting palette makes it a
    per-frame property rather than a one-off.

    Every offset the palette can take, not a sample of them: the drift is
    quantised onto :func:`spacr.qt.theme._drift_grid` precisely so that
    "every point on the drift" is a finite set that can be exhausted.

    All three checks, because they fail differently. Contrast survives a
    re-hue by construction; the page separation and the scrims do not,
    because both composite in sRGB where equal luminances of different hue
    do not stay equal — and both are therefore SOLVED over the same grid.
    """
    for drift in theme._drift_grid():
        with theme._dressed_at(drift):
            assert theme.contrast_failures(name) == [], f"at {drift}°"
            assert theme.page_separation_failures(name) == [], f"at {drift}°"
            if name in theme.IMAGE_THEMES:
                assert theme.scrim_failures(name) == [], f"at {drift}°"


def test_holding_the_check_is_what_the_damping_is_for(dressed):
    """The saturation the drift costs, reproduced rather than described.

    Take the damping away and the light theme's faded panels stop separating
    from its page at four of the sixty offsets — which is the bug the solve
    prevents, and the reason the trade is stated as "hold the check" rather
    than as a constant somebody eyeballed.
    """
    assert any(rows for rows in theme._PAGE_DAMPING.values()), \
        "nothing is damped, so this test proves nothing"
    kept = {name: dict(rows) for name, rows in theme._PAGE_DAMPING.items()}
    theme._PAGE_DAMPING.clear()
    try:
        undamped = [(drift, name)
                    for drift in theme._drift_grid()
                    for name in theme.THEMES
                    if _fails_separation(drift, name)]
    finally:
        theme._PAGE_DAMPING.clear()
        theme._PAGE_DAMPING.update(kept)
    assert undamped, "the damping is not doing anything"
    assert all(name == "light" for _drift, name in undamped)


def _fails_separation(drift: float, name: str) -> bool:
    with theme._dressed_at(drift):
        return bool(theme.page_separation_failures(name))


def test_the_damped_roles_are_the_roles_the_rule_is_about(dressed):
    """:data:`spacr.qt.theme.SPACEOUT_DAMPED_ROLES` is written out because
    it is declared before :data:`spacr.qt.theme.PAGE_PANEL_ROLES`. This is
    what stops the two from drifting apart."""
    assert set(theme.SPACEOUT_DAMPED_ROLES) == \
        {"page"} | set(theme.PAGE_PANEL_ROLES)


def test_the_text_went_further_towards_rainbow(dressed):
    """"The colour of the text is pretty good but could be more rainbow
    like."

    The reason it was not is the identity the rest of the dressing rests on:
    hue moves, luminance does not, and a role at the top of the luminance
    scale has no room to carry a hue. Dark's ``fg`` was ``#ffffff`` — the
    body text of the application was the one thing in it that was not in the
    rainbow.
    """
    for name in theme.THEMES:
        palette = theme.palette_for(name)
        for role in theme.SPACEOUT_INK_ROLES:
            red, green, blue = theme._channels(palette[role])
            assert max(red, green, blue) - min(red, green, blue) >= 40, \
                f"{name}.{role} is {palette[role]}, which carries no hue"
    theme.disable_spaceout()
    plain = theme.palette_for("dark")["fg"]
    theme.enable_spaceout()
    assert theme.palette_for("dark")["fg"] != plain
    assert theme._channels(plain) == (255, 255, 255), \
        "the role this was about is no longer white undressed"


def test_the_ink_only_spends_what_the_rules_leave_it(dressed):
    """The band is read off :data:`spacr.qt.theme.CONTRAST_RULES` with
    :data:`spacr.qt.theme.SPACEOUT_INK_HEADROOM` in hand, so the answer is
    the most coloured ink the check allows and never a colour the check was
    bent for."""
    for name in theme.THEMES:
        bands = theme._INK_BANDS.get(name, {})
        assert bands, f"{name} solved no ink band at all"
        for role, (low, high) in bands.items():
            assert 0.0 <= low < high <= 1.0
            for drift in theme._drift_grid():
                with theme._dressed_at(drift):
                    luma = theme.relative_luminance(
                        theme.palette_for(name)[role])
                assert low - 1e-9 <= luma <= high + 1e-9, \
                    f"{name}.{role} left its band at {drift}°"


def test_the_wallpaper_exposure_ceiling_did_not_move(dressed):
    """The ink is allowed to darken, and :func:`max_background_luma` is a
    MINIMUM over ink luminances — so an unconstrained solve would have
    silently darkened every photograph in the application, because that
    ceiling is what :func:`spacr.qt.imagery.solve_dim` exposes every
    wallpaper down to. The band carries that bound, and this is what proves
    it.

    The tolerance is 8-bit rounding and nothing else. It is not slack for
    the ink: solved from the contrast rules ALONE, ``fg_dim`` takes Cell's
    ceiling from 0.109 to 0.063 and Glass's from 0.081 to 0.070 — every
    wallpaper in those two themes exposed down to little more than half what
    it was, to buy a hue on hint text. What is left after the bound is the
    rounding the whole dressing already has: ``error`` is not an ink role
    and is not solved, and its hue shift moves it by 0.0003 of a luminance
    level, which is the same rounding
    ``test_spaceout_palette_stays_readable.py`` bounds at 0.006.
    """
    theme.disable_spaceout()
    plain = {name: theme.max_background_luma(name) for name in theme.THEMES}
    theme.enable_spaceout()
    for name in theme.THEMES:
        for drift in theme._drift_grid():
            with theme._dressed_at(drift):
                ceiling = theme.max_background_luma(name)
                assert ceiling >= plain[name] - 0.001, \
                    f"{name} at {drift}° dimmed the wallpaper ceiling"


def test_an_ordinary_start_has_no_drift_at_all():
    """The clock does not run and the hue does not move when the dressing is
    off, whatever anybody calls."""
    was = theme.spaceout_enabled()
    theme.disable_spaceout()
    try:
        before = theme.palette_for("dark")
        theme.advance_spaceout_drift(600.0)
        assert theme.spaceout_drift() == 0.0
        assert theme.spaceout_drift_seconds() == 0.0
        assert theme.palette_for("dark") == before
    finally:
        if was:
            theme.enable_spaceout()


# ---------------------------------------------------------------------------
# 6. The mark that follows the cursor
# ---------------------------------------------------------------------------

def _card_pixels(card: SetupCard) -> np.ndarray:
    from PySide6.QtGui import QImage, QPainter

    image = QImage(card.width(), card.height(), QImage.Format_RGB32)
    image.fill(0)
    painter = QPainter(image)
    painter.end()
    card.render(image)
    raw = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4)
    return raw[:, :image.width(), :3].astype(np.int16).copy()


def _lit_hues(pixels: np.ndarray) -> set:
    """The hues of the brightest pixels — the lit run of rim."""
    import colorsys

    flat = pixels.reshape(-1, 3)
    brightness = flat.max(axis=1).astype(int) - flat.min(axis=1).astype(int)
    chosen = flat[brightness > 40]
    return {round(colorsys.rgb_to_hsv(*(channel / 255.0
                                        for channel in pixel))[0], 2)
            for pixel in chosen}


def test_the_cursor_accent_oscillates_under_the_dressing(qtbot, dressed):
    """"The little blue window thing that follows the mouse should also
    oscillate in colour under this special theme."

    Measured on the card's own pixels at two frames, with the mode pinned to
    the shipped ``glow`` so the comparison is about the dressing and not
    about a rim preference somebody set.
    """
    card = SetupCard(mode="glow")
    qtbot.addWidget(card)
    card.resize(420, 300)
    card.show()
    qtbot.waitExposed(card)
    first = _card_pixels(card)
    card._phase += SPACEOUT_RIM_PERIOD / 3.0
    second = _card_pixels(card)
    assert not np.array_equal(first, second), "the rim did not repaint"
    assert _lit_hues(first) and _lit_hues(second)
    assert _lit_hues(first) != _lit_hues(second), \
        "the lit rim is the same colour in both frames"
    assert card.animates(), "a still card would never show the oscillation"


def test_the_cursor_accent_is_constant_on_an_ordinary_start(qtbot):
    """And the other half: without the dressing the shipped rim is the
    theme's accent, frame after frame."""
    was = theme.spaceout_enabled()
    theme.disable_spaceout()
    card = SetupCard(mode="glow")
    qtbot.addWidget(card)
    card.resize(420, 300)
    card.show()
    qtbot.waitExposed(card)
    try:
        first = _card_pixels(card)
        card._phase += SPACEOUT_RIM_PERIOD / 3.0
        second = _card_pixels(card)
        assert np.array_equal(first, second), \
            "the rim changed colour without the dressing"
        assert not card.animates()
    finally:
        if was:
            theme.enable_spaceout()


def test_the_accent_hue_belongs_to_the_palettes_own_spectrum(dressed):
    """It carries the palette's drift as well as its own cycle, so the mark
    is part of the spectrum the window is travelling through rather than a
    second unrelated rainbow."""
    card = SetupCard(mode="glow")
    card.resize(420, 300)
    theme.set_spaceout_drift_seconds(0.0)
    still = card.spaceout_hue(0.5)
    theme.set_spaceout_drift_seconds(200.0)
    assert card.spaceout_hue(0.5) != still
    # And it spreads a piece of spectrum along its own length.
    assert card.spaceout_hue(0.0) != card.spaceout_hue(1.0)


# ---------------------------------------------------------------------------
# 7. The fast paths are the published answers
# ---------------------------------------------------------------------------

def _hue_shift_by_scan(colour: str, hue: float, saturation: float) -> str:
    """:func:`spacr.qt.theme._hue_shift` as it was: all 512 candidates."""
    target = theme._relative_luminance(colour)
    base = theme._hue_rgb(hue, saturation)
    best, error = (0, 0, 0), float("inf")
    for level in range(256):
        candidate = (int(round(base[0] * level)), int(round(base[1] * level)),
                     int(round(base[2] * level)))
        miss = abs(theme._rgb_luminance(candidate) - target)
        if miss < error:
            best, error = candidate, miss
    for step in range(256):
        weight = step / 255.0
        candidate = tuple(int(round(255.0 * (1.0 - weight + weight * channel)))
                          for channel in base)
        miss = abs(theme._rgb_luminance(candidate) - target)
        if miss < error:
            best, error = candidate, miss
    return "#%02x%02x%02x" % best


def test_the_crossing_search_is_the_full_scan(dressed):
    """:func:`spacr.qt.theme._hue_shift` visits about twenty of its 512
    candidates now, because the drift asks for the whole palette at sixty
    offsets rather than once. That is only allowed to be a speed-up.

    Every triple the dressing can produce, which is a finite set: each
    role's colour in each theme, at each offset on the grid.
    """
    checked = 0
    for name in theme.THEMES:
        palette = dict(theme._PALETTES[name])
        palette.update(theme.CONSTANT_ROLES)
        for role, colour in palette.items():
            if role not in theme.SPACEOUT_HUES:
                continue
            saturation = theme.SPACEOUT_SATURATION.get(role, 1.0)
            for drift in theme._drift_grid():
                hue = (theme.SPACEOUT_HUES[role] + drift) % 360.0
                assert theme._hue_shift(colour, hue, saturation) == \
                    _hue_shift_by_scan(colour, hue, saturation), \
                    f"{name}.{role} at {drift}°"
                checked += 1
    assert checked > 2000


def test_the_scrim_bounds_are_the_published_solvers(dressed):
    """:func:`spacr.qt.theme._scrim_bounds` is the two documented solvers in
    one pass over 8-bit channels, and the drift calls it sixty times where
    they were called once. Same answers or it is a different rule."""
    for name in theme.IMAGE_THEMES:
        for drift in (0.0, 42.0, 126.0, 288.0):
            with theme._dressed_at(drift):
                palette = theme.palette_for(name)
                under = theme._channels(theme.scrim_under(name))
                for role, colour_role in theme.SCRIM_ROLES.items():
                    floor, ceiling = theme._scrim_bounds(
                        palette, role, colour_role, under)
                    assert floor == pytest.approx(
                        theme.legible_scrim_floor(name, role, colour_role))
                    assert ceiling == pytest.approx(
                        theme.present_scrim_ceiling(name, role, colour_role))


# ---------------------------------------------------------------------------
# 8. Still only through spaceout
# ---------------------------------------------------------------------------

def test_none_of_this_is_offered_in_preferences():
    """Every piece of the dressing is reachable from the launcher and from
    nowhere else. The animation dropdown is built from
    :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES` and a stored value is
    validated against it, so a settings file that names the fractal is
    rejected like any other name this build does not offer."""
    from spacr.qt import preferences

    assert amb.SPACEOUT_THEME not in amb.AMBIENT_THEMES
    assert amb.SPACEOUT_THEME not in amb.ANIMATION_CHOICES
    assert not amb.is_valid_theme(amb.SPACEOUT_THEME)
    assert not hasattr(preferences, "set_spaceout")
    for name in dir(preferences):
        assert "spaceout" not in name.lower()
        assert "fractal" not in name.lower()
