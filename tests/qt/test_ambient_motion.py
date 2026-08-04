"""Blur, speed and size — and the aurora that is actually an aurora.

Three groups, and each one is here because reading the code cannot tell you
whether it is true:

``the three controls``
    They are multipliers, and the whole promise is that 1.0 is *exactly* the
    animation that shipped. That is asserted against the real pre-change
    engine, pulled out of git and rendered side by side, byte for byte — see
    :func:`shipped_module`. Then each control is moved and the rendered
    frame is measured to have changed in the direction the control claims.

``the aurora``
    An aurora is a recognisable thing, so the tests are about what makes it
    recognisable rather than about the code that draws it: the folds travel
    *along* the arc while the rays stay put, several frequencies are
    superposed, the colour runs green through the body to red at the top, and
    the lower edge is sharp where the upper one is diffuse.

``the palette``
    ``borealis`` is a set of real emission lines. The test names the
    wavelengths and pins the sRGB values, because "the aurora palette" is
    only meaningful if those are the numbers in it.

Everything renders offscreen into a QImage and asserts on pixels or on
measured geometry. Nothing asserts that a setting round-trips through
QSettings and stops there.
"""
from __future__ import annotations

import math
import subprocess
import sys
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings                            # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter              # noqa: E402

from spacr.qt.widgets import ambient as amb                     # noqa: E402
from spacr.qt.widgets.ambient import (AMBIENT_THEMES,           # noqa: E402
                                      BLUR_RANGE, BUFFER_MAX_EDGE,
                                      DEFAULT_BLUR, DEFAULT_SIZE,
                                      DEFAULT_SPEED, SIZE_RANGE,
                                      SPEED_RANGE, AmbientWidget,
                                      make_engine, palette_colors,
                                      palettes_for)

DARK = "#101418"
LIGHT = "#f6f7f9"

#: Big enough that the buffer is genuinely upscaled (which is what the blur
#: control acts on) and small enough to render a few dozen frames per test.
W, H = 640, 400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render(engine, width=W, height=H, background=DARK) -> QImage:
    """One frame, painted exactly as the widget paints it."""
    image = QImage(width, height, QImage.Format_RGB32)
    painter = QPainter(image)
    painter.fillRect(image.rect(), QColor(background))
    engine.paint(painter, width, height)
    painter.end()
    return image


def rows(image: QImage):
    """The frame as a list of rows of ``(r, g, b)``."""
    return [[image.pixelColor(x, y).getRgb()[:3]
             for x in range(image.width())]
            for y in range(image.height())]


def luminance(pixel) -> float:
    r, g, b = pixel
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def sharpness(image: QImage) -> float:
    """Mean absolute difference between horizontally adjacent pixels.

    A blur is exactly the removal of high spatial frequencies, so this is the
    quantity the blur control has to move — and it moves it whatever the
    theme draws, which one measure of "is it softer" has to.
    """
    grid = rows(image)
    total, count = 0.0, 0
    for row in grid:
        for left, right in zip(row, row[1:]):
            total += abs(luminance(left) - luminance(right))
            count += 1
    return total / max(1, count)


def lit_pixels(image: QImage, background=DARK) -> int:
    flat = QColor(background).getRgb()[:3]
    return sum(1 for row in rows(image) for pixel in row if pixel != flat)


def best_shift(a, b, limit):
    """The lag, in samples, that best aligns profile ``a`` onto ``b``.

    Plain sum-of-squares over integer lags on mean-removed profiles. Small
    and obvious beats fast here: the profiles are a few hundred samples and
    this runs a handful of times.
    """
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    centred_a = [v - mean_a for v in a]
    centred_b = [v - mean_b for v in b]
    best, score = 0, None
    for lag in range(-limit, limit + 1):
        overlap = range(max(0, -lag), min(len(a), len(a) - lag))
        if len(overlap) < len(a) // 2:
            continue
        error = sum((centred_a[i] - centred_b[i + lag]) ** 2 for i in overlap)
        error /= len(overlap)
        if score is None or error < score:
            best, score = lag, error
    return best, score


# ---------------------------------------------------------------------------
# The animation that shipped
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shipped_module():
    """The ambient module as it was *before* blur/speed/size existed.

    Found by walking this file's history back to the last revision that does
    not mention ``DEFAULT_BLUR``, so it keeps pointing at the pre-change
    engine no matter how many commits land on top. Exec'd in memory under a
    sibling module name — nothing is ever written into the package.

    Skips rather than fails where there is no git checkout to read (a source
    tarball, a wheel test): the assertion this feeds is worth having and is
    not worth breaking someone else's packaging over.
    """
    path = "spacr/qt/widgets/ambient.py"
    try:
        revisions = subprocess.run(
            ["git", "log", "--format=%H", "--", path],
            capture_output=True, text=True, check=True,
            cwd=_repo_root()).stdout.split()
    except Exception as exc:                       # no git, no history
        pytest.skip(f"cannot read the shipped engine from git: {exc}")
    for revision in revisions:
        try:
            source = subprocess.run(
                ["git", "show", f"{revision}:{path}"],
                capture_output=True, text=True, check=True,
                cwd=_repo_root()).stdout
        except Exception:
            # A shallow clone lists revisions whose blobs it does not have.
            break
        if "DEFAULT_BLUR" in source:
            continue
        name = "spacr.qt.widgets._ambient_shipped"
        module = types.ModuleType(name)
        module.__package__ = "spacr.qt.widgets"
        module.__file__ = path
        sys.modules[name] = module
        exec(compile(source, path, "exec"), module.__dict__)
        return module
    pytest.skip("no revision of the ambient module predates the controls")


def _repo_root():
    from pathlib import Path
    # .../<repo>/spacr/qt/widgets/ambient.py
    return str(Path(amb.__file__).resolve().parents[3])


#: The aurora is a deliberate redesign, so it is the one theme whose frames
#: are *supposed* to differ from the shipped ones.
UNCHANGED_THEMES = tuple(t for t in AMBIENT_THEMES if t != "aurora")


@pytest.mark.parametrize("background", [DARK, LIGHT])
@pytest.mark.parametrize("theme", UNCHANGED_THEMES)
def test_the_defaults_are_the_animation_that_shipped(shipped_module, theme,
                                                     background):
    """Byte for byte, over several frames, on both kinds of page.

    This is the promise the whole feature rests on: a user who never opens
    the three new controls must not be able to tell they exist.
    """
    was = shipped_module.make_engine(theme, "spacr", background, seed=7)
    now = make_engine(theme, "spacr", background, seed=7)
    assert (now.blur, now.speed, now.size) == (DEFAULT_BLUR, DEFAULT_SPEED,
                                               DEFAULT_SIZE)
    for _ in range(6):
        was.advance(1 / 24)
        now.advance(1 / 24)
        assert now.time == pytest.approx(was.time), "the clock drifted"
        assert render(now, background=background) == \
            render(was, background=background), \
            f"{theme} on {background} changed at t={now.time:.2f}"


def test_the_aurora_is_deliberately_not_what_shipped(shipped_module):
    """The one theme that must have changed. Asserted so that a refactor
    which quietly restored the old curtains would fail here rather than in
    somebody's eyes."""
    was = shipped_module.make_engine("aurora", "spacr", DARK, seed=7)
    now = make_engine("aurora", "spacr", DARK, seed=7)
    was.set_time(11.0)
    now.set_time(11.0)
    assert render(now) != render(was)


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_the_default_multipliers_are_the_identity(theme):
    """Each mechanism, checked against the constant it is a multiple of.

    Separate from the frame comparison above because this one keeps working
    when there is no git history to compare against, and because it says
    *which* mechanism broke.
    """
    engine = make_engine(theme, "spacr", DARK, seed=3)

    # Blur: the buffer is the shipped one.
    if isinstance(engine, amb._BufferedEngine):
        assert engine.blur_edge() == BUFFER_MAX_EDGE
        longest = max(1920, 1080)
        scale = max(1, int(math.ceil(longest / BUFFER_MAX_EDGE)))
        assert engine.buffer_size(1920, 1080) == (1920 // scale, 1080 // scale)

    # Speed: the clock counts real seconds.
    engine.advance(0.25)
    engine.advance(0.5)
    assert engine.time == pytest.approx(0.75)

    # Size: the elements are the size the theme's own constants declare.
    engine.set_time(4.0)
    if theme == "blobs":
        short = min(W, H)
        biggest = max(r for _x, _y, r in engine.geometry(W, H))
        ceiling = max(amb.BLOB_LARGE_RADIUS) * short * (
            1.0 + max(amb.BLOB_PULSE))
        assert biggest <= ceiling + 1e-6
    elif theme == "drift":
        assert {round(d, 6) for _x, _y, d in engine.geometry(W, H)} == \
            {round(layer[0], 6) for layer in amb.DRIFT_LAYERS}
    elif theme == "ripple":
        reach = max(amb.RIPPLE_REACH) * 0.5 * math.hypot(W, H)
        assert max(r for _x, _y, r, _f in engine.geometry(W, H)) <= reach + 1
    else:
        assert max(h for _x, _y, h, _b in engine.geometry(W, H)) \
            <= max(amb.AURORA_THICKNESS) * H * 1.5


# ---------------------------------------------------------------------------
# Each control moves the picture, in its own direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme", ["blobs", "aurora", "ripple"])
def test_blur_softens_the_frame(theme):
    """More blur, fewer high frequencies. Measured on the pixels, because
    "the buffer got smaller" is an implementation detail and this is the
    thing the user asked for.

    Not ``drift``: a starfield's high frequencies are its dots, and blurring
    a dot spreads it over *more* pixels, which raises this measure while
    lowering the thing it stands for. It gets its own test below, which is
    the honest way round — one number cannot mean "softer" for a gradient
    field and for a two-pixel star at the same time.
    """
    def frame(blur):
        engine = make_engine(theme, "spacr", DARK, seed=5, blur=blur)
        engine.set_time(9.0)
        return render(engine)

    sharp, normal, soft = (sharpness(frame(b)) for b in (0.4, 1.0, 2.5))
    assert sharp > normal > soft, \
        f"{theme}: sharpness went {sharp:.3f} -> {normal:.3f} -> {soft:.3f}"


def test_blur_is_the_buffer_resolution_not_a_per_frame_filter():
    """The mechanism matters as much as the effect: a Gaussian over two
    million pixels every frame is the thing this must not be."""
    coarse = make_engine("blobs", "spacr", DARK, seed=5, blur=2.5)
    fine = make_engine("blobs", "spacr", DARK, seed=5, blur=0.4)
    assert coarse.blur_edge() < BUFFER_MAX_EDGE < fine.blur_edge()
    assert coarse.buffer_size(1920, 1080)[0] < fine.buffer_size(1920, 1080)[0]

    # ... and it is still allocated once, not per frame.
    render(coarse, 1920, 1080)
    buffer = coarse._buffer
    for _ in range(5):
        coarse.advance(1 / 24)
        render(coarse, 1920, 1080)
    assert coarse._buffer is buffer


def test_blur_spreads_the_starfield_rather_than_resizing_it():
    """Drift is not painted through the blur buffer, so it needs its own
    mechanism — and that mechanism must not just be "bigger dots", which is
    what the size control does."""
    def engine(blur):
        made = make_engine("drift", "spacr", DARK, seed=5, blur=blur)
        made.set_time(9.0)
        return made

    hard, normal, soft = engine(0.5), engine(1.0), engine(2.5)
    # The dots themselves are the same size at every blur.
    assert {round(d, 6) for _x, _y, d in hard.geometry(W, H)} == \
        {round(d, 6) for _x, _y, d in soft.geometry(W, H)}
    # But more of the page is lit, and the halo is dimmer than the core.
    assert lit_pixels(render(hard)) < lit_pixels(render(normal)) \
        < lit_pixels(render(soft))


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_speed_multiplies_the_motion_and_nothing_else(theme):
    """Twice the speed is twice the animation per second of wall clock — and
    the same frame at the same *animation* time, which is what makes it a
    multiplier on the motion rather than a different animation."""
    slow = make_engine(theme, "spacr", DARK, seed=5, speed=0.5)
    normal = make_engine(theme, "spacr", DARK, seed=5)
    fast = make_engine(theme, "spacr", DARK, seed=5, speed=2.0)
    for _ in range(24):
        for engine in (slow, normal, fast):
            engine.advance(1 / 24)
    assert slow.time == pytest.approx(0.5)
    assert normal.time == pytest.approx(1.0)
    assert fast.time == pytest.approx(2.0)

    # Same clock, same frame: speed changes how fast the clock runs, never
    # what the animation looks like at a given point in it.
    fast.set_time(1.0)
    assert render(fast) == render(normal)


def test_changing_the_speed_does_not_teleport_the_animation(qtbot):
    """Dragging the slider in Preferences must not make the backdrop jump."""
    widget = AmbientWidget(theme="blobs", palette="spacr", background=DARK,
                           seed=5, speed=1.0)
    qtbot.addWidget(widget)
    widget.resize(W, H)
    for _ in range(12):
        widget.advance_frame(1 / 24)
    before = widget.engine.geometry(W, H)
    widget.set_speed(3.0)
    assert widget.engine.geometry(W, H) == before, "the animation jumped"
    widget.advance_frame(1 / 24)
    assert widget.engine.time == pytest.approx(0.5 + 3.0 / 24)


@pytest.mark.parametrize("theme,measure", [
    ("blobs", lambda g: max(r for _x, _y, r in g)),
    ("aurora", lambda g: max(h for _x, _y, h, _b in g)),
    ("ripple", lambda g: max(r for _x, _y, r, _f in g)),
    ("drift", lambda g: max(d for _x, _y, d in g)),
])
def test_size_scales_each_themes_own_range(theme, measure):
    """One control, four different meanings — blob radius, curtain height,
    ripple wavelength, star diameter — and it has to move all four."""
    sizes = (0.4, 1.0, 2.0)
    measured = []
    for size in sizes:
        engine = make_engine(theme, "spacr", DARK, seed=5, size=size)
        engine.set_time(7.0)
        measured.append(measure(engine.geometry(W, H)))
    small, normal, large = measured
    assert small < normal < large, f"{theme}: {measured}"
    # Proportional, not merely monotonic: it is a scale on the theme's range.
    assert large / normal == pytest.approx(2.0, rel=0.25)


def test_size_moves_the_ripple_wavelength_not_just_the_radius():
    """"Bigger ripples" has to mean the rings are further apart, or the
    control is a zoom rather than a wavelength."""
    def spacing(size):
        engine = make_engine("ripple", "ocean", DARK, seed=5, size=size)
        engine.set_time(6.0)
        radii = sorted(r for _x, _y, r, _f in engine.geometry(W, H)[:4])
        return [b - a for a, b in zip(radii, radii[1:])]

    normal, large = spacing(1.0), spacing(2.0)
    assert all(b > a for a, b in zip(normal, large))


@pytest.mark.parametrize("theme", AMBIENT_THEMES)
def test_the_three_controls_are_reversible(theme):
    """Set them, put them back, get the shipped frame again. Cheap insurance
    that nothing cached under a setting outlives it."""
    engine = make_engine(theme, "spacr", DARK, seed=5)
    engine.set_time(8.0)
    before = render(engine)
    for blur, size in ((2.5, 0.4), (0.4, 2.5)):
        engine.set_blur(blur)
        engine.set_size(size)
        engine.set_speed(3.0)
        render(engine)
    engine.set_blur(DEFAULT_BLUR)
    engine.set_size(DEFAULT_SIZE)
    engine.set_speed(DEFAULT_SPEED)
    engine.set_time(8.0)
    assert render(engine) == before


@pytest.mark.parametrize("value,expected", [
    (-4.0, BLUR_RANGE[0]), (99.0, BLUR_RANGE[1]),
])
def test_absurd_values_are_clamped_not_obeyed(value, expected):
    engine = make_engine("blobs", "spacr", DARK, seed=1, blur=value)
    assert engine.blur == expected
    assert BUFFER_MAX_EDGE // 8 <= engine.blur_edge() <= BUFFER_MAX_EDGE * 2
    render(engine, 1920, 1080)          # and it still paints


def test_the_widget_exposes_all_three_and_they_reach_the_engine(qtbot):
    widget = AmbientWidget(theme="ripple", palette="ocean", background=DARK,
                           seed=2, blur=1.0, speed=1.0, size=1.0)
    qtbot.addWidget(widget)
    widget.set_blur(2.0)
    widget.set_speed(0.5)
    widget.set_size_scale(1.5)
    assert (widget.blur(), widget.speed(), widget.size_scale()) == \
        (2.0, 0.5, 1.5)
    assert (widget.engine.blur, widget.engine.speed, widget.engine.size) == \
        (2.0, 0.5, 1.5)
    # ... and they survive the engine being replaced under a theme switch.
    widget.set_theme("drift")
    assert (widget.engine.blur, widget.engine.speed, widget.engine.size) == \
        (2.0, 0.5, 1.5)


# ---------------------------------------------------------------------------
# The aurora
# ---------------------------------------------------------------------------

def aurora(palette="borealis", seed=7, curtains=None, **kwargs):
    engine = make_engine("aurora", palette, DARK, seed=seed, **kwargs)
    if curtains is not None:
        engine.curtains = engine.curtains[:curtains]
    return engine


def lower_edge(engine, curtain_index=0, samples=240):
    """The curtain's lower edge, sampled finely along the arc.

    Straight from the model rather than from ``geometry``, so the profile can
    be sampled more finely than the forty columns the painter uses.
    """
    curtain = engine.curtains[curtain_index]
    return [engine.fold(curtain, i / (samples - 1), engine.time)
            for i in range(samples)]


SAMPLES = 240


def smooth(profile, window):
    """Moving average. Used to isolate the slow fold from the ripples riding
    on it — the fast ones alias badly under cross-correlation, being several
    of their own wavelengths along in six seconds."""
    out = []
    for i in range(len(profile)):
        low, high = max(0, i - window), min(len(profile), i + window + 1)
        out.append(sum(profile[low:high]) / (high - low))
    return out


def test_the_folds_travel_along_the_arc():
    """The property that makes it an aurora rather than a flag.

    Two frames six seconds apart, cross-correlated: the fold pattern lines up
    again after a shift ALONG the arc, and that shift is forwards and about
    as far as the dominant fold's declared travel speed says it should be.
    The profile is low-passed first, which leaves the long fold — the fast
    ripples travel more than a wavelength in six seconds and would alias.
    """
    engine = aurora()
    dt = 6.0
    engine.set_time(0.0)
    first = smooth(lower_edge(engine, samples=SAMPLES), 30)
    engine.set_time(dt)
    second = smooth(lower_edge(engine, samples=SAMPLES), 30)

    lag, _error = best_shift(first, second, limit=60)
    speed = amb.AURORA_FOLDS[0][2] * amb.AURORA_DEPTHS[0][0]
    expected = speed * dt * (SAMPLES - 1)
    assert lag > 0, "the folds did not travel along the arc"
    assert lag == pytest.approx(expected, rel=0.4, abs=3), \
        f"shifted {lag} samples, the dominant fold travels {expected:.1f}"


def test_the_fold_is_several_frequencies_and_not_one_sine():
    """A single sine, whatever its speed, would realign perfectly under some
    rigid shift. This one cannot, because its components travel at different
    speeds — which is the same fact as 'it is a wave train, not an object
    being moved'."""
    engine = aurora()
    engine.set_time(0.0)
    first = lower_edge(engine, samples=SAMPLES)
    engine.set_time(6.0)
    second = lower_edge(engine, samples=SAMPLES)
    _lag, residual = best_shift(first, second, limit=SAMPLES // 3)

    mean = sum(first) / len(first)
    variance = sum((v - mean) ** 2 for v in first) / len(first)
    assert residual > 0.05 * variance, (
        f"no rigid shift left more than {residual / variance:.1%} of the "
        "profile unexplained — this is one frequency")

    # ... and the ripples are there in the shape itself, not only in its
    # motion: a single slow sine turns twice across the arc.
    slope = [b - a for a, b in zip(first, first[1:])]
    turns = sum(1 for a, b in zip(slope, slope[1:]) if a * b < 0)
    assert turns >= 5, f"the arc turns {turns} times; one sine turns twice"


def test_the_curtains_travel_at_different_rates():
    """Three curtains folding in step is one curtain drawn three times."""
    engine = aurora()
    engine.set_time(0.0)
    first = [smooth(lower_edge(engine, i, SAMPLES), 30) for i in range(3)]
    engine.set_time(6.0)
    second = [smooth(lower_edge(engine, i, SAMPLES), 30) for i in range(3)]
    lags = [best_shift(a, b, limit=60)[0] for a, b in zip(first, second)]
    assert len(set(lags)) > 1, f"all three curtains shifted alike: {lags}"


def _ray_columns(image: QImage):
    """Which columns of the frame the rays are in."""
    profile = _striation(image)
    top = max(profile)
    return [i for i in range(1, len(profile) - 1)
            if profile[i] > profile[i - 1] and profile[i] >= profile[i + 1]
            and profile[i] > 0.25 * top]


def _agreement(a, b, tolerance=1):
    """Share of ``a``'s columns that have a partner in ``b``."""
    if not a:
        return 0.0
    return sum(1 for i in a
               if any(abs(i - j) <= tolerance for j in b)) / len(a)


def test_the_rays_do_not_slide_sideways():
    """The folds travel; the curtain does not.

    The rays come from a brush that is never translated horizontally, so the
    striations must stay in the same columns of the frame while the folds run
    along them. If the whole band were sliding — the thing this animation was
    asked *not* to do — this is what would move.

    The last two lines are the control: displace the same measurement by four
    pixels and the agreement collapses, which is what says the first
    assertion is measuring position and not merely counting rays.
    """
    engine = aurora(curtains=1)
    engine.set_time(0.0)
    first = _ray_columns(render(engine))
    engine.set_time(23.0)
    later = _ray_columns(render(engine))
    assert len(first) > 20, "no rays to track"
    assert _agreement(first, later) > 0.8, "the ray pattern slid sideways"
    assert _agreement(first, [c + 4 for c in later]) < 0.5


def _striation(image: QImage):
    """Column-wise high-frequency energy: where the rays are."""
    grid = rows(image)
    columns = []
    for x in range(image.width()):
        column = [luminance(grid[y][x]) for y in range(image.height())]
        columns.append(sum(column) / len(column))
    # High-pass, so the answer is about the ray comb and not about the
    # curtain's overall brightness sloping across the frame.
    window = 9
    smoothed = [
        sum(columns[max(0, i - window):i + window + 1])
        / len(columns[max(0, i - window):i + window + 1])
        for i in range(len(columns))]
    return [c - s for c, s in zip(columns, smoothed)]


def test_the_curtain_is_made_of_vertical_rays():
    """Striations are the defining feature, so their absence is a bug and
    not a style change."""
    engine = aurora(curtains=1)
    engine.set_time(9.0)
    profile = _striation(render(engine))
    crossings = sum(1 for a, b in zip(profile, profile[1:]) if a * b < 0)
    assert crossings >= 12, \
        f"only {crossings} bright/dark alternations across the frame"


def test_the_brightness_pulse_travels_and_is_not_the_fold():
    """Surges run along the arc on their own schedule. If they were driven
    by the folding they would shift by the same amount, and they must not."""
    engine = aurora()
    curtain = engine.curtains[0]
    # Short enough that the surge travels less than half its own wavelength:
    # it is periodic, and a correlation over a longer step would come back
    # with the alias rather than the travel.
    dt = 2.0

    def pulse_profile(t):
        return [engine.pulse(curtain, i / (SAMPLES - 1), t)
                for i in range(SAMPLES)]

    engine.set_time(0.0)
    fold_first = smooth(lower_edge(engine, samples=SAMPLES), 30)
    pulse_first = pulse_profile(0.0)
    engine.set_time(dt)
    fold_second = smooth(lower_edge(engine, samples=SAMPLES), 30)
    pulse_second = pulse_profile(dt)

    fold_lag = best_shift(fold_first, fold_second, 60)[0]
    pulse_lag = best_shift(pulse_first, pulse_second, 40)[0]
    assert pulse_lag > 0, "the surge does not travel"
    assert pulse_lag == pytest.approx(
        amb.AURORA_PULSE[2] * dt * (SAMPLES - 1), rel=0.25, abs=2)
    assert pulse_lag > 2 * fold_lag, \
        f"the surge ({pulse_lag}) is not faster than the fold ({fold_lag})"

    # And it is visible: the frame's brightness varies along the arc.
    engine.set_time(9.0)
    grid = rows(render(aurora(curtains=1)))
    band = [sum(luminance(grid[y][x]) for y in range(len(grid)))
            for x in range(len(grid[0]))]
    assert max(band) > min(band) * 1.15, "the arc is evenly lit end to end"


def test_the_colour_runs_green_through_the_body_and_red_at_the_top():
    """The vertical order is atomic physics: 557.7 nm oxygen through the
    body, 630.0 nm oxygen only where the air is thin enough for it."""
    engine = aurora(curtains=1)
    engine.set_time(9.0)
    grid = rows(render(engine))
    page = luminance(QColor(DARK).getRgb()[:3])
    lit = [y for y, row in enumerate(grid)
           if max(luminance(p) for p in row) > page + 6]
    assert lit, "nothing was painted"
    top, bottom = min(lit), max(lit)
    height = bottom - top
    assert height > 40

    def balance(y0, y1):
        """Mean green minus mean red over a horizontal slice."""
        slice_rows = grid[y0:y1]
        greens = sum(p[1] for row in slice_rows for p in row)
        reds = sum(p[0] for row in slice_rows for p in row)
        return (greens - reds) / max(1, len(slice_rows) * len(grid[0]))

    body = balance(bottom - int(0.35 * height), bottom)
    crown = balance(top, top + int(0.2 * height))
    assert body > 0, f"the body is not green (g-r = {body:.1f})"
    assert crown < body, \
        f"the top is not redder than the body ({crown:.1f} vs {body:.1f})"


def test_the_lower_edge_is_sharp_and_the_top_is_diffuse():
    """The asymmetry is as recognisable as the colour: the bottom is where
    the particles run out of altitude, the top is where the emission just
    thins away."""
    engine = aurora(curtains=1)
    engine.set_time(9.0)
    grid = rows(render(engine))
    page = luminance(QColor(DARK).getRgb()[:3])
    # Above the page, not above zero: a fifth of the peak is below the page
    # colour itself, so thresholding raw luminance selects the whole frame.
    column = [sum(luminance(p) for p in row) / len(row) - page for row in grid]
    peak = max(column)
    lit = [y for y, v in enumerate(column) if v > peak * 0.2]
    lowest, highest = max(lit), min(lit)

    def steepest(y0, y1):
        y0, y1 = max(0, y0), min(len(column), y1)
        return max(abs(b - a)
                   for a, b in zip(column[y0:y1], column[y0 + 1:y1]))

    bottom_edge = steepest(lowest - 10, lowest + 8)
    top_edge = steepest(highest - 8, highest + 10)
    assert bottom_edge > top_edge * 2.5, \
        f"lower edge {bottom_edge:.2f}/px, upper edge {top_edge:.2f}/px"


def test_three_curtains_at_different_depths():
    engine = aurora()
    assert len(engine.curtains) == 3
    bases = {round(c.y, 3) for c in engine.curtains}
    assert len(bases) == 3
    engine.set_time(5.0)
    one = render(aurora(curtains=1))
    three = render(engine)
    assert one != three
    assert lit_pixels(three) > lit_pixels(one)


def test_the_aurora_geometry_is_the_model_it_documents():
    """``geometry`` inlines ``fold`` and ``pulse`` for speed. This is what
    stops the two copies drifting apart."""
    engine = aurora()
    engine.set_time(13.0)
    stride = amb.AURORA_COLUMNS + 1
    samples = engine.geometry(W, H)
    for index, curtain in enumerate(engine.curtains):
        zero, ray = engine.anchor(curtain, H)
        for i in range(stride):
            u = i / amb.AURORA_COLUMNS
            x, y, height, bright = samples[index * stride + i]
            expected_y = (
                curtain.y
                + curtain.drift * math.sin(curtain.rate * engine.time
                                           + curtain.phase)
                + curtain.tilt * (u - 0.5) * engine.size
                + engine.fold(curtain, u, engine.time)) * H
            assert y == pytest.approx(expected_y, abs=1e-6)
            assert height == pytest.approx(max(0.0, y - (zero - ray)),
                                           abs=1e-6)
            depth = amb.AURORA_DEPTHS[curtain.depth
                                      % len(amb.AURORA_DEPTHS)][2]
            assert bright == pytest.approx(
                depth * engine.pulse(curtain, u, engine.time), abs=1e-9)


# ---------------------------------------------------------------------------
# The borealis palette
# ---------------------------------------------------------------------------

def test_the_borealis_palette_is_the_real_emission_lines():
    """Named wavelengths, pinned values. A palette called Aurora borealis
    whose colours were chosen by eye would be a different thing."""
    assert palette_colors("aurora", "borealis") == (
        "#7CFC9E",      # atomic oxygen, 557.7 nm — the dominant green
        "#FF3C5A",      # atomic oxygen, 630.0 nm — the high-altitude red
        "#5B6BFF",      # ionised nitrogen, 427.8 nm — the violet fringe
        "#D9FFA8",      # green over red — the pale yellow-green overlap
    )
    spec = amb.PALETTE_SETS["borealis"]
    assert spec.label == "Aurora borealis"
    for wavelength in ("557.7", "630.0", "427.8"):
        assert wavelength in spec.note


def test_borealis_is_offered_where_the_animation_reads_as_sky():
    """Aurora, blobs and the starfield: all three are a sky. Ripple is rain
    on water, and a set named after the northern lights would be decoration
    there rather than a colour choice."""
    for theme in ("aurora", "blobs", "drift"):
        assert "borealis" in palettes_for(theme), theme
    assert "borealis" not in palettes_for("ripple")


def test_the_borealis_palette_reaches_the_curtain():
    """Offered is not the same as used."""
    borealis = aurora(palette="borealis", curtains=1)
    mono = aurora(palette="mono", curtains=1)
    borealis.set_time(9.0)
    mono.set_time(9.0)
    assert borealis.geometry(W, H) == mono.geometry(W, H), \
        "the palette moved something"
    assert render(borealis) != render(mono)

    # The green is the body colour, not one of three interchangeable inks.
    for curtain in borealis.curtains:
        roles = borealis.ramp_colors(curtain)
        assert roles["main"].green() > roles["main"].red()
        assert roles["high"].red() > roles["high"].green()
        assert roles["fringe"].blue() > roles["fringe"].green()


# ---------------------------------------------------------------------------
# The preferences
# ---------------------------------------------------------------------------

@pytest.fixture
def prefs(monkeypatch, tmp_path):
    """Route the preference store into ``tmp_path`` and prove it landed there.

    ``QSettings("spacr", "qt")`` — the (organization, application) form the
    module uses — resolves to the NATIVE location whatever ``setPath`` says,
    so redirecting the class is not isolation. Replacing the accessor is, and
    the assertion below is what makes that claim checkable rather than
    hopeful: this project has already erased a developer's real preferences
    once by assuming the redirect worked.
    """
    from spacr.qt import preferences as module

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(module, "_settings", lambda: store)
    resolved = module._settings().fileName()
    assert str(tmp_path) in resolved, (
        f"preference isolation failed: {resolved} is outside {tmp_path}")
    return module


def test_the_preferences_default_to_the_shipped_animation(prefs):
    assert prefs.get_ambient_blur() == DEFAULT_BLUR
    assert prefs.get_ambient_speed() == DEFAULT_SPEED
    assert prefs.get_ambient_size() == DEFAULT_SIZE


@pytest.mark.parametrize("getter,setter,limits", [
    ("get_ambient_blur", "set_ambient_blur", BLUR_RANGE),
    ("get_ambient_speed", "set_ambient_speed", SPEED_RANGE),
    ("get_ambient_size", "set_ambient_size", SIZE_RANGE),
])
def test_the_preferences_round_trip_and_clamp(prefs, getter, setter, limits):
    low, high = limits
    middle = (low + high) / 2
    getattr(prefs, setter)(middle)
    assert getattr(prefs, getter)() == pytest.approx(middle)
    getattr(prefs, setter)(high * 100)
    assert getattr(prefs, getter)() == high
    getattr(prefs, setter)(-999)
    assert getattr(prefs, getter)() == low
    # A hand-edited INI is a real source of these.
    prefs._settings().setValue(f"prefs/{getter[4:]}", "not a number")
    assert low <= getattr(prefs, getter)() <= high


def test_a_new_backdrop_picks_the_preferences_up(prefs, qtbot):
    """Screens are built long after Preferences is saved, and they must not
    come up with the shipped motion and wait for the next apply."""
    prefs.set_ambient_blur(2.0)
    prefs.set_ambient_speed(0.5)
    prefs.set_ambient_size(1.5)
    widget = AmbientWidget(theme="blobs", palette="spacr", background=DARK,
                           seed=1)
    qtbot.addWidget(widget)
    assert (widget.blur(), widget.speed(), widget.size_scale()) == \
        (2.0, 0.5, 1.5)
    assert widget.engine.blur == 2.0


def test_the_dialog_offers_the_three_controls_and_saves_them(prefs, qtbot,
                                                             qt_theme_applied):
    from PySide6.QtWidgets import QDialogButtonBox, QSlider

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    sliders = {s.objectName(): s for s in dialog.findChildren(QSlider)}
    for name in ("AmbientBlur", "AmbientSpeed", "AmbientSize"):
        assert name in sliders, sorted(sliders)
        assert sliders[name].value() == 100, "does not open on the default"

    sliders["AmbientBlur"].setValue(180)
    sliders["AmbientSpeed"].setValue(60)
    sliders["AmbientSize"].setValue(140)
    dialog.findChild(QDialogButtonBox).button(
        QDialogButtonBox.Save).click()

    assert prefs.get_ambient_blur() == pytest.approx(1.8)
    assert prefs.get_ambient_speed() == pytest.approx(0.6)
    assert prefs.get_ambient_size() == pytest.approx(1.4)


def test_the_controls_grey_out_with_the_animation(prefs, qtbot,
                                                  qt_theme_applied):
    from PySide6.QtWidgets import QSlider

    from spacr.qt.widgets.toggle import Toggle

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    toggle = dialog.findChild(Toggle, "AmbientEnabled")
    sliders = [s for s in dialog.findChildren(QSlider)
               if s.objectName() in ("AmbientBlur", "AmbientSpeed",
                                     "AmbientSize")]
    assert len(sliders) == 3
    toggle.setChecked(False)
    assert not any(s.isEnabled() for s in sliders)
    toggle.setChecked(True)
    assert all(s.isEnabled() for s in sliders)
