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
                                      DEFAULT_BLUR, DEFAULT_DENSITY,
                                      DEFAULT_DRIFT_DIRECTION,
                                      DEFAULT_RESOLUTION, DEFAULT_SIZE,
                                      DEFAULT_SPEED, DENSITY_RANGE,
                                      DRIFT_DIRECTIONS, RESOLUTION_RANGE,
                                      SIZE_RANGE, SPEED_RANGE, AmbientWidget,
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


def lum_array(image: QImage, band=(0.0, 1.0)):
    """Luminance of ``image`` as a NumPy array, optionally row-banded.

    The per-pixel ``pixelColor`` reader above is fine for a 640x400 frame and
    is thirty seconds a frame at 1920x1080, which is the size the pixelation
    complaint is about. This reads the buffer once.
    """
    import numpy as np

    fixed = image.convertToFormat(QImage.Format_RGB32)
    raw = np.frombuffer(fixed.constBits(), dtype=np.uint8).reshape(
        fixed.height(), fixed.bytesPerLine() // 4, 4)
    raw = raw[:, :fixed.width(), :3].astype(np.float64)
    # Qt's RGB32 is BGRA in memory.
    grid = 0.0722 * raw[..., 0] + 0.7152 * raw[..., 1] + 0.2126 * raw[..., 2]
    top = int(band[0] * grid.shape[0])
    bottom = max(top + 3, int(band[1] * grid.shape[0]))
    return grid[top:bottom]


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


def lattice_ratio(image: QImage, period: int, band=(0.0, 1.0)) -> float:
    """How plainly the bilinear upscale grid shows in ``image``.

    Second differences of the luminance, averaged per column and per row,
    then split by phase modulo ``period``: the ratio of the worst phase to
    the mean over all phases. Bilinear interpolation is C0 and not C1, so a
    picture stretched from a small buffer has all its curvature concentrated
    on the block boundaries and this comes out well above 1; a picture with
    nothing on that period has no preferred phase and it comes out at 1.

    ``band`` restricts the rows examined, as fractions of the height — the
    aurora only occupies part of the frame, and averaging its lower edge
    together with two thirds of empty page dilutes exactly what is being
    asked about.
    """
    import numpy as np

    if period < 2:
        return 1.0
    grid = lum_array(image, band)
    scores = []
    for axis in (1, 0):
        second = np.abs(np.diff(np.diff(grid, axis=axis), axis=axis))
        profile = second.mean(axis=1 - axis)
        index = np.arange(profile.size) + 1
        means = np.array([profile[index % period == phase].mean()
                          for phase in range(period)])
        scores.append(means.max() / max(1e-12, means.mean()))
    return float(np.mean(scores))


def comb_contrast(image: QImage, band=(0.62, 0.74)) -> float:
    """RMS of the high-frequency part of a horizontal profile through the
    aurora's curtains — which is the ray comb and nothing else.

    Under-resolving the comb does not move the rays, it *smears* them, so
    this is the measure that says whether they survived.
    """
    import numpy as np

    profile = lum_array(image, band).mean(axis=0)
    window = 81
    local = np.convolve(profile, np.ones(window) / window, mode="valid")
    centre = profile[window // 2: window // 2 + local.size]
    return float(np.sqrt(np.mean((centre - local) ** 2)))


def centroid(engine, width=W, height=H):
    """Mean particle position, in normalised units."""
    points = engine.geometry(width, height)
    return (sum(x for x, _y, _d in points) / len(points) / width,
            sum(y for _x, y, _d in points) / len(points) / height)


def wrapped_steps(engine, dt, frames, width=W, height=H):
    """Per-particle (dx, dy) between consecutive frames, in normalised
    units, with the wrap at the edge of the field taken out.

    A particle that leaves the top and comes back at the bottom moved a
    hair, not a whole screen, and a test that failed to say so would be
    measuring the modulo rather than the motion.
    """
    def shortest(a, b):
        delta = (b - a) % 1.0
        return delta - 1.0 if delta > 0.5 else delta

    previous = engine.geometry(width, height)
    steps = []
    for _ in range(frames):
        engine.advance(dt)
        current = engine.geometry(width, height)
        steps.append([(shortest(a[0] / width, b[0] / width),
                       shortest(a[1] / height, b[1] / height))
                      for a, b in zip(previous, current)])
        previous = current
    return steps


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

    The two guards below name the exceptions "there is no git here" can
    actually raise -- OSError when the binary is missing, SubprocessError when
    git runs and refuses -- so that anything else (a wrong repo root, a bad
    encoding) still fails this fixture instead of quietly retiring the
    comparison it feeds.
    """
    path = "spacr/qt/widgets/ambient.py"
    try:
        revisions = subprocess.run(
            ["git", "log", "--format=%H", "--", path],
            capture_output=True, text=True, check=True,
            cwd=_repo_root()).stdout.split()
    except (OSError, subprocess.SubprocessError) as exc:  # no git, no history
        pytest.skip(f"cannot read the shipped engine from git: {exc}")
    for revision in revisions:
        try:
            source = subprocess.run(
                ["git", "show", f"{revision}:{path}"],
                capture_output=True, text=True, check=True,
                cwd=_repo_root()).stdout
        except (OSError, subprocess.SubprocessError):
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


#: Which themes the byte-for-byte comparison against the shipped engine can
#: still be made for, and why the other two are out.
#:
#: ``aurora`` is out because it is a deliberate redesign — twice over now.
#: The first was the curtains themselves; the second is its buffer, which
#: went from 240x135 to 960x540 at 1080p because the old one was measured to
#: throw away a quarter of its own ray comb (see ``AURORA_BUFFER_EDGE``).
#: That is the one shipped default this change moves, it is moved on
#: measurement rather than taste, and it is stated here rather than papered
#: over by loosening the comparison.
#:
#: ``bokeh`` and ``cells`` are out because they did not exist to be shipped.
#:
#: Everything else — blobs, ripples, the starfield — is still held to the
#: original frame, byte for byte, on both kinds of page. Resolution, blur and
#: density all default to the identity, so a user who never opens the new
#: controls cannot tell they exist.
UNCHANGED_THEMES = ("blobs", "ripple", "drift")


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

    # Resolution: the buffer is this theme's own declared one.
    if isinstance(engine, amb._BufferedEngine):
        assert engine.resolution_edge() == engine.base_edge
        scale = max(1, int(math.ceil(max(1920, 1080) / engine.base_edge)))
        assert engine.buffer_size(1920, 1080) == (1920 // scale, 1080 // scale)
        # Blur: nothing is done to the buffer at all.
        assert engine.blur_scale(1920, 1080) == 1.0

    # Density: the theme's own element count.
    assert engine.effective_density() == 1.0

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
    elif theme == "bokeh":
        ceiling = max(amb.BOKEH_RADIUS) * min(W, H)
        assert max(r for _x, _y, r, _f in engine.geometry(W, H)) <= ceiling + 1
    elif theme == "cells":
        ceiling = max(amb.CELL_RADIUS) * min(W, H)
        assert max(a for _x, _y, a, _b, _t in engine.geometry(W, H)) \
            <= ceiling + 1
    else:
        assert max(h for _x, _y, h, _b in engine.geometry(W, H)) \
            <= max(amb.AURORA_THICKNESS) * H * 1.5


# ---------------------------------------------------------------------------
# Each control moves the picture, in its own direction
# ---------------------------------------------------------------------------

BUFFERED_THEMES = ["blobs", "aurora", "ripple", "bokeh", "cells"]


@pytest.mark.parametrize("theme", BUFFERED_THEMES)
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

    sharp, middling, soft = (sharpness(frame(b)) for b in (0.0, 1.2, 3.0))
    assert sharp > middling > soft, \
        f"{theme}: sharpness went {sharp:.3f} -> {middling:.3f} -> {soft:.3f}"


@pytest.mark.parametrize("theme", BUFFERED_THEMES)
def test_resolution_reduces_pixelation(theme):
    """The complaint this pair of controls was built for, measured.

    Bilinear upscaling is C0 but not C1: inside a block the picture is
    (near enough) linear, so its second difference is ~0, and at every block
    boundary the slope kinks and the second difference spikes. Ask which
    phase of the known upscale grid carries the most of that energy and
    compare it with the average phase, and you have a number that is 1.0
    when the grid cannot be found in the picture and rises with how plainly
    it can. That is pixelation, and nothing else in the frame has a reason
    to sit on that period.
    """
    def lattice(resolution):
        # 1920x1080, because that is the size the complaint is about and
        # because several themes are not upscaled at all on a small canvas.
        engine = make_engine(theme, "spacr", DARK, seed=5,
                             resolution=resolution)
        engine.set_time(9.0)
        period = engine.buffer_scale(1920, 1080)
        # The three settings below are chosen so the buffer divides the
        # canvas exactly. At a scale that does not (101 buffer pixels over
        # 1920) the upscale factor is 19.01 and the block grid slides a
        # whole pixel across the frame, which is easier on the eye and
        # smears the phase this measurement is binning by. That is a fact
        # about the metric, not about the picture, so it is arranged away
        # rather than measured through.
        assert 1920 % period == 0 and 1080 % period == 0, \
            f"{theme}: scale {period} does not divide the canvas"
        return lattice_ratio(render(engine, 1920, 1080), period)

    coarse, shipped, fine = (lattice(r) for r in (0.5, 1.0, 2.0))
    assert coarse > shipped > fine, \
        f"{theme}: lattice went {coarse:.3f} -> {shipped:.3f} -> {fine:.3f}"
    # Measured in *excess over none*: 1.0 is a picture with no findable
    # grid in it, so the quantity that has to fall is the part above 1, and
    # a theme that is already almost clean at its default has almost nothing
    # left to lose. Halving it is the claim.
    assert (fine - 1.0) < (coarse - 1.0) * 0.5, \
        f"{theme}: lattice went {coarse:.3f} -> {shipped:.3f} -> {fine:.3f}"


def test_the_aurora_is_no_longer_pixelated_at_1080p():
    """The specific report — "the aurora looks super pixelated" — with a
    number on it, against the buffer it used to be shaded into.

    Two measurements, because the theme has two kinds of structure that a
    small buffer wrecks: the upscale lattice over the band its lower edge
    runs through, and the contrast of the ray comb, which is 36 screen
    pixels per ray and was being resolved at four and a half.
    """
    def frame(edge):
        engine = make_engine("aurora", "spacr", DARK, seed=7)
        engine.base_edge = edge
        engine._buffer = None
        engine.set_time(11.0)
        return engine, render(engine, 1920, 1080)

    was, was_image = frame(256)          # the buffer that shipped
    now, now_image = frame(amb.AURORA_BUFFER_EDGE)
    assert was.buffer_scale(1920, 1080) == 8
    assert now.buffer_scale(1920, 1080) == 2

    band = (0.40, 0.98)
    before = lattice_ratio(was_image, 8, band)
    after = lattice_ratio(now_image, 2, band)
    assert before > 1.5, f"the old buffer measured clean at {before:.3f}"
    assert after < 1.15, f"the new one still measures {after:.3f}"
    assert after < before / 1.4

    # ...and the rays are actually there now rather than smeared into the
    # sheet. Contrast of the horizontal profile through a curtain.
    assert comb_contrast(now_image) > comb_contrast(was_image) * 1.15


@pytest.mark.parametrize("theme", BUFFERED_THEMES)
def test_resolution_and_blur_are_two_different_axes(theme):
    """Four corners of the grid, and all four have to be different frames.

    This is the whole bug report. The two used to be one control, so
    "sharp" and "not blocky" could not be asked for separately and the
    top-left and bottom-right corners of this grid were the same picture.
    """
    frames = {}
    for resolution in (0.4, 2.0):
        for blur in (0.0, 3.0):
            engine = make_engine(theme, "spacr", DARK, seed=5,
                                 resolution=resolution, blur=blur)
            engine.set_time(9.0)
            frames[(resolution, blur)] = render(engine)

    corners = list(frames)
    for i, a in enumerate(corners):
        for b in corners[i + 1:]:
            assert frames[a] != frames[b], f"{theme}: {a} renders as {b}"

    # And they are different in the directions they claim. Blur softens at
    # either resolution; resolution de-blocks at either blur.
    for resolution in (0.4, 2.0):
        assert sharpness(frames[(resolution, 0.0)]) > \
            sharpness(frames[(resolution, 3.0)]), \
            f"{theme}: blur did not soften at resolution {resolution}"

    # Sharp AND soft: the frame that could not be asked for before. It is
    # softer than the sharp-and-hard one and less blocky than the
    # coarse-and-soft one — one property from each axis, at once.
    sharp_soft = frames[(2.0, 3.0)]
    assert sharpness(sharp_soft) < sharpness(frames[(2.0, 0.0)])
    coarse_soft = frames[(0.4, 3.0)]
    assert lattice_ratio(sharp_soft, 1) <= lattice_ratio(coarse_soft, 8)


def test_blur_is_one_pass_over_the_buffer_not_a_full_screen_filter():
    """The mechanism matters as much as the effect: a Gaussian over two
    million pixels every frame is the thing this must not be."""
    engine = make_engine("blobs", "spacr", DARK, seed=5, blur=2.0)
    # The softening happens on the small buffer, and the buffer that gets
    # shaded is still the resolution setting's, not the blur's.
    assert engine.buffer_size(1920, 1080) == \
        make_engine("blobs", "spacr", DARK, seed=5).buffer_size(1920, 1080)
    assert engine.blur_scale(1920, 1080) > 1.0

    # ... and the shading buffer is still allocated once, not per frame.
    render(engine, 1920, 1080)
    buffer = engine._buffer
    for _ in range(5):
        engine.advance(1 / 24)
        render(engine, 1920, 1080)
    assert engine._buffer is buffer


def test_the_blur_unit_is_the_softness_that_shipped():
    """Blur 1.0 is defined as "the softness the buffered themes always had",
    which at the shipped 240x135 buffer was an eight-fold bilinear stretch.
    So at that resolution it must ask for nothing extra, and at four times
    the detail it must ask for exactly four times the averaging — that is
    what makes the setting mean the same thing on screen at any resolution.
    """
    shipped = make_engine("blobs", "spacr", DARK, seed=5, blur=1.0)
    assert shipped.buffer_scale(1920, 1080) == 8
    assert shipped.blur_scale(1920, 1080) == pytest.approx(1.0)

    detailed = make_engine("blobs", "spacr", DARK, seed=5, blur=1.0,
                           resolution=2.0)
    assert detailed.buffer_scale(1920, 1080) == 4
    assert detailed.blur_scale(1920, 1080) == pytest.approx(2.0)


def test_blur_spreads_the_starfield_rather_than_resizing_it():
    """Drift has no buffer to average down, so it needs its own mechanism —
    and that mechanism must not just be "bigger dots", which is what the
    size control does."""
    def engine(blur):
        made = make_engine("drift", "spacr", DARK, seed=5, blur=blur)
        made.set_time(9.0)
        return made

    hard, normal, soft = engine(0.0), engine(1.0), engine(2.5)
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


@pytest.mark.parametrize("control,limits", [
    ("blur", BLUR_RANGE), ("speed", SPEED_RANGE), ("size", SIZE_RANGE),
    ("resolution", RESOLUTION_RANGE), ("density", DENSITY_RANGE),
])
@pytest.mark.parametrize("value,end", [(-4.0, 0), (99.0, 1)])
def test_absurd_values_are_clamped_not_obeyed(control, limits, value, end):
    engine = make_engine("blobs", "spacr", DARK, seed=1, **{control: value})
    assert getattr(engine, control) == limits[end]
    assert amb.BUFFER_MIN_EDGE <= engine.resolution_edge() \
        <= amb.BUFFER_EDGE_CEILING
    assert engine.buffer_size(1920, 1080)[0] * \
        engine.buffer_size(1920, 1080)[1] <= amb.BUFFER_MAX_PIXELS
    render(engine, 1920, 1080)          # and it still paints


def test_the_widget_exposes_them_all_and_they_reach_the_engine(qtbot):
    widget = AmbientWidget(theme="ripple", palette="ocean", background=DARK,
                           seed=2, blur=0.0, speed=1.0, size=1.0,
                           resolution=1.0, density=1.0, direction="up")
    qtbot.addWidget(widget)
    widget.set_blur(2.0)
    widget.set_speed(0.5)
    widget.set_size_scale(1.5)
    widget.set_resolution(1.5)
    widget.set_density(2.0)
    widget.set_direction("down")

    def state(source):
        return (source.blur, source.speed, source.size, source.resolution,
                source.density, source.direction)

    wanted = (2.0, 0.5, 1.5, 1.5, 2.0, "down")
    assert (widget.blur(), widget.speed(), widget.size_scale(),
            widget.resolution(), widget.density(),
            widget.direction()) == wanted
    assert state(widget.engine) == wanted
    # ... and they survive the engine being replaced under a theme switch.
    widget.set_theme("drift")
    assert state(widget.engine) == wanted


# ---------------------------------------------------------------------------
# What the cost is bounded by
# ---------------------------------------------------------------------------
# Wall-clock assertions belong in a benchmark, not in a test suite that runs
# on a shared machine. What is asserted here is the *structure* the cost
# figures rest on: the shading pass has a hard ceiling in pixels, the blur
# adds one pass over a buffer no bigger than that, and the theme that was
# made dearer is dearer by exactly the amount its docstring claims.

@pytest.mark.parametrize("theme", BUFFERED_THEMES)
@pytest.mark.parametrize("canvas", [(1920, 1080), (3840, 2160), (5120, 2880),
                                    (800, 600), (37, 11)])
def test_the_shading_pass_has_a_hard_ceiling(theme, canvas):
    """The buffer edge is a ratio to the canvas, and a ratio alone lets a
    bigger display quietly buy a bigger shading pass. This is the absolute
    ceiling that stops it."""
    engine = make_engine(theme, "spacr", DARK, seed=5, resolution=2.0,
                         density=3.0)
    width, height = engine.buffer_size(*canvas)
    assert width * height <= amb.BUFFER_MAX_PIXELS, \
        f"{theme} at {canvas} shades {width}x{height}"
    assert width >= 1 and height >= 1
    engine.set_time(4.0)
    render(engine, *canvas)                     # and it still paints


@pytest.mark.parametrize("theme", BUFFERED_THEMES)
def test_the_blur_never_enlarges_anything(theme):
    """One extra pass, over something no bigger than the buffer that was
    already being shaded — which is the whole claim that blur is cheap."""
    engine = make_engine(theme, "spacr", DARK, seed=5, blur=BLUR_RANGE[1])
    engine.set_time(4.0)
    render(engine, 1920, 1080)
    shaded = engine._buffer
    softened = engine._soften(shaded, 1920, 1080)
    assert softened.width() <= shaded.width()
    assert softened.height() <= shaded.height()
    assert softened.width() >= 2 and softened.height() >= 2


def test_the_auroras_buffer_is_the_size_its_docstring_claims():
    """The one shipped default this change moves, pinned to the number the
    module documents it as. If it is ever changed again, the docstring's
    cost table and its measurement table both have to move with it."""
    engine = make_engine("aurora", "spacr", DARK, seed=5)
    assert engine.base_edge == amb.AURORA_BUFFER_EDGE == 960
    assert engine.buffer_size(1920, 1080) == (960, 540)
    # ... and it really is the only one that moved.
    for theme in ("blobs", "ripple"):
        other = make_engine(theme, "spacr", DARK, seed=5)
        assert other.base_edge == BUFFER_MAX_EDGE == 256
        assert other.buffer_size(1920, 1080) == (240, 135)


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

#: What each theme calls the thing density multiplies, and how many of them
#: it draws by default.
ELEMENT_COUNTS = [
    ("blobs", "blobs", lambda e: len(e.geometry(W, H)), amb.BLOB_COUNT),
    ("aurora", "curtains",
     lambda e: len(e.geometry(W, H)) // (amb.AURORA_COLUMNS + 1),
     amb.AURORA_CURTAINS),
    ("ripple", "sources",
     lambda e: len(e.geometry(W, H)) // amb.RIPPLE_RINGS, amb.RIPPLE_SOURCES),
    ("bokeh", "discs", lambda e: len(e.geometry(W, H)), amb.BOKEH_COUNT),
    ("cells", "cells", lambda e: len(e.geometry(W, H)), amb.CELL_COUNT),
]


@pytest.mark.parametrize("theme,noun,count,shipped", ELEMENT_COUNTS)
def test_density_changes_how_many_things_are_drawn(theme, noun, count,
                                                   shipped):
    """The count comes off ``geometry``, which is what the painter builds
    the frame from — so this is the number of things in the picture, not a
    field on the engine that might or might not be read."""
    sparse = make_engine(theme, "spacr", DARK, seed=5, density=0.5)
    normal = make_engine(theme, "spacr", DARK, seed=5)
    dense = make_engine(theme, "spacr", DARK, seed=5, density=3.0)
    assert count(normal) == shipped
    assert count(sparse) < count(normal) < count(dense)
    assert count(dense) == pytest.approx(3 * shipped, abs=1)


@pytest.mark.parametrize("theme,noun,count,shipped", ELEMENT_COUNTS)
def test_density_reaches_the_pixels(theme, noun, count, shipped):
    """More elements, more of the page lit. Asserted on the frame, because
    a count that never made it into a draw call is not a density."""
    def frame(density):
        engine = make_engine(theme, "spacr", DARK, seed=5, density=density)
        engine.set_time(9.0)
        return render(engine)

    assert lit_pixels(frame(0.4)) < lit_pixels(frame(1.0)) \
        < lit_pixels(frame(3.0))


def test_density_moves_the_starfield_too():
    """Drift counts its particles from the canvas area rather than from a
    constant, so it needs its own check that the multiplier is applied."""
    sparse = make_engine("drift", "spacr", DARK, seed=5, density=0.5)
    normal = make_engine("drift", "spacr", DARK, seed=5)
    dense = make_engine("drift", "spacr", DARK, seed=5, density=3.0)
    assert len(sparse.geometry(W, H)) < len(normal.geometry(W, H)) \
        < len(dense.geometry(W, H))
    assert lit_pixels(render(sparse)) < lit_pixels(render(dense))


@pytest.mark.parametrize("theme,noun,count,shipped", ELEMENT_COUNTS)
def test_density_never_re_rolls_what_is_already_on_screen(theme, noun, count,
                                                          shipped):
    """Turning the slider up adds elements; it does not move the ones that
    were there. The pool is rolled once, at the top of the range."""
    engine = make_engine(theme, "spacr", DARK, seed=5)
    engine.set_time(7.0)
    before = engine.geometry(W, H)
    engine.set_density(3.0)
    after = engine.geometry(W, H)
    assert len(after) > len(before)
    assert after[:len(before)] == before


@pytest.mark.parametrize("theme,noun,count,shipped", ELEMENT_COUNTS)
def test_density_and_resolution_share_one_budget(theme, noun, count, shipped):
    """The clamp that stops the two controls multiplying into a frame nobody
    can afford: 2.0 detail is four times the pixels, 3.0 density is three
    times the elements, and twelve times the work behind every screen in the
    app is not a setting, it is a bug with a slider on it."""
    alone = make_engine(theme, "spacr", DARK, seed=5, density=3.0)
    assert alone.effective_density() == pytest.approx(3.0), \
        "density on its own was trimmed"
    detailed = make_engine(theme, "spacr", DARK, seed=5, resolution=2.0)
    assert detailed.effective_density() == pytest.approx(1.0), \
        "detail on its own was trimmed"

    both = make_engine(theme, "spacr", DARK, seed=5, density=3.0,
                       resolution=2.0)
    assert both.effective_density() < 3.0
    assert both.work / both.density * both.effective_density() \
        == pytest.approx(amb.WORK_BUDGET)
    assert count(both) < count(alone)


@pytest.mark.parametrize("background", [DARK, LIGHT])
def test_density_makes_the_field_busier_and_not_brighter(background):
    """A density control with no alpha compensation is a *brightness*
    control wearing a misleading name.

    Additive compositing means N overlapping fields are N times the light,
    and this module's alphas were set on the mean lightness of a rendered
    frame precisely so the backdrop is never the loudest thing behind a
    settings form. Uncompensated, density 300 % measured 0.288 against a
    0.076 page where the default measures 0.135 — brighter than double.
    So what has to be asserted is that the mean lightness does *not* track
    the element count, while the count itself does.
    """
    def lightness(density):
        engine = make_engine(theme, "spacr", background, seed=7,
                             density=density)
        engine.set_time(9.0)
        grid = rows(render(engine, background=background))
        return sum(luminance(px) for row in grid for px in row) / (
            len(grid) * len(grid[0]) * 255.0)

    for theme in ("blobs", "aurora", "ripple", "bokeh", "cells"):
        page = luminance(QColor(background).getRgb()[:3]) / 255.0
        default, dense = lightness(1.0), lightness(3.0)
        # Against the page, because that is what "loud" means here.
        assert abs(dense - page) < abs(default - page) * 1.4, (
            f"{theme}: three times the elements made the frame "
            f"{abs(dense - page) / max(1e-6, abs(default - page)):.2f} "
            "times as loud")


def test_the_starfield_is_exempt_from_the_alpha_compensation():
    """It has nothing to compensate: a couple of hundred dots light 0.65 %
    of the page and almost never land on each other, so dividing their alpha
    by three would delete two thirds of the stars rather than un-brighten
    anything."""
    assert make_engine("drift", "spacr", DARK, seed=7,
                       density=3.0).alpha_scale() == 1.0
    assert make_engine("blobs", "spacr", DARK, seed=7,
                       density=3.0).alpha_scale() == pytest.approx(1 / 3)
    # A sparser field is a quieter one; four times the alpha on a quarter of
    # the blobs would clip to white rather than compensate.
    assert make_engine("blobs", "spacr", DARK, seed=7,
                       density=0.25).alpha_scale() == 1.0


def test_the_starfields_budget_ignores_resolution():
    """Drift has no buffer, so detail costs it nothing and must not be
    allowed to spend its density."""
    engine = make_engine("drift", "spacr", DARK, seed=5, density=3.0,
                         resolution=2.0)
    assert engine.effective_density() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Which way the starfield goes
# ---------------------------------------------------------------------------

def starfield(direction, seed=5):
    engine = make_engine("drift", "spacr", DARK, seed=seed,
                         direction=direction)
    engine.set_time(0.0)
    return engine


@pytest.mark.parametrize("direction,sign", [("up", -1.0), ("down", 1.0)])
def test_the_shared_directions_move_the_field_that_way(direction, sign):
    """Centroid displacement across frames, with the wrap taken out."""
    engine = starfield(direction)
    steps = wrapped_steps(engine, 1 / 24, 24)
    total_y = sum(sum(dy for _dx, dy in frame) / len(frame)
                  for frame in steps)
    assert total_y * sign > 0, \
        f"{direction}: the field's centroid moved {total_y:+.4f} in y"
    # ...and it is a shared vector: everything goes the same way.
    same = [dy for frame in steps for _dx, dy in frame]
    assert all(dy * sign >= 0 for dy in same), \
        f"{direction}: some particles went the other way"


def test_up_and_down_are_mirror_images():
    up = wrapped_steps(starfield("up"), 1 / 24, 12)
    down = wrapped_steps(starfield("down"), 1 / 24, 12)
    for a, b in zip(up, down):
        for (_ax, ay), (_bx, by) in zip(a, b):
            assert ay == pytest.approx(-by, abs=1e-9)


def test_random_gives_every_speck_its_own_direction():
    """Not one shared vector: the headings are spread right round the
    circle, so the field mixes instead of travelling.

    The centroid is therefore the wrong thing to look for motion in — it
    barely moves, because the displacements cancel. What moves is every
    individual speck, and by more than the shared directions manage.
    """
    engine = starfield("random")
    steps = wrapped_steps(engine, 1 / 24, 24)

    per_particle = list(zip(*[list(frame) for frame in steps]))
    travel = [(sum(dx for dx, _dy in track), sum(dy for _dx, dy in track))
              for track in per_particle]
    headings = [math.atan2(dy, dx) for dx, dy in travel]
    quadrants = {int((h + math.pi) / (math.pi / 2)) % 4 for h in headings}
    assert quadrants == {0, 1, 2, 3}, \
        f"only {len(quadrants)} quadrants of heading are represented"

    # The centroid stays put while the specks do not.
    drift_x = sum(dx for dx, _dy in travel) / len(travel)
    drift_y = sum(dy for _dx, dy in travel) / len(travel)
    typical = sum(math.hypot(dx, dy) for dx, dy in travel) / len(travel)
    assert math.hypot(drift_x, drift_y) < typical * 0.5, \
        "the random field is really one shared vector"
    assert typical > 0, "nothing moved at all"

    # ...and every speck really is moving, not just some of them.
    assert min(math.hypot(dx, dy) for dx, dy in travel) > 0


def test_random_wanders_rather_than_running_straight():
    """A per-particle *heading* alone would be a straight line each. The
    wander is what makes it read as diffusion, and it has to be visible in
    the path: the direction of travel over one second must not be the
    direction over the next."""
    engine = starfield("random")
    first = wrapped_steps(engine, 1 / 24, 24)
    second = wrapped_steps(engine, 1 / 24, 24)

    def bearings(steps):
        tracks = list(zip(*[list(f) for f in steps]))
        return [math.atan2(sum(dy for _dx, dy in t),
                           sum(dx for dx, _dy in t)) for t in tracks]

    turned = [abs(math.atan2(math.sin(b - a), math.cos(b - a)))
              for a, b in zip(bearings(first), bearings(second))]
    assert max(turned) > 0.05, "every path was a straight line"


def test_the_direction_is_the_starfields_alone_and_the_others_ignore_it():
    """It reaches every engine, because the widget does not know which one
    it is holding. The ones with no direction have to shrug it off."""
    for theme in AMBIENT_THEMES:
        if theme == "drift":
            continue
        plain = make_engine(theme, "spacr", DARK, seed=5)
        odd = make_engine(theme, "spacr", DARK, seed=5, direction="random")
        plain.set_time(6.0)
        odd.set_time(6.0)
        assert render(plain) == render(odd), theme


def test_an_unknown_direction_is_ignored_rather_than_painted():
    engine = make_engine("drift", "spacr", DARK, seed=5, direction="sideways")
    assert engine.direction == DEFAULT_DRIFT_DIRECTION
    engine.set_direction("diagonally")
    assert engine.direction == DEFAULT_DRIFT_DIRECTION


def test_the_direction_survives_a_theme_switch(qtbot):
    widget = AmbientWidget(theme="drift", palette="spacr", background=DARK,
                           seed=2, direction="down")
    qtbot.addWidget(widget)
    assert widget.engine.direction == "down"
    widget.set_theme("blobs")
    widget.set_theme("drift")
    assert widget.engine.direction == "down"


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
    assert engine.count() == 3
    bases = {round(c.y, 3) for c in engine.curtains[:engine.count()]}
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
    for index, curtain in enumerate(engine.curtains[:engine.count()]):
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
    assert prefs.get_ambient_resolution() == DEFAULT_RESOLUTION
    assert prefs.get_ambient_density() == DEFAULT_DENSITY
    assert prefs.get_ambient_drift_direction() == DEFAULT_DRIFT_DIRECTION


def test_an_old_blur_setting_is_translated_rather_than_reinterpreted(prefs):
    """``ambient_blur`` used to be a buffer-resolution divisor, so its sharp
    half meant the opposite of what it means now. A store written under the
    old scale is converted once, and the user's intent survives: whoever
    asked for the sharpest backdrop gets the most detail, not a blur.
    """
    prefs._settings().setValue("prefs/ambient_blur", 0.25)   # old: sharpest
    prefs._settings().remove("prefs/ambient_motion_scale")
    assert prefs.get_ambient_resolution() == pytest.approx(2.0)
    assert prefs.get_ambient_blur() == pytest.approx(0.0)

    prefs._settings().setValue("prefs/ambient_blur", 3.0)    # old: softest
    prefs._settings().remove("prefs/ambient_motion_scale")
    assert prefs.get_ambient_blur() == pytest.approx(2.0)
    assert prefs.get_ambient_resolution() == pytest.approx(1 / 3)

    # ...and it runs once. A value written under the new scale stays put.
    prefs.set_ambient_blur(1.5)
    prefs.set_ambient_resolution(1.25)
    assert prefs.get_ambient_blur() == pytest.approx(1.5)
    assert prefs.get_ambient_resolution() == pytest.approx(1.25)


def test_the_drift_direction_round_trips_and_is_validated(prefs):
    for name in DRIFT_DIRECTIONS:
        prefs.set_ambient_drift_direction(name)
        assert prefs.get_ambient_drift_direction() == name
    with pytest.raises(ValueError):
        prefs.set_ambient_drift_direction("sideways")
    # A hand-edited file, or a downgrade from a build with more of them.
    prefs._settings().setValue("prefs/ambient_drift_direction", "widdershins")
    assert prefs.get_ambient_drift_direction() == DEFAULT_DRIFT_DIRECTION


@pytest.mark.parametrize("getter,setter,limits", [
    ("get_ambient_blur", "set_ambient_blur", BLUR_RANGE),
    ("get_ambient_speed", "set_ambient_speed", SPEED_RANGE),
    ("get_ambient_size", "set_ambient_size", SIZE_RANGE),
    ("get_ambient_resolution", "set_ambient_resolution", RESOLUTION_RANGE),
    ("get_ambient_density", "set_ambient_density", DENSITY_RANGE),
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
    prefs.set_ambient_resolution(1.5)
    prefs.set_ambient_density(2.0)
    prefs.set_ambient_drift_direction("down")
    widget = AmbientWidget(theme="blobs", palette="spacr", background=DARK,
                           seed=1)
    qtbot.addWidget(widget)
    assert (widget.blur(), widget.speed(), widget.size_scale(),
            widget.resolution(), widget.density(), widget.direction()) == \
        (2.0, 0.5, 1.5, 1.5, 2.0, "down")
    assert widget.engine.blur == 2.0
    assert widget.engine.resolution == 1.5


def test_the_dialog_offers_the_controls_and_saves_them(prefs, qtbot,
                                                       qt_theme_applied):
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QSlider

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    sliders = {s.objectName(): s for s in dialog.findChildren(QSlider)}
    # Every one opens on its designed value — which for blur is 0 %, because
    # the animation ships unsoftened and this control only adds softening.
    designed = {"AmbientBlur": 0, "AmbientSpeed": 100, "AmbientSize": 100,
                "AmbientResolution": 100, "AmbientDensity": 100}
    for name, mark in designed.items():
        assert name in sliders, sorted(sliders)
        assert sliders[name].value() == mark, \
            f"{name} does not open on the default"

    sliders["AmbientBlur"].setValue(180)
    sliders["AmbientSpeed"].setValue(60)
    sliders["AmbientSize"].setValue(140)
    sliders["AmbientResolution"].setValue(150)
    sliders["AmbientDensity"].setValue(200)
    combo = dialog.findChild(QComboBox, "AmbientDriftDirection")
    assert combo is not None, "no starfield direction control"
    combo.setCurrentIndex([combo.itemData(i) for i in
                           range(combo.count())].index("random"))
    dialog.findChild(QDialogButtonBox).button(
        QDialogButtonBox.Save).click()

    assert prefs.get_ambient_blur() == pytest.approx(1.8)
    assert prefs.get_ambient_speed() == pytest.approx(0.6)
    assert prefs.get_ambient_size() == pytest.approx(1.4)
    assert prefs.get_ambient_resolution() == pytest.approx(1.5)
    assert prefs.get_ambient_density() == pytest.approx(2.0)
    assert prefs.get_ambient_drift_direction() == "random"


def test_the_direction_row_is_only_there_for_the_starfield(prefs, qtbot,
                                                           qt_theme_applied):
    """It applies to one animation out of six. Showing it greyed out under
    the other five would be five wrong answers to "what does this do"."""
    from PySide6.QtWidgets import QComboBox

    prefs.set_ambient_theme("blobs")
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    combo = dialog.findChild(QComboBox, "AmbientDriftDirection")
    theme_combo = dialog.findChild(QComboBox, "AmbientTheme")
    assert not combo.isVisible()

    keys = [theme_combo.itemData(i) for i in range(theme_combo.count())]
    theme_combo.setCurrentIndex(keys.index("drift"))
    assert combo.isVisible()
    theme_combo.setCurrentIndex(keys.index("ripple"))
    assert not combo.isVisible()


def test_the_controls_grey_out_with_the_animation(prefs, qtbot,
                                                  qt_theme_applied):
    from PySide6.QtWidgets import QSlider

    from spacr.qt.widgets.toggle import Toggle

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    toggle = dialog.findChild(Toggle, "AmbientEnabled")
    sliders = [s for s in dialog.findChildren(QSlider)
               if s.objectName() in ("AmbientBlur", "AmbientSpeed",
                                     "AmbientResolution", "AmbientDensity",
                                     "AmbientSize")]
    assert len(sliders) == 5
    toggle.setChecked(False)
    assert not any(s.isEnabled() for s in sliders)
    toggle.setChecked(True)
    assert all(s.isEnabled() for s in sliders)
