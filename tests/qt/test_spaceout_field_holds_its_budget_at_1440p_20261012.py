"""The Spaceout field keeps its colour waves inside a 60 fps frame at 2560x1440.

The capture gate on the workstation measured the Spaceout field painting at
about 42 fps on a 2560x1440 GNOME session while Home painted 57. The paint
count is the count of fresh frames (``AmbientWidget._on_tick`` repaints only
when the shading thread has published one), so a frame that shades for longer
than the 16.7 ms tick is a frame the screen never gets.

Measured offscreen at 2560x1440 on 2026-10-12 (thread CPU time, medians,
on a 32-thread machine at load 60-80): a plain Spaceout field frame shaded
in 7-25 ms, and a frame with a colour wave or a spiral in 105-164 ms, 6.2-9.5
times a plain one. Lane two of the scheduler picks one of those two for
about half of its events. The cost was the crossfade between the coloured
grains and the ordinary ones: a whole-frame opaque pass, then a mix of two
whole frames in 65536-word NumPy chunks with the interpreter lock taken
between operations. Only the words a grain touched differ between the two
frames (6136 grains at this size), so the mix now runs in a compiled kernel
without the lock, built by the same warmup thread as the grain kernels,
writing only those words into the ordinary frame; the opaque pass is applied
only where a flash compares against it. A colour-wave frame now costs
46-53 ms there, 1.8-3.1 plain frames. A plain frame costs about 2.5 ms at
this size on a quiet machine (0.80 ms at 1400x900, 35a44a19a), so the bound
below keeps a colour-wave frame inside the 16.7 ms interval of a 60 fps
field.

The ratio is measured interleaved in one process, so a busy machine slows
both sides alike.
"""
from __future__ import annotations

import math
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from spacr.qt.widgets import ambient

W, H = 2560, 1440

#: Most a colour-wave frame may cost, in plain frames of the same field.
#: Under load (60-80 on 32 threads) 1.8-3.1 with the kernel and 6.2-9.5
#: without it. On a quiet machine a plain frame shrinks to 1.5-2.3 ms while
#: a colour-wave frame stays at 6.5-8.7 ms, so the ratio grows: 2.9-5.4 per
#: round with the kernel (per-run medians 3.7-4.9, the same at 69b7ae0d0,
#: which set the first bound of 4.5 from the loaded figures) and 14-20
#: without it, measured 2026-10-13; CI measured 5.00 and 5.05. The bound
#: sits between the two in either regime.
COLOUR_WAVE_PLAIN_FRAMES = 6.0


def _engine(background="#101418"):
    return ambient._SpaceoutFieldEngine(ambient.PALETTE_SETS["spacr"].colors,
                                        background, seed=42, blur=0)


def _only(engine, key):
    engine.set_field_effects(dict.fromkeys(engine._effect_keys, False))
    engine.set_field_effects({key: True})


def _event(engine, kind, clock=12.0):
    engine.set_time(clock)
    row = (kind, 2., 36., ((.3, .4, .24, 1., .2), (.7, .6, .26, -1., 1.)), .7)
    engine._phenomena_schedule = (
        tuple(math.floor(clock / period) for period in engine._lane_periods),
        (row,))
    return row


def _numpy_mix(output, base, blend):
    """The crossfade as the NumPy path in ``_point_material`` computes it."""
    previous, colored = base, output | np.uint32(0xff000000)
    mixed = np.full(output.size, np.uint32(0xff000000), dtype=np.uint32)
    for shift in (16, 8, 0):
        channel = ((((previous >> shift) & 255) * (255 - blend)
                    + ((colored >> shift) & 255) * blend + 127) // 255)
        mixed |= channel << shift
    return mixed


def _every_channel_pair():
    """Words whose three channels between them cover every (base, colour) pair.

    The base is opaque, as the ordinary frame is; the colour words carry a
    rank byte, and the 256 with equal channels are the words a mix leaves.
    """
    pairs = np.arange(65536, dtype=np.uint32)
    first, second = pairs >> 8, pairs & 255
    base = (np.uint32(0xff000000) | first << 16 | second << 8 | first)
    colour = (np.uint32(0x13000000) | second << 16 | first << 8 | second)
    return base, colour


def _compiled_mixer():
    ambient._complete_ambient_startup()
    ambient._warm_packed_scatter()
    if ambient._COLOR_BLEND is None:
        pytest.skip("numba is not available")
    return ambient._COLOR_BLEND


def test_the_python_mixer_rounds_exactly_like_the_numpy_one():
    base, colour = _every_channel_pair()
    rows = np.random.default_rng(7).choice(base.size, 2048, replace=False)
    rows = np.concatenate((rows, np.arange(0, 65536, 257)))
    for blend in (0, 1, 2, 127, 128, 200, 254, 255):
        expected = _numpy_mix(colour[rows], base[rows], blend)
        actual = base[rows].copy()
        ambient._blend_colored_words(colour[rows], actual, blend)
        assert np.array_equal(actual, expected), blend


def test_the_compiled_mixer_matches_numpy_for_every_pair_and_blend():
    mixer = _compiled_mixer()
    base, colour = _every_channel_pair()
    for blend in range(256):
        expected = _numpy_mix(colour, base, blend)
        actual = base.copy()
        mixer(colour, actual, blend)
        assert np.array_equal(actual, expected), blend


@pytest.mark.parametrize("background", ["#101418", "#eeeeee"])
@pytest.mark.parametrize("kind", ["color_waves", "spirals"])
def test_a_frame_mixed_by_the_kernel_is_the_numpy_frame(monkeypatch,
                                                       background, kind):
    made = _engine(background)
    made.set_blink_percent(40.0)
    _only(made, kind)
    _event(made, kind, 6.0)
    monkeypatch.setattr(ambient, "_ready_color_blend", lambda: None)
    reference = bytes(made.shade(320, 180).constBits())
    monkeypatch.setattr(ambient, "_ready_color_blend",
                        lambda: ambient._blend_colored_words)
    assert bytes(made.shade(320, 180).constBits()) == reference


def test_the_mixer_is_offered_once_the_warmup_built_it(monkeypatch):
    asked = []
    monkeypatch.setattr(ambient, "_ready_colored_scatter",
                        lambda: asked.append(1))
    monkeypatch.setattr(ambient, "_COLOR_BLEND", None)
    assert ambient._ready_color_blend() is None
    assert asked == [1], "a missing mixer starts the shared warmup"
    monkeypatch.setattr(ambient, "_COLOR_BLEND", ambient._blend_colored_words)
    assert ambient._ready_color_blend() is ambient._blend_colored_words
    assert asked == [1]


@pytest.mark.parametrize("failure", ["raise", "interpreted"])
def test_a_mixer_the_compiler_cannot_build_leaves_the_numpy_mix(
        monkeypatch, failure):
    def kernel(*_args):
        pass

    kernel.nopython_signatures = (True,)

    def decorator(*_args, **_kwargs):
        def compile(function):
            if function is ambient._blend_colored_words:
                if failure == "raise":
                    raise RuntimeError("no compiler for the mixer")
                return function
            return kernel
        return compile

    monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(njit=decorator))
    for name in ("_PACKED_SCATTER", "_SUBPIXEL_SCATTER", "_COLORED_SCATTER",
                 "_COLOR_BLEND"):
        monkeypatch.setattr(ambient, name, None)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER_FAILED", False)
    monkeypatch.setattr(ambient, "_COLORED_SCATTER_FAILED", False)
    ambient._complete_ambient_startup()
    ambient._warm_packed_scatter()
    assert ambient._COLOR_BLEND is None
    assert ambient._COLORED_SCATTER is kernel, (
        "the grain kernels are kept whatever the mixer did")


def _shade_ms(engine, clock):
    engine.set_time(clock)
    engine.shade(W, H)
    spent = []
    for _frame in range(5):
        engine.advance(1.0 / 60.0)
        started = time.thread_time()
        engine.shade(W, H)
        spent.append((time.thread_time() - started) * 1000.0)
    return statistics.median(spent)


@pytest.mark.skipif(
    sys.gettrace() is not None,
    reason="wall-clock performance budgets require an uninstrumented process",
)
def test_a_colour_wave_frame_stays_within_a_few_plain_frames_at_1440p():
    """A first round is shaded and dropped: the first plain frames of a new
    engine cost 6.5 ms against 1.5-2.3 after, which made the first of three
    rounds read 1.2 and the median the lower of the other two."""
    _compiled_mixer()
    plain, waved = _engine(), _engine()
    for engine in (plain, waved):
        engine.set_max_pixels(W * H)
    plain.set_field_effects(dict.fromkeys(plain._effect_keys, False))
    _only(waved, "color_waves")
    row = _event(waved, "color_waves")
    assert 0 < round(abs(waved._event_strength(row)) * 255) < 255, (
        "the measured frame must be a crossfade, not an endpoint")
    _shade_ms(plain, 12.0)
    _event(waved, "color_waves")
    _shade_ms(waved, 12.0)
    ratios = []
    for _round in range(3):
        cost = _shade_ms(plain, 12.0)
        _event(waved, "color_waves")
        ratios.append(_shade_ms(waved, 12.0) / cost)
    ratio = statistics.median(ratios)
    assert ratio <= COLOUR_WAVE_PLAIN_FRAMES, (
        f"a colour-wave frame costs {ratio:.2f} plain frames at {W}x{H} "
        f"(rounds {', '.join(f'{r:.2f}' for r in ratios)})")
