"""Live DNA rain backdrop — engine, widget, settings bar, integration.

Everything here is deterministic (fixed seeds) and driven by calling
``advance_frame``/``advance`` directly, so no test ever waits on a real
timer.
"""
from __future__ import annotations

import statistics

import pytest

from PySide6.QtCore import QEvent, QRect, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QVBoxLayout, QWidget

from spacr.qt.widgets import dna_rain as dr
from spacr.qt.widgets.dna_rain import (BASES, DnaRainEngine,
                                       DnaRainSettingsBar, DnaRainWidget,
                                       MAX_FONT_PX, MAX_STRING_CELLS,
                                       MIN_FONT_PX, MIN_STRING_CELLS,
                                       SPACR_SPLICE_PROBABILITY, SPACR_TOKEN,
                                       blend, derive_head_color,
                                       install_dna_rain)

DT = 1.0 / 24.0
# Every cell holds exactly one character now: the four bases, or one
# letter of the spaCR splice, which is spelled down the column rather
# than packed into a single cell.
ALLOWED = set(BASES) | set(SPACR_TOKEN)


def make_widget(qtbot, **kwargs):
    kwargs.setdefault("seed", 1234)
    width = kwargs.pop("_w", 640)
    height = kwargs.pop("_h", 480)
    widget = DnaRainWidget(**kwargs)
    qtbot.addWidget(widget)
    widget.resize(width, height)
    return widget


# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------

def test_only_atgc_glyphs_over_many_frames():
    """Nothing but A, T, G, C is ever a glyph when splices are off."""
    engine = DnaRainEngine(960, 640, 16, seed=7, spacr_probability=0.0)
    seen = set()
    for _ in range(600):
        engine.advance(DT)
        seen.update(engine.tokens())
    assert seen == set(BASES), seen
    assert engine.spacr_splices == 0
    # Plenty of turnover happened, so this really covered fresh strings.
    assert engine.respawns > 100


def test_alphabet_is_uppercase_atgc_only():
    """The offline renderer's lowercase mixing is gone."""
    assert BASES == ("A", "T", "G", "C")
    engine = DnaRainEngine(400, 400, 16, seed=3, spacr_probability=0.0)
    text = "".join(engine.column_text(i) for i in range(engine.n_columns))
    assert text and text == text.upper()


def test_tokens_are_bases_or_the_word():
    engine = DnaRainEngine(960, 640, 16, seed=11, spacr_probability=0.5)
    for _ in range(300):
        engine.advance(DT)
        assert set(engine.tokens()) <= ALLOWED


# ---------------------------------------------------------------------------
# Asynchrony
# ---------------------------------------------------------------------------

def test_columns_differ_in_speed_length_and_start_time():
    engine = DnaRainEngine(1920, 1080, 16, seed=99)
    speeds = [c.speed for c in engine.columns]
    lengths = [c.length for c in engine.columns]
    heads = [c.head for c in engine.columns]
    assert len(speeds) == 120

    # Speeds are continuous, so essentially all of them are distinct.
    assert len(set(speeds)) >= 118
    assert statistics.pstdev(speeds) > 1.0
    assert min(speeds) >= dr.MIN_SPEED_CELLS_PER_S
    assert max(speeds) <= dr.MAX_SPEED_CELLS_PER_S

    # Lengths span most of the allowed range.
    assert len(set(lengths)) > 30
    assert min(lengths) >= MIN_STRING_CELLS
    assert max(lengths) <= MAX_STRING_CELLS
    assert max(lengths) - min(lengths) > 40

    # Start times: heads are spread above and below the canvas, so
    # columns enter at wildly different moments.
    assert len(set(heads)) >= 118
    assert min(heads) < -50
    assert max(heads) > 0
    assert statistics.pstdev(heads) > 10


def test_columns_do_not_march_in_lockstep():
    """After many frames the integer rows are still spread out."""
    engine = DnaRainEngine(1920, 1080, 16, seed=5)
    for _ in range(200):
        engine.advance(DT)
    rows = [c.row for c in engine.columns]
    assert len(set(rows)) > 40, "columns collapsed onto the same row"


def test_slow_columns_still_move_a_pixel_at_a_time(qtbot):
    """Sub-pixel motion: a slow column creeps rather than stepping.

    It used to be painted at `row * cell`, so a 4 cells/s column held still
    for five frames and jumped a whole glyph on the sixth — the stepping that
    reads as choppy. Raising the frame rate cannot fix that on its own,
    because the position simply has fewer places it is allowed to be.

    The old test asserted the opposite (that slow columns are skipped on most
    ticks), which is exactly the behaviour that had to go.
    """
    engine = DnaRainEngine(640, 480, 16, seed=3)
    slow = min(engine.columns, key=lambda c: c.speed)
    index = engine.columns.index(slow)

    seen = set()
    for _ in range(40):
        engine.advance(1 / 60)
        seen.add(engine.columns[index].y_px)

    assert len(seen) > 8, (
        f"the slowest column only occupied {len(seen)} distinct pixel "
        f"positions over 40 frames — it is still quantised to cells")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_reproduces_the_same_animation():
    a = DnaRainEngine(800, 600, 16, seed=2024)
    b = DnaRainEngine(800, 600, 16, seed=2024)
    assert a.snapshot() == b.snapshot()
    for _ in range(250):
        a.advance(DT)
        b.advance(DT)
    assert a.snapshot() == b.snapshot()
    assert a.respawns == b.respawns


def test_different_seed_gives_a_different_animation():
    a = DnaRainEngine(800, 600, 16, seed=1)
    b = DnaRainEngine(800, 600, 16, seed=2)
    assert a.snapshot() != b.snapshot()
    for _ in range(120):
        a.advance(DT)
        b.advance(DT)
    assert a.snapshot() != b.snapshot()


def test_seedless_engine_still_works():
    engine = DnaRainEngine(320, 240, 16)
    assert engine.seed is None
    assert engine.n_columns == 20
    engine.advance(DT)


# ---------------------------------------------------------------------------
# The spaCR splice
# ---------------------------------------------------------------------------

def test_spacr_is_spelled_down_the_column_one_letter_per_cell():
    """s / p / a / C / R, falling like the bases around it.

    It used to live in a single cell and be drawn horizontally across its
    neighbours, which read as a label pasted over the rain rather than as part
    of it. Casing stays load-bearing; orientation is what makes it belong.
    """
    engine = DnaRainEngine(320, 240, 16, seed=4, spacr_probability=1.0)
    column = engine.columns[0]
    assert column.has_word

    start = column.word_index
    letters = column.tokens[start:start + len("spaCR")]
    assert letters == ["s", "p", "a", "C", "R"]

    # EVERY cell is exactly one character now — the invariant the old
    # renderer could not rely on, and the reason the second paint pass,
    # the measured token widths and the widened dirty rectangles are gone.
    assert len(column.tokens) == column.length
    assert all(len(tok) == 1 for tok in column.tokens)

    text = engine.column_text(0)
    assert "spaCR" in text
    assert "spaCR" not in column.tokens and "spacr" not in column.tokens


def test_spacr_absent_when_probability_is_zero():
    engine = DnaRainEngine(320, 240, 16, seed=4, spacr_probability=0.0)
    for column in engine.columns:
        assert not column.has_word
        assert column.word_index == -1
    assert engine.spacr_splices == 0


def test_spacr_appears_at_approximately_the_configured_rate():
    """Over a long run the observed rate matches the constant."""
    respawns = 0
    splices = 0
    for seed in (1, 2, 3, 4):
        engine = DnaRainEngine(1920, 1080, 16, seed=seed)
        assert engine.spacr_probability == SPACR_SPLICE_PROBABILITY
        # A generous dt retires every column each step, so this is
        # ~48000 respawns without simulating hours of animation.
        for _ in range(100):
            engine.advance(20.0)
        respawns += engine.respawns
        splices += engine.spacr_splices

    expected = respawns * SPACR_SPLICE_PROBABILITY
    assert respawns > 40000
    assert 0.6 * expected < splices < 1.6 * expected, (splices, expected)


def test_spacr_rate_works_out_to_about_a_minute():
    """The documented "roughly once a minute" is really what happens."""
    seconds = 900.0
    engine = DnaRainEngine(1920, 1080, 16, seed=17)
    for _ in range(int(seconds / DT)):
        engine.advance(DT)
    interval = seconds / max(1, engine.spacr_splices)
    assert 25.0 < interval < 150.0, interval


def test_a_word_column_is_no_wider_than_any_other(qtbot):
    """The dirty rectangle stops being a special case.

    A spliced column used to be wider than its stride, because the word was
    drawn horizontally out of one cell and bled over its neighbours — so the
    repaint rectangle had to be widened and columns to the left had to be
    dragged into every repaint. Five one-letter cells cannot overdraw
    anything, so a column carrying the word is exactly as wide as one that
    does not.
    """
    widget = make_widget(qtbot, font_size=16, spacr_probability=1.0)
    cell = widget.engine.cell_size
    assert all(c.has_word for c in widget.engine.columns)
    rects = widget._coalesce([(0, 0, 3)])
    assert len(rects) == 1
    assert rects[0].width() == cell

    plain = make_widget(qtbot, font_size=16, spacr_probability=0.0)
    assert plain._coalesce([(0, 0, 3)])[0].width() == cell


def test_word_is_painted_into_the_output(qtbot):
    widget = make_widget(qtbot, font_size=20, spacr_probability=1.0,
                         color="#00ff00", background="#000000", opacity=1.0)
    widget.show()
    qtbot.waitExposed(widget)
    for _ in range(60):
        widget.advance_frame(DT)
    image = widget.grab().toImage()
    assert image.width() == widget.width()
    # Something got drawn.
    assert any(image.pixelColor(x, y).green() > 40
               for x in range(0, image.width(), 7)
               for y in range(0, image.height(), 7))


# ---------------------------------------------------------------------------
# Live settings — speed
# ---------------------------------------------------------------------------

def test_changing_speed_preserves_relative_asynchrony():
    engine = DnaRainEngine(1920, 1080, 16, seed=8)
    base = [c.head for c in engine.columns]
    engine.advance(DT)
    slow = [c.head - b for c, b in zip(engine.columns, base)]

    engine2 = DnaRainEngine(1920, 1080, 16, seed=8)
    engine2.set_speed_multiplier(3.0)
    base2 = [c.head for c in engine2.columns]
    engine2.advance(DT)
    fast = [c.head - b for c, b in zip(engine2.columns, base2)]

    assert len(slow) == len(fast) == 120
    for a, b in zip(slow, fast):
        assert b == pytest.approx(a * 3.0, rel=1e-9)
    # Still asynchronous afterwards: the steps are all different.
    assert len(set(round(v, 9) for v in fast)) >= 118


def test_speed_multiplier_is_clamped_and_zero_freezes():
    engine = DnaRainEngine(320, 240, 16, seed=1)
    engine.set_speed_multiplier(-5.0)
    assert engine.speed_multiplier == 0.0
    heads = [c.head for c in engine.columns]
    engine.advance(DT)
    assert [c.head for c in engine.columns] == heads


def test_widget_speed_applies_live(qtbot):
    widget = make_widget(qtbot)
    assert widget.speed() == 1.0
    widget.set_speed(2.5)
    assert widget.speed() == 2.5
    assert widget.engine.speed_multiplier == 2.5


# ---------------------------------------------------------------------------
# Live settings — font size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("font_px,expected", [
    (4, 1920 // 4),
    (8, 1920 // 8),
    (16, 1920 // 16),
    (32, 1920 // 32),
    (96, 1920 // 96),
])
def test_font_size_relayouts_the_column_count(qtbot, font_px, expected):
    widget = make_widget(qtbot, _w=1920, _h=1080)
    widget.set_font_size(font_px)
    assert widget.font_size() == font_px
    assert widget.engine.n_columns == expected
    assert widget.engine.cell_size == font_px


def test_font_size_extremes_are_clamped(qtbot):
    widget = make_widget(qtbot, _w=1920, _h=1080)
    widget.set_font_size(0)
    assert widget.font_size() == MIN_FONT_PX
    assert widget.engine.n_columns == 1920 // MIN_FONT_PX
    widget.set_font_size(100000)
    assert widget.font_size() == MAX_FONT_PX
    assert widget.engine.n_columns == 1920 // MAX_FONT_PX
    # A huge font leaves few rows, so strings are capped to something
    # that still reads as rain rather than a solid bar.
    assert widget.engine.max_length <= 1080 // MAX_FONT_PX * 2
    assert all(c.length <= widget.engine.max_length
               for c in widget.engine.columns)


def test_font_size_no_op_when_unchanged(qtbot):
    widget = make_widget(qtbot)
    before = widget.engine.snapshot()
    widget.set_font_size(widget.font_size())
    assert widget.engine.snapshot() == before


def test_engine_font_size_no_op_when_unchanged():
    engine = DnaRainEngine(640, 480, 16, seed=1)
    before = engine.snapshot()
    engine.set_font_size(16)
    assert engine.snapshot() == before
    assert engine.size == (640, 480)
    engine.set_font_size(8)
    assert engine.snapshot() != before
    assert engine.n_columns == 80


def test_huge_font_still_paints(qtbot):
    widget = make_widget(qtbot, _w=1920, _h=1080, font_size=MAX_FONT_PX)
    widget.show()
    qtbot.waitExposed(widget)
    for _ in range(10):
        widget.advance_frame(DT)
    assert not widget.grab().isNull()


# ---------------------------------------------------------------------------
# Live settings — colour
# ---------------------------------------------------------------------------

def test_color_reaches_the_painted_output(qtbot):
    widget = make_widget(qtbot, _w=320, _h=240, font_size=16,
                         color="#ff0000", background="#000000", opacity=1.0,
                         spacr_probability=0.0)
    widget.show()
    qtbot.waitExposed(widget)
    for _ in range(30):
        widget.advance_frame(DT)
    image = widget.grab().toImage()
    reds = [image.pixelColor(x, y)
            for x in range(image.width())
            for y in range(0, image.height(), 3)]
    assert any(c.red() > 100 and c.green() < 80 for c in reds), "no red glyphs"

    widget.set_color("#0000ff")
    assert widget.color().name() == "#0000ff"
    image = widget.grab().toImage()
    blues = [image.pixelColor(x, y)
             for x in range(image.width())
             for y in range(0, image.height(), 3)]
    assert any(c.blue() > 100 and c.red() < 80 for c in blues), "no blue glyphs"


@pytest.mark.parametrize("base,background", [
    ("#000000", "#000000"),     # trail identical to the background
    ("#ffffff", "#ffffff"),     # ...at the other end
    ("#808080", "#808080"),     # mid grey, nowhere obvious to go
    ("#ff0000", "#000000"),
    ("#4a9eff", "#0d0e10"),
    ("#00ff00", "#ffffff"),
    ("#123456", "#123456"),
    ("#e6e6e6", "#808080"),     # a step up hits the ceiling, a step
    ("#1a1a1a", "#808080"),     # down lands on the background
])
def test_head_stays_distinguishable_at_the_extremes(base, background):
    trail = QColor(base)
    bg = QColor(background)
    head = derive_head_color(trail, bg)
    assert head.isValid()
    assert abs(head.lightnessF() - trail.lightnessF()) >= 0.2, "head == trail"
    assert abs(head.lightnessF() - bg.lightnessF()) >= 0.2, "head == background"


def test_head_pen_differs_from_trail_pens_when_color_equals_background(qtbot):
    widget = make_widget(qtbot, color="#000000", background="#000000",
                         opacity=1.0)
    head = widget._head_pen.color()
    assert head.name() != "#000000"
    for pen in widget._trail_pens:
        assert pen.color().name() != head.name()
    assert widget.head_color().name() != widget.color().name()


def test_color_equal_to_background_still_paints_something(qtbot):
    widget = make_widget(qtbot, _w=320, _h=240, color="#000000",
                         background="#000000", opacity=1.0,
                         spacr_probability=0.0)
    widget.show()
    qtbot.waitExposed(widget)
    for _ in range(30):
        widget.advance_frame(DT)
    image = widget.grab().toImage()
    assert any(image.pixelColor(x, y) != QColor("#000000")
               for x in range(0, image.width(), 3)
               for y in range(0, image.height(), 3)), "head glyphs invisible"


def test_derive_head_color_keeps_hue_for_chromatic_input():
    head = derive_head_color(QColor("#801010"), QColor("#000000"))
    assert head.hue() == QColor("#801010").hue()


def test_blend_endpoints_and_clamping():
    a, b = QColor("#000000"), QColor("#ffffff")
    assert blend(a, b, 0.0).name() == "#000000"
    assert blend(a, b, 1.0).name() == "#ffffff"
    assert blend(a, b, -3.0).name() == "#000000"
    assert blend(a, b, 9.0).name() == "#ffffff"
    assert blend(a, b, 0.5).red() == 128


def test_invalid_colors_fall_back(qtbot):
    widget = make_widget(qtbot, color="not-a-colour")
    # Fell back to the palette accent rather than blowing up.
    assert widget.color().isValid()
    before = widget.color().name()
    widget.set_color("also-not-a-colour")
    assert widget.color().name() == before


def test_background_is_forced_opaque(qtbot):
    widget = make_widget(qtbot, background=QColor(10, 20, 30, 5))
    assert widget.background_color().alpha() == 255
    widget.set_background_color(QColor(1, 2, 3, 0))
    assert widget.background_color().alpha() == 255


def test_opacity_setter_clamps_and_restyles(qtbot):
    widget = make_widget(qtbot)
    widget.set_opacity(5.0)
    assert widget.opacity() == 1.0
    widget.set_opacity(-1.0)
    assert widget.opacity() == 0.0
    widget.set_opacity(0.5)
    assert widget.opacity() == 0.5


def test_apply_theme_takes_colours_from_the_palette(qtbot):
    from spacr.qt.theme import palette_for
    widget = make_widget(qtbot)
    for theme in ("dark", "light", "space"):
        widget.apply_theme(theme)
        palette = palette_for(theme)
        assert widget.color().name() == QColor(palette["accent"]).name()
        assert widget.background_color().name() == QColor(palette["bg"]).name()


def test_the_background_default_comes_from_the_theme_palette(qtbot):
    """The fill follows the theme. The glyph colour deliberately does not.

    They used to be one rule — both from the palette — which made the
    rain the theme accent, i.e. the exact blue of the Run button and the
    AI toggle two inches away. Only the background has to match the
    page; the glyphs are the effect.
    """
    from spacr.qt.theme import palette_for
    widget = make_widget(qtbot, theme="light")
    palette = palette_for("light")
    assert widget.background_color().name() == QColor(palette["bg"]).name()
    assert widget.color().name() == QColor(dr.DEFAULT_COLOR).name()
    assert widget.color().name() != QColor(palette["accent"]).name()


def test_effective_theme_falls_back_when_preferences_explode(monkeypatch):
    import spacr.qt.preferences as prefs

    def boom():
        raise RuntimeError("no settings backend")

    monkeypatch.setattr(prefs, "resolve_effective_theme", boom)
    assert dr._effective_theme() == "dark"


def test_effective_theme_uses_preferences(monkeypatch):
    import spacr.qt.preferences as prefs
    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "space")
    assert dr._effective_theme() == "space"


# ---------------------------------------------------------------------------
# The CPU guarantee
# ---------------------------------------------------------------------------

def test_timer_stops_when_hidden_and_restarts_when_shown(qtbot):
    widget = make_widget(qtbot)
    assert not widget.is_running(), "must not tick before it is on screen"
    widget.show()
    qtbot.waitExposed(widget)
    assert widget.is_running()

    frames = widget.engine.frames
    widget.hide()
    assert not widget.is_running(), "hidden widget must not burn CPU"
    assert widget.engine.frames == frames

    widget.show()
    qtbot.waitExposed(widget)
    assert widget.is_running()


def test_timer_stops_when_the_window_is_minimised(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    layout = QVBoxLayout(window)
    widget = DnaRainWidget(window, seed=5)
    layout.addWidget(widget)
    window.resize(400, 300)
    window.show()
    qtbot.waitExposed(window)
    assert widget.is_running()

    window.setWindowState(Qt.WindowMinimized)
    widget.eventFilter(window, QEvent(QEvent.WindowStateChange))
    assert not widget.is_running()

    window.setWindowState(Qt.WindowNoState)
    widget.eventFilter(window, QEvent(QEvent.WindowStateChange))
    assert widget.is_running()


def test_reparenting_moves_the_window_watch(qtbot):
    """Shown standalone, then adopted by a window — the old filter goes."""
    widget = make_widget(qtbot)
    widget.show()
    qtbot.waitExposed(widget)
    assert widget._watched is widget

    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    layout.addWidget(widget)
    host.resize(400, 300)
    host.show()
    qtbot.waitExposed(host)
    assert widget._watched is host
    assert widget.is_running()

    host.setWindowState(Qt.WindowMinimized)
    widget.eventFilter(host, QEvent(QEvent.WindowStateChange))
    assert not widget.is_running()


def test_start_and_stop_are_idempotent(qtbot):
    widget = make_widget(qtbot)
    widget.start()
    assert widget.is_running()
    widget.start()
    assert widget.is_running()
    widget.stop()
    assert not widget.is_running()
    widget.stop()
    assert not widget.is_running()


def test_fps_is_capped(qtbot):
    widget = make_widget(qtbot)
    assert widget.fps() == dr.DEFAULT_FPS
    widget.set_fps(1000)
    assert widget.fps() == dr.MAX_FPS
    assert widget._timer.interval() >= 1
    widget.set_fps(0)
    assert widget.fps() == dr.MIN_FPS


def test_tick_clamps_a_long_stall(qtbot):
    """A frozen app must not teleport every column down the screen."""
    widget = make_widget(qtbot)
    widget.show()
    qtbot.waitExposed(widget)
    widget._clock.restart()
    heads = [c.head for c in widget.engine.columns]
    widget._on_tick()
    moved = max(c.head - h for c, h in zip(widget.engine.columns, heads))
    assert moved <= dr.MAX_DT * dr.MAX_SPEED_CELLS_PER_S + 1e-6


def test_the_frame_cost_stays_small_even_though_more_columns_move(qtbot):
    """The dirty-rect optimisation is now about pixels, not cells.

    Sub-pixel motion means most columns move every frame, so the old
    "fewer than six columns are dirty" assertion no longer holds — and it was
    a PROXY for cost, not cost itself. Measured directly instead: a full frame
    at 1920x1080 costs about half a millisecond, because the expensive part is
    blitting cached strips rather than deciding which to blit.

    3.1% of one core at 60 fps, against the 4.4% the module documents for the
    old cell-quantised version at 24 fps. Smoother AND cheaper.
    """
    import time

    widget = make_widget(qtbot, font_size=16)
    widget.resize(1920, 1080)
    widget.show()
    qtbot.waitExposed(widget)
    for _ in range(30):            # warm the strip cache
        widget.advance_frame(1 / 60)

    start = time.perf_counter()
    frames = 120
    for _ in range(frames):
        widget.advance_frame(1 / 60)
    per_frame = (time.perf_counter() - start) / frames

    assert per_frame < 0.004, (
        f"a frame costs {per_frame * 1000:.2f} ms; at 60 fps that is "
        f"{per_frame * 60 * 100:.1f}% of a core")


def test_busy_frames_fall_back_to_one_full_repaint(qtbot):
    widget = make_widget(qtbot, _w=1920, _h=1080)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_speed(4.0)
    fulls = 0
    for _ in range(30):
        widget.advance_frame(DT)
        fulls += bool(widget.last_full_repaint)
    assert fulls > 20


def test_a_zero_length_step_schedules_nothing(qtbot):
    """dt of zero must not repaint. Still the cheapest possible case."""
    widget = make_widget(qtbot, font_size=16)
    widget.show()
    qtbot.waitExposed(widget)
    widget.advance_frame(1 / 60)
    assert widget.advance_frame(0.0) == []


def test_coalesce_merges_adjacent_columns(qtbot):
    widget = make_widget(qtbot, font_size=16)
    rects = widget._coalesce([(0, 0, 3), (1, 2, 5), (2, 1, 4), (7, 0, 1)])
    assert len(rects) == 2
    assert rects[0] == QRect(0, 0, 48, 96)
    assert rects[1] == QRect(112, 0, 16, 32)
    assert widget._coalesce([]) == []


def test_highlight_run_always_fits_the_string():
    engine = DnaRainEngine(1920, 1080, 16, seed=6)
    for _ in range(200):
        engine.advance(0.7)
        for column in engine.columns:
            assert 0 <= column.hi_start <= column.hi_end <= column.length
            assert column.hi_end - column.hi_start == min(
                dr.HIGHLIGHT_RUN_CELLS, column.length)


def test_advance_with_non_positive_dt_does_nothing():
    engine = DnaRainEngine(320, 240, 16, seed=1)
    heads = [c.head for c in engine.columns]
    assert engine.advance(0.0) == []
    assert engine.advance(-1.0) == []
    assert [c.head for c in engine.columns] == heads


def test_strips_are_cached_and_invalidated(qtbot):
    widget = make_widget(qtbot, _w=320, _h=240)
    widget.show()
    qtbot.waitExposed(widget)
    widget.grab()
    first = widget._strip_for(0)
    assert widget._strip_for(0) is first, "strip re-rendered for no reason"

    widget.set_color("#ff00ff")
    assert widget._strip_for(0) is not first, "colour change ignored"

    cached = widget._strip_for(0)
    generation = widget.engine.columns[0].generation
    while widget.engine.columns[0].generation == generation:
        widget.engine.advance(1.0)
    assert widget._strip_for(0) is not cached, "respawn ignored"


def test_region_rects_uses_every_rectangle_of_the_region():
    from PySide6.QtGui import QRegion
    region = QRegion()
    for column in range(0, 10, 2):
        region += QRect(column * 16, 0, 16, 100)
    rects = dr._region_rects(region, QRect(0, 0, 160, 100))
    assert len(rects) == 5


def test_region_rects_falls_back_to_the_bounding_box():
    class Hostile:
        def __iter__(self):
            raise TypeError("this binding does not expose the rectangles")

    fallback = QRect(1, 2, 3, 4)
    assert dr._region_rects(Hostile(), fallback) == [fallback]
    assert dr._region_rects([], fallback) == [fallback]


# ---------------------------------------------------------------------------
# Legibility / not being in the way
# ---------------------------------------------------------------------------

def test_widget_takes_no_focus_and_no_mouse(qtbot):
    widget = make_widget(qtbot)
    widget.show()
    qtbot.waitExposed(widget)
    assert widget.focusPolicy() == Qt.NoFocus
    assert widget.testAttribute(Qt.WA_TransparentForMouseEvents)
    widget.setFocus()
    assert not widget.hasFocus()


def test_widget_is_opaque_and_default_dimmed(qtbot):
    widget = make_widget(qtbot)
    assert widget.testAttribute(Qt.WA_OpaquePaintEvent)
    assert widget.opacity() == dr.DEFAULT_OPACITY
    assert 0.0 < widget.opacity() < 0.5, "default must not fight the content"


def test_follow_parent_lowers_and_tracks_geometry(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(300, 200)
    front = QWidget(host)
    rain = DnaRainWidget(host, seed=3)
    rain.follow_parent()
    assert rain.geometry() == host.rect()
    # Lowered: first in the child order is bottom of the stack.
    children = [c for c in host.children() if isinstance(c, QWidget)]
    assert children[0] is rain
    assert children.index(front) > children.index(rain)

    host.show()
    qtbot.waitExposed(host)
    host.resize(600, 400)
    assert rain.size() == host.size()
    assert rain.engine.n_columns == 600 // rain.font_size()


def test_follow_parent_without_a_parent_is_harmless(qtbot):
    rain = make_widget(qtbot)
    assert rain.parent() is None
    rain.follow_parent()
    assert rain.size() == rain.size()


# ---------------------------------------------------------------------------
# Degenerate geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,height", [(0, 0), (0, 400), (400, 0), (1, 1)])
def test_zero_sized_widget_does_not_crash(qtbot, width, height):
    widget = make_widget(qtbot, _w=width, _h=height)
    widget.show()
    assert widget.advance_frame(DT) == []
    widget.advance_frame(DT)
    widget.grab()             # forces a paintEvent
    widget.set_font_size(32)
    widget.set_color("#ff0000")
    assert widget.engine.n_rows >= 0


@pytest.mark.parametrize("width,height", [(0, 0), (0, 400), (400, 0)])
def test_zero_sized_engine_is_inert(width, height):
    engine = DnaRainEngine(width, height, 16, seed=1)
    assert engine.n_columns == 0 or engine.n_rows == 0
    for _ in range(5):
        assert engine.advance(DT) == []
    assert engine.max_length >= MIN_STRING_CELLS
    if engine.n_columns == 0:
        assert engine.tokens() == []
        assert engine.snapshot() == ()


def test_engine_resize_is_a_no_op_when_unchanged():
    engine = DnaRainEngine(320, 240, 16, seed=1)
    before = engine.snapshot()
    engine.resize(320, 240)
    assert engine.snapshot() == before
    engine.resize(-10, -10)
    assert engine.n_columns == 0


def test_negative_geometry_is_clamped():
    engine = DnaRainEngine(-100, -100, 16, seed=1)
    assert engine.n_columns == 0
    assert engine.n_rows == 0


# ---------------------------------------------------------------------------
# Settings bar
# ---------------------------------------------------------------------------

def test_settings_bar_drives_the_widget_live(qtbot):
    widget = make_widget(qtbot, _w=640, _h=480)
    bar = DnaRainSettingsBar()
    qtbot.addWidget(bar)
    bar.bind(widget)

    assert bar.color().name() == widget.color().name()
    assert bar.speed() == pytest.approx(widget.speed())
    assert bar.font_size() == widget.font_size()

    bar.set_color("#12ab34")
    assert widget.color().name() == "#12ab34"

    bar.set_speed(2.5)
    assert widget.speed() == pytest.approx(2.5)

    bar.set_font_size(24)
    assert widget.font_size() == 24
    assert widget.engine.n_columns == 640 // 24


def test_settings_bar_readouts_track_the_sliders(qtbot):
    bar = DnaRainSettingsBar(speed=1.5, font_size=20)
    qtbot.addWidget(bar)
    assert bar._speed_value.text() == "1.5x"
    assert bar._font_value.text() == "20 px"
    bar.set_speed(3.0)
    assert bar._speed_value.text() == "3.0x"
    bar.set_font_size(12)
    assert bar._font_value.text() == "12 px"


def test_settings_bar_clamps_out_of_range_values(qtbot):
    bar = DnaRainSettingsBar(speed=99.0, font_size=9999)
    qtbot.addWidget(bar)
    assert bar.speed() == pytest.approx(dr.MAX_SPEED_MULTIPLIER)
    assert bar.font_size() == MAX_FONT_PX
    bar.set_speed(-4.0)
    assert bar.speed() == pytest.approx(dr.MIN_SPEED_MULTIPLIER)
    bar.set_font_size(-4)
    assert bar.font_size() == MIN_FONT_PX


def test_settings_bar_color_picker_applies_and_cancels(qtbot, monkeypatch):
    widget = make_widget(qtbot)
    bar = DnaRainSettingsBar()
    qtbot.addWidget(bar)
    bar.bind(widget)

    monkeypatch.setattr(QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: QColor("#ff8800")))
    bar._swatch.click()
    assert widget.color().name() == "#ff8800"
    assert "#ff8800" in bar._swatch.styleSheet()

    monkeypatch.setattr(QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: QColor()))
    bar.pick_color()
    assert widget.color().name() == "#ff8800", "cancel must change nothing"


def test_settings_bar_emits_its_signals(qtbot):
    """Each setter emits — carrying the value it was given, not just a ping.

    A bar that fired its signals with stale values would still satisfy a
    bare ``waitSignal``, and the rain bound to it would then be driven by
    whatever the bar used to hold. The emitted argument is checked, and
    so is the bar's own state afterwards, readout labels included: those
    are the numbers the user reads back off the popover.
    """
    bar = DnaRainSettingsBar()
    qtbot.addWidget(bar)
    assert bar.color().name() != "#010203"
    assert bar.speed() != pytest.approx(2.0)
    assert bar.font_size() != 28

    with qtbot.waitSignal(bar.color_changed,
                          check_params_cb=lambda c: c.name() == "#010203"):
        bar.set_color("#010203")
    assert bar.color().name() == "#010203"
    assert "#010203" in bar._swatch.styleSheet()

    with qtbot.waitSignal(bar.speed_changed,
                          check_params_cb=lambda v: v == pytest.approx(2.0)):
        bar.set_speed(2.0)
    assert bar.speed() == pytest.approx(2.0)
    assert bar._speed_value.text() == "2.0x"

    with qtbot.waitSignal(bar.font_size_changed,
                          check_params_cb=lambda px: px == 28):
        bar.set_font_size(28)
    assert bar.font_size() == 28
    assert bar._font_value.text() == "28 px"


def test_settings_bar_defaults_to_the_shipped_teal_whatever_the_theme(qtbot):
    """Not the palette accent — that is the Run button's blue.

    The bar used to open on ``palette['accent']``, so binding it to a
    rain that was already teal was the only thing that made the swatch
    agree with the glyphs.
    """
    bar = DnaRainSettingsBar(theme="space")
    qtbot.addWidget(bar)
    assert bar.color().name() == QColor(dr.DEFAULT_COLOR).name()


# ---------------------------------------------------------------------------
# Integration hook
# ---------------------------------------------------------------------------

def test_install_dna_rain_wires_everything(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    content = QWidget()
    layout.addWidget(content)
    host.resize(800, 600)

    rain = install_dna_rain(host, layout, seed=42)
    assert rain.parent() is host
    assert rain.geometry() == host.rect()
    assert rain.settings_bar is not None
    # The BUTTON goes in the layout; the settings themselves do not.
    assert layout.indexOf(rain.settings_button) >= 0
    assert layout.indexOf(rain.settings_bar) < 0
    # Behind the content, and out of the way of the pointer.
    children = [c for c in host.children() if isinstance(c, QWidget)]
    assert children[0] is rain
    assert children.index(rain) < children.index(content)
    assert rain.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert rain.focusPolicy() == Qt.NoFocus

    # The bar drives the rain.
    rain.settings_bar.set_font_size(32)
    assert rain.font_size() == 32


def test_install_dna_rain_without_a_layout(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(300, 200)
    rain = install_dna_rain(host, None, seed=1)
    # Nowhere to put the button, so it is left for the caller — but the
    # settings are still built, and still inside the popover.
    assert rain.settings_button.parentWidget() is host
    assert rain.settings_bar.parentWidget() is rain.settings_popover


def test_install_dna_rain_is_inert_until_shown(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(300, 200)
    rain = install_dna_rain(host, None, seed=1)
    assert not rain.is_running(), "must not tick while its host is hidden"
    host.show()
    qtbot.waitExposed(host)
    assert rain.is_running()


def test_hook_attaches_to_the_real_sequencing_screen(qtbot, qt_theme_applied):
    """Exactly what the app_screen.py hook does, on the real screen."""
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)
    screen.resize(1200, 800)

    rain = install_dna_rain(screen, screen.layout(), seed=7)
    assert rain.parent() is screen
    assert rain.geometry() == screen.rect()
    children = [c for c in screen.children() if isinstance(c, QWidget)]
    assert children[0] is rain, "rain must be at the bottom of the stack"
    # The button found the AI toggle on its own, so it went to the
    # actions row rather than to the layout it was handed.
    assert screen.layout().indexOf(rain.settings_button) < 0
    row = screen._ai_switch.parentWidget().layout()
    assert row.indexOf(rain.settings_button) >= 0

    screen.show()
    qtbot.waitExposed(screen)
    assert rain.is_running()
    for _ in range(5):
        rain.advance_frame(DT)
    screen.hide()
    assert not rain.is_running()


# ---------------------------------------------------------------------------
# Visibility
#
# The user reported the rain was too faint to read as an effect. It was 0.22 --
# a low-alpha accent colour, which has almost nothing to spend on a light
# theme in particular. Raised, and exposed as a control so it can be dialled
# per theme rather than guessed at once in a constant.
# ---------------------------------------------------------------------------

def test_the_default_visibility_is_the_one_that_was_asked_for():
    """20%, chosen by the user after seeing it.

    This value has moved twice on feedback: 0.22 was called too faint, it was
    raised to 0.42, and then set to 0.20 once the rain was teal and the
    surrounding panels had page opacity — a brighter colour behind more
    translucent chrome needs LESS alpha, not more. The bounds are kept wide
    enough to allow another adjustment without rewriting the test, and narrow
    enough to catch a stray edit.
    """
    from spacr.qt.widgets import dna_rain as dr
    assert 0.10 <= dr.DEFAULT_OPACITY <= 0.6, (
        "the rain sits BEHIND the settings; past ~0.6 it competes with them, "
        "and under ~0.10 it is not an effect at all")
    assert dr.DEFAULT_OPACITY == pytest.approx(0.20)


def test_the_bar_exposes_a_visibility_control(qtbot):
    from spacr.qt.widgets.dna_rain import DnaRainSettingsBar, DEFAULT_OPACITY
    bar = DnaRainSettingsBar()
    qtbot.addWidget(bar)
    assert bar.opacity() == pytest.approx(DEFAULT_OPACITY, abs=0.01)


def test_the_control_reaches_the_painted_output(qtbot):
    """A slider that does not change the render is decoration."""
    from spacr.qt.widgets.dna_rain import DnaRainSettingsBar, DnaRainWidget
    rain = DnaRainWidget(seed=11)
    bar = DnaRainSettingsBar()
    qtbot.addWidget(rain)
    qtbot.addWidget(bar)
    bar.bind(rain)

    bar.set_opacity(0.75)
    assert rain.opacity() == pytest.approx(0.75, abs=0.01)
    bar.set_opacity(0.10)
    assert rain.opacity() == pytest.approx(0.10, abs=0.01)


def test_binding_seeds_the_control_from_the_rain(qtbot):
    from spacr.qt.widgets.dna_rain import DnaRainSettingsBar, DnaRainWidget
    rain = DnaRainWidget(seed=3, opacity=0.33)
    bar = DnaRainSettingsBar()
    qtbot.addWidget(rain)
    qtbot.addWidget(bar)
    bar.bind(rain)
    assert bar.opacity() == pytest.approx(0.33, abs=0.01)


def test_the_slider_cannot_reach_fully_transparent_or_opaque(qtbot):
    """0% is an invisible effect; 100% would bury the settings behind it."""
    from spacr.qt.widgets.dna_rain import (DnaRainSettingsBar,
                                           MIN_OPACITY_PCT, MAX_OPACITY_PCT)
    bar = DnaRainSettingsBar()
    qtbot.addWidget(bar)
    assert MIN_OPACITY_PCT > 0
    assert MAX_OPACITY_PCT < 100
    bar.set_opacity(0.0)
    assert bar.opacity() >= MIN_OPACITY_PCT / 100
    bar.set_opacity(1.0)
    assert bar.opacity() <= MAX_OPACITY_PCT / 100


# ---------------------------------------------------------------------------
# Backdrop
#
# The rain is opaque by construction: it repaints only the cells that
# changed, so it has to be able to *clear* them, and clearing to a
# translucent colour smears the previous frame. On dark and light that is
# free — the thing behind it is a flat `bg` the rain reproduces exactly.
# On Space and Cell it is not, and the consequence was that the one screen
# with a rain was the one screen that never showed the theme's wallpaper:
# `map_barcodes` under Cell painted flat black over the micrograph.
#
# Handing the wallpaper in keeps the dirty-rectangle repaint and gets the
# picture back.
# ---------------------------------------------------------------------------

def _backdrop(width=64, height=48, color="#c81e64"):
    from PySide6.QtGui import QPixmap
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(color))
    return pixmap


def test_no_backdrop_by_default(qtbot):
    rain = make_widget(qtbot)
    assert rain.backdrop() is None


def test_a_backdrop_can_be_handed_in_at_construction(qtbot):
    rain = make_widget(qtbot, backdrop=_backdrop())
    assert rain.backdrop() is not None
    assert rain.backdrop().size().width() == 64


def test_set_backdrop_round_trips_and_can_be_cleared(qtbot):
    rain = make_widget(qtbot)
    rain.set_backdrop(_backdrop())
    assert rain.backdrop() is not None
    rain.set_backdrop(None)
    assert rain.backdrop() is None, "clearing must restore the fast path"


def test_a_backdrop_accepts_a_path_a_pixmap_or_an_image(qtbot, tmp_path):
    from PySide6.QtGui import QImage, QPixmap
    path = tmp_path / "wall.png"
    _backdrop().save(str(path))
    for source in (str(path), path, _backdrop(),
                   QImage(str(path))):
        rain = make_widget(qtbot)
        rain.set_backdrop(source)
        assert rain.backdrop() is not None, f"{source!r} did not load"


def test_a_missing_or_broken_wallpaper_is_not_fatal(qtbot, tmp_path):
    """The file can be deleted between the stylesheet being built and the
    screen being constructed. That is a cosmetic miss, not a crash."""
    rain = make_widget(qtbot)
    rain.set_backdrop(tmp_path / "does-not-exist.png")
    assert rain.backdrop() is None
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image")
    rain.set_backdrop(junk)
    assert rain.backdrop() is None


def test_a_backdrop_that_cannot_be_coerced_at_all_is_survivable(qtbot):
    class Hostile:
        def __str__(self):
            raise RuntimeError("not stringifiable")

    rain = make_widget(qtbot)
    rain.set_backdrop(object())          # coerces, loads nothing
    assert rain.backdrop() is None
    rain.set_backdrop(Hostile())         # cannot even be coerced
    assert rain.backdrop() is None


def test_strips_stay_opaque_without_a_backdrop(qtbot):
    """The fast path: the trail alphas are baked against a constant
    background so the per-frame blit is a straight copy."""
    rain = make_widget(qtbot)
    rain.advance_frame(DT)
    strip = rain._strip_for(0)
    image = strip.toImage()
    assert not image.hasAlphaChannel() or \
        QColor(image.pixelColor(0, 0)).alpha() == 255


def test_strips_carry_alpha_once_there_is_a_picture_under_them(qtbot):
    """There is nothing constant to bake against when the thing under a
    string changes as the string falls."""
    rain = make_widget(qtbot, backdrop=_backdrop())
    rain.advance_frame(DT)
    image = rain._strip_for(0).toImage()
    assert image.hasAlphaChannel()
    corners = [image.pixelColor(0, y).alpha()
               for y in range(0, image.height(), 7)]
    assert min(corners) < 255, "a strip over a picture must not be opaque"


def test_the_backdrop_lands_where_the_window_paints_it(qtbot):
    """The window centres its wallpaper on itself and does not repeat it.
    The rain has to hit exactly the same pixels or the picture visibly
    jumps at the widget's edge."""
    from PySide6.QtCore import QPoint
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(200, 100)
    rain = DnaRainWidget(host, seed=1, backdrop=_backdrop(100, 40))
    rain.setGeometry(20, 10, 180, 90)
    # Image centred on the 200x100 window -> (50, 30); the rain starts at
    # (20, 10), so in its own coordinates the image starts at (30, 20).
    assert rain._backdrop_origin() == QPoint(30, 20)


def test_a_rain_with_no_window_still_places_its_backdrop(qtbot):
    from PySide6.QtCore import QPoint
    rain = make_widget(qtbot, backdrop=_backdrop(64, 48), _w=100, _h=80)
    assert rain._backdrop_origin() == QPoint(18, 16)


def test_clearing_paints_the_picture_not_the_flat_colour(qtbot):
    from PySide6.QtGui import QImage, QPainter
    rain = make_widget(qtbot, backdrop=_backdrop(640, 480), _w=640, _h=480)
    image = QImage(640, 480, QImage.Format_RGB32)
    image.fill(QColor("#000000"))
    painter = QPainter(image)
    rain._clear(painter, QRect(0, 0, 640, 480))
    painter.end()
    assert QColor(image.pixel(320, 240)).name() == "#c81e64"


def test_a_backdrop_smaller_than_the_widget_still_covers_it(qtbot):
    """`WA_OpaquePaintEvent` promises every pixel of the dirty rect is
    written. A wallpaper narrower than the window must not leave the
    previous frame showing through the margins."""
    from PySide6.QtGui import QImage, QPainter
    rain = make_widget(qtbot, backdrop=_backdrop(40, 30), _w=200, _h=150)
    image = QImage(200, 150, QImage.Format_RGB32)
    image.fill(QColor("#00ff00"))
    painter = QPainter(image)
    rain._clear(painter, QRect(0, 0, 200, 150))
    painter.end()
    assert QColor(image.pixel(100, 75)).name() == "#c81e64", "picture"
    assert QColor(image.pixel(2, 2)).name() == \
        QColor(rain.background_color()).name(), "flat fill outside it"


def test_a_backdrop_entirely_outside_the_dirty_rect_falls_back_to_flat(qtbot):
    from PySide6.QtGui import QImage, QPainter
    rain = make_widget(qtbot, backdrop=_backdrop(20, 20), _w=400, _h=400)
    image = QImage(400, 400, QImage.Format_RGB32)
    image.fill(QColor("#00ff00"))
    painter = QPainter(image)
    rain._clear(painter, QRect(0, 0, 40, 40))
    painter.end()
    assert QColor(image.pixel(5, 5)).name() == \
        QColor(rain.background_color()).name()


def test_the_whole_frame_paints_over_a_backdrop(qtbot):
    """End to end: a real paintEvent, splices included, with a picture
    under it. Nothing here may raise and the picture must survive."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QPainter
    rain = make_widget(qtbot, backdrop=_backdrop(640, 480), _w=640, _h=480,
                       spacr_probability=1.0)
    for _ in range(20):
        rain.advance_frame(DT)
    image = QImage(640, 480, QImage.Format_RGB32)
    image.fill(QColor("#000000"))
    painter = QPainter(image)
    rain.render(painter, QPoint(0, 0))
    painter.end()
    pixels = {QColor(image.pixel(x, y)).name()
              for x in range(0, 640, 17) for y in range(0, 480, 13)}
    assert "#c81e64" in pixels, "the wallpaper must survive the rain"
    assert len(pixels) > 1, "and the rain must survive the wallpaper"


def test_the_sequencing_screen_hands_the_rain_its_wallpaper(qtbot,
                                                            qt_theme_applied,
                                                            tmp_path,
                                                            monkeypatch):
    """The wiring, on the real screen: `map_barcodes` under an image
    theme used to paint flat black over the theme's own picture."""
    from spacr.qt.screens import app_screen
    from spacr.qt.screens.app_screen import AppScreen
    path = tmp_path / "wall.png"
    _backdrop(320, 240).save(str(path))
    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: str(path))
    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)
    assert screen._dna_rain is not None
    assert screen._dna_rain.backdrop() is not None


def test_the_opaque_themes_keep_the_cheap_path(qtbot, qt_theme_applied,
                                               monkeypatch):
    from spacr.qt.screens import app_screen
    from spacr.qt.screens.app_screen import AppScreen
    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: None)
    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)
    assert screen._dna_rain is not None
    assert screen._dna_rain.backdrop() is None


def test_a_theme_switch_re_backdrops_the_rain(qtbot, qt_theme_applied,
                                              tmp_path, monkeypatch):
    """Only Home is rebuilt on a theme change; every other screen is
    re-styled in place. That covers everything whose colours come from
    the QSS and nothing that paints itself, so the rain kept the flat
    fill and the wallpaper it was constructed with — a black rectangle
    on the light page, or flat black over a freshly-loaded micrograph.

    The expected fill is ``page_colour``, not ``palette_for(theme)["bg"]``.
    28e9662c (2026-08-04) gave the page a palette role of its own, because
    ``bg`` is the WINDOW colour and on dark it is ``#000000`` — pinning
    ``bg`` here meant asserting that every palette event pushed black back
    into a backdrop that had been built with the page colour. The sibling
    assertions in test_ambient_wiring moved with that commit; this file
    was missed.
    """
    from PySide6.QtCore import QEvent
    from spacr.qt.screens import app_screen
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.theme import page_colour, palette_for

    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: None)
    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)
    rain = screen._dna_rain
    assert rain is not None and rain.backdrop() is None
    chosen = QColor("#ff00ff")
    rain.set_color(chosen)

    path = tmp_path / "wall.png"
    _backdrop(320, 240).save(str(path))
    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: str(path))
    monkeypatch.setattr(
        "spacr.qt.preferences.resolve_effective_theme", lambda: "light")
    screen.changeEvent(QEvent(QEvent.ApplicationPaletteChange))

    assert rain.backdrop() is not None, "the new wallpaper must reach it"
    assert rain.background_color().name() == \
        QColor(page_colour("light")).name()
    # The two roles really are distinct on this theme, so the assertion
    # above is not accidentally satisfied by a palette where page == bg.
    assert QColor(page_colour("light")).name() != \
        QColor(palette_for("light")["bg"]).name()
    assert rain.color().name() == chosen.name(), \
        "a colour the user picked must survive a theme switch"


def test_unrelated_change_events_are_ignored(qtbot, qt_theme_applied,
                                             monkeypatch, tmp_path):
    """Enabled/font changes must not re-theme the backdrop.

    Raising from the stub proves nothing — ``_retheme_backdrops`` wraps
    the whole resolve in ``except Exception``, so an AssertionError from
    in there is swallowed and the test passes either way. What is
    measured instead is the drawn state: a fill nothing in any theme
    would pick is pushed at the rain, and it has to still be there.
    Re-theming costs the rain its pre-rendered strip cache and a full
    repaint, which is why it may not happen on every stylesheet touch.

    The last three lines are the control: an event that *does* mean a
    theme change moves the very same numbers.
    """
    from PySide6.QtCore import QEvent
    from spacr.qt.screens import app_screen
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: None)
    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)
    rain = screen._dna_rain
    assert rain is not None

    rain.set_background_color("#123456")
    applied_before = screen._backdrop_applied

    asked: list = []
    monkeypatch.setattr(app_screen, "_theme_wallpaper",
                        lambda: asked.append("resolved"))
    screen.changeEvent(QEvent(QEvent.EnabledChange))
    screen.changeEvent(QEvent(QEvent.FontChange))
    assert asked == [], "the theme was re-resolved on an unrelated event"
    assert rain.background_color().name() == "#123456", (
        "the backdrop was repainted by an unrelated change event")
    assert rain.backdrop() is None
    assert screen._backdrop_applied == applied_before

    # Control — the same three measurements, on an event that does mean
    # "the theme moved".
    path = tmp_path / "wall.png"
    _backdrop(320, 240).save(str(path))
    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: str(path))
    screen.changeEvent(QEvent(QEvent.ApplicationPaletteChange))
    assert rain.background_color().name() != "#123456"
    assert rain.backdrop() is not None
    assert screen._backdrop_applied != applied_before


def test_a_screen_with_no_rain_shrugs_off_a_theme_switch(qtbot,
                                                         qt_theme_applied):
    from PySide6.QtCore import QEvent
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("measure")
    qtbot.addWidget(screen)
    assert screen._dna_rain is None
    screen.changeEvent(QEvent(QEvent.ApplicationPaletteChange))


def test_a_broken_theme_lookup_does_not_break_the_switch(qtbot,
                                                         qt_theme_applied,
                                                         monkeypatch):
    """A theme that cannot be resolved leaves the backdrop as it was.

    Two things have to hold, and neither is "did not raise". The rain
    must be left with a whole, valid appearance rather than a fill from
    the new theme and a wallpaper from the old one — so the drawn state
    is read back. And the failure must not poison
    ``_backdrop_applied``: recording the tuple before resolving it would
    make the *next*, working switch a no-op, which is what the control
    at the end drives.

    Two expectations were relaxed to say what they meant. This used to
    assert ``_backdrop_applied is None`` after the failed event, which
    was a proxy for "not poisoned" that only worked while the cache
    happened to be empty — Qt delivers an ApplicationPaletteChange during
    AppScreen construction, so by the time the test runs the cache
    legitimately holds the dark page colour that really was painted. The
    honest statement of "not poisoned" is that the failed event leaves
    the cache exactly as it found it, so that is what is asserted, along
    with the cache being truthful in the first place. The final
    expectation moved from ``bg`` to ``page_colour`` for the reason given
    on the sibling test above (28e9662c, 2026-08-04).
    """
    from PySide6.QtCore import QEvent
    from spacr.qt.screens import app_screen
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.theme import page_colour

    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: None)
    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)
    rain = screen._dna_rain
    assert rain is not None

    # The cache is not empty here, and what is in it is not a guess: the
    # construction-time palette event painted that fill. A cache holding
    # a colour the rain was never given would already be the poisoning
    # this test is about.
    cached_fill, cached_wallpaper = screen._backdrop_applied
    assert cached_fill == rain.background_color().name()
    assert cached_wallpaper is None

    rain.set_color("#ff00ff")
    rain.set_background_color("#123456")
    rain.set_backdrop(_backdrop(64, 48))
    applied_before = screen._backdrop_applied

    def boom():
        raise RuntimeError("preferences are gone")
    monkeypatch.setattr("spacr.qt.preferences.resolve_effective_theme", boom)
    screen.changeEvent(QEvent(QEvent.ApplicationPaletteChange))

    assert rain.background_color().isValid()
    assert rain.background_color().name() == "#123456", (
        "a half-applied theme: the fill moved without a theme to move to")
    assert rain.backdrop() is not None, "the wallpaper was cleared"
    assert rain.color().name() == "#ff00ff"
    assert screen._backdrop_applied == applied_before, (
        "the failed lookup was cached and will skip the next real switch")

    # Control — with the lookup working again, the same event does move
    # the same numbers. This is what the poisoned cache would have eaten.
    monkeypatch.setattr("spacr.qt.preferences.resolve_effective_theme",
                        lambda: "light")
    screen.changeEvent(QEvent(QEvent.ApplicationPaletteChange))
    assert rain.background_color().name() == \
        QColor(page_colour("light")).name()
    assert screen._backdrop_applied == (page_colour("light"), None)
    assert screen._backdrop_applied != applied_before


# ---------------------------------------------------------------------------
# The DNA button and its popover
#
# The settings used to be a bar pinned across the bottom of the sequencing
# screen: five controls permanently on show under the settings form the user
# is actually there to fill in. They now live behind a DNA toggle built from
# the same class as the AI toggle it sits beside.
# ---------------------------------------------------------------------------

def _sequencing_screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)
    screen.resize(1200, 800)
    return screen


def test_the_dna_button_sits_immediately_beside_the_ai_button(
        qtbot, qt_theme_applied):
    """Beside, and before: AI keeps its provider chevron next to it."""
    screen = _sequencing_screen(qtbot)
    rain = screen._dna_rain
    assert rain is not None
    button = rain.settings_button
    row = screen._ai_switch.parentWidget().layout()
    assert row.indexOf(button) == row.indexOf(screen._ai_switch) - 1


def test_the_dna_button_is_the_ai_button_not_a_lookalike(qtbot,
                                                        qt_theme_applied):
    """Same class, same object name, so the QSS cannot drift apart."""
    from spacr.qt.widgets.ai_toggle_label import AiToggleLabel
    screen = _sequencing_screen(qtbot)
    button = screen._dna_rain.settings_button
    assert isinstance(button, AiToggleLabel)
    assert button.objectName() == screen._ai_switch.objectName()
    assert button.text() == "DNA"
    # And it inks like one: off is the theme fg, on is the accent.
    button.setChecked(False)
    screen._ai_switch.setChecked(False)
    off = button.styleSheet()
    button.setChecked(True)
    assert button.styleSheet() != off
    # The real AI toggle's signal opens/configures the provider and may
    # immediately turn it back off on a clean machine with no provider.
    # This assertion is about the shared widget's ink, so compare equal
    # logical states without invoking that unrelated application policy.
    was_blocked = screen._ai_switch.blockSignals(True)
    try:
        screen._ai_switch.setChecked(True)
    finally:
        screen._ai_switch.blockSignals(was_blocked)
    assert button.styleSheet() == screen._ai_switch.styleSheet()


def test_the_settings_are_hidden_until_the_button_is_clicked(
        qtbot, qt_theme_applied):
    """The whole point of the change: no permanent strip of controls."""
    screen = _sequencing_screen(qtbot)
    rain = screen._dna_rain
    screen.show()
    qtbot.waitExposed(screen)

    assert not rain.settings_popover.isVisible()
    assert not rain.settings_bar.isVisible()
    # It is nowhere in the screen's own widget tree either — not merely
    # hidden inside it.
    assert rain.settings_bar.window() is rain.settings_popover

    qtbot.mouseClick(rain.settings_button, Qt.LeftButton)
    assert rain.settings_button.isChecked()
    assert rain.settings_button.is_open()
    assert rain.settings_popover.isVisible()
    assert rain.settings_bar.isVisible()

    qtbot.mouseClick(rain.settings_button, Qt.LeftButton)
    assert not rain.settings_button.isChecked()
    assert not rain.settings_button.is_open()
    assert not rain.settings_popover.isVisible()


def test_the_popover_re_themes_itself_every_time_it_opens(qtbot, monkeypatch):
    """It is a top-level window, so the screen's re-style never reaches it.

    Both the popup frame and the panel inside it state their own
    surface; left alone across a theme switch that is a dark card under
    freshly light text.
    """
    from spacr.qt.theme import palette_for
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(400, 300)
    host.show()
    qtbot.waitExposed(host)
    monkeypatch.setattr(
        "spacr.qt.preferences.resolve_effective_theme", lambda: "dark")
    rain = install_dna_rain(host, None, seed=8, theme="dark")
    rain.settings_button.setChecked(True)
    assert palette_for("dark")["surface"] in rain.settings_bar.styleSheet()
    rain.settings_button.setChecked(False)

    monkeypatch.setattr(
        "spacr.qt.preferences.resolve_effective_theme", lambda: "light")
    rain.settings_button.setChecked(True)
    assert palette_for("light")["surface"] in rain.settings_bar.styleSheet()
    assert palette_for("light")["surface_alt"] in \
        rain.settings_popover.styleSheet()


def test_closing_the_popover_any_other_way_un_toggles_the_button(
        qtbot, qt_theme_applied):
    """Escape, a click elsewhere, a tab switch — the button must follow."""
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent as _QEvent
    screen = _sequencing_screen(qtbot)
    rain = screen._dna_rain
    screen.show()
    qtbot.waitExposed(screen)
    button = rain.settings_button

    button.setChecked(True)
    assert rain.settings_popover.isVisible()
    rain.settings_popover.keyPressEvent(
        QKeyEvent(_QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert not rain.settings_popover.isVisible()
    assert not button.isChecked(), "the button was left lit with nothing open"

    # And the screen going away takes the popover with it, rather than
    # leaving it floating over whichever module the user switched to.
    button.setChecked(True)
    assert rain.settings_popover.isVisible()
    screen.hide()
    assert not rain.settings_popover.isVisible()


def _press_popover_at(popover, global_pos):
    """The press Qt delivers to a Qt.Popup before it closes it."""
    from PySide6.QtCore import QEvent as _QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    popover.mousePressEvent(QMouseEvent(
        _QEvent.MouseButtonPress,
        QPointF(popover.mapFromGlobal(global_pos)), QPointF(global_pos),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))


def _open_rain_in_a_window(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(400, 300)
    host.show()
    qtbot.waitExposed(host)
    return install_dna_rain(host, None, seed=5)


def test_the_click_that_closes_the_popover_does_not_reopen_it(qtbot):
    """A Qt.Popup closes on the press, which is then replayed underneath.

    Without the guard that is open-close-open inside one click, and the
    popover looks like it will not close.
    """
    rain = _open_rain_in_a_window(qtbot)
    button = rain.settings_button
    button.setChecked(True)
    assert rain.settings_popover.isVisible()

    # Exactly what Qt does: press -> popup -> close -> replay at button.
    _press_popover_at(rain.settings_popover,
                      button.mapToGlobal(button.rect().center()))
    rain.settings_popover.hide()
    assert not button.isChecked()
    button.setChecked(True)                 # the replayed half
    assert not rain.settings_popover.isVisible()
    assert not button.isChecked()


def test_a_click_somewhere_else_does_not_arm_the_guard(qtbot):
    """Dismiss by clicking away, then click DNA — that must open it."""
    rain = _open_rain_in_a_window(qtbot)
    button = rain.settings_button
    button.setChecked(True)

    away = button.mapToGlobal(button.rect().center())
    away.setY(away.y() + 400)
    _press_popover_at(rain.settings_popover, away)
    rain.settings_popover.hide()
    assert not button.isChecked()
    button.setChecked(True)
    assert rain.settings_popover.isVisible()


def test_a_deliberate_second_click_still_opens_it(qtbot):
    """A close the button asked for never arms the guard."""
    rain = _open_rain_in_a_window(qtbot)
    button = rain.settings_button
    button.setChecked(True)
    button.setChecked(False)
    assert not rain.settings_popover.isVisible()
    button.setChecked(True)
    assert rain.settings_popover.isVisible()


def test_a_press_inside_the_popover_is_not_a_close(qtbot):
    """Moving a slider must not look like the dismissing click."""
    rain = _open_rain_in_a_window(qtbot)
    popover = rain.settings_popover
    rain.settings_button.setChecked(True)
    _press_popover_at(popover, popover.mapToGlobal(popover.rect().center()))
    assert popover.isVisible()
    assert not popover.just_closed()


def test_every_setting_in_the_popover_reaches_the_renderer(qtbot):
    """Four controls, four pieces of renderer state. No decoration."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(800, 600)
    rain = install_dna_rain(host, None, seed=9)
    bar = rain.settings_button.settings_bar
    assert bar is rain.settings_bar

    bar.set_color("#7733aa")
    assert rain.color().name() == "#7733aa"
    assert bar.color().name() == "#7733aa"

    bar.set_speed(2.5)
    assert rain.speed() == pytest.approx(2.5)
    assert bar.speed() == pytest.approx(2.5)

    bar.set_opacity(0.45)
    assert rain.opacity() == pytest.approx(0.45, abs=0.01)
    assert bar.opacity() == pytest.approx(0.45, abs=0.01)

    bar.set_font_size(28)
    assert rain.font_size() == 28
    assert bar.font_size() == 28
    assert rain.engine.n_columns == 800 // 28

    bar.set_random_color(True)
    assert rain.random_colors() is True
    assert bar.random_color() is True


def test_the_popover_hands_the_button_a_bound_bar(qtbot):
    """Opened cold, the controls must already show what the rain is doing."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(600, 400)
    rain = install_dna_rain(host, None, seed=2, color="#123456",
                            opacity=0.33, font_size=22, random_colors=True)
    rain.set_speed(1.75)
    bar = DnaRainSettingsBar()
    qtbot.addWidget(bar)
    bar.bind(rain)
    assert bar.color().name() == "#123456"
    assert bar.opacity() == pytest.approx(0.33, abs=0.01)
    assert bar.font_size() == 22
    assert bar.speed() == pytest.approx(1.75)
    assert bar.random_color() is True


def test_the_grid_layout_carries_the_same_controls_as_the_row(qtbot):
    """The popover's shape is a layout choice, not a different widget."""
    row = DnaRainSettingsBar()
    grid = DnaRainSettingsBar(vertical=True)
    qtbot.addWidget(row)
    qtbot.addWidget(grid)
    for bar in (row, grid):
        assert bar._swatch is not None and bar._random is not None
        assert bar.speed() == pytest.approx(1.0)
        assert bar.opacity() == pytest.approx(dr.DEFAULT_OPACITY, abs=0.01)
        assert bar.font_size() == dr.DEFAULT_FONT_PX
    assert grid.sizeHint().height() > row.sizeHint().height(), \
        "a grid is taller than a row"
    assert grid.sizeHint().width() < row.sizeHint().width(), \
        "and narrower — a row of five controls is half a screen wide"


def test_a_host_without_an_ai_toggle_falls_back_to_the_layout(qtbot):
    """Discovery is a default, not a requirement."""
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    host.resize(400, 300)
    rain = install_dna_rain(host, layout, seed=3)
    assert dr._find_ai_toggle(host) is None
    assert layout.indexOf(rain.settings_button) >= 0


def test_an_explicit_anchor_wins_over_discovery(qtbot):
    from spacr.qt.widgets.ai_toggle_label import AiToggleLabel
    host = QWidget()
    qtbot.addWidget(host)
    outer = QVBoxLayout(host)
    row_host = QWidget()
    row = QVBoxLayout(row_host)
    ai = AiToggleLabel()
    chosen = AiToggleLabel(text="Live")
    row.addWidget(ai)
    row.addWidget(chosen)
    outer.addWidget(row_host)
    host.resize(400, 300)

    assert dr._find_ai_toggle(host) is ai
    rain = install_dna_rain(host, outer, anchor=chosen, seed=3)
    assert row.indexOf(rain.settings_button) == row.indexOf(chosen) - 1


def test_the_button_survives_an_anchor_with_no_layout(qtbot):
    """A widget that is not in a layout cannot be sat beside."""
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    orphan = QWidget(host)
    rain = install_dna_rain(host, layout, anchor=orphan, seed=3)
    assert layout.indexOf(rain.settings_button) >= 0


# ---------------------------------------------------------------------------
# Random colour
#
# Per column, and re-rolled on every respawn — not one random colour per
# session. A column is the unit the effect is built out of: it already has its
# own length, speed, start row and highlight run, so a colour of its own is
# the same kind of variation, and it costs nothing because the strips are
# already cached per column.
# ---------------------------------------------------------------------------

def test_random_colour_gives_the_columns_more_than_one_colour(qtbot):
    widget = make_widget(qtbot, _w=640, _h=480, random_colors=True)
    colors = {widget.column_color(i).name()
              for i in range(widget.engine.n_columns)}
    assert len(colors) > 1, "random colour produced one colour"
    assert len(colors) > 10, f"only {len(colors)} colours across 40 columns"


def test_without_random_colour_every_column_is_the_picked_one(qtbot):
    widget = make_widget(qtbot, _w=640, _h=480, color="#00ff88")
    colors = {widget.column_color(i).name()
              for i in range(widget.engine.n_columns)}
    assert colors == {"#00ff88"}


def test_random_colour_reaches_the_painted_output(qtbot):
    """The accessor could agree with itself and still paint one colour."""
    widget = make_widget(qtbot, _w=320, _h=240, font_size=16, opacity=1.0,
                         background="#000000", spacr_probability=0.0,
                         random_colors=True)
    widget.show()
    qtbot.waitExposed(widget)
    for _ in range(30):
        widget.advance_frame(DT)
    image = widget.grab().toImage()
    hues = set()
    for x in range(image.width()):
        for y in range(0, image.height(), 3):
            pixel = image.pixelColor(x, y)
            if pixel.lightnessF() > 0.12 and pixel.saturationF() > 0.3:
                hues.add(pixel.hue() // 15)
    assert len(hues) > 3, f"only {len(hues)} hue families were painted"


def test_random_colour_is_per_column_not_per_session(qtbot):
    """Two neighbours differ, and a column changes when it respawns."""
    widget = make_widget(qtbot, _w=640, _h=480, random_colors=True)
    first = widget.column_color(0).name()
    assert any(widget.column_color(i).name() != first
               for i in range(1, widget.engine.n_columns))

    column = widget.engine.columns[0]
    seen = {first}
    for _ in range(40):
        widget.engine._respawn(column)
        seen.add(widget.column_color(0).name())
    assert len(seen) > 5, "a column keeps its colour forever"


def test_random_colours_borrow_vividness_from_the_picked_colour(qtbot):
    """Only the hue is random; a grey pick still gives visible colours."""
    widget = make_widget(qtbot, _w=640, _h=480, color="#7a7a7a",
                         random_colors=True)
    colors = [widget.column_color(i)
              for i in range(widget.engine.n_columns)]
    assert all(c.saturationF() >= dr.RANDOM_MIN_SATURATION - 1e-6
               for c in colors)
    assert all(dr.RANDOM_MIN_LIGHTNESS - 1e-6 <= c.lightnessF()
               <= dr.RANDOM_MAX_LIGHTNESS + 1e-6 for c in colors)
    assert len({c.hue() for c in colors}) > 5


def test_the_picked_colour_still_steers_a_random_field(qtbot):
    """Moving the swatch has to change something while random is on."""
    widget = make_widget(qtbot, _w=640, _h=480, color="#009B9B",
                         random_colors=True)
    before = [widget.column_color(i) for i in range(widget.engine.n_columns)]
    widget.set_color("#ffd0d0")             # pale: lighter, less saturated
    after = [widget.column_color(i) for i in range(widget.engine.n_columns)]
    assert [c.name() for c in before] != [c.name() for c in after]
    assert (sum(c.lightnessF() for c in after)
            > sum(c.lightnessF() for c in before))
    # Hues are untouched by the pick; only saturation and lightness are.
    assert [c.hue() for c in before] == [c.hue() for c in after]


def test_toggling_random_colour_re_renders_the_strips(qtbot):
    """The colours are baked into the cached strips, so they must drop."""
    widget = make_widget(qtbot, _w=320, _h=240)
    widget.show()
    qtbot.waitExposed(widget)
    widget.grab()
    generation = widget._style_gen
    widget.set_random_colors(True)
    assert widget._style_gen > generation
    assert widget._strips == []
    widget.set_random_colors(True)          # no-op, no re-render
    assert widget._style_gen == generation + 1


def test_random_colour_does_not_change_where_anything_falls():
    """The hues come from their own RNG, so a seed still reproduces a rain."""
    plain = DnaRainEngine(960, 640, 16, seed=99)
    for _ in range(50):
        plain.advance(DT)
    coloured = DnaRainEngine(960, 640, 16, seed=99)
    for _ in range(50):
        coloured.advance(DT)
    assert plain.snapshot() == coloured.snapshot()
    assert [c.hue for c in plain.columns] == [c.hue for c in coloured.columns]
    # ... and a different seed gives different hues.
    other = DnaRainEngine(960, 640, 16, seed=100)
    assert [c.hue for c in other.columns] != [c.hue for c in plain.columns]


def test_a_seedless_engine_still_rolls_hues():
    engine = DnaRainEngine(320, 240, 16, seed=None)
    hues = [c.hue for c in engine.columns]
    assert len(set(hues)) > 1
    assert all(0.0 <= h < 1.0 for h in hues)


def test_the_random_pen_cache_is_bounded(qtbot):
    """One entry per whole degree of hue, however long the rain runs."""
    widget = make_widget(qtbot, _w=640, _h=480, random_colors=True)
    widget.show()
    qtbot.waitExposed(widget)
    for _ in range(200):
        widget.advance_frame(DT)
        widget.grab()
    assert widget._hue_pens, "random mode never built a pen"
    assert len(widget._hue_pens) <= 360
    assert all(0 <= key < 360 for key in widget._hue_pens)


def test_a_colour_change_drops_the_random_pens(qtbot):
    widget = make_widget(qtbot, _w=320, _h=240, random_colors=True)
    widget.show()
    qtbot.waitExposed(widget)
    widget.grab()
    assert widget._hue_pens
    widget.set_color("#ff0000")
    assert widget._hue_pens == {}


def test_the_random_toggle_round_trips_through_the_bar(qtbot):
    widget = make_widget(qtbot)
    bar = DnaRainSettingsBar()
    qtbot.addWidget(bar)
    bar.bind(widget)
    assert bar.random_color() is False
    with qtbot.waitSignal(bar.random_color_changed):
        bar.set_random_color(True)
    assert widget.random_colors() is True
    bar.set_random_color(False)
    assert widget.random_colors() is False


def test_the_bar_opens_on_whatever_the_rain_is_already_doing(qtbot):
    widget = make_widget(qtbot, random_colors=True)
    bar = DnaRainSettingsBar()
    qtbot.addWidget(bar)
    bar.bind(widget)
    assert bar.random_color() is True


# ---------------------------------------------------------------------------
# The shipped defaults
#
# Every one of these was chosen by the user after seeing it on screen, and
# every one of them has been silently wrong at least once — the teal was a
# constant nothing read, and the rain ran in the theme accent for weeks.
# ---------------------------------------------------------------------------

def test_the_shipped_defaults_are_the_ones_that_were_asked_for():
    assert dr.DEFAULT_COLOR.lower() == "#009b9b"
    assert QColor(dr.DEFAULT_COLOR).getRgb()[:3] == (0, 155, 155)
    assert dr.DEFAULT_OPACITY == pytest.approx(0.20)
    assert dr.DEFAULT_FONT_PX == 16
    assert dr.DEFAULT_FPS == 60


def test_a_rain_built_with_no_arguments_uses_them(qtbot):
    widget = make_widget(qtbot)
    assert widget.color().name() == "#009b9b"
    assert widget.opacity() == pytest.approx(0.20)
    assert widget.font_size() == 16
    assert widget.speed() == pytest.approx(1.0)
    assert widget.fps() == 60
    assert widget.random_colors() is False


def test_the_defaults_survive_all_the_way_to_the_real_screen(
        qtbot, qt_theme_applied):
    """Constants are only defaults if the thing the user sees uses them."""
    screen = _sequencing_screen(qtbot)
    rain = screen._dna_rain
    bar = rain.settings_bar
    assert rain.color().name() == "#009b9b"
    assert rain.opacity() == pytest.approx(0.20)
    assert rain.font_size() == 16
    assert rain.speed() == pytest.approx(1.0)
    assert rain.random_colors() is False
    # And the popover opens showing exactly that, rather than its own
    # idea of a default.
    assert bar.color().name() == "#009b9b"
    assert round(bar.opacity() * 100) == 20
    assert bar.font_size() == 16
    assert bar.speed() == pytest.approx(1.0)
    assert bar.random_color() is False
    assert "#009b9b" in bar._swatch.styleSheet().lower()


def test_the_teal_actually_reaches_the_glyphs(qtbot):
    """The default is a painted colour, not a docstring."""
    widget = make_widget(qtbot, _w=320, _h=240, background="#000000",
                         opacity=1.0, spacr_probability=0.0)
    widget.show()
    qtbot.waitExposed(widget)
    for _ in range(30):
        widget.advance_frame(DT)
    image = widget.grab().toImage()
    teal = 0
    for x in range(image.width()):
        for y in range(0, image.height(), 3):
            pixel = image.pixelColor(x, y)
            if pixel.green() > 60 and pixel.blue() > 60 and pixel.red() < 40:
                teal += 1
    assert teal > 20, f"only {teal} teal-ish pixels were painted"
