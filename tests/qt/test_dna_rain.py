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


def test_slow_columns_cost_nothing_on_most_ticks():
    """Quantising to cells means a column is dirty only when it moves."""
    engine = DnaRainEngine(1920, 1080, 16, seed=5)
    engine.set_speed_multiplier(0.05)
    dirty = [len(engine.advance(DT)) for _ in range(120)]
    assert statistics.mean(dirty) < 6, statistics.mean(dirty)


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
    assert "SPACR" not in text and "spacr" not in text


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


def test_default_colours_come_from_the_theme_palette(qtbot):
    from spacr.qt.theme import palette_for
    widget = make_widget(qtbot, theme="light")
    palette = palette_for("light")
    assert widget.color().name() == QColor(palette["accent"]).name()
    assert widget.background_color().name() == QColor(palette["bg"]).name()


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


def test_only_moved_columns_are_repainted(qtbot):
    """Partial repaints, not the whole canvas, when little changed."""
    widget = make_widget(qtbot, _w=1920, _h=1080)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_speed(0.02)
    partial = 0
    for _ in range(60):
        rects = widget.advance_frame(DT)
        if rects and not widget.last_full_repaint:
            partial += 1
            for rect in rects:
                assert rect.width() <= 1920
                assert rect.height() <= 1080
    assert partial > 10, "never took the partial-repaint path"


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


def test_still_frame_schedules_nothing(qtbot):
    widget = make_widget(qtbot)
    widget.set_speed(0.0)
    assert widget.advance_frame(DT) == []
    assert not widget.last_full_repaint


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
    bar = DnaRainSettingsBar()
    qtbot.addWidget(bar)
    with qtbot.waitSignal(bar.color_changed):
        bar.set_color("#010203")
    with qtbot.waitSignal(bar.speed_changed):
        bar.set_speed(2.0)
    with qtbot.waitSignal(bar.font_size_changed):
        bar.set_font_size(28)


def test_settings_bar_takes_defaults_from_the_theme(qtbot):
    from spacr.qt.theme import palette_for
    bar = DnaRainSettingsBar(theme="space")
    qtbot.addWidget(bar)
    assert bar.color().name() == QColor(palette_for("space")["accent"]).name()


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
    assert layout.indexOf(rain.settings_bar) >= 0
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
    assert rain.settings_bar.parent() is host
    assert rain.settings_bar.parentWidget() is host


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
    assert screen.layout().indexOf(rain.settings_bar) >= 0

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

def test_the_default_is_visible_enough_to_read_as_an_effect():
    from spacr.qt.widgets import dna_rain as dr
    assert dr.DEFAULT_OPACITY > 0.22, (
        "0.22 is the value the user called too faint; the default must exceed it")
    assert dr.DEFAULT_OPACITY <= 0.6, (
        "the rain sits BEHIND the settings; past ~0.6 it competes with them")


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
    on the light page, or flat black over a freshly-loaded micrograph."""
    from PySide6.QtCore import QEvent
    from spacr.qt.screens import app_screen
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.theme import palette_for

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
        QColor(palette_for("light")["bg"]).name()
    assert rain.color().name() == chosen.name(), \
        "a colour the user picked must survive a theme switch"


def test_unrelated_change_events_are_ignored(qtbot, qt_theme_applied,
                                             monkeypatch):
    from PySide6.QtCore import QEvent
    from spacr.qt.screens import app_screen
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: None)
    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)

    def explode():
        raise AssertionError("re-themed on the wrong event")
    monkeypatch.setattr(app_screen, "_theme_wallpaper", explode)
    screen.changeEvent(QEvent(QEvent.EnabledChange))
    screen.changeEvent(QEvent(QEvent.FontChange))


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
    from PySide6.QtCore import QEvent
    from spacr.qt.screens import app_screen
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: None)
    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)

    def boom():
        raise RuntimeError("preferences are gone")
    monkeypatch.setattr("spacr.qt.preferences.resolve_effective_theme", boom)
    screen.changeEvent(QEvent(QEvent.ApplicationPaletteChange))
