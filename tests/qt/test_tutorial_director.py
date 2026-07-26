"""Director / Narrator tests — the tutorial render pipeline end to end.

Piper is stubbed (a separate binary plus a 63 MB voice model is a true
externality) but it is stubbed by writing a *real* WAV, so every step
downstream of it — duration measurement, frame budgeting, the ffmpeg
concat list, the SRT sidecar and the mux — runs for real against real
files. ffmpeg itself is used for real where it is installed.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG,
                                    reason="ffmpeg not installed")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write_wav(path: Path, seconds: float, rate: int = 22050) -> float:
    """Write a real silent mono 16-bit WAV. Returns its exact duration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(seconds * rate))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n_frames)
    return n_frames / float(rate)


class FakeNarrator:
    """Piper stand-in that emits real WAVs of a known length."""

    def __init__(self, seconds: float = 0.4, **kwargs):
        self.seconds = seconds
        self.calls: list[tuple[str, Path]] = []
        self.kwargs = kwargs

    def synth(self, text: str, out_wav: Path) -> float:
        dur = write_wav(Path(out_wav), self.seconds)
        self.calls.append((text, Path(out_wav)))
        return dur


class SpyRecorder:
    """Wraps the real Recorder, remembering every cursor position it was
    asked to draw. The engine code under test is untouched."""

    def __init__(self, inner):
        self._inner = inner
        self.positions: list[tuple[float, float]] = []
        self.highlights: list = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def cursor_pos(self):
        return self._inner.cursor_pos

    @cursor_pos.setter
    def cursor_pos(self, value):
        self._inner.cursor_pos = value

    def snap(self, cursor_pos=None, highlight_rect=None):
        path = self._inner.snap(cursor_pos=cursor_pos,
                                  highlight_rect=highlight_rect)
        self.positions.append(self._inner.cursor_pos)
        self.highlights.append(highlight_rect)
        return path


@pytest.fixture
def geo_window(qtbot, qt_theme_applied):
    """A 400x300 QMainWindow with a button at a known place inside it."""
    from PySide6.QtWidgets import QMainWindow, QPushButton, QWidget
    win = QMainWindow()
    central = QWidget()
    win.setCentralWidget(central)
    btn = QPushButton("Run", central)
    btn.setGeometry(30, 40, 80, 24)
    win.resize(400, 300)
    qtbot.addWidget(win)
    win.show()
    qt_theme_applied.processEvents()
    return win, btn


def make_director(window, steps, out_dir, narrator=None, fps=6):
    from spacr.qt.tutorial.engine import Director
    return Director(window, steps, out_dir=out_dir,
                     narrator=narrator or FakeNarrator(), fps=fps)


# ---------------------------------------------------------------------------
# Narrator
# ---------------------------------------------------------------------------

def test_wav_duration_reads_the_real_header(tmp_path):
    from spacr.qt.tutorial.engine import _wav_duration
    exact_a = write_wav(tmp_path / "a.wav", 0.25, rate=22050)
    assert _wav_duration(tmp_path / "a.wav") == pytest.approx(exact_a, abs=1e-9)
    assert _wav_duration(tmp_path / "a.wav") == pytest.approx(0.25, abs=1e-4)
    exact_b = write_wav(tmp_path / "b.wav", 1.5, rate=16000)
    assert _wav_duration(tmp_path / "b.wav") == pytest.approx(exact_b, abs=1e-9)
    assert _wav_duration(tmp_path / "b.wav") == pytest.approx(1.5, abs=1e-4)


def test_narrator_accepts_an_existing_model_and_keeps_length_scale(tmp_path):
    from spacr.qt.tutorial.engine import Narrator
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not really an onnx, but it exists")
    n = Narrator(voice_model=model, length_scale=0.85)
    assert n.voice_model == model
    assert n.length_scale == 0.85
    assert Narrator(voice_model=model).length_scale == 1.0


def test_narrator_error_names_the_missing_model(tmp_path):
    from spacr.qt.tutorial.engine import Narrator
    missing = tmp_path / "no-such-voice.onnx"
    with pytest.raises(FileNotFoundError) as exc:
        Narrator(voice_model=missing)
    assert str(missing) in str(exc.value)


def test_narrator_falls_back_to_the_default_voice_path(tmp_path,
                                                         monkeypatch):
    """With no voice_model= the Narrator uses DEFAULT_VOICE."""
    from spacr.qt.tutorial import engine
    fallback = tmp_path / "default.onnx"
    fallback.write_bytes(b"x")
    monkeypatch.setattr(engine, "DEFAULT_VOICE", fallback)
    assert engine.Narrator().voice_model == fallback

    monkeypatch.setattr(engine, "DEFAULT_VOICE", tmp_path / "gone.onnx")
    with pytest.raises(FileNotFoundError):
        engine.Narrator()


def test_narrator_synth_builds_the_piper_command_and_returns_duration(
        tmp_path, monkeypatch):
    from spacr.qt.tutorial import engine
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    seen = {}

    def fake_run(cmd, input=None, capture_output=False, timeout=None,
                   **kwargs):
        seen["cmd"] = cmd
        seen["input"] = input
        seen["timeout"] = timeout
        seen["written"] = write_wav(Path(cmd[cmd.index("-f") + 1]), 0.75)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(engine.subprocess, "run", fake_run)
    n = engine.Narrator(voice_model=model, length_scale=0.9)
    out = tmp_path / "deep" / "nested" / "step.wav"
    dur = n.synth("hello spaCR", out)

    assert dur == pytest.approx(seen["written"], abs=1e-9)
    assert dur == pytest.approx(0.75, abs=1e-4)
    assert out.exists(), "synth must create the parent directory"
    assert seen["input"] == b"hello spaCR"
    assert seen["timeout"] == 120
    cmd = seen["cmd"]
    assert cmd[0] == "piper"
    assert cmd[cmd.index("-m") + 1] == str(model)
    assert cmd[cmd.index("--length-scale") + 1] == "0.9"
    assert cmd[cmd.index("-f") + 1] == str(out)


def test_narrator_synth_surfaces_piper_stderr(tmp_path, monkeypatch):
    from spacr.qt.tutorial import engine
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, b"",
                                             "phoneme table missing".encode())

    monkeypatch.setattr(engine.subprocess, "run", fake_run)
    n = engine.Narrator(voice_model=model)
    with pytest.raises(RuntimeError) as exc:
        n.synth("hi", tmp_path / "x.wav")
    assert "phoneme table missing" in str(exc.value)


# ---------------------------------------------------------------------------
# Director: construction, budgets, SRT
# ---------------------------------------------------------------------------

def test_director_builds_its_own_narrator_when_none_given(
        geo_window, tmp_path, monkeypatch):
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    voice = tmp_path / "v.onnx"
    voice.write_bytes(b"x")
    monkeypatch.setattr(engine, "DEFAULT_VOICE", voice)

    d = engine.Director(win, [], out_dir=tmp_path / "out")
    assert isinstance(d.narrator, engine.Narrator)
    assert d.narrator.voice_model == voice
    assert (tmp_path / "out").is_dir(), "out_dir is created up front"
    assert d._workdir.is_dir()
    assert d.fps == engine.FRAME_RATE
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_prerender_audio_totals_narration_plus_holds(geo_window, tmp_path):
    from spacr.qt.tutorial.engine import Step
    win, _ = geo_window
    steps = [Step("first beat", hold_ms=500),
             Step("second beat", hold_ms=0),
             Step("third beat", hold_ms=250)]
    narrator = FakeNarrator(seconds=0.4)
    d = make_director(win, steps, tmp_path / "out", narrator=narrator)

    total = d._prerender_audio()

    assert total == pytest.approx(3 * 0.4 + 0.5 + 0.0 + 0.25, abs=1e-3)
    assert [t for t, _ in narrator.calls] == [
        "first beat", "second beat", "third beat"]
    assert [p.name for _, p in narrator.calls] == [
        "step_000.wav", "step_001.wav", "step_002.wav"]
    # The engine kept the narration text alongside each clip for the SRT.
    assert [text for _d, _p, text in d._audio_wavs] == [
        "first beat", "second beat", "third beat"]
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_frames_for_covers_narration_and_hold(geo_window, tmp_path):
    from spacr.qt.tutorial.engine import Step
    win, _ = geo_window
    steps = [Step("a", hold_ms=500), Step("b", hold_ms=0)]
    d = make_director(win, steps, tmp_path / "out",
                       narrator=FakeNarrator(seconds=0.4), fps=6)
    d._prerender_audio()

    assert d._frames_for(0) == math.ceil((0.4 + 0.5) * 6)   # 6
    assert d._frames_for(1) == math.ceil(0.4 * 6)           # 3
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_frames_for_never_returns_zero(geo_window, tmp_path):
    """A silent step still gets at least one frame — a zero-frame step
    would desync every later step from its audio."""
    from spacr.qt.tutorial.engine import Step
    win, _ = geo_window
    d = make_director(win, [Step("x", hold_ms=0)], tmp_path / "out",
                       narrator=FakeNarrator(seconds=0.0), fps=30)
    d._prerender_audio()
    assert d._frames_for(0) == 1
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_write_srt_emits_sequential_cues_offset_by_holds(geo_window,
                                                           tmp_path):
    from spacr.qt.tutorial.engine import Step
    win, _ = geo_window
    steps = [Step("one", hold_ms=500), Step("two", hold_ms=0),
             Step("three", hold_ms=1000)]
    d = make_director(win, steps, tmp_path / "out",
                       narrator=FakeNarrator(seconds=2.0))
    d._prerender_audio()

    srt = d._write_srt("demo")
    assert srt == tmp_path / "out" / "demo.srt"
    blocks = [b for b in srt.read_text().split("\n\n") if b.strip()]
    assert len(blocks) == 3
    assert blocks[0].splitlines() == [
        "1", "00:00:00,000 --> 00:00:02,000", "one"]
    # +0.5 s hold before cue 2
    assert blocks[1].splitlines() == [
        "2", "00:00:02,500 --> 00:00:04,500", "two"]
    # no hold after cue 2
    assert blocks[2].splitlines() == [
        "3", "00:00:04,500 --> 00:00:06,500", "three"]
    shutil.rmtree(d._workdir, ignore_errors=True)


@pytest.mark.parametrize("seconds,expected", [
    (0, "00:00:00,000"),
    (1.5, "00:00:01,500"),
    (65.25, "00:01:05,250"),
    (3661.001, "01:01:01,001"),
    # Regression: rounding the fraction on its own gave "00:00:00,1000",
    # a four-digit millisecond field that breaks every SRT parser.
    (0.9999, "00:00:01,000"),
    (1.9996, "00:00:02,000"),
    (59.9999, "00:01:00,000"),
    (3599.9999, "01:00:00,000"),
])
def test_srt_timestamps_never_overflow_the_millisecond_field(seconds,
                                                               expected):
    from spacr.qt.tutorial.engine import _srt_ts
    out = _srt_ts(seconds)
    assert out == expected
    assert len(out) == 12 and out[8] == ","
    assert len(out.split(",")[1]) == 3


# ---------------------------------------------------------------------------
# Director: target / highlight resolution
# ---------------------------------------------------------------------------

def test_resolve_target_centres_on_the_widget(geo_window, tmp_path,
                                                monkeypatch):
    from spacr.qt.tutorial import engine
    win, btn = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (400, 300))
    d = make_director(win, [], tmp_path / "out")

    step = engine.Step("s", target=(btn, None))
    got = d._resolve_target(step)
    # mapTo walks the parent chain directly — an independent route to the
    # same answer the engine takes via global coords.
    expect = btn.mapTo(win, btn.rect().center())
    assert got == (float(expect.x()), float(expect.y()))
    # …and the point really sits inside the button, in window coords.
    top_left = btn.mapTo(win, btn.rect().topLeft())
    assert top_left.x() <= got[0] <= top_left.x() + btn.width()
    assert top_left.y() <= got[1] <= top_left.y() + btn.height()
    assert got[0] == pytest.approx(top_left.x() + btn.width() / 2, abs=1.0)
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_resolve_target_honours_an_explicit_offset(geo_window, tmp_path,
                                                     monkeypatch):
    from spacr.qt.tutorial import engine
    win, btn = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (400, 300))
    d = make_director(win, [], tmp_path / "out")

    got = d._resolve_target(engine.Step("s", target=(btn, (5, 7))))
    expect = btn.mapTo(win, btn.rect().topLeft())
    assert got == (float(expect.x() + 5), float(expect.y() + 7))
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_resolve_target_accepts_a_bare_widget(geo_window, tmp_path,
                                                monkeypatch):
    from spacr.qt.tutorial import engine
    win, btn = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (400, 300))
    d = make_director(win, [], tmp_path / "out")
    assert (d._resolve_target(engine.Step("s", target=btn))
             == d._resolve_target(engine.Step("s", target=(btn, None))))
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_resolve_target_scales_window_pixels_to_frame_pixels(
        geo_window, tmp_path, monkeypatch):
    """Frames are always VIDEO_SIZE, so a 400x300 window's coordinates
    must be doubled to land in an 800x600 frame."""
    from spacr.qt.tutorial import engine
    win, btn = geo_window
    d = make_director(win, [], tmp_path / "out")

    monkeypatch.setattr(engine, "VIDEO_SIZE", (400, 300))
    unscaled = d._resolve_target(engine.Step("s", target=(btn, None)))
    monkeypatch.setattr(engine, "VIDEO_SIZE", (800, 600))
    scaled = d._resolve_target(engine.Step("s", target=(btn, None)))

    assert scaled == (unscaled[0] * 2, unscaled[1] * 2)
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_resolve_target_returns_none_when_unset_or_empty(geo_window,
                                                           tmp_path):
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    d = make_director(win, [], tmp_path / "out")
    assert d._resolve_target(engine.Step("s")) is None
    assert d._resolve_target(engine.Step("s", target=(None, None))) is None
    assert d._resolve_target(engine.Step("s", target=(None, (1, 2)))) is None
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_resolve_target_defers_a_callable_until_capture_time(
        geo_window, tmp_path, monkeypatch):
    """Regression: scripts name widgets that only exist after an earlier
    step has run. Passing them eagerly evaluated to None while the Step
    list was being built and the step silently targeted nothing, so the
    engine accepts a zero-arg callable and resolves it here instead."""
    from spacr.qt.tutorial import engine
    win, btn = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (400, 300))
    d = make_director(win, [], tmp_path / "out")

    holder = [None]
    step = engine.Step("s", target=(lambda: holder[0], None))
    assert d._resolve_target(step) is None      # not there yet
    holder[0] = btn
    assert d._resolve_target(step) == d._resolve_target(
        engine.Step("s", target=(btn, None)))
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_resolve_target_warns_instead_of_dying_on_a_broken_widget(
        geo_window, tmp_path, caplog):
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    d = make_director(win, [], tmp_path / "out")

    class Broken:
        def rect(self):
            raise RuntimeError("widget was deleted")

    with caplog.at_level("WARNING", logger="spacr.qt.tutorial"):
        assert d._resolve_target(
            engine.Step("narration text", target=(Broken(), None))) is None
    assert any("target did not resolve" in r.message for r in caplog.records)
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_resolve_highlight_rect_matches_the_widget_geometry(
        geo_window, tmp_path, monkeypatch):
    from spacr.qt.tutorial import engine
    win, btn = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (400, 300))
    d = make_director(win, [], tmp_path / "out")

    rect = d._resolve_highlight_rect(engine.Step("s", highlight=btn))
    top_left = btn.mapTo(win, btn.rect().topLeft())
    assert rect == (top_left.x(), top_left.y(), btn.width(), btn.height())
    assert btn.width() == 80
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_resolve_highlight_rect_scales_with_video_size(geo_window,
                                                         tmp_path,
                                                         monkeypatch):
    from spacr.qt.tutorial import engine
    win, btn = geo_window
    d = make_director(win, [], tmp_path / "out")
    monkeypatch.setattr(engine, "VIDEO_SIZE", (800, 600))
    x, y, w, h = d._resolve_highlight_rect(engine.Step("s", highlight=btn))
    assert (w, h) == (btn.width() * 2, btn.height() * 2)   # doubled
    top_left = btn.mapTo(win, btn.rect().topLeft())
    assert (x, y) == (top_left.x() * 2, top_left.y() * 2)
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_resolve_highlight_rect_handles_none_and_deferred(geo_window,
                                                            tmp_path):
    from spacr.qt.tutorial import engine
    win, btn = geo_window
    d = make_director(win, [], tmp_path / "out")
    assert d._resolve_highlight_rect(engine.Step("s")) is None
    assert d._resolve_highlight_rect(
        engine.Step("s", highlight=lambda: None)) is None
    assert (d._resolve_highlight_rect(engine.Step("s", highlight=lambda: btn))
             == d._resolve_highlight_rect(engine.Step("s", highlight=btn)))
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_resolve_highlight_rect_warns_on_a_broken_widget(geo_window,
                                                           tmp_path, caplog):
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    d = make_director(win, [], tmp_path / "out")

    class Broken:
        def mapToGlobal(self, _pt):
            raise RuntimeError("gone")

    with caplog.at_level("WARNING", logger="spacr.qt.tutorial"):
        assert d._resolve_highlight_rect(
            engine.Step("n", highlight=Broken())) is None
    assert any("highlight did not resolve" in r.message
                 for r in caplog.records)
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_deref_passes_widgets_through_and_calls_callables(geo_window,
                                                            tmp_path):
    from spacr.qt.tutorial.engine import Director
    win, btn = geo_window
    assert Director._deref(btn) is btn
    assert Director._deref(None) is None
    assert Director._deref(lambda: btn) is btn
    # A non-widget, non-callable is handed back untouched rather than
    # swallowed, so the caller's except-branch can report it.
    sentinel = object()
    assert Director._deref(sentinel) is sentinel


# ---------------------------------------------------------------------------
# Director: cursor animation
# ---------------------------------------------------------------------------

def test_animate_cursor_eases_from_rest_to_the_target(geo_window,
                                                        tmp_path):
    from spacr.qt.tutorial.engine import Recorder
    win, _ = geo_window
    d = make_director(win, [], tmp_path / "out")
    inner = Recorder(win, d._workdir / "frames", size=(400, 300))
    inner.cursor_pos = (0.0, 0.0)
    spy = SpyRecorder(inner)
    d._recorder = spy

    d._animate_cursor((100.0, 200.0), frames=4, highlight_rect=None)

    assert len(spy.positions) == 4
    assert inner.frame_idx == 4
    # Cosine ease-in-out, sampled at t = 1/4 … 4/4.
    for i, (px, py) in enumerate(spy.positions):
        eased = 0.5 * (1 - math.cos(math.pi * (i + 1) / 4))
        assert px == pytest.approx(100.0 * eased)
        assert py == pytest.approx(200.0 * eased)
    # Lands exactly on target, and is *not* a linear ramp.
    assert spy.positions[-1] == pytest.approx((100.0, 200.0))
    assert spy.positions[0][0] < 100.0 / 4      # slow start
    assert spy.positions[1] == pytest.approx((50.0, 100.0))  # halfway at t=.5
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_animate_cursor_starts_from_wherever_the_cursor_rests(geo_window,
                                                                tmp_path):
    from spacr.qt.tutorial.engine import Recorder
    win, _ = geo_window
    d = make_director(win, [], tmp_path / "out")
    inner = Recorder(win, d._workdir / "frames", size=(400, 300))
    inner.cursor_pos = (300.0, 100.0)
    spy = SpyRecorder(inner)
    d._recorder = spy

    d._animate_cursor((100.0, 200.0), frames=4, highlight_rect=(1, 2, 3, 4))

    assert spy.positions[-1] == pytest.approx((100.0, 200.0))
    assert spy.positions[0][0] < 300.0 and spy.positions[0][0] > 100.0
    assert spy.highlights == [(1, 2, 3, 4)] * 4
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_animate_cursor_with_zero_frames_is_a_no_op(geo_window, tmp_path):
    """A step whose whole budget is under 3 frames gets frames=0 from
    budget//3; that must not divide by zero or emit frames."""
    from spacr.qt.tutorial.engine import Recorder
    win, _ = geo_window
    d = make_director(win, [], tmp_path / "out")
    rec = Recorder(win, d._workdir / "frames", size=(400, 300))
    d._recorder = rec
    d._animate_cursor((10.0, 10.0), frames=0, highlight_rect=None)
    assert rec.frame_idx == 0
    shutil.rmtree(d._workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Director: the capture loop
# ---------------------------------------------------------------------------

def test_run_capture_spends_exactly_the_frame_budget_per_step(
        geo_window, tmp_path, monkeypatch):
    from spacr.qt.tutorial import engine
    win, btn = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (320, 240))

    fired: list[int] = []
    holder: dict = {}

    def action():
        fired.append(holder["d"]._recorder.frame_idx)

    steps = [
        engine.Step("targeted step", action=action, target=(btn, None),
                     highlight=btn, hold_ms=0),
        engine.Step("untargeted step", hold_ms=0),
    ]
    d = make_director(win, steps, tmp_path / "out",
                       narrator=FakeNarrator(seconds=2.0), fps=6)
    holder["d"] = d
    d._prerender_audio()
    budget = d._frames_for(0)                    # ceil(2.0*6) = 12
    assert budget == 12

    d._run_capture()

    move = min(engine.CURSOR_MOVE_FRAMES, budget // 3)   # 4
    assert move == 4
    # Total frames == sum of budgets: the animation frames are *inside*
    # the budget, not extra.
    assert d._recorder.frame_idx == budget * 2
    # The action fires after the cursor has finished travelling.
    assert fired == [move]
    frames = sorted((d._workdir / "frames").glob("frame_*.png"))
    assert len(frames) == budget * 2
    assert frames[0].name == "frame_000000.png"
    assert frames[-1].name == f"frame_{budget * 2 - 1:06d}.png"
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_run_capture_starts_the_cursor_out_of_the_way(geo_window,
                                                        tmp_path,
                                                        monkeypatch):
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (320, 240))
    d = make_director(win, [engine.Step("no target", hold_ms=0)],
                       tmp_path / "out",
                       narrator=FakeNarrator(seconds=0.2), fps=5)
    d._prerender_audio()
    d._run_capture()
    # Bottom-right corner, 40 px in — never over the UI it is narrating.
    assert d._recorder.cursor_pos == (320 - 40, 240 - 40)
    assert win.width() == 320 and win.height() == 240
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_run_capture_survives_a_step_action_that_raises(geo_window,
                                                          tmp_path,
                                                          monkeypatch,
                                                          caplog):
    """One broken action must not abandon the rest of the render."""
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (320, 240))

    def boom():
        raise ValueError("script bug")

    later = []
    steps = [engine.Step("bad", action=boom, hold_ms=0),
             engine.Step("good", action=lambda: later.append(1), hold_ms=0)]
    d = make_director(win, steps, tmp_path / "out",
                       narrator=FakeNarrator(seconds=0.2), fps=5)
    d._prerender_audio()
    with caplog.at_level("ERROR", logger="spacr.qt.tutorial"):
        d._run_capture()

    assert later == [1], "the next step still ran"
    assert d._recorder.frame_idx == d._frames_for(0) + d._frames_for(1)
    assert any("step action failed" in r.message for r in caplog.records)
    shutil.rmtree(d._workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Director: audio concat + mux
# ---------------------------------------------------------------------------

def test_concat_audio_list_interleaves_silence_only_for_real_holds(
        geo_window, tmp_path, monkeypatch):
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    steps = [engine.Step("a", hold_ms=500), engine.Step("b", hold_ms=0),
             engine.Step("c", hold_ms=250)]
    d = make_director(win, steps, tmp_path / "out")
    d._prerender_audio()

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(engine.subprocess, "run", fake_run)
    out = d._concat_audio()

    assert out == d._workdir / "audio.wav"
    lines = (d._workdir / "audio_concat.txt").read_text().splitlines()
    assert [Path(l.split("'")[1]).name for l in lines] == [
        "step_000.wav", "silence_000.wav",
        "step_001.wav",                       # hold_ms=0 -> no silence
        "step_002.wav", "silence_002.wav",
    ]
    # Two silences generated at the right lengths, then one concat call.
    silence_cmds = [c for c in calls if "anullsrc=r=22050:cl=mono" in c]
    assert len(silence_cmds) == 2
    assert silence_cmds[0][silence_cmds[0].index("-t") + 1] == "0.5"
    assert silence_cmds[1][silence_cmds[1].index("-t") + 1] == "0.25"
    assert all(c[c.index("-c:a") + 1] == "pcm_s16le" for c in silence_cmds)
    concat_cmd = calls[-1]
    assert concat_cmd[:2] == ["ffmpeg", "-y"]
    assert "concat" in concat_cmd and "-safe" in concat_cmd
    assert concat_cmd[-1] == str(out)
    shutil.rmtree(d._workdir, ignore_errors=True)


@needs_ffmpeg
def test_concat_audio_really_produces_one_wav_of_the_right_length(
        geo_window, tmp_path):
    from spacr.qt.tutorial.engine import Step, _wav_duration
    win, _ = geo_window
    steps = [Step("a", hold_ms=500), Step("b", hold_ms=0)]
    d = make_director(win, steps, tmp_path / "out",
                       narrator=FakeNarrator(seconds=0.4))
    d._prerender_audio()
    out = d._concat_audio()
    assert out.exists()
    # 0.4 + 0.5 silence + 0.4
    assert _wav_duration(out) == pytest.approx(1.3, abs=0.05)
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_mux_video_builds_a_faststart_h264_command(geo_window, tmp_path,
                                                     monkeypatch):
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    d = make_director(win, [], tmp_path / "out", fps=24)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(engine.subprocess, "run", fake_run)
    mp4 = d._mux_video(tmp_path / "a.wav", "mask")

    assert mp4 == tmp_path / "out" / "mask.mp4"
    cmd = seen["cmd"]
    assert cmd[cmd.index("-framerate") + 1] == "24"
    assert cmd[cmd.index("-i") + 1] == str(
        d._workdir / "frames" / "frame_%06d.png")
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert "-shortest" in cmd
    assert cmd[-1] == str(mp4)
    shutil.rmtree(d._workdir, ignore_errors=True)


def test_mux_video_reports_ffmpeg_stderr(geo_window, tmp_path, monkeypatch):
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    d = make_director(win, [], tmp_path / "out")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 1, b"", b"Output file #0 does not contain any stream")

    monkeypatch.setattr(engine.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc:
        d._mux_video(tmp_path / "a.wav", "mask")
    assert "does not contain any stream" in str(exc.value)
    assert "ffmpeg mux failed" in str(exc.value)
    shutil.rmtree(d._workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Director.render — the whole pipeline
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_render_produces_a_playable_mp4_and_matching_srt(geo_window,
                                                           tmp_path,
                                                           monkeypatch):
    from spacr.qt.tutorial import engine
    win, btn = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (320, 240))

    steps = [engine.Step("first thing", target=(btn, None), highlight=btn,
                          hold_ms=0),
             engine.Step("second thing", hold_ms=200)]
    d = make_director(win, steps, tmp_path / "out",
                       narrator=FakeNarrator(seconds=0.5), fps=6)
    workdir = d._workdir

    result = d.render("mytutorial")

    assert result.mp4 == tmp_path / "out" / "mytutorial.mp4"
    assert result.srt == tmp_path / "out" / "mytutorial.srt"
    assert result.mp4.exists() and result.mp4.stat().st_size > 0
    assert result.duration_s == pytest.approx(0.5 + 0.5 + 0.2, abs=1e-3)
    # frames = ceil(0.5*6) + ceil(0.7*6) = 3 + 5
    assert result.frames == 3 + 5
    assert result.srt.read_text().count("-->") == 2
    assert "first thing" in result.srt.read_text()
    # Scratch dir is cleaned up on success.
    assert not workdir.exists()

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_name,width,height", "-of", "csv=p=0",
         str(result.mp4)],
        capture_output=True)
    if probe.returncode == 0:
        text = probe.stdout.decode()
        assert "h264" in text
        assert "320,240" in text.replace(" ", "")


def test_render_calls_the_pipeline_stages_in_order(geo_window, tmp_path,
                                                     monkeypatch):
    """render() = narrate -> capture -> concat -> mux -> srt. Getting the
    order wrong (muxing before capture, say) would silently ship an empty
    video, so the sequence is pinned."""
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (160, 120))
    steps = [engine.Step("only", hold_ms=0)]
    d = make_director(win, steps, tmp_path / "out",
                       narrator=FakeNarrator(seconds=0.2), fps=5)

    order = []
    for name in ("_prerender_audio", "_run_capture", "_concat_audio",
                  "_mux_video", "_write_srt"):
        real = getattr(d, name)

        def wrapper(*a, _n=name, _r=real, **kw):
            order.append(_n)
            return _r(*a, **kw)

        monkeypatch.setattr(d, name, wrapper)
    monkeypatch.setattr(engine.subprocess, "run",
                          lambda cmd, **kw: subprocess.CompletedProcess(
                              cmd, 0, b"", b""))

    result = d.render("ordered")
    assert order == ["_prerender_audio", "_run_capture", "_concat_audio",
                     "_mux_video", "_write_srt"]
    assert result.frames == 1
    assert isinstance(result, engine.RenderResult)


def test_render_result_is_a_plain_dataclass():
    from dataclasses import fields
    from spacr.qt.tutorial.engine import RenderResult
    names = [f.name for f in fields(RenderResult)]
    assert names == ["mp4", "srt", "frames", "duration_s"]
    r = RenderResult(mp4=Path("a.mp4"), srt=Path("a.srt"), frames=7,
                      duration_s=1.25)
    assert (r.frames, r.duration_s) == (7, 1.25)


def test_step_defaults_and_module_constants():
    from spacr.qt.tutorial import engine
    s = engine.Step("hello world")
    assert (s.narration, s.action, s.target, s.highlight) == (
        "hello world", None, None, None)
    assert s.hold_ms == engine.DEFAULT_HOLD_MS == 500
    assert engine.FRAME_RATE == 30
    assert engine.VIDEO_SIZE == (1920, 1080)
    assert engine.CURSOR_MOVE_FRAMES == 12
    assert engine.DEFAULT_VOICE.name.endswith(".onnx")


def test_render_survives_an_undeletable_scratch_dir(geo_window, tmp_path,
                                                      monkeypatch, caplog):
    """A render that produced its outputs must still return them when the
    temp dir refuses to go away — but it has to say so, or frames pile up
    in /tmp unnoticed."""
    import os
    from spacr.qt.tutorial import engine
    win, _ = geo_window
    monkeypatch.setattr(engine, "VIDEO_SIZE", (160, 120))
    monkeypatch.setattr(engine.subprocess, "run",
                          lambda cmd, **kw: subprocess.CompletedProcess(
                              cmd, 0, b"", b""))

    d = make_director(win, [engine.Step("only step", hold_ms=0)],
                       tmp_path / "out",
                       narrator=FakeNarrator(seconds=0.2), fps=5)
    workdir = d._workdir
    real_write_srt = d._write_srt

    def write_srt_then_lock(name):
        out = real_write_srt(name)
        # r-x: contents can be listed but not unlinked.
        os.chmod(workdir, 0o500)
        return out

    monkeypatch.setattr(d, "_write_srt", write_srt_then_lock)

    try:
        with caplog.at_level("WARNING", logger="spacr.qt.tutorial"):
            result = d.render("locked")
        assert result.frames == 1
        assert result.srt.exists()
        assert workdir.exists(), "the scratch dir really did survive"
        assert any("could not remove tutorial scratch dir" in r.message
                     for r in caplog.records)
    finally:
        os.chmod(workdir, 0o700)
        shutil.rmtree(workdir, ignore_errors=True)
