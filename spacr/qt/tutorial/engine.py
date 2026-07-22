"""Tutorial rendering engine — narration, cursor overlay, capture, mux.

The pipeline is intentionally linear and easy to reason about:

  1. Synthesize each Step's narration through Piper → WAV, know duration
  2. Spin up the MainWindow (on real DISPLAY or under Xvfb) at 1920x1080
  3. Walk through the Steps, capturing frames at 30 fps into a scratch dir.
     Each step gets ceil(narration_s * 30) + hold frames budget.
     A synthesized cursor (little arrow drawn onto each frame) animates
     to each step's target widget before the step's action fires.
  4. Concatenate all step WAVs → one audio track
  5. `ffmpeg -framerate 30 -i frames/%06d.png -i audio.wav ... out.mp4`
  6. Emit matching .srt sidecar

The engine has no Qt-specific business logic — that lives in per-app
`scripts.py` functions that return a list of Steps.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

LOG = logging.getLogger("spacr.qt.tutorial")

FRAME_RATE = 30
VIDEO_SIZE = (1920, 1080)
CURSOR_MOVE_FRAMES = 12
DEFAULT_HOLD_MS = 500
DEFAULT_VOICE = (
    Path.home() / ".spacr" / "piper" / "en_US-lessac-medium.onnx"
)


# ---------------------------------------------------------------------------
# Step — the atomic unit of a tutorial
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """A single narrated beat of a tutorial.

    Fields:
        narration:    what the narrator says (also becomes the subtitle)
        action:       optional callable that mutates the UI. Runs AFTER
                      the cursor animation completes but BEFORE the
                      narration finishes playing.
        target:       optional (widget, point-in-widget) the cursor
                      animates to before the action fires. Point is
                      relative to `widget`. Pass a QWidget with point
                      omitted to target its center.
        hold_ms:      extra silence at the end of the step, in ms.
                      Useful to let a UI change settle before the next
                      step begins.
        highlight:    optional widget to draw a soft highlight ring
                      around while this step runs.
    """
    narration: str
    action: Optional[Callable[[], None]] = None
    target: Optional[Tuple[Any, Optional[Tuple[int, int]]]] = None
    hold_ms: int = DEFAULT_HOLD_MS
    highlight: Optional[Any] = None


# ---------------------------------------------------------------------------
# Narrator — Piper wrapper
# ---------------------------------------------------------------------------

class Narrator:
    """Synthesize step narration WAVs using Piper.

    Uses the Piper CLI (already installed via pip install piper-tts).
    Voice model defaults to ~/.spacr/piper/en_US-lessac-medium.onnx
    but any Piper .onnx can be passed via `voice_model=`.
    """

    def __init__(self, voice_model: Optional[Path] = None,
                  length_scale: float = 1.0):
        """Load the Piper voice model and set the narration speed.

        :param voice_model: path to a Piper ``.onnx`` model; defaults to
            ``~/.spacr/piper/en_US-lessac-medium.onnx``.
        :param length_scale: Piper length scale; lower = faster speech.
        :raises FileNotFoundError: when the voice model is missing.
        """
        self.voice_model = Path(voice_model or DEFAULT_VOICE)
        if not self.voice_model.exists():
            raise FileNotFoundError(
                f"Piper voice model not found at {self.voice_model}. "
                "Download one via: "
                "`curl -sL https://huggingface.co/rhasspy/piper-voices/"
                "resolve/main/en/en_US/lessac/medium/"
                "en_US-lessac-medium.onnx -o ~/.spacr/piper/"
                "en_US-lessac-medium.onnx`"
            )
        self.length_scale = length_scale

    def synth(self, text: str, out_wav: Path) -> float:
        """Synthesize `text` into `out_wav`. Returns duration in seconds."""
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["piper", "-m", str(self.voice_model),
             "--length-scale", str(self.length_scale),
             "-f", str(out_wav)],
            input=text.encode(),
            capture_output=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Piper failed: {proc.stderr.decode(errors='replace')}"
            )
        return _wav_duration(out_wav)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ---------------------------------------------------------------------------
# Cursor overlay
# ---------------------------------------------------------------------------

def _draw_cursor_on(pixmap, pos_xy: Tuple[int, int]) -> None:
    """Paint an arrow cursor onto `pixmap` at absolute (x, y)."""
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
    x, y = pos_xy
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    # Shadow — offset arrow, semi-transparent black
    shadow_offset = 2
    shadow = QPolygonF([
        QPointF(x + shadow_offset, y + shadow_offset),
        QPointF(x + 18 + shadow_offset, y + 12 + shadow_offset),
        QPointF(x + 11 + shadow_offset, y + 14 + shadow_offset),
        QPointF(x + 15 + shadow_offset, y + 22 + shadow_offset),
        QPointF(x + 11 + shadow_offset, y + 23 + shadow_offset),
        QPointF(x +  7 + shadow_offset, y + 15 + shadow_offset),
        QPointF(x +  0 + shadow_offset, y + 20 + shadow_offset),
    ])
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(0, 0, 0, 120)))
    painter.drawPolygon(shadow)
    # White fill + black outline arrow
    arrow = QPolygonF([
        QPointF(x, y),
        QPointF(x + 18, y + 12),
        QPointF(x + 11, y + 14),
        QPointF(x + 15, y + 22),
        QPointF(x + 11, y + 23),
        QPointF(x +  7, y + 15),
        QPointF(x +  0, y + 20),
    ])
    painter.setPen(QPen(QColor(20, 20, 20), 1.5))
    painter.setBrush(QBrush(QColor(255, 255, 255)))
    painter.drawPolygon(arrow)
    painter.end()


def _draw_highlight_on(pixmap, rect_xywh: Tuple[int, int, int, int]) -> None:
    """Paint a soft accent ring around `rect_xywh` (absolute coords)."""
    from PySide6.QtCore import Qt, QRect
    from PySide6.QtGui import QPainter, QPen, QColor
    x, y, w, h = rect_xywh
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(74, 158, 255, 220), 4)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(QRect(x - 4, y - 4, w + 8, h + 8), 6, 6)
    painter.end()


# ---------------------------------------------------------------------------
# Recorder — captures the MainWindow at FRAME_RATE
# ---------------------------------------------------------------------------

class Recorder:
    """Grab the MainWindow's rendered pixmap N times a second,
    compositing a synthetic cursor onto each frame.

    :param window: source Qt window to grab.
    :param frames_dir: destination folder for numbered PNG frames.
    :param fps: capture frame rate.
    :param size: fixed output frame size ``(width, height)`` in px.
    """

    def __init__(self, window, frames_dir: Path,
                  fps: int = FRAME_RATE,
                  size: Tuple[int, int] = VIDEO_SIZE):
        self.window = window
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.size = size
        self.frame_idx = 0
        self.cursor_pos: Tuple[float, float] = (
            size[0] / 2, size[1] / 2
        )

    def snap(self, cursor_pos: Optional[Tuple[float, float]] = None,
              highlight_rect: Optional[Tuple[int, int, int, int]] = None
              ) -> Path:
        """Grab one frame, save as PNG, return its path."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap
        # Ensure the window has painted anything queued
        self.window.repaint()
        pm = self.window.grab().scaled(
            self.size[0], self.size[1],
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        # If the grab is smaller than the target frame (e.g. window
        # smaller than VIDEO_SIZE), centre it on a black canvas
        if pm.size().width() != self.size[0] or pm.size().height() != self.size[1]:
            canvas = QPixmap(self.size[0], self.size[1])
            canvas.fill(Qt.black)
            from PySide6.QtGui import QPainter
            painter = QPainter(canvas)
            offset_x = (self.size[0] - pm.width()) // 2
            offset_y = (self.size[1] - pm.height()) // 2
            painter.drawPixmap(offset_x, offset_y, pm)
            painter.end()
            pm = canvas
        if highlight_rect:
            _draw_highlight_on(pm, highlight_rect)
        if cursor_pos is None:
            cursor_pos = self.cursor_pos
        else:
            self.cursor_pos = cursor_pos
        _draw_cursor_on(pm, (int(cursor_pos[0]), int(cursor_pos[1])))
        path = self.frames_dir / f"frame_{self.frame_idx:06d}.png"
        pm.save(str(path), "PNG")
        self.frame_idx += 1
        return path


# ---------------------------------------------------------------------------
# Director — orchestrates the whole render
# ---------------------------------------------------------------------------

@dataclass
class RenderResult:
    """Output paths and metadata for a completed tutorial render.

    :ivar mp4: absolute path to the produced MP4.
    :ivar srt: absolute path to the produced SRT sidecar.
    :ivar frames: total number of frames captured.
    :ivar duration_s: total narration duration in seconds.
    """

    mp4: Path
    srt: Path
    frames: int
    duration_s: float


class Director:
    """Orchestrates narration, capture, and mux into a final MP4 + SRT.

    :param window: live MainWindow the tutorial drives.
    :param steps: ordered list of :class:`Step` beats.
    :param out_dir: destination folder for the rendered mp4/srt.
    :param narrator: optional :class:`Narrator`; a default is built if omitted.
    :param fps: capture frame rate.
    """

    def __init__(self, window, steps: List[Step],
                  out_dir: Path,
                  narrator: Optional[Narrator] = None,
                  fps: int = FRAME_RATE):
        self.window = window
        self.steps = steps
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.narrator = narrator or Narrator()
        self.fps = fps
        self._workdir = Path(tempfile.mkdtemp(prefix="spacr-tut-"))
        self._audio_wavs: List[Tuple[float, Path, str]] = []
        self._recorder: Optional[Recorder] = None

    # -- narration pre-render ------------------------------------------------
    def _prerender_audio(self) -> float:
        """Synth all step narrations up front so we know their durations
        before we start capturing frames. Returns total audio duration."""
        total = 0.0
        for i, step in enumerate(self.steps):
            wav_path = self._workdir / f"step_{i:03d}.wav"
            dur = self.narrator.synth(step.narration, wav_path)
            self._audio_wavs.append((dur, wav_path, step.narration))
            total += dur + step.hold_ms / 1000.0
        LOG.info("prerendered %d narration clips, %.1fs total",
                  len(self._audio_wavs), total)
        return total

    # -- step frame budgets --------------------------------------------------
    def _frames_for(self, step_idx: int) -> int:
        dur, _, _ = self._audio_wavs[step_idx]
        hold = self.steps[step_idx].hold_ms / 1000.0
        return max(1, math.ceil((dur + hold) * self.fps))

    # -- capture loop --------------------------------------------------------
    def _run_capture(self) -> None:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()

        # Force window to VIDEO_SIZE for consistent frames
        self.window.resize(VIDEO_SIZE[0], VIDEO_SIZE[1])
        self.window.show()
        for _ in range(6):
            app.processEvents()

        self._recorder = Recorder(self.window, self._workdir / "frames",
                                    fps=self.fps, size=VIDEO_SIZE)

        # Start cursor at bottom-right (out of the way)
        self._recorder.cursor_pos = (
            VIDEO_SIZE[0] - 40, VIDEO_SIZE[1] - 40
        )

        for i, step in enumerate(self.steps):
            LOG.info("step %d/%d: %s", i + 1, len(self.steps),
                      step.narration[:60])
            budget = self._frames_for(i)

            # Cursor animation (if target set)
            target_pos = self._resolve_target(step)
            highlight_rect = self._resolve_highlight_rect(step)

            if target_pos is not None:
                self._animate_cursor(target_pos,
                                       frames=min(CURSOR_MOVE_FRAMES, budget // 3),
                                       highlight_rect=highlight_rect)

            # Fire action (if any)
            if step.action is not None:
                try:
                    step.action()
                except Exception as e:
                    LOG.exception("step action failed: %s", e)

            # Fill remaining frames, letting the UI catch up between grabs
            for _ in range(budget - (
                    min(CURSOR_MOVE_FRAMES, budget // 3)
                    if target_pos is not None else 0)):
                app.processEvents()
                self._recorder.snap(highlight_rect=highlight_rect)

        LOG.info("captured %d frames", self._recorder.frame_idx)

    def _resolve_target(self, step: Step) -> Optional[Tuple[float, float]]:
        if step.target is None:
            return None
        widget, offset = (step.target if isinstance(step.target, tuple)
                            else (step.target, None))
        if widget is None:
            return None
        try:
            from PySide6.QtCore import QPoint
            if offset is None:
                center = widget.rect().center()
                global_pt = widget.mapToGlobal(center)
            else:
                global_pt = widget.mapToGlobal(QPoint(*offset))
            win_pt = self.window.mapFromGlobal(global_pt)
            # Scale window coords to VIDEO_SIZE (they should already match
            # since we resized, but be defensive)
            wsize = self.window.size()
            sx = VIDEO_SIZE[0] / max(1, wsize.width())
            sy = VIDEO_SIZE[1] / max(1, wsize.height())
            return (win_pt.x() * sx, win_pt.y() * sy)
        except Exception:
            return None

    def _resolve_highlight_rect(self, step: Step
                                  ) -> Optional[Tuple[int, int, int, int]]:
        if step.highlight is None:
            return None
        try:
            widget = step.highlight
            from PySide6.QtCore import QPoint
            top_left = self.window.mapFromGlobal(
                widget.mapToGlobal(QPoint(0, 0))
            )
            wsize = self.window.size()
            sx = VIDEO_SIZE[0] / max(1, wsize.width())
            sy = VIDEO_SIZE[1] / max(1, wsize.height())
            return (int(top_left.x() * sx), int(top_left.y() * sy),
                     int(widget.width() * sx),
                     int(widget.height() * sy))
        except Exception:
            return None

    def _animate_cursor(self, target: Tuple[float, float], frames: int,
                          highlight_rect: Optional[Tuple[int, int, int, int]]
                         ) -> None:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        start = self._recorder.cursor_pos
        for i in range(frames):
            t = (i + 1) / frames
            # Smooth ease-in-out
            eased = 0.5 * (1 - math.cos(math.pi * t))
            pos = (
                start[0] + (target[0] - start[0]) * eased,
                start[1] + (target[1] - start[1]) * eased,
            )
            app.processEvents()
            self._recorder.snap(cursor_pos=pos,
                                  highlight_rect=highlight_rect)

    # -- audio + video mux ---------------------------------------------------
    def _concat_audio(self) -> Path:
        """Concatenate all step WAVs (with hold-ms silences) into one WAV."""
        concat_list = self._workdir / "audio_concat.txt"
        # Use ffmpeg's concat demuxer. Build a list file with a silent
        # WAV inserted after each step for hold_ms.
        lines: List[str] = []
        for i, (dur, wav, _text) in enumerate(self._audio_wavs):
            lines.append(f"file '{wav.absolute()}'")
            hold_ms = self.steps[i].hold_ms
            if hold_ms > 0:
                silence = self._workdir / f"silence_{i:03d}.wav"
                self._make_silence(silence, hold_ms / 1000.0)
                lines.append(f"file '{silence.absolute()}'")
        concat_list.write_text("\n".join(lines))
        audio_out = self._workdir / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list),
             "-c", "copy", str(audio_out)],
            capture_output=True, check=True,
        )
        return audio_out

    def _make_silence(self, path: Path, seconds: float) -> None:
        # Match Piper's sample rate (22050) + mono + 16-bit
        subprocess.run(
            ["ffmpeg", "-y",
             "-f", "lavfi", "-i",
             f"anullsrc=r=22050:cl=mono",
             "-t", f"{seconds}",
             "-c:a", "pcm_s16le",
             str(path)],
            capture_output=True, check=True,
        )

    def _mux_video(self, audio: Path, name: str) -> Path:
        frames_dir = self._workdir / "frames"
        mp4 = self.out_dir / f"{name}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-i", str(audio),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(mp4),
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg mux failed: "
                + proc.stderr.decode(errors="replace")[-2000:]
            )
        return mp4

    # -- SRT sidecar ---------------------------------------------------------
    def _write_srt(self, name: str) -> Path:
        srt = self.out_dir / f"{name}.srt"
        with open(srt, "w") as f:
            t = 0.0
            for i, (dur, _wav, text) in enumerate(self._audio_wavs):
                start = t
                end = t + dur
                f.write(f"{i + 1}\n")
                f.write(f"{_srt_ts(start)} --> {_srt_ts(end)}\n")
                f.write(f"{text}\n\n")
                t = end + self.steps[i].hold_ms / 1000.0
        return srt

    # -- main entry point ----------------------------------------------------
    def render(self, name: str) -> RenderResult:
        """Run the full narrate → capture → mux pipeline for this director.

        :param name: base filename for the produced ``<name>.mp4`` and ``.srt``.
        :returns: :class:`RenderResult` with paths and duration metadata.
        """
        total_audio = self._prerender_audio()
        self._run_capture()
        audio = self._concat_audio()
        mp4 = self._mux_video(audio, name)
        srt = self._write_srt(name)
        result = RenderResult(
            mp4=mp4, srt=srt,
            frames=self._recorder.frame_idx,
            duration_s=total_audio,
        )
        # Cleanup scratch dir
        try:
            shutil.rmtree(self._workdir)
        except Exception:
            pass
        return result


def _srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def render_tutorial(app_key: str, out_dir: Optional[Path] = None,
                     voice_model: Optional[Path] = None,
                     length_scale: float = 1.0) -> RenderResult:
    """Boot MainWindow, run the per-app tutorial script, render MP4.

    Returns a RenderResult describing where the MP4 and SRT ended up.
    """
    from PySide6.QtWidgets import QApplication
    from ..app import MainWindow
    from .scripts import build_steps

    out_dir = Path(out_dir or Path.home() / "spacr-tutorials")
    out_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    for _ in range(10):
        app.processEvents()

    steps = build_steps(app_key, window)
    narrator = Narrator(voice_model=voice_model, length_scale=length_scale)
    director = Director(window, steps, out_dir=out_dir, narrator=narrator)
    result = director.render(name=app_key)
    window.close()
    return result
