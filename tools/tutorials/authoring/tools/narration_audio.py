#!/usr/bin/env python3
"""Pure helpers for natural, reproducible tutorial narration assembly.

The renderer intentionally leaves synthesized speech samples untouched.  It
only removes redundant silence at sentence edges, retains a conservative
amount of room around audible speech, and inserts explicit pauses.  Keeping
that work here makes it possible to test cadence and clipping protection
without loading Kokoro or rendering production media.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


SAMPLE_RATE = 24_000
DEFAULT_SPEED = 1.00

# Kokoro voices have noticeably different native rates.  These values were
# selected from active/voiced WPM measurements; ``--speed`` remains an
# explicit global override in both renderers.
ENGLISH_VOICE_SPEEDS = {
    "af_aoede": 0.96,
    "af_bella": 1.03,
    "af_heart": 1.00,
    "af_jessica": 0.88,
    "af_river": 0.88,
    "af_sarah": 0.97,
    "af_sky": 0.96,
    "am_adam": 0.95,
    "am_echo": 0.94,
    "am_eric": 0.88,
    "am_fenrir": 0.99,
    "am_liam": 0.88,
    "am_michael": 1.03,
    "am_onyx": 0.98,
    "am_puck": 1.00,
    "am_santa": 1.03,
    "bf_alice": 0.97,
    "bf_emma": 0.92,
    "bf_isabella": 0.93,
    "bf_lily": 0.95,
    "bm_daniel": 0.92,
    "bm_fable": 0.98,
    "bm_george": 1.03,
    "bm_lewis": 1.03,
}

# Retained Kokoro edge room plus the explicit pause gives an effective
# sentence gap of approximately 0.40 seconds:
# 0.18 second tail + 0.12 second pause + 0.10 second lead.
LEAD_SILENCE_SECONDS = 0.100
TAIL_SILENCE_SECONDS = 0.180
SENTENCE_PAUSE_SECONDS = 0.120
SCENE_HOLD_MIN_SECONDS = 0.320
SCENE_HOLD_MAX_SECONDS = 0.520
DEFAULT_SCENE_HOLD_SECONDS = 0.420
VAD_FRAME_SECONDS = 0.005
VAD_PEAK_RATIO = 0.002
VAD_ABSOLUTE_FLOOR = 0.00002
ASSEMBLY_VERSION = "2026-08-11-natural-sentences-v1"
CADENCE_PROFILE_PATH = Path(__file__).with_name("english_scene_cadence.json")

_CLOSING_PUNCTUATION = (
    "\"'\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK})]}"
)
_ABBREVIATIONS = {
    "dr.",
    "fig.",
    "i.e.",
    "jr.",
    "mr.",
    "mrs.",
    "ms.",
    "no.",
    "prof.",
    "sr.",
    "st.",
    "vs.",
    "e.g.",
}


@dataclass(frozen=True)
class FrozenSourceSnapshot:
    """Exact source bytes used by a renderer and their disk identity."""

    path: Path
    role: str
    sha256: str
    size: int
    mtime_ns: int
    content: bytes = field(repr=False, compare=False)

    def fingerprint(self, version: str | None = None) -> dict[str, Any]:
        """Return deterministic fields suitable for render metadata."""
        return {
            "role": self.role,
            "version": version or "undeclared",
            "sha256": self.sha256,
            "bytes": self.size,
        }

    def assert_current(self) -> None:
        """Abort if disk no longer matches the source loaded in memory."""
        try:
            current = freeze_source_snapshot(self.path, self.role)
        except OSError as error:
            raise RuntimeError(
                f"Frozen {self.role} source is no longer readable: {self.path}"
            ) from error
        if current.sha256 != self.sha256 or current.size != self.size:
            raise RuntimeError(
                f"Frozen {self.role} source changed during rendering: "
                f"{self.path}. Restart the renderer before continuing."
            )


def freeze_source_snapshot(path: Path, role: str) -> FrozenSourceSnapshot:
    """Read, hash, and stat one source snapshot from a single open file."""
    with path.open("rb") as handle:
        content = handle.read()
        stat = os.fstat(handle.fileno())
    return FrozenSourceSnapshot(
        path=path.resolve(),
        role=role,
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        mtime_ns=stat.st_mtime_ns,
        content=content,
    )


def load_module_snapshot(
    path: Path, module_name: str, role: str
) -> tuple[ModuleType, FrozenSourceSnapshot]:
    """Execute exactly the Python bytes recorded by the source snapshot."""
    snapshot = freeze_source_snapshot(path, role)
    module = ModuleType(module_name)
    module.__file__ = str(snapshot.path)
    module.__package__ = ""
    exec(
        compile(snapshot.content, str(snapshot.path), "exec"), module.__dict__
    )
    return module, snapshot


CADENCE_PROFILE_SNAPSHOT = freeze_source_snapshot(
    CADENCE_PROFILE_PATH, "english_cadence_profile"
)


def load_cadence_profile(
    path: Path = CADENCE_PROFILE_PATH,
    content: bytes | None = None,
) -> dict[str, Any]:
    """Load and validate the reviewed English scene-speed profile."""
    source = path.read_bytes() if content is None else content
    profile = json.loads(source.decode("utf-8"))
    if profile.get("schema") != 1 or profile.get("language") != "en":
        raise ValueError(f"Unsupported narration cadence profile: {path}")
    if not profile.get("version") or not isinstance(
        profile.get("scenes"), dict
    ):
        raise ValueError(f"Incomplete narration cadence profile: {path}")
    bounds = profile.get("safe_multiplier_bounds", {})
    minimum = float(bounds.get("minimum", 0))
    maximum = float(bounds.get("maximum", 0))
    if not 0 < minimum <= 1 <= maximum:
        raise ValueError(f"Invalid cadence multiplier bounds: {bounds}")
    for key, entry in profile["scenes"].items():
        if ":" not in key or not isinstance(entry, dict):
            raise ValueError(f"Invalid cadence scene entry: {key!r}")
        multiplier = float(entry.get("multiplier", 0))
        if not minimum <= multiplier <= maximum:
            raise ValueError(
                f"Cadence multiplier {multiplier} leaves bounds for {key}"
            )
    return profile


ENGLISH_CADENCE_PROFILE = load_cadence_profile(
    CADENCE_PROFILE_PATH, CADENCE_PROFILE_SNAPSHOT.content
)


def cadence_profile_identity() -> dict[str, Any]:
    """Return profile version, exact content hash, and reviewed bounds."""
    return {
        "version": ENGLISH_CADENCE_PROFILE["version"],
        "sha256": CADENCE_PROFILE_SNAPSHOT.sha256,
        "bytes": CADENCE_PROFILE_SNAPSHOT.size,
        "enabled": bool(ENGLISH_CADENCE_PROFILE.get("enabled", True)),
        "targets": ENGLISH_CADENCE_PROFILE["targets"],
        "safe_multiplier_bounds": ENGLISH_CADENCE_PROFILE[
            "safe_multiplier_bounds"
        ],
    }


def assert_cadence_profile_current() -> None:
    """Fail before another track if the loaded profile changed on disk."""
    CADENCE_PROFILE_SNAPSHOT.assert_current()


def scene_speed_multiplier(
    language: str, lesson_id: str, scene_number: int
) -> float:
    """Return the reviewed scene cadence multiplier, or one when unlisted."""
    if language != "en" or not ENGLISH_CADENCE_PROFILE.get(
        "enabled", True
    ):
        return 1.0
    key = f"{lesson_id}:{int(scene_number)}"
    entry = ENGLISH_CADENCE_PROFILE["scenes"].get(key)
    return 1.0 if entry is None else float(entry["multiplier"])


def effective_scene_speed(
    base_speed: float,
    language: str,
    lesson_id: str,
    scene_number: int,
) -> tuple[float, float]:
    """Return ``(effective speed, scene multiplier)`` for synthesis."""
    multiplier = scene_speed_multiplier(language, lesson_id, scene_number)
    effective = float(base_speed) * multiplier
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError(f"Invalid effective narration speed {effective!r}")
    return effective, multiplier


def file_sha256(path: Path) -> str:
    """Hash a file without loading the complete media asset into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def frozen_asset_identity(path_text: str, role: str) -> dict[str, Any]:
    """Hash a model/tool asset once per renderer process."""
    path = Path(path_text).resolve()
    stat = path.stat()
    return {
        "role": role,
        "name": path.name,
        "bytes": stat.st_size,
        "sha256": file_sha256(path),
    }


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


@lru_cache(maxsize=None)
def voice_tensor_sha256(path_text: str) -> str:
    """Hash the exact float tensor Kokoro loads from a voice pack."""
    import torch

    tensor = torch.load(Path(path_text), weights_only=True, map_location="cpu")
    return hashlib.sha256(
        tensor.detach().contiguous().numpy().tobytes()
    ).hexdigest()


@lru_cache(maxsize=1)
def ffmpeg_runtime_identity() -> dict[str, Any]:
    """Freeze the exact ffmpeg executable and build used for mastering."""
    executable_text = shutil.which("ffmpeg")
    if not executable_text:
        raise RuntimeError("ffmpeg is required for tutorial mastering")
    executable = Path(executable_text).resolve()
    result = subprocess.run(
        [str(executable), "-version"],
        capture_output=True,
        text=True,
        check=True,
    )
    version_output = result.stdout.strip()
    return {
        "version": version_output.splitlines()[0],
        "version_output_sha256": hashlib.sha256(
            version_output.encode("utf-8")
        ).hexdigest(),
        "executable": frozen_asset_identity(
            str(executable), "ffmpeg_executable"
        ),
    }


def synthesis_runtime_identity(
    snapshot: Path, voice: str, device: str
) -> dict[str, Any]:
    """Identify every model, voice, library, and mastering runtime input."""
    snapshot = snapshot.resolve()
    voice_path = snapshot / "voices" / f"{voice}.pt"
    voice_identity = dict(
        frozen_asset_identity(str(voice_path), "kokoro_voice_pack")
    )
    voice_identity["tensor_sha256"] = voice_tensor_sha256(str(voice_path))
    return {
        "model": {
            "repository": "hexgrad/Kokoro-82M",
            "snapshot": snapshot.name,
            "config": frozen_asset_identity(
                str(snapshot / "config.json"), "kokoro_config"
            ),
            "weights": frozen_asset_identity(
                str(snapshot / "kokoro-v1_0.pth"), "kokoro_weights"
            ),
        },
        "voice_pack": voice_identity,
        "software": {
            "kokoro": _installed_version("kokoro"),
            "misaki": _installed_version("misaki"),
            "torch": _installed_version("torch"),
            "soundfile": _installed_version("soundfile"),
            "python": platform.python_version(),
            "implementation": sys.implementation.name,
        },
        "inference": {"device": device},
        "mastering_tool": ffmpeg_runtime_identity(),
    }


@contextmanager
def staged_output_path(target: Path):
    """Yield a same-directory temporary path and clean it after any failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=f"{target.suffix}.part",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a text sidecar only after its complete content reaches disk."""
    with staged_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def resolve_voice_speed(voice: str, override: float | None = None) -> float:
    """Return an explicit override or the measured per-voice default."""
    speed = (
        override
        if override is not None
        else ENGLISH_VOICE_SPEEDS.get(voice, DEFAULT_SPEED)
    )
    speed = float(speed)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError(f"Narration speed must be positive, got {speed!r}")
    return speed


def narration_dialect(language: str, lang_code: str, voice: str) -> str:
    """Resolve the English pronunciation profile used by a Kokoro voice."""
    if language == "en" and (
        lang_code == "b" or voice.startswith(("bf_", "bm_"))
    ):
        return "uk"
    return "us"


def _ends_abbreviation(prefix: str) -> bool:
    bare = prefix.rstrip().rstrip(_CLOSING_PUNCTUATION).lower()
    # Module comparison prose uses literal UI labels such as ``Model B.``.
    # Here the letter is the end of a complete label, not a person's initial.
    if re.search(r"\bmodel\s+[a-z]\.$", bare):
        return False
    token_match = re.search(r"(?:^|\s)([^\s]+)$", bare)
    token = token_match.group(1) if token_match else bare
    if token in _ABBREVIATIONS:
        return True
    # Initials and dotted initialisms ("A." and "U.S.") are not boundaries.
    return bool(re.fullmatch(r"(?:[a-z]\.)+", token))


def split_english_sentences(text: str) -> list[str]:
    """Split display prose at boundaries while retaining punctuation."""
    text = text.strip()
    if not text:
        return []
    sentences = []
    start = 0
    for whitespace in re.finditer(r"\s+", text):
        prefix = text[start : whitespace.start()].rstrip()
        if not prefix:
            continue
        terminal = prefix.rstrip(_CLOSING_PUNCTUATION)[-1:]
        if terminal not in ".!?" or _ends_abbreviation(prefix):
            continue
        following = text[whitespace.end() :].lstrip(
            _CLOSING_PUNCTUATION + "({["
        )
        starts_sentence = bool(following) and (
            following[0].isupper()
            or following[0].isdigit()
            # The product's official mixed-case spelling starts with a
            # lowercase letter even when it begins a sentence.
            or re.match(r"spaCR\b", following) is not None
        )
        if not starts_sentence:
            continue
        sentences.append(text[start : whitespace.start()].strip())
        start = whitespace.end()
    sentences.append(text[start:].strip())
    return [sentence for sentence in sentences if sentence]


def split_sentences(text: str, language: str) -> list[str]:
    """Use sentence synthesis for English and scene synthesis elsewhere."""
    stripped = text.strip()
    if not stripped:
        return []
    return (
        split_english_sentences(stripped) if language == "en" else [stripped]
    )


def clamp_scene_hold(authored: float | int | None) -> float:
    """Keep the effective between-scene gap in the 0.6--0.8 second range."""
    value = DEFAULT_SCENE_HOLD_SECONDS if authored is None else float(authored)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"Scene hold must be finite and non-negative, got {value!r}"
        )
    return min(SCENE_HOLD_MAX_SECONDS, max(SCENE_HOLD_MIN_SECONDS, value))


def trim_edge_silence(audio, np, sample_rate: int = SAMPLE_RATE):
    """Trim redundant edge silence without fading or editing active samples.

    The threshold is deliberately low and frame activity is peak-based so
    quiet fricatives are retained.  The returned audio is a contiguous slice
    of the input: no crossfade, resampling, or word-duration surgery occurs.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono narration, got shape {samples.shape}")
    if not len(samples):
        return samples.copy(), {
            "source_samples": 0,
            "trim_start_sample": 0,
            "trim_end_sample": 0,
            "first_active_sample": None,
            "last_active_sample": None,
            "kept_lead_samples": 0,
            "kept_tail_samples": 0,
        }

    peak = float(np.max(np.abs(samples)))
    threshold = max(VAD_ABSOLUTE_FLOOR, peak * VAD_PEAK_RATIO)
    frame_size = max(1, int(round(VAD_FRAME_SECONDS * sample_rate)))
    active_frames = []
    for frame_start in range(0, len(samples), frame_size):
        frame_end = min(len(samples), frame_start + frame_size)
        if float(np.max(np.abs(samples[frame_start:frame_end]))) >= threshold:
            active_frames.append((frame_start, frame_end))

    if not active_frames:
        # Silence-only results are unusual but retaining them is safer than
        # manufacturing an empty clip that could collapse scene timing.
        return samples.copy(), {
            "source_samples": len(samples),
            "trim_start_sample": 0,
            "trim_end_sample": len(samples),
            "first_active_sample": None,
            "last_active_sample": None,
            "kept_lead_samples": len(samples),
            "kept_tail_samples": 0,
        }

    first_active = active_frames[0][0]
    last_active_exclusive = active_frames[-1][1]
    lead = int(round(LEAD_SILENCE_SECONDS * sample_rate))
    tail = int(round(TAIL_SILENCE_SECONDS * sample_rate))
    trim_start = max(0, first_active - lead)
    trim_end = min(len(samples), last_active_exclusive + tail)
    return samples[trim_start:trim_end].copy(), {
        "source_samples": len(samples),
        "trim_start_sample": trim_start,
        "trim_end_sample": trim_end,
        "first_active_sample": first_active,
        "last_active_sample": last_active_exclusive - 1,
        "kept_lead_samples": first_active - trim_start,
        "kept_tail_samples": trim_end - last_active_exclusive,
        "threshold": threshold,
    }


def assemble_sentence_audio(
    sentence_audio: Iterable[Any], np, sample_rate: int = SAMPLE_RATE
):
    """Trim and join sentences with deterministic natural pauses."""
    source = list(sentence_audio)
    if not source:
        raise ValueError("At least one synthesized sentence is required")
    pause_samples = int(round(SENTENCE_PAUSE_SECONDS * sample_rate))
    pause = np.zeros(pause_samples, dtype=np.float32)
    parts = []
    segments = []
    cursor = 0
    previous_audible_end = None
    for index, audio in enumerate(source, start=1):
        trimmed, trim = trim_edge_silence(audio, np, sample_rate)
        clip_start = cursor
        clip_end = clip_start + len(trimmed)
        first_active = trim["first_active_sample"]
        last_active = trim["last_active_sample"]
        if first_active is None or last_active is None:
            audible_start = clip_start
            audible_end = clip_end
        else:
            audible_start = (
                clip_start + first_active - trim["trim_start_sample"]
            )
            audible_end = (
                clip_start + last_active + 1 - trim["trim_start_sample"]
            )
        segment = {
            "sentence": index,
            "clip_start": clip_start / sample_rate,
            "clip_end": clip_end / sample_rate,
            "audible_start": audible_start / sample_rate,
            "audible_end": audible_end / sample_rate,
            "duration": len(trimmed) / sample_rate,
            "trimmed_lead": trim["trim_start_sample"] / sample_rate,
            "trimmed_tail": (trim["source_samples"] - trim["trim_end_sample"])
            / sample_rate,
        }
        if previous_audible_end is not None:
            segment["gap_from_previous"] = (
                audible_start - previous_audible_end
            ) / sample_rate
        segments.append(segment)
        parts.append(trimmed)
        cursor = clip_end
        previous_audible_end = audible_end
        if index < len(source):
            parts.append(pause)
            cursor += pause_samples
    return np.concatenate(parts).astype(np.float32, copy=False), segments


def audio_assembly_config() -> dict[str, Any]:
    """Return the stable, JSON-safe assembly settings used in fingerprints."""
    return {
        "version": ASSEMBLY_VERSION,
        "sample_rate": SAMPLE_RATE,
        "lead_silence_seconds": LEAD_SILENCE_SECONDS,
        "tail_silence_seconds": TAIL_SILENCE_SECONDS,
        "sentence_pause_seconds": SENTENCE_PAUSE_SECONDS,
        "scene_hold_min_seconds": SCENE_HOLD_MIN_SECONDS,
        "scene_hold_max_seconds": SCENE_HOLD_MAX_SECONDS,
        "default_scene_hold_seconds": DEFAULT_SCENE_HOLD_SECONDS,
        "vad_frame_seconds": VAD_FRAME_SECONDS,
        "vad_peak_ratio": VAD_PEAK_RATIO,
        "vad_absolute_floor": VAD_ABSOLUTE_FLOOR,
    }


def render_fingerprint(payload: dict[str, Any]) -> str:
    """Hash a render plan so completed-but-stale tracks never skip."""
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
