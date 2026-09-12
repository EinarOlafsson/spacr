#!/usr/bin/env python3
"""Verify every supported narration track by fully decoding its AAC stream."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import subprocess
from threading import Lock
import warnings
from typing import Any

from narration_audio import (
    file_sha256,
    narration_dialect,
    resolve_voice_speed,
    synthesis_runtime_identity,
)
from render_all_voices import (
    LANGUAGES,
    SNAPSHOT,
    assert_frozen_sources_current,
    prepare_scene_plans,
    track_fingerprint,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import audioop
    except ImportError:  # pragma: no cover - Python 3.13 fallback below
        audioop = None


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
CATALOG = ROOT / "catalog" / "lessons_en.json"
SAMPLE_RATE = 24_000
CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2
DECODED_DURATION_TOLERANCE_SECONDS = 0.075
TIMING_TOLERANCE_SECONDS = 0.001
MIN_SILENT_HOLD_SECONDS = 0.1
MAX_SILENT_HOLD_SECONDS = 5.0
MIN_SCENE_RMS_PCM = 96
MIN_SCENE_PEAK_PCM = 512
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUNTIME_IDENTITY_LOCK = Lock()


@dataclass(frozen=True)
class TrackSpec:
    """Current renderer inputs for one supported release track."""

    path: Path
    lesson: dict[str, Any]
    language: str
    lang_code: str
    voice: str
    dialect: str
    speed: float
    scene_plans: list[dict[str, Any]]
    freshness_required: bool = True


@lru_cache(maxsize=None)
def _current_runtime_identity(voice: str, device: str) -> dict[str, Any]:
    """Build each expensive voice/runtime identity once, without hash races."""
    with _RUNTIME_IDENTITY_LOCK:
        identity = dict(synthesis_runtime_identity(SNAPSHOT, voice, device))
        voice_pack = dict(identity["voice_pack"])
        voice_pack["loaded_tensor_sha256"] = voice_pack["tensor_sha256"]
        identity["voice_pack"] = voice_pack
        return identity


def _error_excerpt(stderr: bytes | str, limit: int = 600) -> str:
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    compact = " | ".join(line.strip() for line in stderr.splitlines() if line.strip())
    return compact[:limit] or "no decoder diagnostic"


def _finite_number(
    value: Any,
    *,
    field: str,
    path: Path,
    errors: list[str],
) -> float | None:
    if isinstance(value, bool):
        errors.append(f"{path}: {field} must be a finite number")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{path}: {field} must be a finite number")
        return None
    if not math.isfinite(number):
        errors.append(f"{path}: {field} must be a finite number")
        return None
    return number


def _numbers_match(left: float, right: float) -> bool:
    return abs(left - right) <= TIMING_TOLERANCE_SECONDS


def _validate_sentences(
    path: Path,
    scene: dict[str, Any],
    *,
    scene_number: int,
    scene_start: float,
    scene_speech_end: float,
    plan: dict[str, Any] | None,
) -> list[str]:
    """Validate exact, monotonic sentence coverage within one scene."""
    errors: list[str] = []
    label = f"scene {scene_number}"
    sentences = scene.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        return [f"{path}: {label} sentences must be a non-empty list"]

    planned_sentences = plan["sentences"] if plan is not None else None
    if planned_sentences is not None and len(sentences) != len(planned_sentences):
        errors.append(
            f"{path}: {label} has {len(sentences)} sentence(s); current "
            f"catalog/render plan requires {len(planned_sentences)}"
        )

    previous_speech_end: float | None = None
    previous_audible_end: float | None = None
    first_speech_start: float | None = None
    last_speech_end: float | None = None
    for sentence_number, sentence in enumerate(sentences, start=1):
        sentence_label = f"{label} sentence {sentence_number}"
        if not isinstance(sentence, dict):
            errors.append(f"{path}: {sentence_label} must be a JSON object")
            continue
        if sentence.get("sentence") != sentence_number:
            errors.append(
                f"{path}: {sentence_label} has sentence="
                f"{sentence.get('sentence')!r}; expected {sentence_number}"
            )

        speech_start = _finite_number(
            sentence.get("speech_start"),
            field=f"{sentence_label} speech_start",
            path=path,
            errors=errors,
        )
        speech_end = _finite_number(
            sentence.get("speech_end"),
            field=f"{sentence_label} speech_end",
            path=path,
            errors=errors,
        )
        audible_start = _finite_number(
            sentence.get("audible_start"),
            field=f"{sentence_label} audible_start",
            path=path,
            errors=errors,
        )
        audible_end = _finite_number(
            sentence.get("audible_end"),
            field=f"{sentence_label} audible_end",
            path=path,
            errors=errors,
        )
        duration = _finite_number(
            sentence.get("duration"),
            field=f"{sentence_label} duration",
            path=path,
            errors=errors,
        )
        if None not in (
            speech_start,
            speech_end,
            audible_start,
            audible_end,
            duration,
        ):
            assert speech_start is not None
            assert speech_end is not None
            assert audible_start is not None
            assert audible_end is not None
            assert duration is not None
            if speech_start < scene_start - TIMING_TOLERANCE_SECONDS:
                errors.append(f"{path}: {sentence_label} begins before its scene")
            if speech_end > scene_speech_end + TIMING_TOLERANCE_SECONDS:
                errors.append(f"{path}: {sentence_label} ends after scene speech_end")
            if speech_end <= speech_start:
                errors.append(
                    f"{path}: {sentence_label} speech_end must be after " "speech_start"
                )
            if audible_start < speech_start - TIMING_TOLERANCE_SECONDS:
                errors.append(
                    f"{path}: {sentence_label} audible_start precedes " "speech_start"
                )
            if audible_end <= audible_start:
                errors.append(
                    f"{path}: {sentence_label} audible segment must be " "positive"
                )
            if audible_end > speech_end + TIMING_TOLERANCE_SECONDS:
                errors.append(
                    f"{path}: {sentence_label} audible_end exceeds speech_end"
                )
            if not _numbers_match(duration, speech_end - speech_start):
                errors.append(
                    f"{path}: {sentence_label} duration={duration:.6f}s "
                    f"does not match its speech interval "
                    f"({speech_end - speech_start:.6f}s)"
                )
            if (
                previous_speech_end is not None
                and speech_start < previous_speech_end - TIMING_TOLERANCE_SECONDS
            ):
                errors.append(
                    f"{path}: {sentence_label} overlaps the previous "
                    "sentence speech interval"
                )
            if (
                previous_audible_end is not None
                and audible_start < previous_audible_end - TIMING_TOLERANCE_SECONDS
            ):
                errors.append(
                    f"{path}: {sentence_label} has non-monotonic audible " "timing"
                )

            gap = sentence.get("gap_from_previous")
            if sentence_number == 1:
                if gap is not None:
                    errors.append(
                        f"{path}: {sentence_label} gap_from_previous must " "be null"
                    )
            else:
                gap_value = _finite_number(
                    gap,
                    field=f"{sentence_label} gap_from_previous",
                    path=path,
                    errors=errors,
                )
                if gap_value is not None and previous_audible_end is not None:
                    measured_gap = audible_start - previous_audible_end
                    if not _numbers_match(gap_value, measured_gap):
                        errors.append(
                            f"{path}: {sentence_label} gap_from_previous="
                            f"{gap_value:.6f}s does not match audible gap "
                            f"({measured_gap:.6f}s)"
                        )

            if first_speech_start is None:
                first_speech_start = speech_start
            last_speech_end = speech_end
            previous_speech_end = speech_end
            previous_audible_end = audible_end

        if (
            not isinstance(sentence.get("phonemes"), str)
            or not sentence["phonemes"].strip()
        ):
            errors.append(f"{path}: {sentence_label} phonemes are empty")
        if planned_sentences is not None and sentence_number <= len(planned_sentences):
            planned = planned_sentences[sentence_number - 1]
            for field in ("text", "speech_text"):
                if sentence.get(field) != planned[field]:
                    errors.append(
                        f"{path}: {sentence_label} {field} does not match "
                        "the current catalog/render plan"
                    )

    if first_speech_start is not None and not _numbers_match(
        first_speech_start, scene_start
    ):
        errors.append(
            f"{path}: {label} first sentence begins at "
            f"{first_speech_start:.6f}s instead of scene speech_start "
            f"{scene_start:.6f}s"
        )
    if last_speech_end is not None and not _numbers_match(
        last_speech_end, scene_speech_end
    ):
        errors.append(
            f"{path}: {label} last sentence ends at {last_speech_end:.6f}s "
            f"instead of scene speech_end {scene_speech_end:.6f}s"
        )
    return errors


def validate_timing(
    path: Path,
    timing: Any,
    spec: TrackSpec | None = None,
    *,
    require_sentence_metadata: bool | None = None,
) -> tuple[float | None, list[str]]:
    """Validate sidecar coverage and return its declared total duration."""
    errors: list[str] = []
    if not isinstance(timing, dict):
        return None, [f"{path}: timing sidecar must contain a JSON object"]

    if timing.get("schema") != 1:
        errors.append(f"{path}: timing schema={timing.get('schema')!r}, expected 1")
    expected_language = spec.language if spec is not None else path.parent.name
    expected_voice = spec.voice if spec is not None else path.stem
    if timing.get("language") != expected_language:
        errors.append(
            f"{path}: sidecar language={timing.get('language')!r}, "
            f"expected {expected_language!r}"
        )
    if timing.get("voice") != expected_voice:
        errors.append(
            f"{path}: sidecar voice={timing.get('voice')!r}, "
            f"expected {expected_voice!r}"
        )
    if timing.get("sample_rate") != SAMPLE_RATE:
        errors.append(
            f"{path}: sidecar sample_rate={timing.get('sample_rate')!r}, "
            f"expected {SAMPLE_RATE}"
        )

    total = _finite_number(
        timing.get("total_duration"),
        field="total_duration",
        path=path,
        errors=errors,
    )
    if total is not None and total <= 0:
        errors.append(f"{path}: total_duration must be positive")

    scenes = timing.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        errors.append(f"{path}: scenes must be a non-empty list")
        return total, errors
    if spec is not None and len(scenes) != len(spec.lesson["scenes"]):
        errors.append(
            f"{path}: sidecar has {len(scenes)} scene(s); current catalog "
            f"requires exactly {len(spec.lesson['scenes'])}"
        )

    if spec is not None:
        if timing.get("dialect") != spec.dialect:
            errors.append(
                f"{path}: sidecar dialect={timing.get('dialect')!r}, "
                f"expected {spec.dialect!r}"
            )
        sidecar_speed = _finite_number(
            timing.get("speed"), field="speed", path=path, errors=errors
        )
        if sidecar_speed is not None and not _numbers_match(sidecar_speed, spec.speed):
            errors.append(
                f"{path}: sidecar speed={sidecar_speed!r}, expected current "
                f"base speed {spec.speed!r}"
            )

    expected_start = 0.0
    for index, scene in enumerate(scenes, start=1):
        label = f"scene {index}"
        if not isinstance(scene, dict):
            errors.append(f"{path}: {label} must be a JSON object")
            continue
        if scene.get("scene") != index:
            errors.append(
                f"{path}: {label} has scene={scene.get('scene')!r}; "
                f"expected {index}"
            )

        start = _finite_number(
            scene.get("speech_start"),
            field=f"{label} speech_start",
            path=path,
            errors=errors,
        )
        speech_end = _finite_number(
            scene.get("speech_end"),
            field=f"{label} speech_end",
            path=path,
            errors=errors,
        )
        scene_end = _finite_number(
            scene.get("scene_end"),
            field=f"{label} scene_end",
            path=path,
            errors=errors,
        )
        duration = _finite_number(
            scene.get("duration"),
            field=f"{label} duration",
            path=path,
            errors=errors,
        )
        if None in (start, speech_end, scene_end, duration):
            continue

        assert start is not None
        assert speech_end is not None
        assert scene_end is not None
        assert duration is not None
        plan = (
            spec.scene_plans[index - 1]
            if spec is not None and index <= len(spec.scene_plans)
            else None
        )
        if abs(start - expected_start) > TIMING_TOLERANCE_SECONDS:
            relationship = "gap" if start > expected_start else "overlap"
            errors.append(
                f"{path}: {label} begins at {start:.6f}s instead of "
                f"{expected_start:.6f}s ({relationship} in scene coverage)"
            )
        if speech_end <= start:
            errors.append(f"{path}: {label} speech_end must be after speech_start")
        if scene_end <= speech_end:
            errors.append(f"{path}: {label} has no silent hold after speech")
        else:
            hold = scene_end - speech_end
            if not MIN_SILENT_HOLD_SECONDS <= hold <= MAX_SILENT_HOLD_SECONDS:
                errors.append(
                    f"{path}: {label} silent hold={hold:.3f}s; expected "
                    f"{MIN_SILENT_HOLD_SECONDS:.3f}s to "
                    f"{MAX_SILENT_HOLD_SECONDS:.3f}s"
                )
        if abs(duration - (scene_end - start)) > TIMING_TOLERANCE_SECONDS:
            errors.append(
                f"{path}: {label} duration={duration:.6f}s does not cover "
                f"speech_start through scene_end ({scene_end - start:.6f}s)"
            )
        if plan is not None:
            expected_speech_text = " ".join(
                sentence["speech_text"] for sentence in plan["sentences"]
            )
            if scene.get("text") != plan["display_text"]:
                errors.append(
                    f"{path}: {label} text does not match the current catalog"
                )
            if scene.get("speech_text") != expected_speech_text:
                errors.append(
                    f"{path}: {label} speech_text does not match the current "
                    "render plan"
                )
            numeric_expectations = {
                "hold_after": plan["hold"],
                "base_speed": spec.speed,
                "speed_multiplier": plan["speed_multiplier"],
                "effective_speed": plan["effective_speed"],
            }
            for field, expected in numeric_expectations.items():
                actual = _finite_number(
                    scene.get(field),
                    field=f"{label} {field}",
                    path=path,
                    errors=errors,
                )
                if actual is not None and not _numbers_match(actual, expected):
                    errors.append(
                        f"{path}: {label} {field}={actual!r}, expected "
                        f"{expected!r} from the current render plan"
                    )
            authored_hold = scene.get("authored_hold_after")
            if plan["authored_hold"] is None:
                if authored_hold is not None:
                    errors.append(f"{path}: {label} authored_hold_after must be null")
            else:
                actual_authored_hold = _finite_number(
                    authored_hold,
                    field=f"{label} authored_hold_after",
                    path=path,
                    errors=errors,
                )
                if actual_authored_hold is not None and not _numbers_match(
                    actual_authored_hold, plan["authored_hold"]
                ):
                    errors.append(
                        f"{path}: {label} authored_hold_after does not match "
                        "the current catalog"
                    )
        validate_sentences = (
            spec is not None
            if require_sentence_metadata is None
            else require_sentence_metadata
        )
        if validate_sentences:
            errors.extend(
                _validate_sentences(
                    path,
                    scene,
                    scene_number=index,
                    scene_start=start,
                    scene_speech_end=speech_end,
                    plan=plan,
                )
            )
        if total is not None and scene_end > total + TIMING_TOLERANCE_SECONDS:
            errors.append(
                f"{path}: {label} ends after total_duration "
                f"({scene_end:.6f}s > {total:.6f}s)"
            )
        expected_start = scene_end

    if total is not None and abs(expected_start - total) > TIMING_TOLERANCE_SECONDS:
        errors.append(
            f"{path}: final scene ends at {expected_start:.6f}s, "
            f"total_duration={total:.6f}s"
        )
    return total, errors


def _probe_track(path: Path) -> tuple[int | None, int | None, list[str]]:
    errors: list[str] = []
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return None, None, [f"{path}: ffprobe could not run: {error}"]
    if probe.returncode != 0:
        errors.append(f"{path}: ffprobe failed: {_error_excerpt(probe.stderr)}")
        return None, None, errors
    try:
        data = json.loads(probe.stdout)
        stream = data["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        errors.append(f"{path}: invalid ffprobe result: {error}")
        return None, None, errors

    if stream.get("codec_name") != "aac":
        errors.append(f"{path}: codec={stream.get('codec_name')}, expected aac")
    try:
        sample_rate = int(stream.get("sample_rate", 0))
        channels = int(stream.get("channels", 0))
    except (TypeError, ValueError):
        sample_rate = channels = 0
    if sample_rate != SAMPLE_RATE:
        errors.append(f"{path}: sample_rate={sample_rate}, expected {SAMPLE_RATE}")
    if channels != CHANNELS:
        errors.append(f"{path}: channels={channels}, expected {CHANNELS}")
    return sample_rate or None, channels or None, errors


def _decode_track(
    path: Path,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> tuple[int, float, bytes, list[str]]:
    """Strictly decode all AAC packets and count decoded PCM sample frames."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-threads",
        "1",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        decoded = subprocess.run(command, check=False, capture_output=True)
    except OSError as error:
        return 0, 0.0, b"", [f"{path}: ffmpeg could not run: {error}"]

    errors: list[str] = []
    # Some corrupt AAC streams make FFmpeg print decoder errors but still exit
    # zero. Error-level stderr is therefore independently release-fatal.
    if decoded.returncode != 0:
        errors.append(
            f"{path}: strict AAC decode exited {decoded.returncode}: "
            f"{_error_excerpt(decoded.stderr)}"
        )
    elif decoded.stderr.strip():
        errors.append(
            f"{path}: strict AAC decoder reported errors: "
            f"{_error_excerpt(decoded.stderr)}"
        )

    bytes_per_frame = PCM_BYTES_PER_SAMPLE * max(channels, 1)
    sample_frames, remainder = divmod(len(decoded.stdout), bytes_per_frame)
    if remainder:
        errors.append(
            f"{path}: decoded PCM has {remainder} trailing byte(s) outside "
            "a complete sample frame"
        )
    if sample_frames == 0:
        errors.append(f"{path}: strict AAC decode produced no samples")
    duration = sample_frames / sample_rate if sample_rate > 0 else 0.0
    return sample_frames, duration, decoded.stdout, errors


def _pcm_activity(fragment: bytes) -> tuple[int, int]:
    """Return integer PCM16 RMS and peak, with a Python 3.13 fallback."""
    if not fragment:
        return 0, 0
    if audioop is not None:
        return audioop.rms(fragment, PCM_BYTES_PER_SAMPLE), audioop.max(
            fragment, PCM_BYTES_PER_SAMPLE
        )

    # ``audioop`` was removed in Python 3.13. Keep the release verifier
    # self-contained instead of requiring NumPy merely for two statistics.
    from array import array
    import sys

    samples = array("h")
    samples.frombytes(fragment)
    if sys.byteorder != "little":  # decoded PCM is always s16le
        samples.byteswap()
    if not samples:
        return 0, 0
    peak = max(abs(sample) for sample in samples)
    rms = int(math.sqrt(sum(sample * sample for sample in samples) / len(samples)))
    return rms, peak


def _scene_activity_errors(
    path: Path,
    timing: Any,
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> list[str]:
    """Require credible decoded signal activity inside every scene's speech."""
    if not isinstance(timing, dict) or not isinstance(timing.get("scenes"), list):
        return []
    errors: list[str] = []
    bytes_per_frame = PCM_BYTES_PER_SAMPLE * max(channels, 1)
    frame_count = len(pcm) // bytes_per_frame
    for index, scene in enumerate(timing["scenes"], start=1):
        if not isinstance(scene, dict):
            continue
        try:
            start_seconds = float(scene["speech_start"])
            end_seconds = float(scene["speech_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
            continue
        start_frame = max(0, min(frame_count, math.floor(start_seconds * sample_rate)))
        end_frame = max(0, min(frame_count, math.ceil(end_seconds * sample_rate)))
        if end_frame <= start_frame:
            errors.append(
                f"{path}: scene {index} has no decoded samples in its "
                "declared speech interval"
            )
            continue
        fragment = pcm[start_frame * bytes_per_frame : end_frame * bytes_per_frame]
        rms, peak = _pcm_activity(fragment)
        if rms < MIN_SCENE_RMS_PCM or peak < MIN_SCENE_PEAK_PCM:
            errors.append(
                f"{path}: scene {index} has no credible decoded speech "
                f"activity (PCM16 rms={rms}, peak={peak}; minimum rms="
                f"{MIN_SCENE_RMS_PCM}, peak={MIN_SCENE_PEAK_PCM})"
            )
    return errors


def _media_identity_errors(path: Path, timing: dict[str, Any]) -> list[str]:
    """Bind the sidecar to the exact M4A bytes being released."""
    errors: list[str] = []
    declared_bytes = timing.get("media_bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
        errors.append(f"{path}: media_bytes must be an integer")
    else:
        try:
            actual_bytes = path.stat().st_size
        except OSError as error:
            errors.append(f"{path}: could not stat media for identity: {error}")
        else:
            if declared_bytes != actual_bytes:
                errors.append(
                    f"{path}: media_bytes={declared_bytes}, actual M4A has "
                    f"{actual_bytes} bytes"
                )

    declared_hash = timing.get("media_sha256")
    if not isinstance(declared_hash, str) or not SHA256_PATTERN.fullmatch(
        declared_hash
    ):
        errors.append(f"{path}: media_sha256 must be a lowercase 64-character SHA-256")
    else:
        try:
            actual_hash = file_sha256(path)
        except OSError as error:
            errors.append(f"{path}: could not hash media for identity: {error}")
        else:
            if declared_hash != actual_hash:
                errors.append(
                    f"{path}: media_sha256 does not match the actual M4A "
                    f"({declared_hash} != {actual_hash})"
                )
    return errors


def _expected_render_metadata(
    spec: TrackSpec,
    timing: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    """Rebuild the current frozen renderer identity for a supported track."""
    try:
        device = timing["render_inputs"]["synthesis_runtime"]["inference"]["device"]
    except (KeyError, TypeError):
        return (
            None,
            None,
            [f"{spec.path}: render_inputs does not record the synthesis device"],
        )
    if device not in {"cpu", "cuda"}:
        return (
            None,
            None,
            [
                f"{spec.path}: recorded synthesis device={device!r}; expected "
                "'cpu' or 'cuda'"
            ],
        )
    try:
        runtime_identity = _current_runtime_identity(spec.voice, device)
        fingerprint, render_inputs = track_fingerprint(
            spec.lesson,
            spec.language,
            spec.lang_code,
            spec.dialect,
            spec.voice,
            spec.speed,
            spec.scene_plans,
            runtime_identity,
        )
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        return (
            None,
            None,
            [
                f"{spec.path}: could not reconstruct current render identity: "
                f"{error}"
            ],
        )
    return fingerprint, render_inputs, []


def _render_identity_errors(
    spec: TrackSpec,
    timing: dict[str, Any],
) -> list[str]:
    expected_fingerprint, expected_inputs, errors = _expected_render_metadata(
        spec, timing
    )
    if errors:
        return errors
    if timing.get("render_inputs") != expected_inputs:
        errors.append(
            f"{spec.path}: render_inputs are stale or do not match the "
            "current catalog, pronunciation, cadence, model, voice, and tool "
            "identities"
        )
    if timing.get("render_fingerprint") != expected_fingerprint:
        errors.append(
            f"{spec.path}: render_fingerprint is stale; expected "
            f"{expected_fingerprint}, found "
            f"{timing.get('render_fingerprint')!r}"
        )
    return errors


def _dead_air_errors(path: Path) -> list[str]:
    silence = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-threads",
            "1",
            "-i",
            str(path),
            "-af",
            "silencedetect=noise=-50dB:d=8",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    errors: list[str] = []
    if silence.returncode != 0:
        errors.append(f"{path}: dead-air analysis failed")
    if "silence_duration:" in silence.stderr:
        errors.append(f"{path}: contains at least 8 seconds of dead air")
    return errors


def check_track(track: Path | TrackSpec) -> list[str]:
    spec = track if isinstance(track, TrackSpec) else None
    path = spec.path if spec is not None else track
    errors: list[str] = []
    timing_path = path.with_suffix(".json")
    expected_duration: float | None = None
    timing: Any = None
    try:
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{path}: timing sidecar error: {error}")
    else:
        freshness_spec = spec if spec is not None and spec.freshness_required else None
        expected_duration, timing_errors = validate_timing(path, timing, freshness_spec)
        errors.extend(timing_errors)
        if isinstance(timing, dict):
            if spec is not None and spec.freshness_required:
                errors.extend(_media_identity_errors(path, timing))
                errors.extend(_render_identity_errors(spec, timing))

    sample_rate, channels, probe_errors = _probe_track(path)
    errors.extend(probe_errors)
    decoded_samples, decoded_duration, decoded_pcm, decode_errors = _decode_track(
        path,
        sample_rate=sample_rate or SAMPLE_RATE,
        channels=channels or CHANNELS,
    )
    errors.extend(decode_errors)
    if expected_duration is not None and (
        abs(decoded_duration - expected_duration) > DECODED_DURATION_TOLERANCE_SECONDS
    ):
        errors.append(
            f"{path}: decoded_samples={decoded_samples} "
            f"decoded_duration={decoded_duration:.6f}s "
            f"sidecar={expected_duration:.6f}s exceeds "
            f"{DECODED_DURATION_TOLERANCE_SECONDS * 1000:.0f}ms tolerance"
        )
    if not decode_errors:
        errors.extend(
            _scene_activity_errors(
                path,
                timing,
                decoded_pcm,
                sample_rate=sample_rate or SAMPLE_RATE,
                channels=channels or CHANNELS,
            )
        )
        errors.extend(_dead_air_errors(path))
    return errors


def supported_track_paths(
    *,
    production: Path = PRODUCTION,
    catalog_path: Path = CATALOG,
    languages: dict[str, tuple[str, list[str]]] | None = None,
) -> list[Path]:
    """Return the exact lesson/language/voice matrix offered to users."""
    matrix = LANGUAGES if languages is None else languages
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    lesson_ids = [lesson["id"] for lesson in catalog["lessons"]]
    return [
        production / lesson_id / "audio" / language / f"{voice}.m4a"
        for lesson_id in lesson_ids
        for language, (_, voices) in matrix.items()
        for voice in voices
    ]


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lessons = data.get("lessons") if isinstance(data, dict) else None
    if not isinstance(lessons, list) or not lessons:
        raise ValueError(f"{path}: lessons must be a non-empty list")
    if not all(isinstance(lesson, dict) for lesson in lessons):
        raise ValueError(f"{path}: every lesson must be a JSON object")
    identifiers = [lesson.get("id") for lesson in lessons]
    if not all(
        isinstance(identifier, str) and identifier for identifier in identifiers
    ):
        raise ValueError(f"{path}: every lesson must have a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path}: duplicate lesson ids are not allowed")
    return lessons


def supported_track_specs(
    *,
    production: Path = PRODUCTION,
    catalog_path: Path = CATALOG,
    languages: dict[str, tuple[str, list[str]]] | None = None,
    freshness_languages: set[str] | None = None,
) -> list[TrackSpec]:
    """Plan current renderer inputs for the exact supported release matrix."""
    matrix = LANGUAGES if languages is None else languages
    freshness_scope = (
        set(matrix) if freshness_languages is None else freshness_languages
    )
    unknown_freshness = freshness_scope - set(matrix)
    if unknown_freshness:
        raise ValueError(
            "unsupported freshness language(s): " + ", ".join(sorted(unknown_freshness))
        )
    canonical_lessons = _read_catalog(catalog_path)
    canonical_ids = [lesson["id"] for lesson in canonical_lessons]
    localized_by_language: dict[str, dict[str, dict[str, Any]]] = {}
    for language in matrix:
        localized_path = (
            catalog_path
            if language == "en"
            else catalog_path.with_name(f"lessons_{language}.json")
        )
        localized_lessons = _read_catalog(localized_path)
        localized_ids = [lesson["id"] for lesson in localized_lessons]
        if localized_ids != canonical_ids:
            raise ValueError(
                f"{localized_path}: lesson ids/order do not exactly match "
                f"{catalog_path}"
            )
        localized_by_language[language] = {
            lesson["id"]: lesson for lesson in localized_lessons
        }

    specs: list[TrackSpec] = []
    plan_cache: dict[tuple[str, str, str, float, str], list[dict[str, Any]]] = {}
    for lesson_id in canonical_ids:
        for language, (lang_code, voices) in matrix.items():
            lesson = localized_by_language[language][lesson_id]
            scenes = lesson.get("scenes")
            if not isinstance(scenes, list) or not scenes:
                raise ValueError(
                    f"{language}/{lesson_id}: scenes must be a non-empty list"
                )
            for voice in voices:
                voice_lang_code = (
                    "b" if language == "en" and voice.startswith("b") else lang_code
                )
                dialect = narration_dialect(language, voice_lang_code, voice)
                speed = resolve_voice_speed(voice)
                cache_key = (lesson_id, language, dialect, speed, voice)
                if cache_key not in plan_cache:
                    plan_cache[cache_key] = prepare_scene_plans(
                        lesson, language, dialect, speed, voice=voice
                    )
                specs.append(
                    TrackSpec(
                        path=(
                            production / lesson_id / "audio" / language / f"{voice}.m4a"
                        ),
                        lesson=lesson,
                        language=language,
                        lang_code=voice_lang_code,
                        voice=voice,
                        dialect=dialect,
                        speed=speed,
                        scene_plans=plan_cache[cache_key],
                        freshness_required=language in freshness_scope,
                    )
                )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--freshness-languages",
        default="all",
        help=(
            "comma-separated languages that must match current renderer "
            "inputs and media identities; default: all"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")

    if args.freshness_languages == "all":
        freshness_languages = set(LANGUAGES)
    else:
        freshness_languages = {
            language.strip()
            for language in args.freshness_languages.split(",")
            if language.strip()
        }
        if not freshness_languages:
            parser.error("--freshness-languages must not be empty")
        unknown_languages = freshness_languages - set(LANGUAGES)
        if unknown_languages:
            parser.error(
                "unsupported --freshness-languages value(s): "
                + ", ".join(sorted(unknown_languages))
            )

    try:
        assert_frozen_sources_current()
        tracks = supported_track_specs(freshness_languages=freshness_languages)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERROR: could not build supported audio matrix: {error}")
        return 1
    if not tracks:
        print("ERROR: supported audio matrix is empty")
        return 1

    matrix_errors: list[str] = []
    present_tracks: list[TrackSpec] = []
    for track in tracks:
        if not track.path.is_file():
            matrix_errors.append(f"{track.path}: supported narration track is missing")
            continue
        if not track.path.with_suffix(".json").is_file():
            matrix_errors.append(f"{track.path}: supported timing sidecar is missing")
            continue
        present_tracks.append(track)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check_track, present_tracks))
    try:
        assert_frozen_sources_current()
    except RuntimeError as error:
        matrix_errors.append(
            f"frozen renderer source changed during verification: {error}"
        )
    errors = matrix_errors + [error for result in results for error in result]
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors[:100]))
        print(f"errors={len(errors)} tracks={len(tracks)}")
        return 1

    voice_count = sum(len(voices) for _, voices in LANGUAGES.values())
    lesson_count = len(tracks) // voice_count
    print(
        f"verified supported audio tracks={len(tracks)} "
        f"lessons={lesson_count} voices={voice_count} codec=aac "
        f"sample_rate={SAMPLE_RATE} channels={CHANNELS} "
        f"decoded_duration_tolerance_ms="
        f"{DECODED_DURATION_TOLERANCE_SECONDS * 1000:.0f} dead_air=none "
        f"freshness_languages={','.join(sorted(freshness_languages))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
