#!/usr/bin/env python3
"""Render resumable Kokoro narration tracks for the complete tutorial matrix.

The output is intentionally audio-only: one 4K silent visual master is shared
by every language and voice in the browser. This keeps the published course
small enough to host while preserving every approved distinct voice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from narration_audio import (
    SAMPLE_RATE,
    assemble_sentence_audio,
    assert_cadence_profile_current,
    atomic_write_text,
    audio_assembly_config,
    cadence_profile_identity,
    clamp_scene_hold,
    effective_scene_speed,
    file_sha256,
    load_module_snapshot,
    narration_dialect,
    render_fingerprint,
    resolve_voice_speed,
    split_sentences,
    staged_output_path,
    synthesis_runtime_identity,
)


PRONUNCIATION_PATH = Path(__file__).with_name("pronunciation.py")
pronunciation_profile, PRONUNCIATION_SOURCE_SNAPSHOT = load_module_snapshot(
    PRONUNCIATION_PATH,
    "tutorial_pronunciation_snapshot",
    "pronunciation_profile",
)
PRONUNCIATION_IDENTITY = PRONUNCIATION_SOURCE_SNAPSHOT.fingerprint(
    getattr(pronunciation_profile, "PRONUNCIATION_VERSION", None)
)
assert_pronunciation_safe = pronunciation_profile.assert_pronunciation_safe
spoken_form = pronunciation_profile.spoken_form


def assert_frozen_sources_current() -> None:
    """Stop a run before source edits can mix identities between tracks."""
    PRONUNCIATION_SOURCE_SNAPSHOT.assert_current()
    assert_cadence_profile_current()


AUTHORING_ROOT = Path(__file__).resolve().parents[1]
# A refresh must not replace the currently published narration while only
# some languages are ready. Keep models shared, but isolate scripts/output.
ROOT = Path(os.environ.get("SPACR_TUTORIAL_WORKSPACE", AUTHORING_ROOT)).resolve()
CATALOG = ROOT / "catalog"
PRODUCTION = ROOT / "production"
MODEL_CACHE = AUTHORING_ROOT / "project" / "kokoro_models"
SNAPSHOT = (
    MODEL_CACHE
    / "hub"
    / "models--hexgrad--Kokoro-82M"
    / "snapshots"
    / "f3ff3571791e39611d31c381e3a41a3af07b4987"
)
# AAC can overshoot the PCM true peak during encoding. A -1.5 dBTP filter
# target produced decoded peaks as high as +1.4 dBFS in the release matrix.
# The measured -3 dBTP target leaves roughly 2 dB of post-codec headroom while
# preserving a broadcast-friendly integrated loudness near -16 LUFS.
LOUDNESS_FILTERS = (
    "loudnorm=I=-16:TP=-3:LRA=11",
    (
        "loudnorm=I=-16:TP=-5:LRA=11,"
        "alimiter=limit=0.7:attack=5:release=50:level=disabled"
    ),
    (
        "loudnorm=I=-16:TP=-5:LRA=11,"
        "alimiter=limit=0.5:attack=5:release=50:level=disabled"
    ),
)
MAX_RELEASE_TRUE_PEAK_DBFS = -1.0
MASTERING_CONFIG = {
    "filters": list(LOUDNESS_FILTERS),
    "codec": "aac",
    "bitrate": "48k",
    "channels": 1,
    "maximum_decoded_true_peak_dbfs": MAX_RELEASE_TRUE_PEAK_DBFS,
}

LANGUAGES = {
    "en": (
        "a",
        [
            "af_heart",
            "af_aoede",
            "af_bella",
            "af_jessica",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
            "bm_daniel",
            "bm_fable",
            "bm_george",
            "bm_lewis",
        ],
    ),
    "es": ("e", ["ef_dora", "em_alex", "em_santa"]),
    "fr": ("f", ["ff_siwis"]),
    "hi": ("h", ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]),
    "it": ("i", ["if_sara", "im_nicola"]),
    "pt-BR": ("p", ["pf_dora", "pm_alex", "pm_santa"]),
    "ja": (
        "j",
        ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    ),
    "zh-CN": (
        "z",
        [
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
            "zm_yunjian",
            "zm_yunxi",
            "zm_yunxia",
            "zm_yunyang",
        ],
    ),
}


def vtt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def write_vtt(path: Path, timings: list[dict]) -> None:
    cues = []
    for timing in timings:
        sentences = timing.get("sentences")
        if sentences:
            cues.extend(
                {
                    "speech_start": sentence["speech_start"],
                    "speech_end": sentence["speech_end"],
                    "text": sentence["text"],
                }
                for sentence in sentences
            )
        else:
            cues.append(timing)
    lines = ["WEBVTT", ""]
    previous_end = 0.0
    for index, timing in enumerate(cues, start=1):
        start = float(timing["speech_start"])
        end = float(timing["speech_end"])
        if start < previous_end - 0.001 or end <= start:
            raise ValueError(
                f"Non-monotonic caption timing at cue {index}: {start}--{end}"
            )
        text = str(timing["text"]).replace("-->", "→").strip()
        lines.extend(
            [
                str(index),
                f"{vtt_timestamp(start)} --> {vtt_timestamp(end)}",
                text,
                "",
            ]
        )
        previous_end = end
    atomic_write_text(path, "\n".join(lines))


def true_peak_dbfs(path: Path) -> float | None:
    """Measure the decoded AAC true peak, or fail closed with ``None``."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    matches = re.findall(
        r"Peak:\s+(-?(?:\d+(?:\.\d+)?|inf)) dBFS", result.stderr
    )
    if result.returncode != 0 or not matches:
        return None
    return float(matches[-1])


def prepare_scene_plans(
    lesson: dict,
    language: str,
    dialect: str,
    base_speed: float = 1.0,
) -> list[dict]:
    """Resolve exact sentence-level TTS text before cache validation."""
    plans = []
    for index, scene in enumerate(lesson["scenes"], start=1):
        display_text = scene["narration"].strip()
        display_sentences = split_sentences(display_text, language)
        if not display_sentences:
            raise ValueError(
                f"Empty narration in {lesson['id']} scene {index}"
            )
        sentences = []
        for display_sentence in display_sentences:
            speech_text = spoken_form(
                display_sentence, language, dialect=dialect
            )
            assert_pronunciation_safe(display_sentence, speech_text)
            sentences.append(
                {
                    "text": display_sentence,
                    "speech_text": speech_text,
                }
            )
        authored_hold = scene.get("hold_after")
        scene_speed, speed_multiplier = effective_scene_speed(
            base_speed, language, lesson["id"], index
        )
        plans.append(
            {
                "display_text": display_text,
                "sentences": sentences,
                "authored_hold": authored_hold,
                "hold": clamp_scene_hold(authored_hold),
                "speed_multiplier": speed_multiplier,
                "effective_speed": scene_speed,
            }
        )
    return plans


def track_fingerprint(
    lesson: dict,
    language: str,
    lang_code: str,
    dialect: str,
    voice: str,
    speed: float,
    scene_plans: list[dict],
    runtime_identity: dict | None = None,
) -> tuple[str, dict]:
    """Return a deterministic digest and the render inputs it represents."""
    payload = {
        "renderer": "render_all_voices-v4",
        "lesson": lesson["id"],
        "language": language,
        "lang_code": lang_code,
        "dialect": dialect,
        "voice": voice,
        "speed": speed,
        "scenes": [
            {
                "narration": plan["display_text"],
                "speech_text": [
                    sentence["speech_text"] for sentence in plan["sentences"]
                ],
                "authored_hold": plan["authored_hold"],
                "resolved_hold": plan["hold"],
                "speed_multiplier": plan["speed_multiplier"],
                "effective_speed": plan["effective_speed"],
            }
            for plan in scene_plans
        ],
        "assembly": audio_assembly_config(),
        "cadence_profile": (
            cadence_profile_identity() if language == "en" else None
        ),
        "mastering": mastering_config(lesson["id"], language, voice),
        "pronunciation": PRONUNCIATION_IDENTITY,
        "synthesis_runtime": runtime_identity or {"state": "test-unbound"},
    }
    return render_fingerprint(payload), payload


def mastering_config(lesson_id: str, language: str, voice: str) -> dict:
    """Keep existing tracks unchanged; pin a measured, narrowly scoped repair.

    Motility's bf_isabella failed all three normal encodes: the last decoded
    AAC true peak was -0.5 dBFS. Add 2 dB of attenuation only for this track,
    before encoding, and still require the existing decoded -1 dBFS gate.
    The extra chain is fingerprinted; no timing, voice or pronunciation changes.
    """
    result = dict(MASTERING_CONFIG)
    result["filters"] = list(MASTERING_CONFIG["filters"])
    if (lesson_id, language, voice) == ("18_motility", "en", "bf_isabella"):
        result["filters"].append(LOUDNESS_FILTERS[-1] + ",volume=-2dB")
    return result


def audio_equivalent_render_inputs(
    previous: object,
    current: object,
) -> bool:
    """Return whether a profile edit left the synthesized audio unchanged.

    The complete pronunciation source identity remains in the release
    fingerprint, so any edit is visible and auditable.  Re-synthesis is only
    necessary when that edit changes the exact sentence speech plan.  Display
    narration may also change without changing the waveform; its timing
    metadata and captions can be refreshed in place.
    """
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    old = json.loads(json.dumps(previous))
    new = json.loads(json.dumps(current))
    old.pop("pronunciation", None)
    new.pop("pronunciation", None)
    for payload in (old, new):
        scenes = payload.get("scenes")
        if not isinstance(scenes, list):
            return False
        for scene in scenes:
            if not isinstance(scene, dict):
                return False
            scene.pop("narration", None)
    return old == new


def refresh_reused_timing(
    timing_data: dict,
    scene_plans: list[dict],
    fingerprint: str,
    fingerprint_inputs: dict,
    media_bytes: int,
) -> bool:
    """Refresh text/profile metadata while preserving verified audio timing."""
    scenes = timing_data.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != len(scene_plans):
        return False
    for timing, plan in zip(scenes, scene_plans):
        sentences = timing.get("sentences")
        planned_sentences = plan["sentences"]
        if (
            not isinstance(sentences, list)
            or len(sentences) != len(planned_sentences)
            or [item.get("speech_text") for item in sentences]
            != [item["speech_text"] for item in planned_sentences]
        ):
            return False
        timing["text"] = plan["display_text"]
        timing["speech_text"] = " ".join(
            item["speech_text"] for item in planned_sentences
        )
        timing["authored_hold_after"] = plan["authored_hold"]
        timing["hold_after"] = plan["hold"]
        timing["speed_multiplier"] = plan["speed_multiplier"]
        timing["effective_speed"] = plan["effective_speed"]
        for sentence_timing, sentence_plan in zip(
            sentences, planned_sentences
        ):
            sentence_timing["text"] = sentence_plan["text"]
            sentence_timing["speech_text"] = sentence_plan["speech_text"]
    timing_data["render_fingerprint"] = fingerprint
    timing_data["render_inputs"] = fingerprint_inputs
    timing_data["media_bytes"] = media_bytes
    return True


def render_track(
    pipeline,
    voice: str,
    lesson: dict,
    language: str,
    lang_code: str,
    dialect: str,
    speed: float,
    np,
    sf,
    force: bool,
    repair_peaks: bool,
    runtime_identity: dict,
    synthesis_voice,
) -> float:
    assert_frozen_sources_current()
    lesson_root = PRODUCTION / lesson["id"]
    audio_dir = lesson_root / "audio" / language
    audio_dir.mkdir(parents=True, exist_ok=True)
    m4a = audio_dir / f"{voice}.m4a"
    timing_path = audio_dir / f"{voice}.json"
    captions_path = lesson_root / "captions" / language / f"{voice}.vtt"
    scene_plans = prepare_scene_plans(lesson, language, dialect, speed)
    fingerprint, fingerprint_inputs = track_fingerprint(
        lesson,
        language,
        lang_code,
        dialect,
        voice,
        speed,
        scene_plans,
        runtime_identity,
    )
    complete = all(
        path.exists() and path.stat().st_size > 200
        for path in (m4a, timing_path)
    )
    if not force and complete:
        try:
            timing_data = json.loads(timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            timing_data = {}
        fingerprint_matches = timing_data.get("render_fingerprint") == fingerprint
        media_matches = False
        if timing_data.get("media_sha256"):
            try:
                media_matches = timing_data["media_sha256"] == file_sha256(m4a)
            except OSError:
                media_matches = False
        previous_inputs = timing_data.get("render_inputs")
        previous_runtime = (
            previous_inputs.get("synthesis_runtime")
            if isinstance(previous_inputs, dict)
            else None
        )
        reuse_fingerprint, reuse_inputs = track_fingerprint(
            lesson,
            language,
            lang_code,
            dialect,
            voice,
            speed,
            scene_plans,
            previous_runtime,
        )
        metadata_reusable = (
            not fingerprint_matches
            and media_matches
            and isinstance(previous_runtime, dict)
            and audio_equivalent_render_inputs(previous_inputs, reuse_inputs)
        )
        peak = (
            true_peak_dbfs(m4a)
            if repair_peaks
            and (fingerprint_matches or metadata_reusable)
            and media_matches
            else None
        )
        peak_is_safe = not repair_peaks or (
            peak is not None and peak <= MAX_RELEASE_TRUE_PEAK_DBFS
        )
        if (
            (fingerprint_matches or metadata_reusable)
            and media_matches
            and peak_is_safe
        ):
            if metadata_reusable:
                refreshed = refresh_reused_timing(
                    timing_data,
                    scene_plans,
                    reuse_fingerprint,
                    reuse_inputs,
                    m4a.stat().st_size,
                )
                if not refreshed:
                    metadata_reusable = False
                else:
                    atomic_write_text(
                        timing_path,
                        json.dumps(timing_data, indent=2, ensure_ascii=False)
                        + "\n",
                    )
            if not fingerprint_matches and not metadata_reusable:
                pass
            else:
                write_vtt(captions_path, timing_data["scenes"])
                return -1.0

    combined = []
    timings = []
    cursor = 0.0
    for index, plan in enumerate(scene_plans, start=1):
        sentence_audio = []
        sentence_phonemes = []
        for sentence_number, sentence in enumerate(plan["sentences"], start=1):
            chunks = []
            phonemes = []
            for result in pipeline(
                sentence["speech_text"],
                voice=synthesis_voice,
                speed=plan["effective_speed"],
            ):
                if result.audio is not None:
                    chunks.append(
                        result.audio.detach().cpu().numpy().astype(np.float32)
                    )
                if result.phonemes:
                    phonemes.append(str(result.phonemes))
            if not chunks:
                raise RuntimeError(
                    f"No audio for {language}/{voice}/{lesson['id']}/"
                    f"{index}/{sentence_number}"
                )
            sentence_audio.append(
                np.concatenate(chunks).astype(np.float32, copy=False)
            )
            sentence_phonemes.append(" ".join(phonemes))
        speech, sentence_segments = assemble_sentence_audio(sentence_audio, np)
        peak = float(np.max(np.abs(speech)))
        if peak > 0.98:
            speech *= 0.98 / peak
        speech_start = cursor
        speech_end = speech_start + len(speech) / SAMPLE_RATE
        hold = plan["hold"]
        scene_end = speech_end + hold
        combined.extend(
            [
                speech,
                np.zeros(int(round(hold * SAMPLE_RATE)), dtype=np.float32),
            ]
        )
        sentence_timings = []
        for sentence, segment, phonemes in zip(
            plan["sentences"], sentence_segments, sentence_phonemes
        ):
            sentence_timings.append(
                {
                    "sentence": segment["sentence"],
                    "speech_start": speech_start + segment["clip_start"],
                    "speech_end": speech_start + segment["clip_end"],
                    "audible_start": speech_start + segment["audible_start"],
                    "audible_end": speech_start + segment["audible_end"],
                    "duration": segment["duration"],
                    "gap_from_previous": segment.get("gap_from_previous"),
                    "trimmed_lead": segment["trimmed_lead"],
                    "trimmed_tail": segment["trimmed_tail"],
                    "text": sentence["text"],
                    "speech_text": sentence["speech_text"],
                    "phonemes": phonemes,
                }
            )
        timings.append(
            {
                "scene": index,
                "speech_start": speech_start,
                "speech_end": speech_end,
                "scene_end": scene_end,
                "duration": scene_end - speech_start,
                "text": plan["display_text"],
                "speech_text": " ".join(
                    sentence["speech_text"] for sentence in plan["sentences"]
                ),
                "phonemes": " ".join(sentence_phonemes),
                "authored_hold_after": plan["authored_hold"],
                "hold_after": hold,
                "base_speed": speed,
                "speed_multiplier": plan["speed_multiplier"],
                "effective_speed": plan["effective_speed"],
                "sentences": sentence_timings,
            }
        )
        cursor = scene_end

    full = np.concatenate(combined).astype(np.float32, copy=False)
    assert_frozen_sources_current()
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
        sf.write(temp.name, full, SAMPLE_RATE, subtype="PCM_16")
        with staged_output_path(m4a) as staged_m4a:
            decoded_peak = None
            for loudness_filter in fingerprint_inputs["mastering"]["filters"]:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        temp.name,
                        "-af",
                        loudness_filter,
                        "-ar",
                        str(SAMPLE_RATE),
                        "-ac",
                        "1",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "48k",
                        "-movflags",
                        "+faststart",
                        "-f",
                        "ipod",
                        str(staged_m4a),
                    ],
                    check=True,
                )
                decoded_peak = true_peak_dbfs(staged_m4a)
                if (
                    decoded_peak is not None
                    and decoded_peak <= MAX_RELEASE_TRUE_PEAK_DBFS
                ):
                    break
            if decoded_peak is None:
                raise RuntimeError(
                    f"Could not measure decoded true peak: {staged_m4a}"
                )
            if decoded_peak > MAX_RELEASE_TRUE_PEAK_DBFS:
                raise RuntimeError(
                    f"Decoded true peak {decoded_peak:.1f} dBFS exceeds "
                    f"{MAX_RELEASE_TRUE_PEAK_DBFS:.1f} dBFS: {m4a}"
                )
            media_sha256 = file_sha256(staged_m4a)
            media_bytes = staged_m4a.stat().st_size
            assert_frozen_sources_current()
            os.replace(staged_m4a, m4a)
    assert_frozen_sources_current()
    atomic_write_text(
        timing_path,
        json.dumps(
            {
                "schema": 1,
                "language": language,
                "lang_code": lang_code,
                "dialect": dialect,
                "voice": voice,
                "speed": speed,
                "sample_rate": SAMPLE_RATE,
                "total_duration": len(full) / SAMPLE_RATE,
                "media_sha256": media_sha256,
                "media_bytes": media_bytes,
                "render_fingerprint": fingerprint,
                "render_inputs": fingerprint_inputs,
                "scenes": timings,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )
    write_vtt(captions_path, timings)
    return len(full) / SAMPLE_RATE


def parse_range(value: str, maximum: int) -> set[int]:
    if value == "all":
        return set(range(1, maximum + 1))
    selected = set()
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages",
        default="all",
        help="Comma-separated catalog language ids",
    )
    parser.add_argument(
        "--lessons", default="all", help="Lesson number, ranges, or all"
    )
    parser.add_argument(
        "--voices", default="all", help="Optional comma-separated voice ids"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help=(
            "override the measured per-voice speed profile "
            "(base default: 1.00)"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--repair-peaks",
        action="store_true",
        help="rerender only tracks whose decoded true peak exceeds -1 dBFS",
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be between 1 and 16")

    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KModel, KPipeline

    torch.set_num_threads(args.threads)
    selected_languages = (
        list(LANGUAGES)
        if args.languages == "all"
        else args.languages.split(",")
    )
    catalog_maximum = max(
        lesson["number"]
        for lesson in json.loads(
            (CATALOG / "lessons_en.json").read_text(encoding="utf-8")
        )["lessons"]
    )
    selected_lessons = parse_range(args.lessons, catalog_maximum)
    voice_filter = (
        None if args.voices == "all" else set(args.voices.split(","))
    )
    model = (
        KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(SNAPSHOT / "config.json"),
            model=str(SNAPSHOT / "kokoro-v1_0.pth"),
        )
        .to(args.device)
        .eval()
    )
    rendered = skipped = 0
    start_time = time.monotonic()
    for language in selected_languages:
        lang_code, voices = LANGUAGES[language]
        catalog = json.loads(
            (CATALOG / f"lessons_{language}.json").read_text(encoding="utf-8")
        )
        lessons = [
            lesson
            for lesson in catalog["lessons"]
            if lesson["number"] in selected_lessons
        ]
        pipelines = {}
        for voice in voices:
            if voice_filter is not None and voice not in voice_filter:
                continue
            voice_lang_code = (
                "b"
                if language == "en" and voice.startswith("b")
                else lang_code
            )
            if voice_lang_code not in pipelines:
                pipelines[voice_lang_code] = KPipeline(
                    lang_code=voice_lang_code,
                    repo_id="hexgrad/Kokoro-82M",
                    model=model,
                    device=args.device,
                )
            pipeline = pipelines[voice_lang_code]
            dialect = narration_dialect(language, voice_lang_code, voice)
            speed = resolve_voice_speed(voice, args.speed)
            runtime_identity = synthesis_runtime_identity(
                SNAPSHOT, voice, args.device
            )
            synthesis_voice = pipeline.load_voice(voice)
            runtime_identity["voice_pack"] = dict(
                runtime_identity["voice_pack"]
            )
            loaded_voice_sha256 = hashlib.sha256(
                synthesis_voice.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            if (
                loaded_voice_sha256
                != runtime_identity["voice_pack"]["tensor_sha256"]
            ):
                raise RuntimeError(
                    f"Loaded Kokoro voice does not match {voice}.pt"
                )
            runtime_identity["voice_pack"][
                "loaded_tensor_sha256"
            ] = loaded_voice_sha256
            for lesson in lessons:
                duration = render_track(
                    pipeline,
                    voice,
                    lesson,
                    language,
                    voice_lang_code,
                    dialect,
                    speed,
                    np,
                    sf,
                    args.force,
                    args.repair_peaks,
                    runtime_identity,
                    synthesis_voice,
                )
                if duration < 0:
                    skipped += 1
                else:
                    rendered += 1
                    elapsed = time.monotonic() - start_time
                    print(
                        f"{language}/{voice}/{lesson['id']} {duration:.1f}s "
                        f"rendered={rendered} elapsed={elapsed / 60:.1f}m",
                        flush=True,
                    )
    print(f"complete rendered={rendered} skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
