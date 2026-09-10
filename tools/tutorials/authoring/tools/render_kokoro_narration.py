#!/usr/bin/env python3
"""Render natural sentence-level Kokoro narration, timings, and subtitles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
    "tutorial_pronunciation_snapshot_legacy",
    "pronunciation_profile",
)
PRONUNCIATION_IDENTITY = PRONUNCIATION_SOURCE_SNAPSHOT.fingerprint(
    getattr(pronunciation_profile, "PRONUNCIATION_VERSION", None)
)
assert_pronunciation_safe = pronunciation_profile.assert_pronunciation_safe
spoken_form = pronunciation_profile.spoken_form
LOUDNESS_FILTER = "loudnorm=I=-16:TP=-3:LRA=11"
MASTERING_CONFIG = {
    "filter": LOUDNESS_FILTER,
    "codec": "pcm_s16le",
    "sample_rate": SAMPLE_RATE,
    "channels": 1,
}
KOKORO_REPOSITORY = "hexgrad/Kokoro-82M"
KOKORO_SNAPSHOT_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
DEFAULT_MODEL_CACHE = (
    Path(__file__).resolve().parents[1] / "project" / "kokoro_models"
)


def assert_frozen_sources_current() -> None:
    PRONUNCIATION_SOURCE_SNAPSHOT.assert_current()
    assert_cadence_profile_current()


def _model_snapshot(model_cache: Path) -> Path:
    """Resolve the only Kokoro snapshot permitted for this renderer."""
    return (
        model_cache.resolve()
        / "hub"
        / "models--hexgrad--Kokoro-82M"
        / "snapshots"
        / KOKORO_SNAPSHOT_REVISION
    )


def _build_synthesis_stack(
    model_class,
    pipeline_class,
    snapshot: Path,
    voice: str,
    lang_code: str,
    device: str,
):
    """Load and identify the exact model and voice used for synthesis."""
    snapshot = snapshot.resolve()
    model = (
        model_class(
            repo_id=KOKORO_REPOSITORY,
            config=str(snapshot / "config.json"),
            model=str(snapshot / "kokoro-v1_0.pth"),
        )
        .to(device)
        .eval()
    )
    pipeline = pipeline_class(
        lang_code=lang_code,
        repo_id=KOKORO_REPOSITORY,
        model=model,
        device=device,
    )
    runtime_identity = synthesis_runtime_identity(snapshot, voice, device)
    synthesis_voice = pipeline.load_voice(voice)
    runtime_identity["voice_pack"] = dict(runtime_identity["voice_pack"])
    loaded_voice_sha256 = hashlib.sha256(
        synthesis_voice.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    if loaded_voice_sha256 != runtime_identity["voice_pack"]["tensor_sha256"]:
        raise RuntimeError(f"Loaded Kokoro voice does not match {voice}.pt")
    runtime_identity["voice_pack"][
        "loaded_tensor_sha256"
    ] = loaded_voice_sha256
    return pipeline, synthesis_voice, runtime_identity


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _scene_plans(
    scenes: list[dict],
    language: str,
    dialect: str,
    lesson_id: str,
    base_speed: float,
) -> list[dict]:
    plans = []
    for index, scene in enumerate(scenes, start=1):
        text = scene["narration"].strip()
        display_sentences = split_sentences(text, language)
        if not display_sentences:
            raise ValueError(f"Empty narration in scene {index}")
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
            base_speed, language, lesson_id, index
        )
        plans.append(
            {
                "text": text,
                "sentences": sentences,
                "authored_hold": authored_hold,
                "hold": clamp_scene_hold(authored_hold),
                "speed_multiplier": speed_multiplier,
                "effective_speed": scene_speed,
            }
        )
    return plans


def _render_inputs(
    scenes_path: Path,
    lesson_id: str,
    language: str,
    dialect: str,
    voice: str,
    lang_code: str,
    speed: float,
    plans: list[dict],
    runtime_identity: dict,
) -> dict:
    return {
        "renderer": "render_kokoro_narration-v5",
        "scenes_file": scenes_path.name,
        "lesson": lesson_id,
        "language": language,
        "dialect": dialect,
        "lang_code": lang_code,
        "voice": voice,
        "speed": speed,
        "scenes": [
            {
                "narration": plan["text"],
                "speech_text": [
                    sentence["speech_text"] for sentence in plan["sentences"]
                ],
                "authored_hold": plan["authored_hold"],
                "resolved_hold": plan["hold"],
                "speed_multiplier": plan["speed_multiplier"],
                "effective_speed": plan["effective_speed"],
            }
            for plan in plans
        ],
        "assembly": audio_assembly_config(),
        "cadence_profile": (
            cadence_profile_identity() if language == "en" else None
        ),
        "mastering": MASTERING_CONFIG,
        "pronunciation": PRONUNCIATION_IDENTITY,
        "synthesis_runtime": runtime_identity,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--lang-code", default="a")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help=(
            "override the measured per-voice speed profile "
            "(base default: 1.00)"
        ),
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=DEFAULT_MODEL_CACHE,
    )
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be between 1 and 16")

    # Hugging Face reads these during import. Pin the CLI cache rather than
    # inheriting a caller's unrelated cache while fingerprinting this one.
    os.environ["HF_HOME"] = str(args.model_cache.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KModel, KPipeline

    torch.set_num_threads(args.threads)
    spec = json.loads(args.scenes.read_text(encoding="utf-8"))
    scenes = spec["scenes"]
    lesson_id = str(spec.get("lesson") or args.scenes.parent.name)
    output_dir = args.output_dir
    clips_dir = output_dir / args.voice
    clips_dir.mkdir(parents=True, exist_ok=True)

    speed = resolve_voice_speed(args.voice, args.speed)
    dialect = narration_dialect(args.language, args.lang_code, args.voice)
    assert_frozen_sources_current()
    plans = _scene_plans(scenes, args.language, dialect, lesson_id, speed)
    snapshot = _model_snapshot(args.model_cache)
    pipeline, synthesis_voice, runtime_identity = _build_synthesis_stack(
        KModel,
        KPipeline,
        snapshot,
        args.voice,
        args.lang_code,
        args.device,
    )
    fingerprint_inputs = _render_inputs(
        args.scenes,
        lesson_id,
        args.language,
        dialect,
        args.voice,
        args.lang_code,
        speed,
        plans,
        runtime_identity,
    )
    fingerprint = render_fingerprint(fingerprint_inputs)
    combined_parts = []
    timings = []
    srt_blocks = []
    cursor = 0.0

    for index, plan in enumerate(plans, start=1):
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
                    f"Kokoro returned no audio for scene {index}, "
                    f"sentence {sentence_number}"
                )
            sentence_audio.append(
                np.concatenate(chunks).astype(np.float32, copy=False)
            )
            sentence_phonemes.append(" ".join(phonemes))

        audio, sentence_segments = assemble_sentence_audio(sentence_audio, np)
        peak = float(np.max(np.abs(audio)))
        if peak > 0.98:
            audio = audio * (0.98 / peak)

        clip_path = clips_dir / f"scene_{index:02d}.wav"
        with staged_output_path(clip_path) as staged_clip:
            sf.write(
                staged_clip,
                audio,
                SAMPLE_RATE,
                subtype="PCM_16",
                format="WAV",
            )
            os.replace(staged_clip, clip_path)

        speech_start = cursor
        speech_end = speech_start + len(audio) / SAMPLE_RATE
        hold_after = plan["hold"]
        scene_end = speech_end + hold_after
        silence = np.zeros(
            int(round(hold_after * SAMPLE_RATE)), dtype=np.float32
        )
        combined_parts.extend([audio, silence])
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
                "clip": str(clip_path),
                "text": plan["text"],
                "speech_text": " ".join(
                    sentence["speech_text"] for sentence in plan["sentences"]
                ),
                "phonemes": " ".join(sentence_phonemes),
                "authored_hold_after": plan["authored_hold"],
                "hold_after": hold_after,
                "base_speed": speed,
                "speed_multiplier": plan["speed_multiplier"],
                "effective_speed": plan["effective_speed"],
                "sentences": sentence_timings,
            }
        )
        for sentence in sentence_timings:
            cue_number = len(srt_blocks) + 1
            srt_blocks.append(
                f"{cue_number}\n"
                f"{_srt_timestamp(sentence['speech_start'])} --> "
                f"{_srt_timestamp(sentence['speech_end'])}\n"
                f"{sentence['text']}\n"
            )
        cursor = scene_end

    combined = np.concatenate(combined_parts).astype(np.float32, copy=False)
    narration_path = output_dir / f"narration_{args.voice}.wav"
    mastered_path = output_dir / f"narration_{args.voice}_mastered.wav"
    timings_path = output_dir / f"timings_{args.voice}.json"
    subtitles_path = output_dir / f"subtitles_{args.voice}.srt"
    assert_frozen_sources_current()
    with staged_output_path(narration_path) as staged_narration:
        sf.write(
            staged_narration,
            combined,
            SAMPLE_RATE,
            subtype="PCM_16",
            format="WAV",
        )
        os.replace(staged_narration, narration_path)
    with staged_output_path(mastered_path) as staged_mastered:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(narration_path),
                "-af",
                LOUDNESS_FILTER,
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(staged_mastered),
            ],
            check=True,
        )
        mastered_sha256 = file_sha256(staged_mastered)
        mastered_bytes = staged_mastered.stat().st_size
        assert_frozen_sources_current()
        os.replace(staged_mastered, mastered_path)
    assert_frozen_sources_current()
    atomic_write_text(
        timings_path,
        json.dumps(
            {
                "schema": 1,
                "voice": args.voice,
                "language": args.language,
                "dialect": dialect,
                "lang_code": args.lang_code,
                "speed": speed,
                "sample_rate": SAMPLE_RATE,
                "total_duration": len(combined) / SAMPLE_RATE,
                "media_sha256": mastered_sha256,
                "media_bytes": mastered_bytes,
                "render_fingerprint": fingerprint,
                "render_inputs": fingerprint_inputs,
                "scenes": timings,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )
    atomic_write_text(subtitles_path, "\n".join(srt_blocks))

    print(narration_path)
    print(mastered_path)
    print(timings_path)
    print(subtitles_path)
    print(f"duration={len(combined) / SAMPLE_RATE:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
