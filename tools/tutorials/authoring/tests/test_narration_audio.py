"""Synthetic coverage for narration cadence, trimming, and cache identity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from narration_audio import (  # noqa: E402
    CADENCE_PROFILE_PATH,
    ENGLISH_CADENCE_PROFILE,
    ENGLISH_VOICE_SPEEDS,
    SENTENCE_PAUSE_SECONDS,
    assemble_sentence_audio,
    atomic_write_text,
    audio_assembly_config,
    cadence_profile_identity,
    clamp_scene_hold,
    effective_scene_speed,
    freeze_source_snapshot,
    load_module_snapshot,
    narration_dialect,
    render_fingerprint,
    resolve_voice_speed,
    split_english_sentences,
    staged_output_path,
    trim_edge_silence,
)
import render_all_voices as voice_renderer  # noqa: E402
import render_kokoro_narration as legacy_renderer  # noqa: E402
from render_all_voices import (  # noqa: E402
    LANGUAGES,
    prepare_scene_plans,
    track_fingerprint,
    write_vtt,
)


def test_english_sentence_split_retains_display_punctuation() -> None:
    text = (
        "Dr. Rivera uses version 1.5.0.4. "
        "Run preprocessing next! Is the assay ready?"
    )

    assert split_english_sentences(text) == [
        "Dr. Rivera uses version 1.5.0.4.",
        "Run preprocessing next!",
        "Is the assay ready?",
    ]


def test_english_sentence_split_accepts_sentence_initial_spacr() -> None:
    assert split_english_sentences(
        "Review the plan first. spaCR writes only after confirmation."
    ) == [
        "Review the plan first.",
        "spaCR writes only after confirmation.",
    ]


def test_english_sentence_split_treats_model_letter_as_a_complete_label() -> (
    None
):
    assert split_english_sentences(
        "Configure Model A and Model B. Keep shared thresholds matched."
    ) == [
        "Configure Model A and Model B.",
        "Keep shared thresholds matched.",
    ]


def test_vad_trim_protects_quiet_fricative_and_complete_word_ending() -> None:
    sample_rate = 1_000
    source = np.zeros(1_600, dtype=np.float32)
    source[600:650] = 0.002  # quiet fricative before the vowel
    source[650:850] = 0.5
    source[850:900] = 0.003  # quiet final consonant

    trimmed, info = trim_edge_silence(source, np, sample_rate)

    assert info["trim_start_sample"] == 500
    assert info["trim_end_sample"] == 1_080
    assert info["first_active_sample"] == 600
    assert info["last_active_sample"] == 899
    # The renderer may only take a contiguous slice; it must not modify speech.
    np.testing.assert_array_equal(trimmed, source[500:1_080])
    assert trimmed[100] == pytest.approx(0.002)
    assert trimmed[399] == pytest.approx(0.003)


def test_sentence_assembly_has_a_natural_effective_gap() -> None:
    sample_rate = 1_000
    first = np.zeros(1_100, dtype=np.float32)
    second = np.zeros(1_100, dtype=np.float32)
    first[500:600] = 0.4
    second[500:600] = -0.4

    assembled, segments = assemble_sentence_audio(
        [first, second], np, sample_rate
    )

    assert SENTENCE_PAUSE_SECONDS == pytest.approx(0.12)
    assert segments[1]["gap_from_previous"] == pytest.approx(0.40)
    assert 0.35 <= segments[1]["gap_from_previous"] <= 0.50
    assert len(assembled) == 880


def test_scene_hold_clamp_targets_natural_scene_gap() -> None:
    assert clamp_scene_hold(0.10) == pytest.approx(0.32)
    assert clamp_scene_hold(0.45) == pytest.approx(0.45)
    assert clamp_scene_hold(0.90) == pytest.approx(0.52)


def test_dialect_and_per_voice_speed_resolution() -> None:
    assert narration_dialect("en", "a", "af_heart") == "us"
    assert narration_dialect("en", "b", "af_heart") == "uk"
    assert narration_dialect("en", "a", "bf_emma") == "uk"
    assert narration_dialect("fr", "f", "bf_emma") == "us"
    assert resolve_voice_speed("af_heart") == pytest.approx(1.00)
    assert resolve_voice_speed("af_jessica") == pytest.approx(0.88)
    assert resolve_voice_speed("unknown_voice") == pytest.approx(1.00)
    assert resolve_voice_speed("af_jessica", 1.07) == pytest.approx(1.07)


def test_reviewed_scene_cadence_profile_is_bounded_and_complete() -> None:
    profile = ENGLISH_CADENCE_PROFILE
    assert profile["version"] == "2026-08-11-af-heart-scene-cadence-v6"
    assert profile["enabled"] is False
    assert profile["measurement"]["scene_count"] == 479
    assert profile["measurement"]["slow_scene_count"] == 26
    assert profile["measurement"]["fast_scene_count"] == 125
    assert len(profile["scenes"]) == 152
    assert profile["safe_multiplier_bounds"] == {
        "minimum": 0.68,
        "maximum": 1.18,
    }
    identity = cadence_profile_identity()
    profile_bytes = CADENCE_PROFILE_PATH.read_bytes()
    assert identity["enabled"] is False
    assert identity["bytes"] == len(profile_bytes)
    assert identity["sha256"] == hashlib.sha256(profile_bytes).hexdigest()
    for entry in profile["scenes"].values():
        assert 0.68 <= entry["multiplier"] <= 1.18
        expected = entry["baseline_active_span_wpm"] * entry["multiplier"]
        assert entry["expected_active_span_wpm"] == pytest.approx(
            expected, abs=0.002
        )


def test_disabled_cadence_is_uniform_for_all_current_scenes_and_24_voices() -> None:
    catalog = json.loads(
        (ROOT / "catalog" / "lessons_en.json").read_text(encoding="utf-8")
    )
    lessons = catalog["lessons"]
    scene_count = sum(len(lesson["scenes"]) for lesson in lessons)
    voices = LANGUAGES["en"][1]

    # Inventory evolves independently of disabled cadence. Exercise every
    # actual scene rather than pinning the pre-refresh catalog's old size.
    assert lessons and scene_count > 0
    assert len({lesson['id'] for lesson in lessons}) == len(lessons)
    assert len(voices) == 24
    assert len(set(voices)) == 24
    assert set(voices) == set(ENGLISH_VOICE_SPEEDS)

    checked = 0
    for voice in voices:
        base_speed = resolve_voice_speed(voice)
        effective_speeds = set()
        for lesson in lessons:
            for scene_number, _scene in enumerate(lesson["scenes"], start=1):
                effective, multiplier = effective_scene_speed(
                    base_speed, "en", lesson["id"], scene_number
                )
                assert multiplier == pytest.approx(1.0)
                assert effective == pytest.approx(base_speed)
                effective_speeds.add(effective)
                checked += 1
        assert effective_speeds == {base_speed}

    assert checked == len(voices) * scene_count


def test_scene_multiplier_combines_with_voice_base_speed() -> None:
    effective, multiplier = effective_scene_speed(
        0.88, "en", "45_project_browser", 2
    )
    assert multiplier == pytest.approx(1.0)
    assert effective == pytest.approx(0.88)
    assert effective_scene_speed(0.88, "en", "05_home", 4) == (0.88, 1.0)
    assert effective_scene_speed(0.88, "fr", "45_project_browser", 2) == (
        0.88,
        1.0,
    )


def test_renderer_excludes_voices_with_unfixable_internal_pauses() -> None:
    release_voices = LANGUAGES["en"][1]
    assert "af_nicole" not in release_voices
    assert "af_alloy" not in release_voices
    assert "af_kore" not in release_voices
    assert "af_nova" not in release_voices


def test_primary_renderer_fingerprint_records_exact_sentence_plan() -> None:
    lesson = {
        "id": "synthetic",
        "scenes": [
            {
                "narration": (
                    "Classify the assay. Then inspect the classifier."
                ),
                "hold_after": 0.7,
            }
        ],
    }
    plans = prepare_scene_plans(lesson, "en", "us")

    digest, payload = track_fingerprint(
        lesson, "en", "a", "us", "af_heart", 1.0, plans
    )

    assert len(digest) == 64
    assert payload["lang_code"] == "a"
    assert (
        payload["scenes"][0]["narration"] == lesson["scenes"][0]["narration"]
    )
    assert len(payload["scenes"][0]["speech_text"]) == 2
    assert payload["scenes"][0]["resolved_hold"] == pytest.approx(0.52)
    assert payload["scenes"][0]["effective_speed"] > 0
    assert payload["cadence_profile"] == cadence_profile_identity()
    assert payload["cadence_profile"]["enabled"] is False
    assert payload["cadence_profile"]["sha256"] == hashlib.sha256(
        CADENCE_PROFILE_PATH.read_bytes()
    ).hexdigest()
    assert payload["pronunciation"]["version"]
    assert len(payload["pronunciation"]["sha256"]) == 64


def test_cadence_enabled_flag_and_source_hash_change_track_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lesson = {
        "id": "synthetic",
        "scenes": [{"narration": "Inspect the assay.", "hold_after": 0.4}],
    }
    plans = prepare_scene_plans(lesson, "en", "us")
    baseline, _ = track_fingerprint(
        lesson, "en", "a", "us", "af_heart", 1.0, plans
    )
    identity = cadence_profile_identity()

    for field, changed_value in (
        ("enabled", True),
        ("sha256", "0" * 64),
    ):
        changed_identity = dict(identity)
        changed_identity[field] = changed_value
        monkeypatch.setattr(
            voice_renderer,
            "cadence_profile_identity",
            lambda identity=changed_identity: identity,
        )
        changed, payload = track_fingerprint(
            lesson, "en", "a", "us", "af_heart", 1.0, plans
        )
        assert payload["cadence_profile"] == changed_identity
        assert changed != baseline


def test_runtime_identity_changes_track_fingerprint() -> None:
    lesson = {
        "id": "synthetic",
        "scenes": [{"narration": "Inspect the assay.", "hold_after": 0.4}],
    }
    plans = prepare_scene_plans(lesson, "en", "us")
    first, _ = track_fingerprint(
        lesson,
        "en",
        "a",
        "us",
        "af_heart",
        1.0,
        plans,
        {"model": {"weights": {"sha256": "one"}}},
    )
    second, _ = track_fingerprint(
        lesson,
        "en",
        "a",
        "us",
        "af_heart",
        1.0,
        plans,
        {"model": {"weights": {"sha256": "two"}}},
    )
    assert first != second


def test_pronunciation_edit_reuses_audio_only_when_speech_plan_is_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lesson = {
        "id": "synthetic",
        "scenes": [{"narration": "Inspect PyPI.", "hold_after": 0.4}],
    }
    plans = prepare_scene_plans(lesson, "en", "us")
    _, previous = track_fingerprint(
        lesson, "en", "a", "us", "af_heart", 1.0, plans
    )
    changed_identity = dict(previous["pronunciation"])
    changed_identity["sha256"] = "0" * 64
    monkeypatch.setattr(
        voice_renderer, "PRONUNCIATION_IDENTITY", changed_identity
    )
    _, current = track_fingerprint(
        lesson, "en", "a", "us", "af_heart", 1.0, plans
    )

    assert voice_renderer.audio_equivalent_render_inputs(previous, current)
    current["scenes"][0]["speech_text"] = ["a different pronunciation"]
    assert not voice_renderer.audio_equivalent_render_inputs(previous, current)


def test_audio_reuse_keeps_the_runtime_that_actually_rendered_the_file() -> None:
    lesson = {
        "id": "synthetic",
        "scenes": [{"narration": "Inspect the assay.", "hold_after": 0.4}],
    }
    plans = prepare_scene_plans(lesson, "en", "us")
    recorded_runtime = {
        "inference": {"device": "cuda"},
        "model": {"weights": {"sha256": "recorded"}},
    }
    _, previous = track_fingerprint(
        lesson, "en", "a", "us", "af_heart", 1.0, plans, recorded_runtime
    )
    _, current_cpu = track_fingerprint(
        lesson,
        "en",
        "a",
        "us",
        "af_heart",
        1.0,
        plans,
        {"inference": {"device": "cpu"}},
    )
    _, refreshed = track_fingerprint(
        lesson, "en", "a", "us", "af_heart", 1.0, plans, recorded_runtime
    )

    assert not voice_renderer.audio_equivalent_render_inputs(
        previous, current_cpu
    )
    assert voice_renderer.audio_equivalent_render_inputs(previous, refreshed)
    assert refreshed["synthesis_runtime"] == recorded_runtime


def test_lang_code_changes_track_fingerprint() -> None:
    lesson = {
        "id": "synthetic",
        "scenes": [{"narration": "Inspect the assay.", "hold_after": 0.4}],
    }
    plans = prepare_scene_plans(lesson, "en", "us")
    us_code, _ = track_fingerprint(
        lesson, "en", "a", "us", "af_heart", 1.0, plans
    )
    uk_code, payload = track_fingerprint(
        lesson, "en", "b", "us", "af_heart", 1.0, plans
    )

    assert us_code != uk_code
    assert payload["lang_code"] == "b"


def test_source_snapshot_executes_and_fingerprints_the_same_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "profile.py"
    source.write_text('VERSION = "loaded"\nVALUE = 1\n', encoding="utf-8")
    module, snapshot = load_module_snapshot(
        source, "synthetic_profile", "synthetic_profile"
    )
    frozen_identity = snapshot.fingerprint(module.VERSION)

    source.write_text('VERSION = "edited"\nVALUE = 2\n', encoding="utf-8")

    assert module.VERSION == "loaded"
    assert module.VALUE == 1
    assert frozen_identity["sha256"] == snapshot.sha256
    with pytest.raises(RuntimeError, match="changed during rendering"):
        snapshot.assert_current()


def test_source_snapshot_detects_same_size_mutation(tmp_path: Path) -> None:
    source = tmp_path / "profile.json"
    source.write_bytes(b'{"value":1}\n')
    snapshot = freeze_source_snapshot(source, "synthetic_profile")
    source.write_bytes(b'{"value":2}\n')

    with pytest.raises(RuntimeError, match="Restart the renderer"):
        snapshot.assert_current()


def test_fingerprint_changes_for_every_audible_render_input() -> None:
    base = {
        "narration": "Classify the assay.",
        "speech_text": "classify the assay.",
        "dialect": "us",
        "voice": "af_heart",
        "speed": 1.0,
        "assembly": audio_assembly_config(),
        "mastering": {"filter": "loudnorm"},
        "pronunciation": {"version": "v1", "sha256": "abc"},
    }
    baseline = render_fingerprint(base)

    for key, changed in (
        ("speech_text", "classify an assay."),
        ("dialect", "uk"),
        ("voice", "bf_emma"),
        ("speed", 0.92),
        ("pronunciation", {"version": "v2", "sha256": "def"}),
    ):
        variant = dict(base)
        variant[key] = changed
        assert render_fingerprint(variant) != baseline

    assert render_fingerprint(base) == baseline


def test_assembly_identity_has_no_obsolete_waveform_edit_switch() -> None:
    assert set(audio_assembly_config()) == {
        "version",
        "sample_rate",
        "lead_silence_seconds",
        "tail_silence_seconds",
        "sentence_pause_seconds",
        "scene_hold_min_seconds",
        "scene_hold_max_seconds",
        "default_scene_hold_seconds",
        "vad_frame_seconds",
        "vad_peak_ratio",
        "vad_absolute_floor",
    }


def test_vtt_uses_monotonic_sentence_cues_without_changing_text(
    tmp_path: Path,
) -> None:
    output = tmp_path / "captions.vtt"
    write_vtt(
        output,
        [
            {
                "speech_start": 0.0,
                "speech_end": 3.0,
                "text": "First sentence. Second sentence.",
                "sentences": [
                    {
                        "speech_start": 0.0,
                        "speech_end": 1.2,
                        "text": "First sentence.",
                    },
                    {
                        "speech_start": 1.6,
                        "speech_end": 3.0,
                        "text": "Second sentence.",
                    },
                ],
            }
        ],
    )

    content = output.read_text(encoding="utf-8")
    assert "00:00:00.000 --> 00:00:01.200" in content
    assert "00:00:01.600 --> 00:00:03.000" in content
    assert "First sentence.\n" in content
    assert "Second sentence.\n" in content
    assert "First sentence. Second sentence." not in content


def test_non_monotonic_vtt_does_not_replace_existing_sidecar(
    tmp_path: Path,
) -> None:
    output = tmp_path / "captions.vtt"
    atomic_write_text(output, "existing\n")
    with pytest.raises(ValueError, match="Non-monotonic"):
        write_vtt(
            output,
            [
                {"speech_start": 1.0, "speech_end": 2.0, "text": "First."},
                {"speech_start": 1.5, "speech_end": 3.0, "text": "Second."},
            ],
        )
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_staged_media_failure_preserves_previous_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "voice.m4a"
    target.write_bytes(b"known-good")

    with pytest.raises(RuntimeError, match="interrupted"):
        with staged_output_path(target) as staged:
            assert staged.name.endswith(".m4a.part")
            assert not staged.match("*.m4a")
            staged.write_bytes(b"partial")
            raise RuntimeError("interrupted")

    assert target.read_bytes() == b"known-good"
    assert list(tmp_path.iterdir()) == [target]


def test_legacy_renderer_loads_the_fingerprinted_snapshot_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voice_values = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    voice_sha256 = hashlib.sha256(voice_values.tobytes()).hexdigest()
    observed_identity = {}

    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return voice_values

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.device = None
            self.evaluated = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    class FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_voice(self, voice):
            self.loaded_voice = voice
            return FakeTensor()

    def fake_runtime_identity(snapshot, voice, device):
        observed_identity.update(
            {"snapshot": snapshot, "voice": voice, "device": device}
        )
        return {"voice_pack": {"tensor_sha256": voice_sha256}}

    monkeypatch.setattr(
        legacy_renderer,
        "synthesis_runtime_identity",
        fake_runtime_identity,
    )
    snapshot = tmp_path / "snapshot"
    pipeline, synthesis_voice, runtime_identity = (
        legacy_renderer._build_synthesis_stack(
            FakeModel,
            FakePipeline,
            snapshot,
            "af_heart",
            "a",
            "cpu",
        )
    )

    model = pipeline.kwargs["model"]
    assert model.kwargs == {
        "repo_id": "hexgrad/Kokoro-82M",
        "config": str(snapshot.resolve() / "config.json"),
        "model": str(snapshot.resolve() / "kokoro-v1_0.pth"),
    }
    assert model.device == "cpu"
    assert model.evaluated is True
    assert pipeline.kwargs["repo_id"] == "hexgrad/Kokoro-82M"
    assert pipeline.kwargs["lang_code"] == "a"
    assert pipeline.kwargs["device"] == "cpu"
    assert pipeline.loaded_voice == "af_heart"
    assert observed_identity == {
        "snapshot": snapshot.resolve(),
        "voice": "af_heart",
        "device": "cpu",
    }
    assert synthesis_voice.numpy() is voice_values
    assert runtime_identity["voice_pack"]["loaded_tensor_sha256"] == (
        voice_sha256
    )


def test_primary_one_track_wiring_commits_validated_media_and_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import soundfile as sf

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakePipeline:
        def __init__(self):
            self.calls = []

        def __call__(self, text, voice, speed):
            self.calls.append((text, voice, speed))
            samples = np.zeros(4_800, dtype=np.float32)
            samples[600:4_000] = 0.2
            return [
                SimpleNamespace(
                    audio=FakeTensor(samples),
                    phonemes="synthetic phonemes",
                )
            ]

    def fake_encode(command, check):
        assert check is True
        staged = Path(command[-1])
        assert staged.name.endswith(".m4a.part")
        assert command[-3:-1] == ["-f", "ipod"]
        staged.write_bytes(b"m" * 512)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(voice_renderer, "PRODUCTION", tmp_path)
    monkeypatch.setattr(voice_renderer.subprocess, "run", fake_encode)
    monkeypatch.setattr(voice_renderer, "true_peak_dbfs", lambda path: -2.0)
    pipeline = FakePipeline()
    synthesis_voice = object()
    lesson = {
        "id": "synthetic_lesson",
        "scenes": [{"narration": "Inspect the assay.", "hold_after": 0.4}],
    }

    duration = voice_renderer.render_track(
        pipeline,
        "af_heart",
        lesson,
        "en",
        "a",
        "us",
        1.0,
        np,
        sf,
        True,
        False,
        {"model": {"weights": {"sha256": "synthetic"}}},
        synthesis_voice,
    )

    audio_dir = tmp_path / "synthetic_lesson" / "audio" / "en"
    timing = (audio_dir / "af_heart.json").read_text(encoding="utf-8")
    assert duration > 0
    assert (audio_dir / "af_heart.m4a").stat().st_size == 512
    assert '"media_sha256"' in timing
    assert '"synthesis_runtime"' in timing
    assert pipeline.calls[0][1] is synthesis_voice
    assert pipeline.calls[0][2] == pytest.approx(1.0)
    assert not list(tmp_path.rglob("*.part"))
